"""End-to-end training orchestrator.

Pipeline:
  1. Preprocess raw data → ProcessedData (train/test, sparse matrix, indices).
  2. Fit classical models: content-based, collaborative, popularity.
  3. Fit hybrid combiner over (1).
  4. Fit Two-Tower + FAISS retrieval.
  5. Fit LTR ranker over stage-1 signals.
  6. (Optional) Fit Tier 2 neural models (BERT4Rec, DeepFM, LightGCN, Mult-VAE).
  7. Fit Tier 3 domain: salary predictor, career path, skill ontology (static).
  8. Evaluate all recommenders on the test holdout.
  9. Save every artifact to {paths.artifacts}/.

Run: `python -m src.models.train` (uses config/config.yaml)."""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from typing import Any
import argparse
import numpy as np
import pandas as pd

from config.settings import Settings, load_settings
from src.data.preprocessing import DataPreprocessor, ProcessedData
from src.features.text_features import EmbeddingFeaturizer
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
from src.models.popularity import PopularityRecommender
from src.models.hybrid import HybridRecommender, hybrid_config_from_settings
from src.models.two_tower import TwoTowerTrainer
from src.models.reciprocal import JobToUserTower, BilateralScorer, ReciprocalConfig
from src.models.ltr_ranker import LTRRanker, LTRConfig, SignalProvider
from src.retrieval.faiss_index import FaissJobIndex
from src.models.bert4rec import BERT4RecTrainer, BERT4RecConfig
from src.models.deepfm import DeepFMRecommender, DeepFMConfig
from src.models.lightgcn import LightGCNRecommender, LightGCNConfig
from src.models.mult_vae import MultVAERecommender, MultVAEConfig
from src.models.salary_predictor import SalaryPredictor
from src.models.career_path import CareerPathPredictor
from src.ontology.skill_ontology import SkillOntology
from src.evaluation.evaluator import Evaluator
from src.utils import tracking
from src.utils.logging import get_logger

log = get_logger(__name__)


def _artifacts_dir(cfg: Settings) -> Path:
    p = cfg.path("artifacts"); p.mkdir(parents=True, exist_ok=True); return p


# Fit classical models + hybrid. Returns (content, collab, popularity, hybrid).
def fit_classical(cfg: Settings, data: ProcessedData, embedder: EmbeddingFeaturizer | None = None):
    mb = cfg.models
    embedder = embedder or EmbeddingFeaturizer(
        model_name=mb["content_based"]["embedding_model"],
        dim=mb["content_based"]["embedding_dim"],
    )
    content = ContentBasedRecommender(embedder=embedder).fit(data.jobs, data.users, data.train)
    collab_cfg = mb["collaborative"]
    collab = CollaborativeRecommender(
        n_factors=collab_cfg["n_factors"], n_epochs=collab_cfg["n_epochs"],
        reg=collab_cfg["reg"], alpha=collab_cfg.get("alpha", 40.0),
    ).fit(data.train, data.jobs["job_id"].to_numpy(), all_user_ids=data.users["user_id"].to_numpy())
    pop = PopularityRecommender(recency_halflife_days=mb["popularity"]["recency_halflife_days"]).fit(
        data.train, data.jobs)
    hybrid = HybridRecommender(content, collab, pop, hybrid_config_from_settings(mb)).fit(
        data.jobs, data.users, data.train)
    return content, collab, pop, hybrid


# Fit Two-Tower + FAISS. Returns (two_tower_trainer, faiss_index).
def fit_neural_retrieval(cfg: Settings, data: ProcessedData, embedder: EmbeddingFeaturizer):
    tt_cfg = cfg.models["two_tower"]
    trainer = TwoTowerTrainer(tt_cfg, text_embedder=embedder)
    trainer.build_features(data.jobs, data.users)
    trainer.train(data.train)
    je = trainer.job_embeddings()
    faiss_idx = FaissJobIndex(
        embedding_dim=je.shape[1],
        index_type=cfg.faiss["index_type"],
        nlist=cfg.faiss["nlist"], nprobe=cfg.faiss["nprobe"],
    ).build(je, trainer.artifacts.job_ids)
    return trainer, faiss_idx


def fit_reciprocal(cfg: Settings, data: ProcessedData, two_tower: TwoTowerTrainer) -> BilateralScorer:
    """Train inverse (job→user) tower on flipped positive pairs and wrap with the
    forward two-tower into a BilateralScorer. Reuses the forward tower's feature
    matrices via shared TwoTowerArtifacts — no recomputation of features."""
    rcfg_dict = cfg.models.get("reciprocal", {})
    rcfg = ReciprocalConfig(
        hidden_dims=tuple(rcfg_dict.get("hidden_dims", [256, 128])),
        embedding_dim=rcfg_dict.get("embedding_dim", 128),
        dropout=rcfg_dict.get("dropout", 0.2),
        epochs=rcfg_dict.get("epochs", 10),
        batch_size=rcfg_dict.get("batch_size", 512),
        lr=rcfg_dict.get("lr", 1e-3),
    )
    inv = JobToUserTower(two_tower.artifacts, rcfg).fit(data.train)
    return BilateralScorer(forward_tower=two_tower, inverse_tower=inv)


def fit_ltr(cfg: Settings, data: ProcessedData, content, collab, popularity, two_tower,
            bilateral: BilateralScorer | None = None) -> LTRRanker:
    sp = SignalProvider(two_tower=two_tower, content=content, collab=collab,
                        popularity=popularity, bilateral=bilateral)
    lcfg = cfg.models["ltr"]
    ltr = LTRRanker(LTRConfig(
        objective=lcfg["objective"], n_estimators=lcfg["n_estimators"],
        learning_rate=lcfg["learning_rate"], max_depth=lcfg["max_depth"],
    ), sp).fit(data.users, data.jobs, data.train)
    return ltr


def fit_tier2(cfg: Settings, data: ProcessedData) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["bert4rec"] = BERT4RecTrainer(BERT4RecConfig()).fit(data.train)
    out["deepfm"] = DeepFMRecommender(DeepFMConfig()).fit(data.users, data.jobs, data.train)
    out["lightgcn"] = LightGCNRecommender(LightGCNConfig()).fit(data.users, data.jobs, data.train)
    out["mult_vae"] = MultVAERecommender(MultVAEConfig()).fit(data.users, data.jobs, data.train)
    return out


def fit_tier3(cfg: Settings, data: ProcessedData) -> dict[str, Any]:
    ontology = SkillOntology()
    # Augment salary training data with Train_rev1 (real UK salary labels) if present.
    salary_jobs = data.jobs
    train_rev1_path = cfg.path("raw") / "Train_rev1.csv"
    if train_rev1_path.exists():
        from src.data.real_data_adapter import adapt_train_rev1
        aux = adapt_train_rev1(train_rev1_path, ontology)
        salary_jobs = pd.concat([data.jobs, aux], ignore_index=True)
    salary = SalaryPredictor().fit(salary_jobs)
    career = CareerPathPredictor().fit(data.train, data.jobs)
    return {"ontology": ontology, "salary": salary, "career": career}


# Save everything we need at inference time.
def save_artifacts(cfg: Settings, *, data: ProcessedData,
                   content, collab, popularity, hybrid,
                   two_tower, faiss_idx, ltr,
                   bilateral: BilateralScorer | None = None,
                   tier2: dict | None = None, tier3: dict | None = None) -> None:
    root = _artifacts_dir(cfg)
    content.save(root / "content_based")
    collab.save(root / "collaborative")
    popularity.save(root / "popularity")
    two_tower.save(root / "two_tower")
    faiss_idx.save(root / "faiss")
    ltr.save(root / "ltr")
    if bilateral is not None:
        bilateral.inv.save(root / "reciprocal")
    if tier3:
        tier3["ontology"].save(root / "ontology")
        tier3["salary"].save(root / "salary")
    log.info("Artifacts saved to %s", root)


def evaluate_all(cfg: Settings, data: ProcessedData, content, collab, popularity, hybrid) -> pd.DataFrame:
    ev = Evaluator(data.test, k_values=cfg.evaluation["k_values"], jobs=data.jobs, positive_rating_threshold=3)
    models = {
        "content": lambda u, k: content.recommend(u, k) if u in content._user_index else [],
        "collab": lambda u, k: collab.recommend(u, k),
        "popularity": lambda u, k: popularity.recommend(k=k),
        "hybrid": lambda u, k: [(j, s) for j, s in hybrid.recommend(u, k=k)],
    }
    return ev.compare_models(models)


def run(cfg: Settings, include_tier2: bool = False) -> dict[str, Any]:
    with tracking.run("rs-overhaul", run_name="full-train"):
        tracking.log_params({
            "split_strategy": cfg.split.strategy,
            "test_size": cfg.split.test_size,
            "embedding_model": cfg.models["content_based"]["embedding_model"],
            "embedding_dim": cfg.models["content_based"]["embedding_dim"],
            "cf_n_factors": cfg.models["collaborative"]["n_factors"],
            "cf_n_epochs": cfg.models["collaborative"]["n_epochs"],
            "two_tower_emb_dim": cfg.models["two_tower"]["embedding_dim"],
            "two_tower_epochs": cfg.models["two_tower"]["epochs"],
            "ltr_objective": cfg.models["ltr"]["objective"],
            "ltr_n_estimators": cfg.models["ltr"]["n_estimators"],
            "include_tier2": include_tier2,
            "data_source": cfg.data.source,
        })
        data = DataPreprocessor(cfg).run(persist=True)
        embedder = EmbeddingFeaturizer(
            model_name=cfg.models["content_based"]["embedding_model"],
            dim=cfg.models["content_based"]["embedding_dim"],
        )
        content, collab, popularity, hybrid = fit_classical(cfg, data, embedder)
        two_tower, faiss_idx = fit_neural_retrieval(cfg, data, embedder)
        bilateral = fit_reciprocal(cfg, data, two_tower)
        ltr = fit_ltr(cfg, data, content, collab, popularity, two_tower, bilateral)
        tier2 = fit_tier2(cfg, data) if include_tier2 else None
        tier3 = fit_tier3(cfg, data)
        save_artifacts(cfg, data=data, content=content, collab=collab, popularity=popularity,
                       hybrid=hybrid, two_tower=two_tower, faiss_idx=faiss_idx, ltr=ltr,
                       bilateral=bilateral, tier2=tier2, tier3=tier3)
        report = evaluate_all(cfg, data, content, collab, popularity, hybrid)
        log.info("Evaluation:\n%s", report.to_string(index=False))
        # Per-model headline metrics into MLflow
        for _, row in report.iterrows():
            name = str(row["model"])
            for col in row.index:
                if col == "model":
                    continue
                try:
                    val = float(row[col])
                except (TypeError, ValueError):
                    continue
                tracking.log_metrics({f"{name}_{col}".replace("@", "_at_"): val})
        tracking.log_artifact(str(_artifacts_dir(cfg)))
        return {"data": data, "content": content, "collab": collab, "popularity": popularity,
                "hybrid": hybrid, "two_tower": two_tower, "faiss": faiss_idx, "ltr": ltr,
                "bilateral": bilateral, "tier2": tier2, "tier3": tier3, "report": report}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-tier2", action="store_true", help="Also fit BERT4Rec/DeepFM/LightGCN/Mult-VAE.")
    args = ap.parse_args()
    cfg = load_settings()
    run(cfg, include_tier2=args.include_tier2)


if __name__ == "__main__":
    main()
