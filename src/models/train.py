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
            bilateral: BilateralScorer | None = None, hybrid=None, bert4rec=None,
            salary=None) -> LTRRanker:
    # User interaction history (chronological, oldest→newest) for BERT4Rec scoring.
    sort_col = "timestamp_days_ago" if "timestamp_days_ago" in data.train.columns else None
    user_history: dict[int, list[int]] = {}
    if sort_col is not None:
        for uid, g in data.train.groupby("user_id"):
            ordered = g.sort_values(sort_col, ascending=False)  # large=old → small=new
            user_history[int(uid)] = [int(j) for j in ordered["job_id"].tolist()]
    else:
        for uid, g in data.train.groupby("user_id"):
            user_history[int(uid)] = [int(j) for j in g["job_id"].tolist()]
    sp = SignalProvider(two_tower=two_tower, content=content, collab=collab,
                        popularity=popularity, bilateral=bilateral,
                        hybrid=hybrid, bert4rec=bert4rec, salary=salary,
                        user_history=user_history)
    lcfg = cfg.models["ltr"]
    ltr = LTRRanker(LTRConfig(
        objective=lcfg["objective"], n_estimators=lcfg["n_estimators"],
        learning_rate=lcfg["learning_rate"], max_depth=lcfg["max_depth"],
        n_random_negatives=lcfg.get("n_random_negatives", 50),
        n_hard_negatives=lcfg.get("n_hard_negatives", 5),
    ), sp).fit(data.users, data.jobs, data.train)
    return ltr


def fit_tier2(cfg: Settings, data: ProcessedData) -> dict[str, Any]:
    """Trim: only BERT4Rec survives. DeepFM/LightGCN/Mult-VAE were trained but never
    consumed downstream — academic theatre. The single sequence model now feeds LTR
    as a feature (LiRank KDD 2024 ships exactly one sequence model + GBDT)."""
    return {"bert4rec": BERT4RecTrainer(BERT4RecConfig()).fit(data.train)}


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


def evaluate_all(cfg: Settings, data: ProcessedData, content, collab, popularity, hybrid,
                 *, two_tower=None, faiss_idx=None, ltr=None, bilateral=None,
                 eval_with_llm: bool = False,
                 eval_llm_sample: int = 200,
                 debug_per_model: bool = True) -> pd.DataFrame:
    """Evaluate the production cascade (FAISS → bilateral → LTR → MMR [→ LLM]) — the
    only metric that actually reflects what we ship. Per-model rows kept for
    debugging so we can see individual stage contributions.

    With binary implicit feedback, every test interaction is a positive
    (positive_rating_threshold=1).

    `eval_with_llm`: when True, the production cascade includes the Groq LLM rerank
    step — the same pipeline served at inference. Off by default for fast/cheap
    eval. Requires GROQ_API_KEY[_2,_3] in env.
    """
    from src.pipeline.multi_stage import MultiStagePipeline, PipelineStages
    from src.models.llm_reranker import LLMReranker, LLMConfig
    user_history_set = data.train.groupby("user_id")["job_id"].apply(
        lambda s: {int(j) for j in s}).to_dict()
    ev = Evaluator(data.test, k_values=cfg.evaluation["k_values"], jobs=data.jobs,
                   positive_rating_threshold=1)
    rows: list[pd.DataFrame] = []
    # Production cascade — only available once the full stack is fit.
    if two_tower is not None and faiss_idx is not None and ltr is not None:
        # Always evaluate the no-LLM cascade on the full test set — fast, free,
        # gives the headline metric. The LLM rerank's job is per-query polish,
        # not metric movement, and Groq's free-tier rate limit makes full-set
        # LLM eval impractical (~28 min best case, regularly 429s out).
        pipe_nollm = MultiStagePipeline(
            PipelineStages(two_tower=two_tower, faiss=faiss_idx, content=content,
                           collab=collab, popularity=popularity, ltr=ltr, llm=None,
                           bilateral=bilateral),
            users=data.users, jobs=data.jobs, user_history=user_history_set,
            n_retrieve=500, n_rank=20, n_final=10,
        )
        models_full: dict[str, Any] = {
            "production": lambda u, k: [
                (r.job_id, r.score) for r in pipe_nollm.recommend(u, exclude_seen=True)
            ]
        }
        if debug_per_model:
            models_full.update({
                "content": lambda u, k: content.recommend(u, k) if u in content._user_index else [],
                "collab": lambda u, k: collab.recommend(u, k),
                "popularity": lambda u, k: popularity.recommend(k=k),
                "hybrid": lambda u, k: [(j, s) for j, s in hybrid.recommend(u, k=k)],
            })
        rows.append(ev.compare_models(models_full))

        # Optional LLM-augmented cascade — run on a sampled subset of test users
        # to estimate the LLM rerank's NDCG impact without burning rate limits.
        if eval_with_llm:
            llm_cfg = cfg.models.get("llm", {})
            llm = LLMReranker(LLMConfig(
                model=llm_cfg.get("model", "llama-3.1-8b-instant"),
                rerank_top_k=llm_cfg.get("rerank_top_k", 20),
                output_top_n=llm_cfg.get("output_top_n", 10),
                max_tokens=llm_cfg.get("max_tokens", 4096),
            ))
            pipe_llm = MultiStagePipeline(
                PipelineStages(two_tower=two_tower, faiss=faiss_idx, content=content,
                               collab=collab, popularity=popularity, ltr=ltr, llm=llm,
                               bilateral=bilateral),
                users=data.users, jobs=data.jobs, user_history=user_history_set,
                n_retrieve=500, n_rank=20, n_final=10,
            )
            test_users = list(data.test["user_id"].unique())
            rng = np.random.default_rng(cfg.split.seed)
            sample = rng.choice(test_users,
                                size=min(eval_llm_sample, len(test_users)),
                                replace=False).tolist()
            log.info("Evaluating LLM cascade on %d/%d sampled users",
                     len(sample), len(test_users))
            llm_models = {
                "production_with_llm": lambda u, k: [
                    (r.job_id, r.score) for r in pipe_llm.recommend(u, exclude_seen=True)
                ],
            }
            rows.append(ev.compare_models(llm_models, user_ids=[int(u) for u in sample]))
    elif debug_per_model:
        # No production cascade — just per-model debug.
        models = {
            "content": lambda u, k: content.recommend(u, k) if u in content._user_index else [],
            "collab": lambda u, k: collab.recommend(u, k),
            "popularity": lambda u, k: popularity.recommend(k=k),
            "hybrid": lambda u, k: [(j, s) for j, s in hybrid.recommend(u, k=k)],
        }
        rows.append(ev.compare_models(models))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run(cfg: Settings, include_tier2: bool = True, eval_with_llm: bool = False,
        eval_llm_sample: int = 200, checkpoint_dir: str | None = None) -> dict[str, Any]:
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
            "eval_with_llm": eval_with_llm,
            "data_source": cfg.data.source,
        })
        data = DataPreprocessor(cfg).run(persist=True)
        embedder = EmbeddingFeaturizer(
            model_name=cfg.models["content_based"]["embedding_model"],
            dim=cfg.models["content_based"]["embedding_dim"],
        )
        import gc

        def _free_between_phases():
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # Incremental checkpointing — saves each component to `checkpoint_dir`
        # immediately after it's fit, so a Colab disconnect mid-training loses
        # only the in-progress phase. checkpoint_dir is typically a Drive path
        # like /content/drive/MyDrive/rs_checkpoints.
        ckpt_root = Path(checkpoint_dir) if checkpoint_dir else None
        if ckpt_root is not None:
            ckpt_root.mkdir(parents=True, exist_ok=True)
            log.info("Checkpointing intermediate artifacts to %s", ckpt_root)

        def _ckpt(name: str, save_fn):
            if ckpt_root is None:
                return
            try:
                save_fn(ckpt_root / name)
                log.info("Checkpointed %s -> %s", name, ckpt_root / name)
            except Exception as e:
                log.warning("Checkpoint of %s failed: %s", name, e)

        content, collab, popularity, hybrid = fit_classical(cfg, data, embedder)
        _ckpt("content_based", content.save)
        _ckpt("collaborative", collab.save)
        _ckpt("popularity", popularity.save)
        _free_between_phases()
        two_tower, faiss_idx = fit_neural_retrieval(cfg, data, embedder)
        _ckpt("two_tower", two_tower.save)
        _ckpt("faiss", faiss_idx.save)
        # MiniLM has done its job (text encoding for content + two-tower) — drop it.
        # Frees ~120MB CPU and any cached GPU activations before bilateral / tier2 fit.
        embedder.release()
        _free_between_phases()
        bilateral = fit_reciprocal(cfg, data, two_tower)
        _ckpt("reciprocal", bilateral.inv.save)
        _free_between_phases()
        # Tier 2 (BERT4Rec) and tier 3 (salary) train first so LTR can consume them.
        tier2 = fit_tier2(cfg, data) if include_tier2 else None
        if tier2 and "bert4rec" in tier2:
            _ckpt("bert4rec", tier2["bert4rec"].save)
        _free_between_phases()
        tier3 = fit_tier3(cfg, data)
        _ckpt("ontology", tier3["ontology"].save)
        _ckpt("salary", tier3["salary"].save)
        _free_between_phases()
        ltr = fit_ltr(cfg, data, content, collab, popularity, two_tower, bilateral,
                      hybrid=hybrid, bert4rec=(tier2 or {}).get("bert4rec"),
                      salary=(tier3 or {}).get("salary"))
        _ckpt("ltr", ltr.save)
        _free_between_phases()
        save_artifacts(cfg, data=data, content=content, collab=collab, popularity=popularity,
                       hybrid=hybrid, two_tower=two_tower, faiss_idx=faiss_idx, ltr=ltr,
                       bilateral=bilateral, tier2=tier2, tier3=tier3)
        report = evaluate_all(cfg, data, content, collab, popularity, hybrid,
                              two_tower=two_tower, faiss_idx=faiss_idx, ltr=ltr,
                              bilateral=bilateral, eval_with_llm=eval_with_llm,
                              eval_llm_sample=eval_llm_sample)
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
    ap.add_argument("--no-tier2", action="store_true",
                    help="Skip BERT4Rec sequence model (faster but loses one LTR feature).")
    ap.add_argument("--eval-with-llm", action="store_true",
                    help="Also evaluate the LLM-augmented production cascade on a sampled "
                         "subset (--eval-llm-sample) of test users. Requires GROQ_API_KEY[_2,_3] "
                         "in env. The non-LLM production cascade is always evaluated on the "
                         "full test set.")
    ap.add_argument("--eval-llm-sample", type=int, default=200,
                    help="When --eval-with-llm is set, evaluate LLM cascade on this many "
                         "randomly-sampled test users (default 200 — ±0.02 NDCG CI, "
                         "fits comfortably under Groq free-tier rate limits).")
    ap.add_argument("--checkpoint-dir", type=str, default=None,
                    help="If set, save each model to this directory immediately after "
                         "it's fit. Use a Drive path on Colab so a disconnect mid-train "
                         "doesn't cost you the whole run.")
    args = ap.parse_args()
    cfg = load_settings()
    run(cfg, include_tier2=not args.no_tier2, eval_with_llm=args.eval_with_llm,
        eval_llm_sample=args.eval_llm_sample, checkpoint_dir=args.checkpoint_dir)


if __name__ == "__main__":
    main()
