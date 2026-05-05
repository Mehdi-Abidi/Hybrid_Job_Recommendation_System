"""LambdaMART ranker via XGBoost rank:ndcg.
Input: training user-job pairs grouped by user. Each pair has stage-1 signals + engineered features."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.ranking_features import FEATURE_NAMES, RankingSignals, build_ranking_features, user_category_apply_rates
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class LTRConfig:
    objective: str = "rank:pairwise"
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    seed: int = 42
    n_random_negatives: int = 50
    n_hard_negatives: int = 5


# SignalProvider: returns (two_tower, content, collab, popularity[, bilateral]) for a
# user over a list of candidate job_ids. Bilateral signals (forward score, inverse
# score, sigmoid-product) are filled when a BilateralScorer is supplied; otherwise
# RankingSignals leaves those fields None and downstream feature-builders treat them
# as zero, keeping the LTR feature vector dimensionality stable.
class SignalProvider:
    def __init__(self, two_tower, content, collab, popularity, bilateral=None,
                 hybrid=None, bert4rec=None, salary=None, user_history=None):
        self.two_tower = two_tower      # TwoTowerTrainer or None
        self.content = content          # ContentBasedRecommender or None
        self.collab = collab            # CollaborativeRecommender or None
        self.popularity = popularity    # PopularityRecommender or None
        self.bilateral = bilateral      # BilateralScorer or None
        self.hybrid = hybrid            # HybridRecommender or None — used as a feature, not a ranker
        self.bert4rec = bert4rec        # BERT4RecTrainer or None — sequence-model score
        self.salary = salary            # SalaryPredictor or None — soft salary alignment feature
        self.user_history = user_history or {}   # {user_id: [job_id, ...]} for BERT4Rec
        # Pre-compute id→row dicts AND the full (n_users, d) and (n_jobs, d)
        # embedding matrices once. Previously every call to _two_tower_scores
        # ran a full corpus forward pass through the tower — for a 24k-pair LTR
        # fit over 2,484 users that's 2,484 redundant full forwards. Caching
        # collapses this to two forward passes total.
        if two_tower is not None:
            art = two_tower.artifacts
            self._u_id_to_row = {int(u): i for i, u in enumerate(art.user_ids)}
            self._j_id_to_row = {int(j): i for i, j in enumerate(art.job_ids)}
            self._tt_user_emb = two_tower.user_embeddings()
            self._tt_job_emb = two_tower.job_embeddings()
        else:
            self._u_id_to_row = {}
            self._j_id_to_row = {}
            self._tt_user_emb = None
            self._tt_job_emb = None

    def compute(self, user_id: int, candidate_job_ids: list[int]) -> RankingSignals:
        n = len(candidate_job_ids)
        tt = self._two_tower_scores(user_id, candidate_job_ids) if self.two_tower is not None else np.zeros(n, dtype=np.float32)
        ct = self.content.score_pairs(user_id, candidate_job_ids) if (self.content is not None and user_id in self.content._user_index) \
            else np.zeros(n, dtype=np.float32)
        cf = self.collab.score_pairs(user_id, candidate_job_ids) if self.collab is not None \
            else np.zeros(n, dtype=np.float32)
        pop = self.popularity.score_pairs(candidate_job_ids) if self.popularity is not None \
            else np.zeros(n, dtype=np.float32)
        s_uj = s_ju = bilat = None
        if self.bilateral is not None:
            s_uj, s_ju, bilat = self.bilateral.score(user_id, candidate_job_ids)
        hyb = self._hybrid_scores(user_id, candidate_job_ids) if self.hybrid is not None else None
        b4r = self._bert4rec_scores(user_id, candidate_job_ids) if self.bert4rec is not None else None
        return RankingSignals(two_tower=tt, content=ct, collab=cf, popularity=pop,
                              s_user_to_job=s_uj, s_job_to_user=s_ju, bilateral=bilat,
                              hybrid=hyb, bert4rec=b4r)

    # Hybrid combiner score per candidate, computed only over the requested subset
    # (~candidate count, typically 50-500) instead of the full corpus (~123k). Saves
    # ~30-90 min on the real-data LTR fit.
    def _hybrid_scores(self, user_id: int, candidate_job_ids: list[int]) -> np.ndarray:
        try:
            return np.asarray(
                self.hybrid.score_pairs(int(user_id), [int(j) for j in candidate_job_ids]),
                dtype=np.float32,
            )
        except Exception:
            return np.zeros(len(candidate_job_ids), dtype=np.float32)

    # BERT4Rec next-item score over the user's interaction history, gathered only
    # over the candidate set (logit-space — fine for a tree booster). Items unknown
    # to the trained vocabulary score 0.
    def _bert4rec_scores(self, user_id: int, candidate_job_ids: list[int]) -> np.ndarray:
        history = self.user_history.get(int(user_id), [])
        try:
            return self.bert4rec.score_pairs(history, [int(j) for j in candidate_job_ids])
        except Exception:
            return np.zeros(len(candidate_job_ids), dtype=np.float32)

    def _two_tower_scores(self, user_id: int, candidate_job_ids: list[int]) -> np.ndarray:
        u_row = self._u_id_to_row.get(int(user_id), -1)
        if u_row < 0 or self._tt_user_emb is None:
            return np.zeros(len(candidate_job_ids), dtype=np.float32)
        ue = self._tt_user_emb[u_row]
        j_idx = np.fromiter(
            (self._j_id_to_row.get(int(j), -1) for j in candidate_job_ids),
            dtype=np.int64, count=len(candidate_job_ids),
        )
        out = np.zeros(len(candidate_job_ids), dtype=np.float32)
        valid = j_idx >= 0
        out[valid] = self._tt_job_emb[j_idx[valid]] @ ue
        return out


class LTRRanker:
    def __init__(self, cfg: LTRConfig, signals: SignalProvider):
        self.cfg = cfg
        self.signals = signals
        self.booster: xgb.Booster | None = None
        self._apply_rates: dict[tuple[int, str], float] = {}
        self._jobs: pd.DataFrame | None = None
        self._users: pd.DataFrame | None = None

    # Build (X, y, group) with binary relevance + sampled negatives. For each user:
    # positives = their interacted jobs (label 1); negatives = `n_random_negatives`
    # uniformly sampled unseen jobs + `n_hard_negatives` per positive drawn from the
    # same category (label 0). LambdaRank then learns to rank positives above
    # both classes of negatives, including near-positives (Facebook KDD 2020).
    def _build_training_matrix(self, users: pd.DataFrame, jobs: pd.DataFrame,
                               train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._apply_rates = user_category_apply_rates(train, jobs)
        rng = np.random.default_rng(self.cfg.seed)
        all_job_ids = jobs["job_id"].astype(int).to_numpy()
        has_cat = "category" in jobs.columns
        job_to_cat: dict[int, str] = (
            dict(zip(jobs["job_id"].astype(int), jobs["category"].astype(str)))
            if has_cat else {}
        )
        cat_to_jobs: dict[str, np.ndarray] = (
            {c: jobs.loc[jobs["category"] == c, "job_id"].astype(int).to_numpy()
             for c in jobs["category"].dropna().unique()}
            if has_cat else {}
        )
        # Pre-index once instead of per-call inside build_ranking_features.
        jobs_indexed = jobs.set_index("job_id")

        X_parts, y_parts, group_sizes = [], [], []
        for uid, g in train.sort_values("user_id").groupby("user_id", sort=False):
            positives = g["job_id"].astype(int).to_numpy()
            seen = set(int(j) for j in positives)
            n_pos = len(positives)
            # Random negatives: oversample to handle the seen-set filter, then truncate.
            target_random = self.cfg.n_random_negatives * n_pos
            pool = rng.choice(all_job_ids, size=target_random * 2, replace=True)
            neg_random = np.array(
                [int(j) for j in pool if int(j) not in seen][:target_random],
                dtype=int,
            )
            # Hard negatives: same category as a positive, not seen.
            hard: list[int] = []
            if has_cat:
                for pj in positives:
                    cat_pool = cat_to_jobs.get(job_to_cat.get(int(pj), ""))
                    if cat_pool is None or len(cat_pool) == 0:
                        continue
                    pick = rng.choice(cat_pool, size=min(self.cfg.n_hard_negatives,
                                                        len(cat_pool)), replace=False)
                    hard.extend(int(j) for j in pick if int(j) not in seen)
            cand = np.concatenate(
                [positives, neg_random, np.array(hard, dtype=int)]
            ).astype(int)
            labels = np.concatenate([
                np.ones(n_pos, dtype=np.float32),
                np.zeros(len(neg_random) + len(hard), dtype=np.float32),
            ])
            sigs = self.signals.compute(int(uid), cand.tolist())
            feats = build_ranking_features(int(uid), cand.tolist(), users, jobs_indexed,
                                           sigs, self._apply_rates,
                                           salary_model=self.signals.salary)
            X_parts.append(feats)
            y_parts.append(labels)
            group_sizes.append(len(cand))
        X = np.vstack(X_parts) if X_parts else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        y = np.concatenate(y_parts) if y_parts else np.zeros(0, dtype=np.float32)
        return X, y, np.array(group_sizes, dtype=np.int64)

    def fit(self, users: pd.DataFrame, jobs: pd.DataFrame, train: pd.DataFrame) -> "LTRRanker":
        # Cache the job_id-indexed frame for rank()'s hot path — same hoist as inside
        # _build_training_matrix, but reused at inference too.
        self._jobs = jobs.set_index("job_id") if jobs.index.name != "job_id" else jobs
        self._users = users
        X, y, groups = self._build_training_matrix(users, jobs, train)
        if len(groups) == 0 or X.shape[0] == 0:
            log.warning("LTR: no training pairs; booster not fit.")
            return self
        dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES)
        dtrain.set_group(groups)
        params = {
            "objective": self.cfg.objective, "eta": self.cfg.learning_rate,
            "max_depth": self.cfg.max_depth, "verbosity": 0, "seed": self.cfg.seed,
            "tree_method": "hist",
        }
        self.booster = xgb.train(params, dtrain, num_boost_round=self.cfg.n_estimators)
        log.info("LTR fit: %d pairs, %d groups", X.shape[0], len(groups))
        return self

    # Rank candidate job_ids for a user. Returns [(job_id, score), ...] descending.
    def rank(self, user_id: int, candidate_job_ids: list[int]) -> list[tuple[int, float]]:
        if self.booster is None or not candidate_job_ids:
            return [(int(j), 0.0) for j in candidate_job_ids]
        sigs = self.signals.compute(int(user_id), candidate_job_ids)
        feats = build_ranking_features(int(user_id), candidate_job_ids, self._users,
                                       self._jobs, sigs, self._apply_rates,
                                       salary_model=self.signals.salary)
        d = xgb.DMatrix(feats, feature_names=FEATURE_NAMES)
        scores = self.booster.predict(d)
        order = np.argsort(-scores)
        return [(int(candidate_job_ids[i]), float(scores[i])) for i in order]

    def feature_importance(self) -> dict[str, float]:
        if self.booster is None:
            return {}
        imp = self.booster.get_score(importance_type="gain")
        return {k: float(v) for k, v in imp.items()}

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if self.booster is not None:
            self.booster.save_model(str(path / "ltr.json"))
        with open(path / "ltr_meta.json", "w") as f:
            json.dump({
                "feature_names": FEATURE_NAMES,
                "apply_rates": {f"{k[0]}|{k[1]}": v for k, v in self._apply_rates.items()},
            }, f)

    def load(self, path: Path, users: pd.DataFrame, jobs: pd.DataFrame) -> "LTRRanker":
        self._users, self._jobs = users, jobs
        model_path = path / "ltr.json"
        if model_path.exists():
            self.booster = xgb.Booster()
            self.booster.load_model(str(model_path))
        with open(path / "ltr_meta.json") as f:
            meta = json.load(f)
        self._apply_rates = {tuple(k.split("|", 1)): v for k, v in meta.get("apply_rates", {}).items()}
        # Coerce the parsed key: (user_id:int, category:str)
        self._apply_rates = {(int(k[0]), str(k[1])): v for k, v in self._apply_rates.items()}
        return self
