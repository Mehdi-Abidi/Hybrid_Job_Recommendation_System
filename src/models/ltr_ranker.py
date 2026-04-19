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
    objective: str = "rank:ndcg"
    n_estimators: int = 200
    learning_rate: float = 0.1
    max_depth: int = 6
    seed: int = 42


# SignalProvider: returns (two_tower, content, collab, popularity) for a user over a list of candidate job_ids.
class SignalProvider:
    def __init__(self, two_tower, content, collab, popularity):
        self.two_tower = two_tower      # TwoTowerTrainer or None
        self.content = content          # ContentBasedRecommender or None
        self.collab = collab            # CollaborativeRecommender or None
        self.popularity = popularity    # PopularityRecommender or None

    def compute(self, user_id: int, candidate_job_ids: list[int]) -> RankingSignals:
        n = len(candidate_job_ids)
        tt = self._two_tower_scores(user_id, candidate_job_ids) if self.two_tower is not None else np.zeros(n, dtype=np.float32)
        ct = self.content.score_pairs(user_id, candidate_job_ids) if (self.content is not None and user_id in self.content._user_index) \
            else np.zeros(n, dtype=np.float32)
        cf = self.collab.score_pairs(user_id, candidate_job_ids) if self.collab is not None \
            else np.full(n, 3.0, dtype=np.float32)
        pop = self.popularity.score_pairs(candidate_job_ids) if self.popularity is not None \
            else np.zeros(n, dtype=np.float32)
        return RankingSignals(two_tower=tt, content=ct, collab=cf, popularity=pop)

    def _two_tower_scores(self, user_id: int, candidate_job_ids: list[int]) -> np.ndarray:
        art = self.two_tower.artifacts
        u_pos = np.where(art.user_ids == user_id)[0]
        if len(u_pos) == 0:
            return np.zeros(len(candidate_job_ids), dtype=np.float32)
        ue = self.two_tower.user_embeddings()[u_pos[0]]
        je = self.two_tower.job_embeddings()
        j_idx = np.array([np.where(art.job_ids == j)[0][0] if j in art.job_ids else -1 for j in candidate_job_ids])
        out = np.zeros(len(candidate_job_ids), dtype=np.float32)
        valid = j_idx >= 0
        out[valid] = je[j_idx[valid]] @ ue
        return out


class LTRRanker:
    def __init__(self, cfg: LTRConfig, signals: SignalProvider):
        self.cfg = cfg
        self.signals = signals
        self.booster: xgb.Booster | None = None
        self._apply_rates: dict[tuple[int, str], float] = {}
        self._jobs: pd.DataFrame | None = None
        self._users: pd.DataFrame | None = None

    # Build a training-ready (X, y, group) from train interactions.
    def _build_training_matrix(self, users: pd.DataFrame, jobs: pd.DataFrame,
                               train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._apply_rates = user_category_apply_rates(train, jobs)
        X_parts, y_parts, group_sizes = [], [], []
        for uid, g in train.sort_values("user_id").groupby("user_id", sort=False):
            cand = g["job_id"].astype(int).tolist()
            labels = g["rating"].astype(float).to_numpy()
            sigs = self.signals.compute(int(uid), cand)
            feats = build_ranking_features(int(uid), cand, users, jobs, sigs, self._apply_rates)
            X_parts.append(feats)
            y_parts.append(labels)
            group_sizes.append(len(cand))
        X = np.vstack(X_parts) if X_parts else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        y = np.concatenate(y_parts) if y_parts else np.zeros(0, dtype=np.float32)
        return X, y, np.array(group_sizes, dtype=np.int64)

    def fit(self, users: pd.DataFrame, jobs: pd.DataFrame, train: pd.DataFrame) -> "LTRRanker":
        self._jobs, self._users = jobs, users
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
        feats = build_ranking_features(int(user_id), candidate_job_ids, self._users, self._jobs, sigs, self._apply_rates)
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
