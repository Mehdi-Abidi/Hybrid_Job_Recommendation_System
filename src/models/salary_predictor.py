"""Salary prediction from job features (category, seniority, location, skills).
Uses gradient boosting. Outputs predicted (salary_min, salary_max) pair from (seniority, skills, location)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.structured_features import CategoricalEncoder, SkillEncoder
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class SalaryPrediction:
    salary_min: float
    salary_max: float
    midpoint: float


class SalaryPredictor:
    def __init__(self, n_estimators: int = 200, max_depth: int = 6, lr: float = 0.1, seed: int = 42):
        self.n_estimators, self.max_depth, self.lr, self.seed = n_estimators, max_depth, lr, seed
        self.booster_min: xgb.Booster | None = None
        self.booster_max: xgb.Booster | None = None
        self.cat_enc: CategoricalEncoder | None = None
        self.sen_enc: CategoricalEncoder | None = None
        self.loc_enc: CategoricalEncoder | None = None
        self.skill_enc: SkillEncoder | None = None

    def _featurize(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        if fit:
            self.cat_enc = CategoricalEncoder().fit(list(df.get("category", pd.Series([""])).astype(str)))
            self.sen_enc = CategoricalEncoder().fit(list(df.get("seniority", pd.Series([""])).astype(str)))
            self.loc_enc = CategoricalEncoder().fit(list(df.get("location", pd.Series([""])).astype(str)))
            self.skill_enc = SkillEncoder().fit(list(df.get("skills", pd.Series([""])).astype(str)))

        cat = self.cat_enc.transform(list(df.get("category", pd.Series([""] * len(df))).astype(str))).reshape(-1, 1)
        sen = self.sen_enc.transform(list(df.get("seniority", pd.Series([""] * len(df))).astype(str))).reshape(-1, 1)
        loc = self.loc_enc.transform(list(df.get("location", pd.Series([""] * len(df))).astype(str))).reshape(-1, 1)
        skill = self.skill_enc.transform(list(df.get("skills", pd.Series([""] * len(df))).astype(str)))
        return np.hstack([cat, sen, loc, skill]).astype(np.float32)

    def fit(self, jobs: pd.DataFrame) -> "SalaryPredictor":
        required = {"salary_min", "salary_max"}
        if not required.issubset(jobs.columns):
            log.warning("SalaryPredictor: missing salary columns; skipping fit.")
            return self
        mask = jobs["salary_min"].notna() & jobs["salary_max"].notna() & (jobs["salary_min"] > 0)
        df = jobs[mask].copy()
        if df.empty:
            log.warning("SalaryPredictor: no valid salary rows."); return self
        X = self._featurize(df, fit=True)
        params = {"objective": "reg:squarederror", "max_depth": self.max_depth,
                  "eta": self.lr, "verbosity": 0, "seed": self.seed, "tree_method": "hist"}
        self.booster_min = xgb.train(params, xgb.DMatrix(X, label=df["salary_min"].to_numpy(np.float32)),
                                     num_boost_round=self.n_estimators)
        self.booster_max = xgb.train(params, xgb.DMatrix(X, label=df["salary_max"].to_numpy(np.float32)),
                                     num_boost_round=self.n_estimators)
        log.info("SalaryPredictor fit on %d jobs", len(df))
        return self

    def predict(self, category: str = "", seniority: str = "", location: str = "",
                skills: str = "") -> SalaryPrediction:
        if self.booster_min is None or self.booster_max is None:
            return SalaryPrediction(0.0, 0.0, 0.0)
        row = pd.DataFrame([{"category": category, "seniority": seniority,
                             "location": location, "skills": skills}])
        X = self._featurize(row, fit=False)
        d = xgb.DMatrix(X)
        lo = float(self.booster_min.predict(d)[0])
        hi = float(self.booster_max.predict(d)[0])
        if hi < lo:
            lo, hi = hi, lo
        return SalaryPrediction(salary_min=lo, salary_max=hi, midpoint=(lo + hi) / 2)

    def save(self, path: Path) -> None:
        import joblib
        path.mkdir(parents=True, exist_ok=True)
        if self.booster_min is not None:
            self.booster_min.save_model(str(path / "salary_min.json"))
            self.booster_max.save_model(str(path / "salary_max.json"))
        joblib.dump({"cat_enc": self.cat_enc, "sen_enc": self.sen_enc,
                     "loc_enc": self.loc_enc, "skill_enc": self.skill_enc},
                    path / "salary_encoders.joblib")

    def load(self, path: Path) -> "SalaryPredictor":
        import joblib
        obj = joblib.load(path / "salary_encoders.joblib")
        self.cat_enc, self.sen_enc, self.loc_enc, self.skill_enc = \
            obj["cat_enc"], obj["sen_enc"], obj["loc_enc"], obj["skill_enc"]
        self.booster_min = xgb.Booster(); self.booster_min.load_model(str(path / "salary_min.json"))
        self.booster_max = xgb.Booster(); self.booster_max.load_model(str(path / "salary_max.json"))
        return self
