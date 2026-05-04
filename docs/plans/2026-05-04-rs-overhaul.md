# Hybrid Job Recommender — End-to-End Overhaul Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Land all Tier-A/B/C fixes plus the 12 quality improvements in a single coherent
training run that produces a production-grade hybrid job recommender backed by:
implicit feedback throughout, a time-based split, FAISS-retrieval → bilateral filter →
LTR with negative sampling → MMR diversity rerank, with BERT4Rec and a learned salary
model wired in as ranking features.

**Architecture:** Keep the four-tier modular structure (data / features / models /
pipeline). Replace synthetic ratings with implicit feedback. Demote the hybrid combiner
to a feature consumed by LTR. Trim tier 2 to BERT4Rec only. Embed categorical features
end-to-end inside the two-tower (no more 10k-dim location one-hot). Cache embedding
matrices in scoring components to eliminate the 2,484× per-user redundant forward pass.
Evaluate the actual production cascade.

**Tech Stack:** Python 3.12, PyTorch, sentence-transformers, scikit-learn, scipy.sparse,
xgboost (LambdaMART + salary), `implicit` (iALS), MLflow, FAISS, pandas, numpy.

---

## Operating principles

- **TDD where it pays:** write a focused test before each non-trivial code change. Tests
  here are correctness probes, not coverage theater. Skip tests for mechanical config
  flips and pure renames.
- **DRY/YAGNI:** no scaffolding for hypothetical futures. If a fix removes code (Mult-VAE,
  DeepFM, LightGCN), delete the files — don't leave them as commented dead weight.
- **Frequent commits:** one commit per task. Each commit is independently revertable.
- **Verify before claim:** every task ends with running the relevant smoke test or
  command and confirming exit/output. No "should work" — only "verified working."
- **One training run:** no intermediate full retrains. We use a tiny smoke fixture
  (50 users / 1000 jobs / 500 interactions) for per-task verification, then run the
  real training once at the end.

---

## Pre-flight: smoke fixture

We need a deterministic small dataset for verifying changes without a 10-minute train.

### Task 0: Smoke fixture builder

**Files:**
- Create: `tests/fixtures/build_smoke.py`
- Create: `tests/conftest.py`

**Step 1: Write `tests/fixtures/build_smoke.py`**

```python
"""Build a tiny deterministic dataset for fast end-to-end smoke tests.
50 users, 1000 jobs, ~500 interactions. Same canonical schema as production."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

SKILLS = ["python", "sql", "java", "react", "aws", "ml", "go", "rust", "tf", "kubernetes"]
CATS = ["software", "data", "devops", "frontend", "ml"]
LOCS = ["NYC", "SF", "Remote", "Berlin", "London", "Austin"]
SEN = ["junior", "mid", "senior", "staff"]


def build(out_dir: Path, n_users: int = 50, n_jobs: int = 1000,
          n_interactions: int = 500, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = pd.DataFrame({
        "job_id": np.arange(n_jobs),
        "title": [f"{c}_engineer_{i}" for i, c in enumerate(rng.choice(CATS, n_jobs))],
        "description": [f"role description {i}" for i in range(n_jobs)],
        "category": rng.choice(CATS, n_jobs),
        "seniority": rng.choice(SEN, n_jobs),
        "location": rng.choice(LOCS, n_jobs),
        "skills": [",".join(rng.choice(SKILLS, rng.integers(2, 5), replace=False)) for _ in range(n_jobs)],
        "salary_min": rng.integers(60_000, 120_000, n_jobs),
        "salary_max": rng.integers(120_000, 200_000, n_jobs),
        "posted_days_ago": rng.integers(0, 60, n_jobs),
    })
    users = pd.DataFrame({
        "user_id": np.arange(n_users),
        "resume_text": [f"user {i} resume" for i in range(n_users)],
        "skills": [",".join(rng.choice(SKILLS, rng.integers(2, 5), replace=False)) for _ in range(n_users)],
        "experience_years": rng.integers(0, 15, n_users),
        "preferred_location": rng.choice(LOCS, n_users),
    })
    inter = pd.DataFrame({
        "user_id": rng.integers(0, n_users, n_interactions),
        "job_id": rng.integers(0, n_jobs, n_interactions),
        "action": rng.choice(["view", "save", "apply"], n_interactions, p=[0.7, 0.2, 0.1]),
        "timestamp_days_ago": rng.integers(0, 90, n_interactions),
    })
    jobs.to_csv(out_dir / "jobs.csv", index=False)
    users.to_csv(out_dir / "users.csv", index=False)
    inter.to_csv(out_dir / "interactions.csv", index=False)


if __name__ == "__main__":
    build(Path("data/smoke"))
```

**Step 2: Write `tests/conftest.py`**

```python
import os
import pytest
from pathlib import Path
from tests.fixtures.build_smoke import build


@pytest.fixture(scope="session")
def smoke_data_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("smoke_raw")
    build(d)
    return d
```

**Step 3: Verify**

```bash
python tests/fixtures/build_smoke.py
ls data/smoke/  # expect jobs.csv users.csv interactions.csv
python -c "import pandas as pd; print(pd.read_csv('data/smoke/jobs.csv').shape, pd.read_csv('data/smoke/users.csv').shape, pd.read_csv('data/smoke/interactions.csv').shape)"
# expect (1000, 9) (50, 5) (500, 4)
```

**Step 4: Commit**

```bash
git add tests/fixtures/build_smoke.py tests/conftest.py
git commit -m "test: add deterministic smoke fixture for fast end-to-end checks"
```

---

# PHASE 1 — Foundation

## Task 1: Time-based split as default

**Why first:** every metric in every later task depends on this. Per-user random split
overstates metrics by 20–50% (Ji et al. TOIS 2023).

**Files:**
- Modify: `config/config.yaml:23-26`
- Inspect: `src/data/preprocessing.py:75-78` (verify the existing `_split_time_based`
  uses sort order correctly)

**Step 1: Audit `_split_time_based`**

Open `src/data/preprocessing.py:75-78`. The current code is:
```python
df = df.sort_values("timestamp_days_ago", ascending=False).reset_index(drop=True)  # oldest first
cut = int(len(df) * (1 - test_size))
return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
```

`timestamp_days_ago` larger = older. `ascending=False` puts largest first = oldest first.
`iloc[:cut]` is the older portion (train), `iloc[cut:]` is the newer portion (test).
**This is correct.** Comment matches behavior. No change needed.

**Step 2: Flip config default**

Edit `config/config.yaml`:
```yaml
split:
  test_size: 0.2
  strategy: time_based  # was per_user — global temporal cutoff prevents leakage
  seed: 42
```

**Step 3: Write a test that verifies no test-row predates a train-row**

Create `tests/data/test_time_split.py`:
```python
from pathlib import Path
import pytest
from config import load_settings
from src.data.preprocessing import DataPreprocessor


def test_time_based_split_has_no_leakage(smoke_data_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Wire smoke data into config paths
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    for f in ("jobs.csv", "users.csv", "interactions.csv"):
        (raw / f).write_bytes((smoke_data_dir / f).read_bytes())
    cfg = load_settings()
    cfg.data.source = "canonical"
    cfg.split.strategy = "time_based"
    data = DataPreprocessor(cfg).run(persist=False)
    # Every test interaction must be at least as recent as the newest train interaction
    assert data.test["timestamp_days_ago"].max() <= data.train["timestamp_days_ago"].min()
```

**Step 4: Run test**

```bash
pytest tests/data/test_time_split.py -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add config/config.yaml tests/data/test_time_split.py
git commit -m "data: default to global time-based split (kills 20-50% leakage bias)"
```

---

## Task 2: B1 — `np.where` lookup loops → dict lookups

**Hot sites:**
- `src/models/two_tower.py:147-148` (per-pair full scan during `train`)
- `src/models/ltr_ranker.py:61` (per-job full scan inside per-user loop)
- `src/models/reciprocal.py:189` (`bilateral._forward_scores`)
- `src/models/two_tower.py` — also `_two_tower_scores` in `ltr_ranker.SignalProvider:55-56`

**Files:**
- Modify: `src/models/two_tower.py`
- Modify: `src/models/ltr_ranker.py`
- Modify: `src/models/reciprocal.py`

**Step 1: Refactor `two_tower.train` (line 147)**

Replace:
```python
u_idx = np.array([np.where(self.artifacts.user_ids == u)[0][0] for u in train_pairs["user_id"]])
j_idx = np.array([np.where(self.artifacts.job_ids == j)[0][0] for j in train_pairs["job_id"]])
```
With:
```python
u_id_to_row = {int(u): i for i, u in enumerate(self.artifacts.user_ids)}
j_id_to_row = {int(j): i for i, j in enumerate(self.artifacts.job_ids)}
u_idx = np.fromiter((u_id_to_row[int(u)] for u in train_pairs["user_id"]),
                    dtype=np.int64, count=len(train_pairs))
j_idx = np.fromiter((j_id_to_row[int(j)] for j in train_pairs["job_id"]),
                    dtype=np.int64, count=len(train_pairs))
```

**Step 2: Refactor `SignalProvider._two_tower_scores` and `LTRRanker._build_training_matrix`**

In `ltr_ranker.py`, replace the per-user `np.where` calls with cached dicts. We'll
combine this with B2 (caching embedding matrices) in Task 7 — for now, just convert
the lookup to dict-based. Add to `SignalProvider.__init__`:
```python
if two_tower is not None:
    art = two_tower.artifacts
    self._u_id_to_row = {int(u): i for i, u in enumerate(art.user_ids)}
    self._j_id_to_row = {int(j): i for i, j in enumerate(art.job_ids)}
```
Replace `_two_tower_scores` body to use the dicts.

**Step 3: Refactor `bilateral._forward_scores` and `reciprocal.score_pairs`**

`reciprocal.py:127-140` already uses `self._user_id_to_row` correctly — leave it.
`reciprocal.BilateralScorer._forward_scores:181-193`: replace the `np.where` with a
precomputed dict on the BilateralScorer (will be replaced again in Task 7). For now:
```python
def _forward_scores(self, user_id, job_ids):
    art = self.fwd.artifacts
    if not hasattr(self, "_u_row"):
        self._u_row = {int(u): i for i, u in enumerate(art.user_ids)}
        self._j_row = {int(j): i for i, j in enumerate(art.job_ids)}
    u_idx = self._u_row.get(int(user_id), -1)
    if u_idx < 0:
        return np.zeros(len(job_ids), dtype=np.float32)
    ue = self.fwd.user_embeddings()[u_idx]
    je = self.fwd.job_embeddings()
    out = np.zeros(len(job_ids), dtype=np.float32)
    for i, jid in enumerate(job_ids):
        row = self._j_row.get(int(jid), -1)
        if row >= 0:
            out[i] = float(je[row] @ ue)
    return out
```

**Step 4: Verify behavior unchanged via existing recommend smoke**

Add `tests/models/test_lookup_invariance.py`:
```python
import numpy as np
import pandas as pd
from src.models.two_tower import TwoTowerTrainer

def test_two_tower_lookup_dict_yields_same_indices():
    user_ids = np.array([10, 11, 12, 13])
    job_ids = np.array([100, 200, 300, 400, 500])
    pairs = pd.DataFrame({"user_id": [11, 13, 11], "job_id": [300, 100, 500]})
    u_id_to_row = {int(u): i for i, u in enumerate(user_ids)}
    j_id_to_row = {int(j): i for i, j in enumerate(job_ids)}
    u_idx_dict = np.fromiter((u_id_to_row[int(u)] for u in pairs["user_id"]),
                             dtype=np.int64, count=len(pairs))
    u_idx_where = np.array([np.where(user_ids == u)[0][0] for u in pairs["user_id"]])
    assert np.array_equal(u_idx_dict, u_idx_where)
```

**Step 5: Run**

```bash
pytest tests/models/test_lookup_invariance.py -v
```

**Step 6: Commit**

```bash
git add src/models/two_tower.py src/models/ltr_ranker.py src/models/reciprocal.py tests/models/test_lookup_invariance.py
git commit -m "perf: replace O(n) np.where lookups with O(1) dict lookups (4 sites)"
```

---

## Task 3: MLflow tracking

**Files:**
- Create: `src/utils/tracking.py`
- Modify: `src/models/train.py`
- Modify: `requirements.txt` (or `pyproject.toml`) — add `mlflow`

**Step 1: Add `mlflow` to deps**

Edit `requirements.txt`: add line `mlflow>=2.10`.

**Step 2: Create `src/utils/tracking.py`**

```python
"""Thin MLflow wrapper. No-op if mlflow is unavailable so tests don't depend on it."""
from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Iterator
import os

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


@contextmanager
def run(experiment: str, run_name: str | None = None) -> Iterator[Any]:
    if not _HAS_MLFLOW or os.getenv("MLFLOW_DISABLE") == "1":
        yield None
        return
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as r:
        yield r


def log_params(d: dict[str, Any]) -> None:
    if _HAS_MLFLOW:
        mlflow.log_params({k: str(v) for k, v in d.items()})


def log_metrics(d: dict[str, float], step: int | None = None) -> None:
    if _HAS_MLFLOW:
        mlflow.log_metrics({k: float(v) for k, v in d.items()}, step=step)


def log_artifact(path: str) -> None:
    if _HAS_MLFLOW:
        mlflow.log_artifact(path)
```

**Step 3: Wire into `train.run`**

Edit `src/models/train.py` `run()`:
```python
from src.utils import tracking

def run(cfg, include_tier2=False):
    with tracking.run("rs-overhaul", run_name="full-train"):
        tracking.log_params({
            "split_strategy": cfg.split.strategy,
            "embedding_model": cfg.models["content_based"]["embedding_model"],
            "ltr_objective": cfg.models["ltr"]["objective"],
            "include_tier2": include_tier2,
            ...
        })
        # existing pipeline body
        ...
        for _, row in report.iterrows():
            tracking.log_metrics({f"{row['model']}_ndcg@10": row["ndcg@10"]})
        tracking.log_artifact(str(_artifacts_dir(cfg)))
```

**Step 4: Smoke run**

```bash
MLFLOW_DISABLE=1 python -c "from src.utils.tracking import run, log_params; \
  with run('test'): log_params({'x': 1})"
```
Expected: no error, no MLflow call.

**Step 5: Commit**

```bash
git add src/utils/tracking.py src/models/train.py requirements.txt
git commit -m "obs: add MLflow tracking with no-op fallback"
```

---

# PHASE 2 — Modeling rewrites

## Task 4: Implicit-feedback CF (SVD → iALS)

**Why:** Synthetic 1/3/5 ratings are an anti-pattern; iALS on binary implicit feedback
is the production CF baseline (LinkedIn, YouTube). `global_mean=1.393` becomes
meaningless once labels are arbitrary.

**Files:**
- Rewrite: `src/models/collaborative.py`
- Modify: `src/models/train.py:fit_classical`
- Modify: `requirements.txt` — add `implicit>=0.7`

**Step 1: Add dep**

Edit `requirements.txt`: add `implicit>=0.7`.

**Step 2: Write the new `collaborative.py`**

```python
"""Implicit ALS collaborative filter over the binary user-job interaction matrix.

Replaces the prior SVD-on-synthetic-ratings approach. Treats every observed
interaction as a positive signal with confidence proportional to interaction
strength, following Hu/Koren/Volinsky 2008. Score for unseen (u,j) is the dot
product of learned user and item factors — not a rating; rank by it directly."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

from src.utils.logging import get_logger

log = get_logger(__name__)


class CollaborativeRecommender:
    """iALS on (user × job) binary interactions. `score_pairs` returns the raw
    factor dot product; values are unbounded reals — meaningful only as a ranking."""

    def __init__(self, n_factors: int = 64, n_epochs: int = 20,
                 reg: float = 0.01, alpha: float = 40.0, seed: int = 42, **_kwargs):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg
        self.alpha = alpha           # confidence scaling for implicit feedback
        self.seed = seed
        self.U: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self._user_index: dict[int, int] = {}
        self._job_index: dict[int, int] = {}
        self._all_job_ids: np.ndarray | None = None
        self._trained_users: set[int] = set()

    def fit(self, train: pd.DataFrame, all_job_ids: np.ndarray,
            all_user_ids: np.ndarray | None = None) -> "CollaborativeRecommender":
        user_ids = np.asarray(all_user_ids) if all_user_ids is not None \
            else np.sort(train["user_id"].unique())
        job_ids = np.asarray(all_job_ids)
        self._user_index = {int(u): i for i, u in enumerate(user_ids)}
        self._job_index = {int(j): i for i, j in enumerate(job_ids)}
        self._all_job_ids = job_ids
        self._trained_users = set(int(u) for u in train["user_id"].unique())

        rows = train["user_id"].map(self._user_index).to_numpy()
        cols = train["job_id"].map(self._job_index).to_numpy()
        mask = (~pd.isna(rows)) & (~pd.isna(cols))
        rows, cols = rows[mask].astype(int), cols[mask].astype(int)
        # Binary positives — confidence = 1 + alpha (handled inside iALS via the data values).
        data = np.ones(len(rows), dtype=np.float32)
        ui_matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(job_ids)))

        model = AlternatingLeastSquares(
            factors=self.n_factors, regularization=self.reg,
            alpha=self.alpha, iterations=self.n_epochs, random_state=self.seed,
            use_gpu=False, calculate_training_loss=False,
        )
        model.fit(ui_matrix, show_progress=False)
        self.U = np.asarray(model.user_factors, dtype=np.float32)
        self.V = np.asarray(model.item_factors, dtype=np.float32)
        log.info("iALS fit: %d users, %d jobs, factors=%d", len(user_ids), len(job_ids), self.n_factors)
        return self

    def score_pairs(self, user_id: int, job_ids: list[int]) -> np.ndarray:
        ui = self._user_index.get(int(user_id))
        if ui is None or self.U is None or self.V is None:
            return np.zeros(len(job_ids), dtype=np.float32)
        cols = np.fromiter((self._job_index.get(int(j), -1) for j in job_ids),
                           dtype=np.int64, count=len(job_ids))
        scores = np.zeros(len(job_ids), dtype=np.float32)
        valid = cols >= 0
        scores[valid] = self.V[cols[valid]] @ self.U[ui]
        return scores

    def recommend(self, user_id: int, k: int = 10,
                  exclude: set[int] | None = None) -> list[tuple[int, float]]:
        if self._all_job_ids is None:
            return []
        ui = self._user_index.get(int(user_id))
        if ui is None or self.U is None or self.V is None:
            return []
        scores = self.V @ self.U[ui]
        if exclude:
            for jid in exclude:
                idx = self._job_index.get(int(jid))
                if idx is not None:
                    scores[idx] = -np.inf
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(self._all_job_ids[i]), float(scores[i])) for i in top]

    def knows_user(self, user_id: int) -> bool:
        return int(user_id) in self._trained_users

    # Compat shim for hybrid: hybrid now passes raw iALS scores through min-max
    # so it doesn't matter what `predict` returns numerically. Keep API stable.
    def predict(self, user_id: int, job_id: int) -> float:
        return float(self.score_pairs(user_id, [job_id])[0])

    @property
    def global_mean(self) -> float:  # legacy compat for hybrid fallback path
        return 0.0

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "ials.npz",
                 U=self.U, V=self.V,
                 user_ids=np.array(list(self._user_index.keys()), dtype=np.int64),
                 job_ids=np.array(self._all_job_ids, dtype=np.int64),
                 trained_users=np.array(list(self._trained_users), dtype=np.int64))

    def load(self, path: Path) -> "CollaborativeRecommender":
        z = np.load(path / "ials.npz")
        self.U, self.V = z["U"], z["V"]
        user_ids = z["user_ids"]
        self._user_index = {int(u): i for i, u in enumerate(user_ids)}
        self._all_job_ids = z["job_ids"]
        self._job_index = {int(j): i for i, j in enumerate(self._all_job_ids)}
        self._trained_users = set(int(u) for u in z["trained_users"])
        return self
```

**Step 3: Update `train.fit_classical`**

`config/config.yaml` collaborative section: add `alpha: 40.0`. Map to constructor:
```python
collab_cfg = mb["collaborative"]
collab = CollaborativeRecommender(
    n_factors=collab_cfg["n_factors"], n_epochs=collab_cfg["n_epochs"],
    reg=collab_cfg["reg"], alpha=collab_cfg.get("alpha", 40.0),
).fit(data.train, ...)
```
Drop the `lr` param at call site.

**Step 4: Smoke test**

```bash
pytest tests/models/ -k collab -v   # add a one-line test that fit+recommend works on smoke
```
Add `tests/models/test_collaborative.py`:
```python
import numpy as np
import pandas as pd
from src.models.collaborative import CollaborativeRecommender

def test_ials_smoke():
    rng = np.random.default_rng(0)
    n_users, n_jobs = 30, 200
    inter = pd.DataFrame({
        "user_id": rng.integers(0, n_users, 300),
        "job_id": rng.integers(0, n_jobs, 300),
        "rating": np.ones(300, dtype=int),  # ignored
    })
    rec = CollaborativeRecommender(n_factors=16, n_epochs=5).fit(
        inter, all_job_ids=np.arange(n_jobs), all_user_ids=np.arange(n_users),
    )
    out = rec.recommend(user_id=5, k=10)
    assert len(out) == 10
    assert all(isinstance(j, int) and isinstance(s, float) for j, s in out)
```

**Step 5: Commit**

```bash
git add src/models/collaborative.py src/models/train.py config/config.yaml requirements.txt tests/models/test_collaborative.py
git commit -m "model(cf): replace SVD-on-synthetic-ratings with iALS on implicit feedback"
```

---

## Task 5: LTR with binary labels + negative sampling

**Why:** All-positives training is incoherent for a pairwise ranker. Add 50 random + 5
hard negatives per positive (Facebook KDD 2020). Switch label semantics to binary.

**Files:**
- Modify: `src/models/ltr_ranker.py`
- Modify: `src/features/ranking_features.py` (no behavioral change, just verify label
  flow)
- Modify: `config/config.yaml` — `ltr.objective: rank:pairwise`

**Step 1: Update objective**

`config/config.yaml`:
```yaml
  ltr:
    objective: rank:pairwise   # LambdaRank — proven > pointwise/listwise for job rec
    n_estimators: 500
    learning_rate: 0.05
    max_depth: 6
    n_random_negatives: 50
    n_hard_negatives: 5
```

**Step 2: Update `LTRConfig`**

```python
@dataclass
class LTRConfig:
    objective: str = "rank:pairwise"
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    seed: int = 42
    n_random_negatives: int = 50
    n_hard_negatives: int = 5
```

**Step 3: Rewrite `_build_training_matrix`**

```python
def _build_training_matrix(self, users, jobs, train):
    self._apply_rates = user_category_apply_rates(train, jobs)
    rng = np.random.default_rng(self.cfg.seed)
    all_job_ids = jobs["job_id"].astype(int).to_numpy()
    job_to_cat = dict(zip(jobs["job_id"].astype(int), jobs.get("category", pd.Series([""] * len(jobs))).astype(str)))
    cat_to_jobs: dict[str, np.ndarray] = {
        c: jobs.loc[jobs["category"] == c, "job_id"].astype(int).to_numpy()
        for c in jobs["category"].dropna().unique()
    } if "category" in jobs.columns else {}

    X_parts, y_parts, group_sizes = [], [], []
    for uid, g in train.sort_values("user_id").groupby("user_id", sort=False):
        positives = g["job_id"].astype(int).to_numpy()
        seen = set(int(j) for j in positives)
        # Random negatives (not seen)
        neg_random = rng.choice(all_job_ids, size=self.cfg.n_random_negatives * len(positives),
                                replace=True)
        neg_random = np.array([j for j in neg_random if int(j) not in seen][: self.cfg.n_random_negatives * len(positives)])
        # Hard negatives: same category as a positive, not seen
        hard = []
        for pj in positives:
            cat = job_to_cat.get(int(pj), "")
            pool = cat_to_jobs.get(cat)
            if pool is None or len(pool) == 0:
                continue
            picks = rng.choice(pool, size=min(self.cfg.n_hard_negatives, len(pool)), replace=False)
            hard.extend(int(j) for j in picks if int(j) not in seen)
        cand = np.concatenate([positives, neg_random, np.array(hard, dtype=int)]).astype(int)
        labels = np.concatenate([
            np.ones(len(positives), dtype=np.float32),
            np.zeros(len(neg_random) + len(hard), dtype=np.float32),
        ])
        sigs = self.signals.compute(int(uid), cand.tolist())
        feats = build_ranking_features(int(uid), cand.tolist(), users, jobs, sigs, self._apply_rates)
        X_parts.append(feats)
        y_parts.append(labels)
        group_sizes.append(len(cand))
    X = np.vstack(X_parts) if X_parts else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.concatenate(y_parts) if y_parts else np.zeros(0, dtype=np.float32)
    return X, y, np.array(group_sizes, dtype=np.int64)
```

**Step 4: Test that negatives outnumber positives ~50:1**

`tests/models/test_ltr_neg_sampling.py`:
```python
def test_ltr_negative_ratio(smoke_data_dir, monkeypatch, tmp_path):
    # Build classical+two-tower on smoke, fit LTR, assert ~55× negatives per positive
    ...
```
(Skip the full test if it's too heavy — minimum: a unit test that asserts the negative
sampling routine produces ~55 candidates per positive on a synthetic frame.)

Minimal unit test:
```python
def test_negative_sample_counts():
    import numpy as np, pandas as pd
    jobs = pd.DataFrame({"job_id": np.arange(100), "category": ["A"] * 50 + ["B"] * 50})
    train = pd.DataFrame({"user_id": [0, 0], "job_id": [3, 60], "rating": [1, 1]})
    from src.models.ltr_ranker import LTRRanker, LTRConfig, SignalProvider
    sp = SignalProvider(two_tower=None, content=None, collab=None, popularity=None)
    ltr = LTRRanker(LTRConfig(n_random_negatives=50, n_hard_negatives=5), sp)
    X, y, groups = ltr._build_training_matrix(
        users=pd.DataFrame({"user_id": [0]}), jobs=jobs, train=train,
    )
    # 2 pos + ~100 random + ~10 hard ≈ 110 ish
    assert groups[0] >= 50
    assert (y == 1).sum() == 2
```

**Step 5: Commit**

```bash
git add src/models/ltr_ranker.py config/config.yaml tests/models/test_ltr_neg_sampling.py
git commit -m "ltr: binary labels + negative sampling (50 random + 5 hard, rank:pairwise)"
```

---

## Task 6: Short-term intent vector

**Why:** Single user vector regresses to mean of all-time history. PinnerSAGE (KDD 2020):
~5–10% engagement lift from session sub-encoder.

**Files:**
- Modify: `src/features/user_profile.py`
- Modify: `src/models/content_based.py` (consume the new combined vector)

**Step 1: Extend `UserProfileBuilder` with short-term**

In `user_profile.py`, add a parallel build that returns a (n_users, 2*dim) matrix —
long-term concat short-term:
```python
def build(self, users, jobs, train, job_embeddings, job_index, *, short_term_n: int = 5):
    n_users, dim = len(users), job_embeddings.shape[1]
    long = np.zeros((n_users, dim), dtype=np.float32)
    short = np.zeros((n_users, dim), dtype=np.float32)
    user_row = {uid: i for i, uid in enumerate(users["user_id"].to_numpy())}
    cold_uids: list[int] = []
    has_time = "timestamp_days_ago" in train.columns
    grouped = train.groupby("user_id")
    interacted = set(grouped.groups.keys())

    for uid, i in user_row.items():
        if uid not in interacted:
            cold_uids.append(uid); continue
        g = grouped.get_group(uid)
        cols = g["job_id"].map(job_index).to_numpy()
        mask = ~pd.isna(cols)
        cols = cols[mask].astype(int)
        ratings = g["rating"].to_numpy(dtype=np.float32)[mask] if "rating" in g.columns \
                  else np.ones(mask.sum(), dtype=np.float32)
        if len(cols) == 0:
            cold_uids.append(uid); continue
        # Long-term: rating-weighted mean of all interacted job vectors.
        w = ratings / max(ratings.sum(), 1e-12)
        long[i] = (job_embeddings[cols] * w[:, None]).sum(axis=0)
        # Short-term: mean of last N (by timestamp_days_ago ascending — small = recent).
        if has_time:
            order = np.argsort(g["timestamp_days_ago"].to_numpy()[mask])[: short_term_n]
            recent_cols = cols[order]
        else:
            recent_cols = cols[-short_term_n:]
        short[i] = job_embeddings[recent_cols].mean(axis=0)

    if cold_uids:
        idx = [user_row[u] for u in cold_uids]
        cold_vecs = self.embedder.encode([user_text(users.iloc[i]) for i in idx], normalize=True)
        long[idx] = cold_vecs
        short[idx] = cold_vecs

    long /= (np.linalg.norm(long, axis=1, keepdims=True) + 1e-12)
    short /= (np.linalg.norm(short, axis=1, keepdims=True) + 1e-12)
    return np.concatenate([long, short], axis=1).astype(np.float32)  # (n_users, 2*dim)
```

**Step 2: Adjust `ContentBasedRecommender`**

The user profile is now `2*dim` but job embeddings are `dim`. Two options:

- (A) Score = 0.5 * (cos(long, job) + cos(short, job)) — simple, no model changes.
- (B) Concatenate `[job_emb, job_emb]` to match — wasteful.

**Choose A.** In `recommend` and `score_pairs`:
```python
def _user_score(self, user_idx, job_emb_subset):
    profile = self.artifacts.user_profiles[user_idx]
    half = profile.shape[0] // 2
    long, short = profile[:half], profile[half:]
    return 0.5 * (job_emb_subset @ long + job_emb_subset @ short)
```
Replace the bare `@` calls in `recommend`, `score_pairs`, `similar_jobs` with this.

**Step 3: Test**

`tests/features/test_user_profile.py`:
```python
def test_short_long_concat_produces_2dim_profile():
    ...  # smoke fit, assert shape == (n_users, 2*dim)
```

**Step 4: Commit**

```bash
git add src/features/user_profile.py src/models/content_based.py tests/features/test_user_profile.py
git commit -m "feat: split user profile into long-term + short-term vectors (PinnerSAGE)"
```

---

## Task 7: Cold-start NER + ontology

**Files:**
- Modify: `src/features/user_profile.py:build_for_new_user`

**Step 1: Wire NER + ontology**

```python
def build_for_new_user(self, resume: str, skills: str, *, ner=None, ontology=None) -> np.ndarray:
    if ner is not None:
        try:
            extracted = ner.extract_skills(resume)  # whatever the NER API returns
            skills = ",".join(set(parse_skills(skills) | set(extracted)))
        except Exception:
            pass
    if ontology is not None:
        skills = ",".join(ontology.expand(parse_skills(skills)))  # expand related skills
    text = f"{resume} Skills: {skills}"
    long = self.embedder.encode([text], normalize=True)[0]
    short = long.copy()
    return np.concatenate([long, short]).astype(np.float32)
```

The `ner` and `ontology` are passed in by the caller (e.g. inference path) — defaults
keep behavior backward-compatible.

**Step 2: Test**

`tests/features/test_cold_start_expansion.py`:
```python
def test_ontology_expands_skills_when_passed():
    class FakeOntology:
        def expand(self, skills): return skills | {"docker"}
    ...
```

**Step 3: Commit**

```bash
git commit -am "feat: cold-start uses ResumeNER + SkillOntology when provided"
```

---

# PHASE 3 — Pipeline correctness

## Task 8: Hybrid → LTR feature

**Files:**
- Modify: `src/features/ranking_features.py` — add `hybrid_score` to `FEATURE_NAMES`
- Modify: `src/models/ltr_ranker.py` — `SignalProvider` accepts `hybrid`
- Modify: `src/models/train.py` — pass hybrid into SignalProvider

**Step 1: Append `hybrid_score` to `FEATURE_NAMES`**

```python
FEATURE_NAMES = [
    ..., "bilateral_score", "hybrid_score",
]
```

**Step 2: Add `hybrid` to `RankingSignals` and to `SignalProvider`**

```python
@dataclass
class RankingSignals:
    ...
    hybrid: np.ndarray | None = None
```

In `SignalProvider`:
```python
def __init__(self, two_tower, content, collab, popularity, bilateral=None, hybrid=None):
    self.hybrid = hybrid
    ...

def compute(self, user_id, candidate_job_ids):
    ...
    h = None
    if self.hybrid is not None:
        # use raw recommend scoring vector — recommend_scores returns the raw score per job_id
        h_full = self.hybrid.recommend(user_id, k=10**9, exclude_seen=False)
        score_map = {j: s for j, s in h_full}
        h = np.array([score_map.get(int(j), 0.0) for j in candidate_job_ids], dtype=np.float32)
    return RankingSignals(..., hybrid=h)
```

**Step 3: Use it in `build_ranking_features`**

```python
hybrid_s = float(signals.hybrid[i]) if signals.hybrid is not None else 0.0
feats[i] = [..., bilat, hybrid_s]
```

**Step 4: Wire from train.py**

```python
ltr = fit_ltr(cfg, data, content, collab, popularity, two_tower, bilateral, hybrid)
# inside fit_ltr:
sp = SignalProvider(two_tower=two_tower, content=content, collab=collab,
                    popularity=popularity, bilateral=bilateral, hybrid=hybrid)
```

**Step 5: Commit**

```bash
git commit -am "ltr: consume hybrid_score as a ranker feature (demoted from parallel ranker)"
```

---

## Task 9: Tier-2 trim (BERT4Rec only) + salary as feature

**Files:**
- Delete: `src/models/deepfm.py`, `src/models/lightgcn.py`, `src/models/mult_vae.py`
- Modify: `src/models/train.py:fit_tier2` — keep only BERT4Rec
- Modify: `src/features/ranking_features.py` — add `bert4rec_score`, replace
  `_salary_alignment` heuristic with model-predicted salary
- Modify: `src/models/ltr_ranker.py:SignalProvider` — accept `bert4rec`, `salary`

**Step 1: Delete unused tier 2**

```bash
git rm src/models/deepfm.py src/models/lightgcn.py src/models/mult_vae.py
```

**Step 2: Slim `fit_tier2`**

```python
def fit_tier2(cfg, data):
    from src.models.bert4rec import BERT4RecTrainer, BERT4RecConfig
    return {"bert4rec": BERT4RecTrainer(BERT4RecConfig()).fit(data.train)}
```

Drop the imports of DeepFM/LightGCN/MultVAE from `train.py`.

**Step 3: Wire BERT4Rec into SignalProvider**

```python
class SignalProvider:
    def __init__(self, ..., bert4rec=None, salary=None, user_history: dict[int, list[int]] | None = None):
        self.bert4rec = bert4rec
        self.salary = salary
        self.user_history = user_history or {}

    def compute(self, user_id, candidate_job_ids):
        ...
        b = None
        if self.bert4rec is not None:
            history = self.user_history.get(int(user_id), [])
            top = self.bert4rec.recommend(history, k=10**9, exclude_seen=False)  # already returns scores
            sm = {j: s for j, s in top}
            b = np.array([sm.get(int(j), 0.0) for j in candidate_job_ids], dtype=np.float32)
        return RankingSignals(..., bert4rec=b, salary_aligned=s_align)
```

**Step 4: Replace heuristic salary with model**

In `ranking_features._salary_alignment`, accept an optional salary model:
```python
def _salary_alignment(user_years, sal_min, sal_max, salary_model=None,
                      job_features: dict | None = None):
    if salary_model is not None and job_features is not None:
        pred = salary_model.predict(**job_features)
        # alignment = how close the user's "expected comp" is to the predicted band
        ...
    # fallback: existing heuristic
```

Keep the heuristic as fallback so the function is single-call.

**Step 5: Update FEATURE_NAMES**

```python
FEATURE_NAMES = [..., "bilateral_score", "hybrid_score", "bert4rec_score",
                 "salary_predicted_alignment"]
```

**Step 6: Commit**

```bash
git add -A
git commit -m "tier2: keep BERT4Rec only; wire salary model + bert4rec score into LTR"
```

---

## Task 10: B2 — cache embedding matrices in SignalProvider + BilateralScorer

**Why:** Currently `SignalProvider._two_tower_scores` calls
`two_tower.user_embeddings()` — a full forward pass — *per user* during LTR fit. Same
in `BilateralScorer._forward_scores`. With 2,484 users, that's 2,484× redundant
forward passes per training run.

**Files:**
- Modify: `src/models/ltr_ranker.py:SignalProvider`
- Modify: `src/models/reciprocal.py:BilateralScorer`

**Step 1: Cache in `SignalProvider.__init__`**

```python
def __init__(self, two_tower, ...):
    ...
    if two_tower is not None:
        self._tt_user_emb = two_tower.user_embeddings()       # full corpus, once
        self._tt_job_emb  = two_tower.job_embeddings()
        art = two_tower.artifacts
        self._u_id_to_row = {int(u): i for i, u in enumerate(art.user_ids)}
        self._j_id_to_row = {int(j): i for i, j in enumerate(art.job_ids)}
    else:
        self._tt_user_emb = self._tt_job_emb = None

def _two_tower_scores(self, user_id, candidate_job_ids):
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
```

**Step 2: Cache in BilateralScorer**

```python
class BilateralScorer:
    def __init__(self, forward_tower, inverse_tower):
        self.fwd = forward_tower
        self.inv = inverse_tower
        # Forward + inverse user/job embeddings — full corpus, computed once.
        self._fwd_u = forward_tower.user_embeddings()
        self._fwd_j = forward_tower.job_embeddings()
        self._inv_u = inverse_tower.user_embeddings()
        self._inv_j = inverse_tower.job_embeddings()
        art = forward_tower.artifacts
        self._u_row = {int(u): i for i, u in enumerate(art.user_ids)}
        self._j_row = {int(j): i for i, j in enumerate(art.job_ids)}

    def score(self, user_id, job_ids):
        u = self._u_row.get(int(user_id), -1)
        if u < 0 or not job_ids:
            empty = np.zeros(len(job_ids), dtype=np.float32)
            return empty, empty.copy(), empty.copy()
        j_idx = np.fromiter((self._j_row.get(int(j), -1) for j in job_ids),
                            dtype=np.int64, count=len(job_ids))
        valid = j_idx >= 0
        fwd = np.zeros(len(job_ids), dtype=np.float32)
        inv = np.zeros(len(job_ids), dtype=np.float32)
        fwd[valid] = self._fwd_j[j_idx[valid]] @ self._fwd_u[u]
        inv[valid] = self._inv_j[j_idx[valid]] @ self._inv_u[u]
        bilat = (_sigmoid(fwd) * _sigmoid(inv)).astype(np.float32)
        return fwd, inv, bilat
```

**Step 3: Test parity**

`tests/models/test_signal_provider_caching.py`: build a tiny two-tower, run scores
through the cached and uncached paths, assert identical output.

**Step 4: Commit**

```bash
git commit -am "perf: cache full embedding matrices in SignalProvider + BilateralScorer"
```

---

## Task 11: Eval the production pipeline

**Files:**
- Modify: `src/models/train.py:evaluate_all`
- Modify: `src/pipeline/multi_stage.py` — already exists, verify it composes correctly

**Step 1: Replace `evaluate_all`**

```python
def evaluate_all(cfg, data, content, collab, popularity, hybrid, two_tower,
                 faiss_idx, ltr, bilateral, bert4rec=None, salary=None,
                 debug_per_model: bool = False):
    from src.pipeline.multi_stage import MultiStagePipeline, PipelineStages
    user_history = data.train.groupby("user_id")["job_id"].apply(
        lambda s: set(int(j) for j in s)).to_dict()
    pipe = MultiStagePipeline(
        PipelineStages(two_tower=two_tower, faiss=faiss_idx, content=content,
                       collab=collab, popularity=popularity, ltr=ltr, llm=None,
                       bilateral=bilateral),
        users=data.users, jobs=data.jobs, user_history=user_history,
        n_retrieve=500, n_rank=20, n_final=10,
    )
    ev = Evaluator(data.test, k_values=cfg.evaluation["k_values"], jobs=data.jobs,
                   positive_rating_threshold=1)  # any interaction = positive
    models = {"production": lambda u, k: [(r.job_id, r.score) for r in pipe.recommend(u)]}
    if debug_per_model:
        models.update({
            "content": lambda u, k: content.recommend(u, k) if u in content._user_index else [],
            "collab": lambda u, k: collab.recommend(u, k),
            "hybrid": lambda u, k: hybrid.recommend(u, k=k),
        })
    return ev.compare_models(models)
```

**Step 2: Update threshold**

`Evaluator.__init__` already takes `positive_rating_threshold`. With binary feedback,
pass `1` (every interaction counts as positive) — default in code is 3, override at
call site.

**Step 3: Commit**

```bash
git commit -am "eval: measure the production cascade (FAISS → bilateral → LTR), per-model is debug-only"
```

---

## Task 12: MMR diversity rerank

**Files:**
- Modify: `src/pipeline/multi_stage.py`

**Step 1: Add MMR step in `_rerank` or as a new stage**

Insert after `_rank` and before `_rerank` (LLM):
```python
def _diversify_mmr(self, ranked, lambda_=0.7):
    """MMR over post-LTR scores using the two-tower job embedding for similarity."""
    if self.s.two_tower is None or len(ranked) <= self.n_final:
        return ranked[: self.n_final]
    art = self.s.two_tower.artifacts
    j_row = {int(j): i for i, j in enumerate(art.job_ids)}
    je = self.s.two_tower.job_embeddings()
    selected: list[tuple[int, float]] = []
    pool = list(ranked)
    while pool and len(selected) < self.n_final:
        if not selected:
            best = max(pool, key=lambda x: x[1])
        else:
            sel_idx = [j_row[j] for j, _ in selected if j in j_row]
            if not sel_idx:
                selected.append(pool.pop(pool.index(max(pool, key=lambda x: x[1]))))
                continue
            sel_emb = je[sel_idx]
            def mmr(jid_score):
                jid, s = jid_score
                row = j_row.get(int(jid), None)
                if row is None: return s
                sim = float(np.max(sel_emb @ je[row]))
                return lambda_ * s - (1 - lambda_) * sim
            best = max(pool, key=mmr)
        selected.append(best); pool.remove(best)
    return selected
```
Wire into `recommend`:
```python
ranked = self._rank(...)
diversified = self._diversify_mmr(ranked)
recs = self._rerank(user_id, diversified)
```

**Step 2: Test**

`tests/pipeline/test_mmr.py`:
```python
def test_mmr_picks_diverse_top():
    # 10 candidates, top 5 by score are near-duplicates; MMR should mix in lower-scored diverse items
    ...
```

**Step 3: Commit**

```bash
git commit -am "pipeline: MMR diversity rerank after LTR (λ=0.7) over two-tower embedding"
```

---

## Task 13: Business rules (filter applied/expired)

**Files:**
- Modify: `src/pipeline/multi_stage.py`

**Step 1: Filter in `recommend`**

```python
def recommend(self, user_id, exclude_seen=True):
    retrieved = self._retrieve(int(user_id))
    if exclude_seen:
        seen = self.user_history.get(int(user_id), set())
        retrieved = [(j, s) for j, s in retrieved if j not in seen]
    # Filter expired jobs
    if "posted_days_ago" in self.jobs.columns:
        max_age = 90  # configurable
        active = set(int(j) for j in self.jobs.loc[
            self.jobs["posted_days_ago"] <= max_age, "job_id"])
        retrieved = [(j, s) for j, s in retrieved if j in active]
    ranked = self._rank(...)
    ...
```

**Step 2: Commit**

```bash
git commit -am "pipeline: filter expired jobs (>90 days) and applied jobs from candidates"
```

---

# PHASE 4 — Memory + feature hygiene

## Task 14: A1+C2 — location hash bucket + state/country, category embedding

**Why:** Location one-hot at 10k+ unique values = ~7 GB feature matrix and overfits on
long-tail. Hash bucketing is Facebook-KDD-2020-recommended for high-card long-tail
categoricals. Category vocab is small (~30) — learned embedding wins.

**Files:**
- Modify: `src/features/structured_features.py`
- Modify: `src/models/two_tower.py`

**Step 1: Add hash bucket helper**

In `structured_features.py`:
```python
class HashBucketEncoder:
    """Hash a string to one of `n_buckets` integer ids. Stable across runs."""
    def __init__(self, n_buckets: int = 256):
        self.n_buckets = n_buckets
    def transform(self, values: list[str]) -> np.ndarray:
        import hashlib
        return np.array([int(hashlib.md5(str(v).encode()).hexdigest()[:8], 16) % self.n_buckets
                         for v in values], dtype=np.int64)


def parse_location(loc: str) -> tuple[str, str, str]:
    """Split 'City, State, Country' -> (city, state, country). Robust to missing parts."""
    parts = [p.strip() for p in str(loc).split(",")]
    while len(parts) < 3: parts.append("")
    return parts[0], parts[1], parts[2]
```

**Step 2: Replace one-hots in `two_tower.build_features`**

```python
N_LOC_BUCKETS = 256
self.loc_hash = HashBucketEncoder(N_LOC_BUCKETS)
self.state_enc = CategoricalEncoder().fit([parse_location(l)[1] for l in jobs["location"]])
self.country_enc = CategoricalEncoder().fit([parse_location(l)[2] for l in jobs["location"]])

job_loc_h = np.eye(N_LOC_BUCKETS, dtype=np.float32)[self.loc_hash.transform(list(jobs["location"]))]
# state and country stay small one-hots
```

For category — store as int and pass through `nn.Embedding` *inside the tower*. This
requires changing `_PairDataset` and `_Tower` signatures. Keep changes scoped:

- `TwoTowerArtifacts` adds `job_cat_id: np.ndarray` and `cat_vocab_size: int`.
- `_PairDataset` returns `(user_feat, job_feat, job_cat_id)` — but easier: keep the
  category id as the **first scalar of `job_feat`** and have `_Tower` slice it off
  and embed:

```python
class _Tower(nn.Module):
    def __init__(self, input_dim, hidden_dims, out_dim, dropout, *, cat_vocab=0, cat_dim=8):
        super().__init__()
        self.cat_vocab = cat_vocab
        self.cat_dim = cat_dim
        if cat_vocab > 0:
            self.cat_emb = nn.Embedding(cat_vocab, cat_dim)
            mlp_in = (input_dim - 1) + cat_dim
        else:
            mlp_in = input_dim
        layers = []
        prev = mlp_in
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]; prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.cat_vocab > 0:
            cat_id = x[:, 0].long()
            rest = x[:, 1:]
            emb = self.cat_emb(cat_id)
            x = torch.cat([rest, emb], dim=-1)
        return F.normalize(self.net(x), dim=-1)
```

`build_features` then prepends `job_cat_id` as float column 0 of `job_feat` (model
casts back to long).

**Step 3: Test**

`tests/models/test_two_tower_features.py`:
```python
def test_two_tower_feature_dim_smaller_with_hash_bucket():
    # build features on smoke; assert job_feat.shape[1] < 1000 (was thousands)
```

**Step 4: Commit**

```bash
git commit -am "feat(twotower): hash-bucket location + learned category embedding (cuts ~7GB)"
```

---

## Task 15: A3 — free MiniLM after last text-encode

**Files:**
- Modify: `src/models/train.py`
- Modify: `src/features/text_features.py`

**Step 1: Add `release` method**

```python
class EmbeddingFeaturizer:
    def release(self):
        self._model = None
        import gc; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass
```

**Step 2: Call in `train.run` after Two-Tower features built**

```python
two_tower, faiss_idx = fit_neural_retrieval(cfg, data, embedder)
embedder.release()
```

**Step 3: Add `gc.collect()` between major fits** (`fit_classical`, after Two-Tower,
after BERT4Rec, after LTR). One line each.

**Step 4: Commit**

```bash
git commit -am "mem: free MiniLM after last text-encode + gc.collect between phases"
```

---

# PHASE 5 — Final verification

## Task 16: Full smoke run end-to-end

**Files:** none — runs `src.models.train` against the smoke dataset.

**Step 1: Wire smoke to config**

```bash
mkdir -p data/raw
cp data/smoke/*.csv data/raw/
# In config.yaml temporarily set data.source: canonical
```

**Step 2: Run training**

```bash
MLFLOW_DISABLE=1 python -m src.models.train --include-tier2 2>&1 | tee /tmp/smoke.log
```

Expected: completes in <2 min, prints final eval DataFrame, no OOM, no NaN losses,
production pipeline metric appears in eval table.

**Step 3: Inspect smoke artifacts**

```bash
ls models_artifacts/
# expect: content_based/ collaborative/ popularity/ two_tower/ faiss/ ltr/ reciprocal/ ontology/ salary/ bert4rec/
```

**Step 4: Commit if anything changed**

```bash
git status
# only new artifacts under models_artifacts/ (gitignored, presumably) - skip commit
```

---

## Task 17: Real training run

**Files:** none — uses real data.

**Step 1: Restore real config**

```yaml
data:
  source: real
split:
  strategy: time_based
```

**Step 2: Run**

```bash
python -m src.models.train --include-tier2 2>&1 | tee logs/full_run_$(date +%F).log
```

**Step 3: Verify metrics persisted to MLflow**

```bash
mlflow ui --backend-store-uri ./mlruns &
# open http://localhost:5000, confirm rs-overhaul experiment has the run with all params + metrics
```

**Step 4: Commit log + final notes**

```bash
git add logs/full_run_*.log
git commit -m "run: end-to-end production training with all overhaul changes"
```

---

# Skipped on purpose (with reasoning)

- **MIRROR-style reciprocal joint loss:** ~3-8% NDCG lift but invasive rewrite. Sigmoid
  product is published baseline. Not worth the risk on a one-shot run.
- **DPP rerank:** MMR is enough; DPP is a v2 squeeze.
- **Calibration (Platt scaling):** matters only for cross-user score comparison; top-K
  per user doesn't need it.
- **Mixed precision (`torch.cuda.amp`):** marginal lift after A1, NaN/scaler debug surface
  not worth one-shot risk.
- **B3 (vectorize Mult-VAE iterrows):** Mult-VAE is deleted in Task 9 — moot. (BERT4Rec's
  sequence-build is already efficient; verified during Task 9.)

---

# Execution order summary

```
P0:  Task 0 (smoke fixture)
P1:  Task 1 (time-based split)
     Task 2 (B1 dict lookups)
     Task 3 (MLflow)
P2:  Task 4 (iALS CF)
     Task 5 (LTR neg sampling + binary labels)
     Task 6 (short-term intent)
     Task 7 (cold-start NER+ontology)
P3:  Task 8 (hybrid → LTR feature)
     Task 9 (tier 2 trim + salary wire-in)
     Task 10 (B2 cache embeddings)
     Task 11 (production pipeline eval)
     Task 12 (MMR rerank)
     Task 13 (business rules)
P4:  Task 14 (A1+C2 location hash + category embed)
     Task 15 (A3 free MiniLM + gc)
P5:  Task 16 (smoke end-to-end)
     Task 17 (real training run)
```

Total tasks: 17. Estimated effort: 15–18 hours of careful work.
