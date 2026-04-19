# Hybrid Job Recommendation System

Multi-stage job recommender demonstrating classical, neural, and LLM-powered techniques across **retrieval → ranking → re-ranking** stages. Built for algorithmic depth as a portfolio project.

## Architecture

```
User request
    ↓
Stage 1 — RETRIEVAL     Two-Tower NN → FAISS ANN           → ~500 candidates
                        (+ content / collab / popularity fallback)
    ↓
Stage 2 — RANKING       LambdaMART (XGBoost rank:ndcg)     → ~20 ranked
                        with engineered cross-features
    ↓
Stage 3 — RE-RANKING    Claude LLM re-order + explanations → top-10
    ↓
              Top-K with per-stage scores + NL explanations
```

## Capabilities

**Classical recommenders**
- Content-based (sentence-transformer embeddings, cosine similarity)
- Collaborative filtering (truncated-SVD matrix factorization)
- Popularity baseline with recency decay
- Tiered hybrid combiner (warm / lukewarm / cold)

**Neural (Tier 1 retrieval + ranking)**
- Two-Tower network (PyTorch, in-batch softmax)
- FAISS ANN index (IVFFlat / Flat fallback)
- LambdaMART ranker with 11 engineered features
- LLM re-ranker (Claude API) with structured-JSON output and graceful fallback

**Neural (Tier 2 advanced)**
- BERT4Rec — bidirectional transformer for sequential recommendation
- DeepFM — factorization machine + MLP over sparse features
- LightGCN — graph neural network on the user-job bipartite graph
- Mult-VAE — variational autoencoder with multinomial likelihood

**Domain (Tier 3)**
- Resume NER (regex + optional spaCy)
- ESCO-inspired skill ontology with alias normalization and relatedness
- Salary prediction (gradient boosting)
- Career-path prediction from apply sequences
- Skill-gap analyzer with ontology-aware "adjacent" skill suggestions
- LLM query understanding (structured filter extraction)

**Evaluation**
- Precision@K, Recall@K, NDCG@K, MAP, coverage, intra-list diversity
- Per-bucket cold-start analysis
- `compare_models()` for A/B comparison

## Setup

```bash
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash / or venv\Scripts\activate on cmd
pip install -r requirements.txt
```

Optional — for LLM re-ranker / query understanding:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Data

`config/config.yaml` controls data source. The default uses the built-in synthetic generator (`src/data/acquire.py`), which produces correlated user-job-interaction triples so models have learnable signal (~36% category match rate vs 10% random baseline).

To use a real Kaggle dataset, set `data.use_synthetic: false` and place CSVs in `data/raw/`:
- `jobs.csv` — `job_id, title, category, seniority, location, skills, description, salary_min, salary_max, posted_days_ago`
- `users.csv` — `user_id, primary_category, seniority, experience_years, preferred_location, skills, resume_text`
- `interactions.csv` — `user_id, job_id, action, timestamp_days_ago` (action ∈ {view, save, apply})

## Train

```bash
# Classical + Two-Tower + FAISS + LTR + Tier 3
python -m src.models.train

# Also fit the Tier 2 advanced neural models (slower)
python -m src.models.train --include-tier2
```

Artifacts are saved under `models_artifacts/`.

## Serve

**FastAPI:**
```bash
uvicorn api.main:app --reload
```

Key endpoints:
- `GET /recommend/{user_id}?model=hybrid|content|collab|popularity` — classical recommendations
- `POST /recommend/new-user` — cold-start via resume + skills
- `GET /recommend/multi-stage/{user_id}` — full pipeline with explanations
- `GET /pipeline/inspect/{user_id}` — per-stage candidate breakdown
- `GET /similar-jobs/{job_id}` — reverse / job-to-job
- `GET /explain/{user_id}/{job_id}` — signal-level explanation
- `POST /query/parse` — free-form search → structured filters
- `POST /salary/predict` — salary range prediction
- `POST /skill-gap` — gap analysis with adjacent-skill suggestions

**Streamlit:**
```bash
streamlit run app/streamlit_app.py
```

UI tabs: Job Seeker · Recruiter · Pipeline Inspector · Tools.

## Test

```bash
pytest tests/ -v
```

Current suite: **56 tests** across preprocessing, classical models, neural retrieval, LTR + pipeline, evaluation, Tier 2 neural, Tier 3 domain, training orchestrator, API routing.

## Repository Layout

```
config/          Typed config loader
src/
  data/          Acquisition + preprocessing
  features/      Text, structured, ranking features
  models/        All recommenders + train orchestrator
  retrieval/     FAISS wrapper
  pipeline/      Multi-stage orchestration
  evaluation/    Metrics + evaluator
  nlp/           Resume NER, query understanding
  ontology/      Skill ontology
  utils/         Shared logger
api/             FastAPI app + schemas
app/             Streamlit UI
tests/           pytest suite
```

## Notable Design Decisions

- **scipy `svds` instead of scikit-surprise** — surprise requires a C compiler on Windows; scipy ships with wheels and `svds` gives a proper truncated-SVD matrix factorization with none of the install friction.
- **In-batch softmax for Two-Tower** — avoids separate negative sampling while still learning discriminative embeddings.
- **LTR on top of pre-computed stage-1 signals** — keeps feature engineering transparent and interpretable (11 named features).
- **LLM re-rank with graceful fallback** — without an API key, the pipeline still produces recommendations from classical + LTR signals.
- **Lazy artifact loading in the API** — `/health` returns fast; the first real request triggers model loading.
