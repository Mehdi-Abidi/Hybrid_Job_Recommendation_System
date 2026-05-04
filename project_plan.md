# Hybrid Job Recommendation System — Project Plan

## 1. Problem Statement

### The Problem
Every day, millions of job seekers scroll through hundreds of irrelevant listings while recruiters sift through thousands of unqualified applications. Both sides waste enormous time. The right jobs exist for the right people — they just can't find each other efficiently.

### Why Job Recommendation?
We chose job recommendation over common alternatives (movie/music recommendation) for several reasons:

- **Real-world impact**: This is a billion-dollar problem that companies like LinkedIn, Indeed, and Glassdoor invest heavily in solving
- **Rich feature space**: Jobs involve both structured data (skills, experience, location, salary) and unstructured data (job descriptions, resumes) — making it technically interesting
- **Natural cold-start problem**: New users sign up daily with no history, and new jobs are posted constantly — the system must handle both gracefully
- **Multi-stakeholder**: Unlike movies (one user, one recommendation), jobs serve two parties — job seekers looking for roles AND recruiters looking for candidates
- **Portfolio differentiation**: Every student builds a Netflix clone. Job matching is harder, more impressive, and demonstrates real product thinking

---

## 2. System Architecture

### High-Level Overview

```
┌──────────────┐     ┌──────────────┐
│  Streamlit   │────▶│   FastAPI    │
│  (Demo UI)   │◀────│   (Backend)  │
└──────────────┘     └──────┬───────┘
                            │
                    ┌───────┼───────┐
                    ▼       ▼       ▼
               Content   Collab  Popularity
               Model     Model    Model
                    │       │       │
                    └───┬───┘───────┘
                        ▼
                  Hybrid Combiner
                        │
                  ┌─────┼─────┐
                  ▼           ▼
            Preprocessed   Trained
            Data (CSV)     Models (pkl)
```

**Justification**: A layered architecture separates concerns cleanly — data processing, model training, API serving, and UI are all independent. This means we can swap out models, change the UI, or retrain without touching other parts. FastAPI sits between the UI and models so the recommendation logic is reusable (any frontend or external system can call the API).

### Project Structure

```
RS proj/
├── config/                    # Configuration files
│   ├── config.yaml            # Hyperparameters, paths, thresholds
│   └── settings.py            # Python dataclass that loads config.yaml
├── data/
│   ├── raw/                   # Original downloaded datasets
│   └── processed/             # Cleaned, split, ready-to-use data
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   └── 02_model_comparison.ipynb  # Side-by-side model evaluation
├── src/
│   ├── data/
│   │   ├── acquire.py         # Dataset download/generation
│   │   └── preprocessing.py   # Cleaning, splitting, matrix building
│   ├── features/
│   │   ├── text_features.py   # TF-IDF and embedding featurizers
│   │   ├── user_profile.py    # Build user representations
│   │   └── structured_features.py  # Encode skills, location, experience
│   ├── models/
│   │   ├── content_based.py   # Content-based recommender
│   │   ├── collaborative.py   # Collaborative filtering (SVD)
│   │   ├── hybrid.py          # Hybrid combiner (the main model)
│   │   ├── popularity.py      # Popularity baseline/fallback
│   │   └── train.py           # Orchestration: preprocess → train → evaluate → save
│   ├── evaluation/
│   │   ├── metrics.py         # Precision@K, NDCG@K, coverage, etc.
│   │   └── evaluator.py       # Run and compare models
│   └── utils/                 # Shared helpers
├── api/
│   ├── main.py                # FastAPI endpoints
│   └── schemas.py             # Request/response Pydantic models
├── app/
│   └── streamlit_app.py       # Demo web interface
├── tests/                     # Unit and integration tests
├── requirements.txt
└── README.md
```

**Justification**: This mirrors production ML project conventions (data → features → models → evaluation → serving). Each module has a single responsibility, making it easy to test, debug, and extend independently.

---

## 3. The Three Models

We use three separate recommendation "brains," each with different strengths. They are combined into a hybrid model that adapts based on how much we know about the user.

### 3.1 Content-Based Model — "Match the description"

**What it does**: Compares the *meaning* of job descriptions against a user's profile/resume to find semantic matches.

**How it works**:
1. Every job description is converted into a **numerical fingerprint** — a vector of 384 numbers — using a pre-trained language model (Sentence-Transformers, specifically `all-MiniLM-L6-v2`)
2. The user's profile (resume text, skills, past job interactions) is converted into the same kind of fingerprint
3. We compute **cosine similarity** between the user's vector and every job's vector
4. Higher similarity = better match. Return the top-N jobs

**Why sentence embeddings over simple keyword matching?**
- Keywords miss synonyms: "Python developer" and "software engineer (Python)" are the same role but share few exact words
- Embeddings understand *meaning* — they know "machine learning" and "deep learning" are related, even though the words differ
- TF-IDF is also included as a fast fallback and for interpretability (showing *which* keywords matched)

**Strengths**:
- Works immediately for new users — just needs a resume or skill list
- Transparent — we can explain "your Python skills matched this job's requirements"

**Weaknesses**:
- Only recommends jobs similar to what you already describe — no serendipity
- Can't leverage what other similar users found useful

**Key class**: `ContentBasedRecommender` in `src/models/content_based.py`
- `fit()` — builds job embeddings and user profiles
- `recommend(user_id, n=10)` — cosine similarity, returns top-N
- `recommend_for_new_user(resume, skills)` — cold-start path, builds profile on the fly
- `similar_jobs(job_id, n=10)` — job-to-job similarity (useful for "more like this")

---

### 3.2 Collaborative Filtering — "People like you applied here"

**What it does**: Finds patterns in *who applied where* to predict what you'd be interested in, without looking at job descriptions at all.

**How it works**:
1. Build a giant matrix: rows = users, columns = jobs, values = interaction strength
   - Viewed a job = 1, Saved = 3, Applied = 5
2. This matrix is >99% empty (sparse) — nobody interacts with every job
3. Use **SVD (Singular Value Decomposition)** to factorize this matrix into two smaller matrices:
   - A "user factors" matrix (what kind of jobs does each user gravitate toward?)
   - An "item factors" matrix (what kind of users does each job attract?)
4. Multiply these factors to **predict the missing values** — i.e., estimate how much User X would like unseen Job Y
5. Recommend the jobs with the highest predicted scores

**Why SVD?**
- It's the gold standard for matrix factorization in recommender systems (won the Netflix Prize)
- Handles sparse data well — learns latent patterns like "this user prefers remote roles" or "this job attracts senior engineers" without anyone explicitly labeling those patterns
- The Surprise library provides a well-tested, efficient implementation
- 100 latent factors, 20 training epochs — a good balance of accuracy and speed

**Strengths**:
- Discovers unexpected but relevant jobs — things you'd never search for but users like you loved
- Gets better as more interaction data accumulates

**Weaknesses**:
- Completely useless for new users (no history = no factors). This is the **cold-start problem**
- Can't explain *why* — the latent factors are abstract numbers, not interpretable features

**Key class**: `CollaborativeRecommender` in `src/models/collaborative.py`
- `fit(interactions_df)` — trains SVD on the interaction matrix
- `recommend(user_id, candidate_job_ids, n=10)` — predict ratings for candidates, return top-N
- `predict(user_id, job_id)` — single score prediction

---

### 3.3 Popularity Model — "What's trending"

**What it does**: Recommends the most-applied-to jobs, optionally filtered by category. It's the "bestseller list" of jobs.

**How it works**:
1. Count weighted interactions per job (views × 1 + saves × 3 + applications × 5)
2. Apply **recency decay** — a job posted yesterday with 50 applications ranks higher than one from 3 months ago with 200 (because jobs expire)
3. Optionally filter by category — if we know you're in "Data Science," show popular Data Science jobs, not popular Marketing jobs

**Why include such a simple model?**
- **It's the safety net.** When a brand new user shows up with no resume, no skills entered, and zero history, both the content-based and collaborative models have nothing to work with. Showing "popular jobs in your area" is infinitely better than showing nothing or random noise.
- **It's a baseline.** Any good recommender must beat popularity. If our fancy hybrid can't outperform "just show the popular stuff," something is wrong.
- It's computationally trivial and always available.

**When it's used**: Only as a fallback for cold-start users with zero information, or blended in at low weight for lukewarm users.

**Key class**: `PopularityRecommender` in `src/models/popularity.py`
- `fit(interactions_df, jobs_df)` — compute popularity scores with recency decay
- `recommend(n=10, category=None)` — global or category-filtered top-N

---

## 4. The Hybrid Combiner — The Core of the System

**What it does**: Intelligently blends all three models based on how much we know about the user.

### Decision Logic

```
User Request
     │
     ▼
┌─────────────────────┐
│  How much do we know │
│  about this user?    │
└─────────┬───────────┘
          │
    ┌─────┼──────────┐
    ▼     ▼          ▼
  WARM  LUKEWARM    COLD
  ≥5      1-4        0
 actions  actions   actions
    │     │          │
    ▼     ▼          ▼
  40%    80%       100%
 content content   content
    +     +       (from resume)
  60%    20%         +
 collab  collab     30%
                  popularity
    │     │          │
    └─────┼──────────┘
          ▼
   Normalize scores
   (put on same scale)
          │
          ▼
   Merge & Rank top 10
          │
          ▼
   Return with explanations
```

### Why these weights?

- **Warm users (≥5 interactions)**: We have enough history for collaborative filtering to be reliable. Give it 60% weight because it captures patterns content-matching can't. Content still gets 40% to keep recommendations grounded in actual skill/description relevance.

- **Lukewarm users (1-4 interactions)**: Collaborative filtering with so few data points is noisy and unreliable. Lean heavily on content (80%) which works well with even a basic profile, but give collab 20% to start learning.

- **Cold users (0 interactions)**: Collaborative filtering is impossible. Use content-based from whatever profile information exists (resume, skills), boosted by 30% popularity to surface generally desirable jobs.

### Score Normalization

Content-based scores are cosine similarities (range 0 to 1). Collaborative scores are predicted ratings (range 1 to 5). You can't just add them — a collab score of 3.5 would drown out a content score of 0.85. We apply **min-max normalization** to put both on a 0-1 scale before combining.

### Explainability

The hybrid model includes an `explain(user_id, job_id)` function that returns:
- Which of the user's skills matched the job
- Similar jobs the user previously applied to
- Score breakdown (how much came from content vs. collab vs. popularity)

**Justification**: Explainability builds trust. "We recommended this because of your Python and SQL skills" is far more useful than a black-box score. Real job platforms (LinkedIn) do this too.

**Key class**: `HybridRecommender` in `src/models/hybrid.py`

---

## 5. Feature Engineering

### Why it matters
Raw data (job descriptions, skill lists) can't be fed directly into models. Feature engineering converts them into numerical representations the models can compute with.

### Text Features (`src/features/text_features.py`)

| Method | Output | Purpose |
|--------|--------|---------|
| TF-IDF | Sparse vector (5000 dims) | Fast keyword-level matching, interpretable |
| Sentence Embeddings | Dense vector (384 dims) | Semantic meaning, handles synonyms |

**Justification for both**: TF-IDF tells us *which specific words* matched (useful for explanations). Embeddings understand *meaning* (useful for accuracy). They complement each other.

### User Profiles (`src/features/user_profile.py`)

For users with history: **weighted average** of the embeddings of jobs they interacted with. A job they applied to (weight=5) matters more than one they just viewed (weight=1).

For new users: directly encode their resume/skills text into an embedding.

### Structured Features (`src/features/structured_features.py`)

- **Skills**: Multi-label binarization (one column per known skill, 0 or 1)
- **Location**: One-hot encoding of normalized city/state
- **Experience**: Binned into levels (entry: 0-2yr, junior: 3-5yr, mid: 6-10yr, senior: 10+yr)

---

## 6. Data

### Dataset
**Primary**: CareerBuilder dataset from Kaggle (~350K job postings with titles, descriptions, locations, and user application data)

**Fallback**: Synthetic interaction generator — creates users with skill vectors and simulates applications based on skill overlap + noise. This ensures the project works even without Kaggle API access.

### Preprocessing Pipeline (`src/data/preprocessing.py`)

1. **Clean jobs**: Remove duplicates, strip HTML from descriptions, normalize locations, extract salary ranges
2. **Clean users**: Parse skill lists, normalize experience years, standardize education levels
3. **Clean interactions**: Remove orphaned records, deduplicate, assign implicit ratings (view=1, save=3, apply=5)
4. **Build interaction matrix**: Sparse user×job matrix for collaborative filtering
5. **Train/test split**: Time-based if timestamps available, otherwise random per-user split (80/20)

**Justification for implicit ratings**: Real job platforms don't have explicit 1-5 star ratings. Users express preference through actions — applying signals much stronger interest than just viewing. The 1/3/5 weighting reflects this gradient.

---

## 7. Evaluation Framework

### Metrics (`src/evaluation/metrics.py`)

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| Precision@K | Of K recommended jobs, how many were relevant? | Measures recommendation quality |
| Recall@K | Of all relevant jobs, how many appeared in top K? | Measures coverage of relevant items |
| NDCG@K | Are the best matches ranked at the top? | Measures ranking quality (position matters) |
| MAP | Average precision across all relevant items | Single-number summary of ranking quality |
| Coverage | % of total jobs that appear in any recommendation | Detects if the system only recommends popular items |
| Diversity | Average dissimilarity among recommended items | Detects if recommendations are too narrow |

**Justification**: Accuracy alone (Precision/Recall) isn't enough. A system could achieve high precision by recommending the same 50 popular jobs to everyone — Coverage and Diversity catch this failure mode.

### Evaluation Strategy (`src/evaluation/evaluator.py`)

- **Model comparison**: Side-by-side metrics for content-based, collaborative, popularity, and hybrid
- **Cold-start analysis**: Metrics broken down by user activity level (0, 1-4, 5-20, 20+ interactions) — the hybrid should gracefully degrade rather than collapse for low-activity users
- **The thesis**: The hybrid model should outperform each individual model. If it doesn't, the combination weights need tuning.

---

## 8. API Layer

### FastAPI Backend (`api/main.py`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/recommend/{user_id}` | GET | Get top-N job recommendations for existing user |
| `/recommend/new-user` | POST | Get recommendations from resume/skills (cold-start) |
| `/similar-jobs/{job_id}` | GET | Find jobs similar to a given job |
| `/explain/{user_id}/{job_id}` | GET | Get explanation for why a job was recommended |
| `/health` | GET | Health check |

**Justification for FastAPI**: Lightweight, async, auto-generates API documentation (Swagger UI), and has native Pydantic integration for request/response validation. It's the standard for serving ML models in Python.

### Request/Response Schemas (`api/schemas.py`)

Pydantic models ensure type safety and clear API contracts:
- `UserProfile`: resume_text, skills, preferred_location, experience_years
- `JobSummary`: job_id, title, company, location, score
- `RecommendationResponse`: user_id, list of JobSummary, strategy used
- `ExplanationResponse`: matching_skills, similar_applied_jobs, score_breakdown

---

## 9. Demo UI

### Streamlit App (`app/streamlit_app.py`)

Three tabs serving different stakeholders:

1. **Job Seeker View**: Select an existing user or enter skills/resume text. See top-10 recommended jobs as cards showing title, company, location, match score, and an expandable "Why this job?" explanation.

2. **Recruiter View**: Select a job posting, see top candidate users. This is **reverse recommendation** — the same content similarity engine running from job-to-user direction instead of user-to-job.

3. **Model Explorer**: Side-by-side comparison of what each model (content, collab, hybrid) recommends for the same user. Shows evaluation metrics dashboard with bar charts.

**Justification for Streamlit**: Fast to build, Python-native (no frontend framework needed), interactive widgets out of the box, and sufficient for a demo/portfolio project. The Recruiter View adds a second dimension that most student projects don't have.

---

## 10. Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Language | Python 3.10+ | Standard for ML/data science |
| Data processing | Pandas, NumPy, SciPy | Industry standard, mature |
| Text features | Scikit-learn (TF-IDF), Sentence-Transformers | TF-IDF for speed, embeddings for accuracy |
| Collaborative filtering | Surprise (SVD) | Purpose-built for recommender systems, well-tested |
| API | FastAPI + Uvicorn | Fast, async, auto-docs, Pydantic integration |
| UI | Streamlit | Rapid prototyping, Python-native |
| Testing | Pytest | Standard Python testing framework |
| Visualization | Matplotlib, Seaborn | EDA and evaluation plots |
| Config | PyYAML | Simple, human-readable configuration |

---

## 11. Implementation Timeline

| Days | Phase | Deliverable |
|------|-------|-------------|
| 1 | Setup | Project structure, config, dependencies |
| 2-3 | Data | Acquisition, preprocessing pipeline |
| 4 | EDA | Exploratory notebook with visualizations |
| 5-6 | Features | TF-IDF, embeddings, user profiles, structured features |
| 7-8 | Content Model | Working content-based recommender |
| 9-10 | Collab Model | SVD collaborative filtering + popularity baseline |
| 11-12 | Hybrid + Eval | Combined model, full evaluation framework |
| 13 | API | FastAPI serving layer |
| 14 | UI | Streamlit demo with 3 tabs |
| 15 | Polish | Tests, training script, README, documentation |

---

## 12. Verification Plan

1. **Training pipeline**: `python -m src.models.train` completes without errors, prints evaluation metrics for all models
2. **Tests**: `pytest tests/` — all pass (preprocessing edge cases, model output shapes, metric correctness, API responses)
3. **API**: `uvicorn api.main:app` → `GET /recommend/1` returns JSON with 10 job recommendations
4. **UI**: `streamlit run app/streamlit_app.py` → loads, shows recommendations, explanations work
5. **Model quality**: Hybrid outperforms individual models on Precision@10 and NDCG@10 in the comparison notebook
6. **Cold-start**: New users with just a skill list receive reasonable, category-relevant recommendations

---

# PART II — Advanced Enhancements (Algorithmic Depth)

The baseline above produces a working classical hybrid recommender. The enhancements below transform it into an **advanced multi-stage production-style pipeline** mirroring how YouTube, LinkedIn, and Meta architect real recommendation systems. Scalability is intentionally out of scope (1-2 users max) — focus is purely algorithmic and domain depth.

> **Deferred work**: Contextual bandits, neural network fine-tuning, MLOps infrastructure (MLflow, A/B testing, drift monitoring, CI/CD), UX polish (production frontend, conversational interface), data infrastructure upgrades (live scraping, vector DB replacement), fairness/ethics metrics, and research-grade extensions (RL, causal evaluation, multi-task, meta-learning, federated learning) have been intentionally excluded from current scope. The full catalog with per-item rationale lives in `future_considerations.md`.

## 13. The New Architecture — Multi-Stage Pipeline

Instead of a single hybrid model producing recommendations, the system becomes a **funnel** with specialized stages:

```
User Request
    │
    ▼
┌──────────────────────────────────┐
│ STAGE 1: RETRIEVAL               │
│  Two-Tower Neural Network        │
│  + FAISS ANN Index               │  → ~500 candidates
│  (+ classical hybrid fallback)   │
└───────────────┬──────────────────┘
                ▼
┌──────────────────────────────────┐
│ STAGE 2: RANKING                 │
│  LambdaMART (XGBoost rank:ndcg)  │  → top 50 ranked
│  + DeepFM feature interactions   │
│  + Skill-ontology match features │
└───────────────┬──────────────────┘
                ▼
┌──────────────────────────────────┐
│ STAGE 3: RE-RANKING              │
│  LLM (Claude) semantic re-order  │  → top 10 + explanations
│  + BERT4Rec sequence prediction  │
└───────────────┬──────────────────┘
                ▼
┌──────────────────────────────────┐
│ STAGE 4: ENRICHMENT              │
│  Salary prediction per job       │
│  Career path suggestions         │
│  Skill-gap analysis              │
│  Bi-directional recruiter scores │
└───────────────┬──────────────────┘
                ▼
         Final recommendations
         with rich context
```

**Justification for multi-stage**: Classical hybrid does everything in one pass, limiting both quality and interpretability. Multi-stage separates concerns — retrieval cares about recall (don't miss good candidates), ranking cares about precision (order them well), re-ranking adds sophistication (LLM reasoning + exploration). This is how every modern production RS works.

---

## 14. Tier 1 Enhancements — Core Algorithmic Depth

### 14.1 Two-Tower Neural Network (PyTorch)

**What it does**: Replaces SVD's fixed latent factors with a **learned joint embedding space** where users and jobs live as comparable vectors. A user tower maps user features → 128-dim vector; a job tower maps job features → 128-dim vector. Recommendation becomes a dot product.

**How it integrates**:
- Replaces the SVD model as the primary retrieval mechanism
- Still uses collaborative signals (application data for positive pairs) but adds content signals (features) inside the same model
- Solves the cold-start problem elegantly — a new user with just skills/resume still gets a meaningful embedding because the tower is feature-based, not ID-based
- Output embeddings get indexed into FAISS for fast retrieval

**Why it's better than SVD**:
- SVD learns from interactions *only*; two-tower learns from interactions *and* features
- SVD can't score new users/jobs (ID-based); two-tower can (feature-based)
- SVD's latent factors are fixed-dim linear combinations; two-tower is a full neural network capturing non-linear patterns

**File**: `src/models/two_tower.py` with classes `UserTower`, `JobTower`, `TwoTowerModel`, `TwoTowerTrainer`. Training uses contrastive loss with in-batch negatives.

---

### 14.2 FAISS Vector Index

**What it does**: Efficient approximate nearest neighbor (ANN) search over the 128-dim embedding space produced by the two-tower model.

**How it integrates**:
- After two-tower training, pre-compute embeddings for all ~350K jobs once
- Build a FAISS `IVFFlat` index over those embeddings
- At recommendation time: compute user embedding → FAISS search → top-500 candidate jobs in milliseconds
- Replaces brute-force cosine similarity (would be O(N) per user)

**Why include it even at 1-2 users scale**:
- Demonstrates awareness of retrieval-stage bottlenecks
- Shows understanding of production RS architecture
- Enables sub-linear search — 350K jobs brute-forced would be slow even for one user

**File**: `src/retrieval/faiss_index.py` — `FaissJobIndex` class with `build()`, `search()`, `save()`, `load()`.

---

### 14.3 Learning-to-Rank (LambdaMART via XGBoost)

**What it does**: A **second-stage ranker** that takes the ~500 candidates from Stage 1 and rescores them using rich engineered features, optimizing NDCG directly.

**How it integrates**:
- Runs *after* two-tower + FAISS retrieval
- Consumes the candidate set + engineered pairwise features:
  - Two-tower dot product score
  - Content similarity score
  - Classical collab predicted rating
  - Popularity score
  - Skill overlap count (from skill ontology — see 16.2)
  - Location match flag
  - Experience-seniority match flag
  - Salary-expectation alignment
  - Days since posting
  - User's historical apply rate in this category
- Trained with XGBoost `objective='rank:ndcg'`, grouped by user
- Outputs the top 50 ranked candidates

**Why LambdaMART over a simple MLP**:
- LambdaMART is the gold-standard pointwise ranker — won multiple LTR competitions
- Directly optimizes NDCG (the metric we evaluate on)
- Feature importance provides interpretability (which signals drive recommendations?)
- Handles categorical and numerical features without heavy preprocessing

**File**: `src/models/ltr_ranker.py` — `LTRRanker` class. Cross-features built in `src/features/ranking_features.py`.

---

### 14.4 LLM-Powered Re-Ranking + Explanations (Claude API)

**What it does**: Takes the top-20 from Stage 2 and sends them to Claude for semantic re-ranking + generation of natural-language explanations per recommendation.

**How it integrates**:
- Final ranking stage of the pipeline
- Input to Claude: user profile summary (extracted via NER — see 16.1) + compact summaries of 20 candidate jobs
- Output: JSON with reordered job IDs, per-job reasoning, and confidence scores
- Uses **prompt caching** on the system prompt (stable) to minimize cost
- Only runs on the final top-20 — keeps token budget bounded

**Example output**: `"This role matches your Python/ETL background, and Seattle aligns with your location preference. The 4-year experience requirement fits your 5 years. Potential concern: no mention of your preferred AWS stack."`

**Why include it**:
- Explanations from classical ML are shallow (feature weights); LLM explanations reason about fit holistically
- Cutting-edge research direction (2024-2026 — LLM4Rec is hot)
- Showcases modern LLM integration beyond basic chatbots

**File**: `src/models/llm_reranker.py` — `LLMReranker` class using the Anthropic SDK.

---

### 14.5 Deferred Components

The following online-learning components were researched and evaluated but deferred from the current scope: **LinUCB Contextual Bandit** (real-time feedback loop) and **Neural Network Fine-Tuning** (incremental retraining without from-scratch cost). See `future_considerations.md` §A for rationale and implementation notes.

---

## 15. Tier 2 Enhancements — Advanced Neural Architectures

### 15.1 BERT4Rec (Sequential/Transformer Model)

**What it does**: Models the user's interaction history as a *sequence* and predicts the next likely interaction using bidirectional transformer attention. Unlike two-tower (which averages past interactions into a single vector), BERT4Rec preserves temporal order — "applied to startup → applied to startup → applied to enterprise" tells a story the two-tower flattens.

**How it integrates**:
- Runs in **Stage 3 (re-ranking)** alongside the LLM re-ranker
- For users with ≥5 interactions: uses their recent interaction sequence as context
- Masks the last item and asks BERT4Rec to predict it; the model's score for each candidate becomes an additional ranking signal
- Combined with LLM rerank scores via weighted ensemble

**Why add it on top of two-tower**:
- Two-tower captures *what* you like; BERT4Rec captures *how your preferences evolve*
- For career progression (e.g., a junior engineer applying to senior roles as they grow), sequence matters
- Demonstrates mastery of transformer-based RS — a hot research area (SIGIR/RecSys papers every year)

**Why BERT4Rec over SASRec**:
- Bidirectional attention sees both past and future context during training → richer representations
- Masked language modeling objective is more robust than autoregressive
- Industry implementations (Alibaba, eBay) prefer BERT4Rec

**File**: `src/models/sequential.py` — `BERT4RecModel`, `SequentialTrainer` classes using PyTorch. Training data: chronological sequences of each user's interactions.

---

### 15.2 DeepFM (Feature Interactions)

**What it does**: Explicitly learns **feature interactions** — combinations like `(user_is_senior × job_is_junior)` = negative signal, or `(user_skills_python × job_category_data_science)` = positive signal. Combines Factorization Machines (low-order interactions) with a deep neural network (high-order interactions).

**How it integrates**:
- Runs as an **alternative ranker in Stage 2**, alongside LambdaMART
- Same feature inputs as LTR (user features, job features, cross-features)
- Outputs a score per candidate
- Ensemble with LambdaMART: final Stage 2 score = 0.6 × LambdaMART + 0.4 × DeepFM (weights tuned on validation)

**Why both LambdaMART and DeepFM**:
- LambdaMART: tree-based, interpretable, strong on tabular features, directly optimizes NDCG
- DeepFM: neural, captures non-linear feature interactions that trees miss
- Ensembling tree-based + neural rankers is a proven production pattern (Airbnb, Alibaba do this)
- Demonstrates understanding that no single algorithm dominates — ensembling is the real production secret

**File**: `src/models/deepfm.py` — `DeepFMModel`, `DeepFMTrainer` using PyTorch.

---

### 15.3 Graph Neural Network (LightGCN)

**What it does**: Treats the recommendation problem as a **graph**: users, jobs, and skills are nodes; applications, skill-requirements, and skill-possessions are edges. LightGCN propagates embeddings across this graph — a user's embedding becomes influenced by the jobs they applied to, which are influenced by similar users, and so on (multi-hop).

**How it integrates**:
- Runs in **Stage 1 (retrieval)** as an alternative/complement to two-tower
- Produces user and job embeddings via graph convolution over the user-job-skill graph
- Candidate pool becomes **union of (two-tower top-500) and (LightGCN top-500)** → ~700-900 unique candidates forwarded to Stage 2
- LTR learns which retrieval source's candidates tend to rank better for which user types

**Why add GNN**:
- Captures relationships two-tower misses: "users who worked at startups tend to apply to other startups" is a graph pattern, not a feature
- Multi-hop reasoning: user → applied to → job → requires skill → possessed by → similar user → applied to → new job
- Very impressive on portfolios — demonstrates graph ML, a rapidly growing area
- LightGCN is specifically proven for recommendations (simpler and better than full GCN for this use case)

**File**: `src/models/graph_rec.py` — `LightGCNModel` using PyTorch Geometric. Graph construction in `src/data/graph_builder.py`.

---

### 15.4 Variational Autoencoder for Collaborative Filtering (Mult-VAE)

**What it does**: A **generative model** over user interaction vectors. Takes a user's sparse interaction vector as input, encodes to a latent distribution, samples, and decodes back to a dense probability distribution over all items. High-probability items the user hasn't seen = recommendations.

**How it integrates**:
- Runs as an **auxiliary retrieval model** alongside two-tower and LightGCN
- Particularly useful for users with rich interaction history (5+ applications)
- Output is blended into Stage 1 candidate pool (top-200 from Mult-VAE)
- Acts as a sanity check: does a probabilistic generative model agree with discriminative two-tower?

**Why include it**:
- State-of-the-art on MovieLens/Netflix benchmarks for implicit feedback CF
- Demonstrates depth in generative modeling (VAE theory: ELBO, reparameterization)
- Complementary to two-tower — different inductive biases, useful for ensembling

**File**: `src/models/mult_vae.py` — `MultVAEModel`, `MultVAETrainer` using PyTorch.

---

## 16. Tier 3 Enhancements — Domain Sophistication (Job-Specific)

### 16.1 Resume NER + Structured Skill Extraction

**What it does**: Parses raw resume text (PDF or plain text) and extracts structured entities: skills, job titles, companies, education, experience duration, certifications. Transforms unstructured resume data into a rich structured user profile.

**How it integrates**:
- Sits **upstream of the entire pipeline** — happens at user onboarding
- Replaces/augments the existing `UserProfileBuilder`
- Extracted skills feed into:
  - Two-tower user tower (as multi-hot skill features)
  - LTR ranking features (skill overlap with job)
  - LLM re-ranker (structured profile summary in prompt)
- Extracted job titles feed into career path prediction (16.5)

**Technical approach**:
- Use spaCy with a custom NER model fine-tuned on a resume dataset (Kaggle has labeled resume NER datasets)
- Entity types: `SKILL`, `JOB_TITLE`, `COMPANY`, `EDUCATION`, `DURATION`, `CERTIFICATION`
- Hybrid approach: spaCy NER + regex patterns + LLM-based extraction for edge cases

**Why it's transformative**:
- Current plan requires users to manually enter skills → friction
- With NER: user uploads PDF → parsed instantly → personalized recommendations in one click
- Massive UX win that directly improves recommendation quality downstream
- Demonstrates domain-specific NLP — critical for HR tech

**File**: `src/nlp/resume_parser.py` — `ResumeParser` class. Training data prep in `notebooks/03_resume_ner.ipynb`.

---

### 16.2 Skill Ontology / Knowledge Graph

**What it does**: Builds a structured **skill taxonomy** (using ESCO — European Skills, Competences, Qualifications and Occupations — or O*NET) where skills have relationships: `"Django" is_a "Python Web Framework"`, `"React" related_to "Frontend"`, `"PyTorch" prerequisite_for "Deep Learning"`. Enables reasoning beyond exact skill matching.

**How it integrates**:
- Loaded once at startup; queried throughout the pipeline
- Affects **feature engineering for LTR** (Stage 2):
  - Instead of binary skill-match, compute *ontology-weighted* skill similarity
  - User has "Django", job wants "Flask" → exact match = 0, ontology match = 0.85 (both Python web frameworks)
- Affects **LLM re-ranker** (Stage 3):
  - Prompt includes skill relationship context so LLM can reason about transferable skills
- Affects **skill-gap analysis** (16.6):
  - Identifies prerequisite skills the user is missing

**Technical approach**:
- Load ESCO skill ontology (freely available, 13K+ skills with relationships)
- Store as a NetworkX graph for in-memory traversal
- Optional: Neo4j for richer queries (overkill for 1-2 users)
- Implement similarity functions: `skill_similarity(skill_a, skill_b)` using shortest path or Wu-Palmer similarity

**Why it matters**:
- Jobs describe requirements in free text with synonyms and adjacent skills — exact matching is brittle
- Enables *transferable skill* reasoning — critical for career transitions
- Shows domain expertise beyond generic RS techniques
- Prerequisite for meaningful skill-gap analysis

**File**: `src/ontology/skill_graph.py` — `SkillOntology` class. ESCO loader in `src/ontology/esco_loader.py`.

---

### 16.3 Salary Prediction Model

**What it does**: Regression model that predicts the expected salary range for any job based on its features (title, location, required skills, experience level, company size, remote/onsite). Predicts even for jobs that don't list a salary.

**How it integrates**:
- Runs as an **enrichment pass in Stage 4** (after ranking is finalized)
- For each of the final top-10 recommendations:
  - Predict salary range
  - Compare to user's salary expectation (if known from profile)
  - Tag as "above market", "at market", or "below market" for the user's skill/experience level
- Displayed in UI as a badge on each recommendation

**Technical approach**:
- Gradient boosting regressor (XGBoost) with features: job title embedding, location one-hot, required skills multi-hot, experience level, remote flag
- Training data: subset of jobs with listed salaries (usually 30-40% of postings)
- Evaluate with MAE and R²
- Output: predicted median salary + confidence interval

**Why include it**:
- Real product value — salary is a top concern for job seekers
- Adds quantitative signal to recommendations beyond "is this a good match"
- Uncovers hidden signals: jobs similar to ones the user likes but with better pay
- Demonstrates regression + RS integration

**File**: `src/models/salary_predictor.py` — `SalaryPredictor` class.

---

### 16.4 Career Path Prediction

**What it does**: Given a user's current role and history, predicts likely next career steps. "Software Engineer → Senior Software Engineer → Staff Engineer" or "Data Analyst → Data Scientist → ML Engineer" — learned from real career trajectories in the dataset.

**How it integrates**:
- Runs as an **enrichment pass in Stage 4**
- Uses the user's most recent job title (extracted via NER) to predict next 1-3 career stages
- Influences retrieval: biases FAISS query toward jobs matching predicted next-step titles (not just current-role titles)
- Displayed in UI as a "Career Trajectory" panel

**Technical approach**:
- Extract (previous_title → current_title) transition pairs from resume data
- Train a Markov chain or small transformer on title sequences
- Augment with title embeddings (sentence-transformers on title text) for generalization beyond seen titles

**Why include it**:
- Goes beyond matching to career planning — genuinely useful product feature
- Addresses a blind spot in most RS: they recommend more of what you have, not what you should grow into
- Demonstrates sequence modeling with domain-specific structure
- Integrates naturally with retrieval — career-aware recommendations

**File**: `src/models/career_path.py` — `CareerPathPredictor` class.

---

### 16.5 Bi-Directional Matching (Recruiter-Side)

**What it does**: The same embedding space enables the reverse direction — given a job, find the best-fit candidates. The recruiter-side dashboard becomes a full-featured sourcing tool.

**How it integrates**:
- Re-uses the two-tower model: compute job embedding → FAISS search over *user* embeddings (requires building a user-embedding FAISS index too)
- LTR ranker is reused but with (job, user) pairs instead of (user, job) — same features, same model
- LLM re-ranker generates **outreach templates** per candidate: "Hi [Name], I noticed your background in X aligns with our need for Y..."

**Why prioritize it**:
- Transforms the project from a one-sided recommender into a **two-sided marketplace**
- Showcases bi-directional embedding spaces (same model serves two use cases)
- LLM-generated outreach is a standout feature for HR tech

**File**: `src/pipeline/recruiter_pipeline.py` — mirror of `MultiStagePipeline` but reversed. UI: existing Streamlit "Recruiter View" tab expanded significantly.

---

### 16.6 Resume-Job Gap Analysis

**What it does**: For each recommended job, identifies the **specific skills the user is missing** to fully qualify, and generates LLM-powered suggestions for how to acquire them (courses, projects, resources).

**How it integrates**:
- Runs as an **enrichment pass in Stage 4** on the final top-10
- For each job:
  - Compute `required_skills - user_skills` (using ontology, so "React wanted, user has Vue" = still a gap but smaller)
  - For each missing skill, query LLM for learning suggestions (prompt-cached common skills)
- Surface as a "Skills to develop" panel per recommendation in the UI

**Why include it**:
- Transforms the recommender from "here are jobs" into "here are jobs *and* here's your roadmap to them"
- Unique product positioning — most job boards don't do this
- Integrates skill ontology (16.2), NER (16.1), and LLM capabilities in one feature
- Demonstrates thinking about the full user journey, not just the recommendation moment

**File**: `src/models/skill_gap.py` — `SkillGapAnalyzer` class.

---

### 16.7 Query Understanding (LLM-Based Search)

**What it does**: Users can type free-text queries like *"remote senior python jobs in healthcare startups that pay >$150k"* and the LLM parses intent into structured filters + semantic query that the pipeline can execute.

**How it integrates**:
- Pre-processing layer **before Stage 1**
- LLM extracts: `{remote: true, seniority: "senior", skills: ["python"], industry: "healthcare", company_stage: "startup", min_salary: 150000}`
- Applies structured filters to the candidate pool (pre-filter before two-tower retrieval)
- Generates a query embedding for semantic portion of search
- Falls back to normal recommendation flow if query is vague

**Why include it**:
- Modern UX pattern combining LLM + classical retrieval
- More flexible than faceted search UI
- Showcases LLM tool-use / structured-output patterns
- Elegant way to integrate user intent with the recommendation pipeline

**File**: `src/nlp/query_parser.py` — `QueryParser` class.

---

## 17. Updated Project Structure

```
RS proj/
├── config/
├── data/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_comparison.ipynb
│   ├── 03_resume_ner.ipynb           # NEW
│   └── 04_multi_stage_evaluation.ipynb  # NEW
├── src/
│   ├── data/
│   │   ├── acquire.py
│   │   ├── preprocessing.py
│   │   └── graph_builder.py          # NEW (for LightGCN)
│   ├── features/
│   │   ├── text_features.py
│   │   ├── user_profile.py
│   │   ├── structured_features.py
│   │   └── ranking_features.py       # NEW (cross-features for LTR)
│   ├── nlp/                          # NEW directory
│   │   ├── resume_parser.py          # Resume NER
│   │   └── query_parser.py           # LLM query understanding
│   ├── ontology/                     # NEW directory
│   │   ├── esco_loader.py
│   │   └── skill_graph.py
│   ├── models/
│   │   ├── content_based.py
│   │   ├── collaborative.py
│   │   ├── popularity.py
│   │   ├── hybrid.py
│   │   ├── two_tower.py              # NEW (Tier 1)
│   │   ├── ltr_ranker.py             # NEW (Tier 1)
│   │   ├── llm_reranker.py           # NEW (Tier 1)
│   │   ├── sequential.py             # NEW (Tier 2 - BERT4Rec)
│   │   ├── deepfm.py                 # NEW (Tier 2)
│   │   ├── graph_rec.py              # NEW (Tier 2 - LightGCN)
│   │   ├── mult_vae.py               # NEW (Tier 2)
│   │   ├── salary_predictor.py       # NEW (Tier 3)
│   │   ├── career_path.py            # NEW (Tier 3)
│   │   ├── skill_gap.py              # NEW (Tier 3)
│   │   └── train.py
│   ├── retrieval/                    # NEW directory
│   │   └── faiss_index.py
│   ├── pipeline/                     # NEW directory
│   │   ├── multi_stage.py            # Main orchestrator
│   │   └── recruiter_pipeline.py     # Bi-directional matching
│   ├── evaluation/
│   └── utils/
├── api/
├── app/
├── tests/
├── requirements.txt
└── README.md
```

---

## 18. End-to-End Request Flow (Final System)

To make the integration crystal clear, here's what happens when a user requests recommendations in the final enhanced system:

**Step 1 — Onboarding (one-time)**:
- User uploads resume PDF
- `ResumeParser` (16.1) extracts: skills list, job titles, experience, education
- `SkillOntology` (16.2) normalizes extracted skills to canonical ESCO IDs
- `CareerPathPredictor` (16.4) predicts next likely career steps
- Structured profile saved

**Step 2 — (Optional) Free-text Query**:
- User types: "remote ML jobs paying >$150k"
- `QueryParser` (16.7) extracts filters via LLM → applied as pre-filter

**Step 3 — Stage 1: Retrieval**:
- Two-tower (14.1) computes user embedding from profile → FAISS (14.2) returns top-500
- LightGCN (15.3) returns top-500 using graph propagation
- Mult-VAE (15.4) returns top-200 for users with rich history
- Union + dedupe → ~800 candidates
- Career path bias applied: boost candidates matching predicted next-step titles (16.4)

**Step 4 — Stage 2: Ranking**:
- For each candidate, build feature vector: retrieval scores + skill-ontology-weighted overlap + location match + experience match + salary alignment + popularity + recency
- LambdaMART (14.3) scores each → top 50
- DeepFM (15.2) scores each → top 50
- Ensemble: final Stage 2 top-20 by weighted combination

**Step 5 — Stage 3: Re-ranking**:
- BERT4Rec (15.1) scores top-20 using user's recent interaction sequence
- LLM re-ranker (14.4) re-orders top-20 with natural-language reasoning, returns top-10 + explanations
- LinUCB bandit (14.5) replaces 2-3 slots with exploratory picks

**Step 6 — Stage 4: Enrichment**:
- `SalaryPredictor` (16.3) adds salary estimate + market comparison per job
- `SkillGapAnalyzer` (16.6) identifies missing skills per job + LLM suggests learning resources
- Final response assembled

**Step 7 — Profile Update**:
- User interactions are logged; the user's profile vector (weighted average of interacted job embeddings) is recomputed so subsequent requests reflect recent behavior
- A live feedback loop (bandit + incremental NN fine-tuning) is **not** included in current scope — see `future_considerations.md` §A

---

## 19. Updated Evaluation Strategy

The evaluation framework extends to compare **multiple pipeline configurations** side-by-side:

| Variant | Retrieval | Ranking | Re-ranking | Enrichment |
|---------|-----------|---------|------------|------------|
| V1 Baseline | Hybrid | — | — | — |
| V2 + Neural Retrieval | Two-Tower + FAISS | — | — | — |
| V3 + LTR | Two-Tower + FAISS | LambdaMART | — | — |
| V4 + LLM | Two-Tower + FAISS | LambdaMART | LLM | — |
| V5 + Sequential | Two-Tower + FAISS | LambdaMART | LLM + BERT4Rec | — |
| V6 + Graph | Two-Tower + LightGCN + FAISS | LambdaMART + DeepFM | LLM + BERT4Rec | — |
| V7 Full | All retrieval sources | Ensemble | Full re-rank | Salary + Career + Gaps |

Each variant evaluated on: Precision@10, NDCG@10, Recall@50, Coverage, Diversity, Serendipity. Expectation: monotonic improvement from V1 to V7, with largest jumps at V2 (neural retrieval) and V4 (LLM reasoning).

---

## 20. Enhancement Timeline

| Days | Phase | Deliverables |
|------|-------|--------------|
| 16-19 | Tier 1a: Two-Tower + FAISS | Neural retrieval working |
| 20-22 | Tier 1b: LambdaMART LTR | Multi-stage pipeline alive |
| 23-24 | Tier 1c: LLM re-ranker | Natural-language explanations |
| 25-30 | Tier 2a: BERT4Rec | Sequential model trained, integrated |
| 31-33 | Tier 2b: DeepFM | Second ranker, ensembled |
| 34-38 | Tier 2c: LightGCN | Graph model, multi-source retrieval |
| 39-41 | Tier 2d: Mult-VAE | Generative CF, auxiliary retrieval |
| 42-44 | Tier 3a: Resume NER | Upload-resume onboarding flow |
| 45-48 | Tier 3b: Skill ontology | ESCO graph integrated into features |
| 49-50 | Tier 3c: Salary prediction | Enrichment pass working |
| 51-53 | Tier 3d: Career path | Prediction + retrieval bias |
| 54-55 | Tier 3e: Bi-directional matching | Recruiter pipeline complete |
| 56-57 | Tier 3f: Skill gap analysis | Per-recommendation skill roadmaps |
| 58-59 | Tier 3g: Query understanding | LLM-parsed free-text search |
| 60-63 | Integration + Evaluation | Multi-variant comparison, final polish |

**Total project timeline: ~63 days (baseline 15 + enhancements 48)**

---

## 21. Final Verification

1. Baseline training completes: `python -m src.models.train`
2. Neural pipeline training completes: `python -m src.pipeline.train_neural`
3. All tests pass: `pytest tests/`
4. API serves multi-stage recommendations: `GET /recommend/multi-stage/{user_id}` returns ranked list with per-stage scores, LLM explanations, salary estimates, skill gaps, career suggestions
5. Streamlit UI shows pipeline inspector tab (funnel visualization V1-V7 comparison)
6. Resume upload → NER extraction → recommendations flow works end-to-end
7. Free-text query like "remote Python ML jobs >$150k" parses correctly and returns filtered recommendations
8. Evaluation notebook shows clear performance gains from V1 → V7 across all metrics
9. Bi-directional matching works — recruiter selects job, sees ranked candidates with LLM-drafted outreach
10. Skill gap analysis produces coherent learning suggestions grounded in ontology

---

# PART III — Differentiation Layer (Reciprocal Recommendation + Defensive Novelty)

This part was added after a supervisor concern that hybrid job RS is well-trodden. It establishes our defensible novelty and adds the one piece of code needed to back it up. See `how_we_are_different.md` for the full competitor analysis and sourced references.

## 22. The Five Differentiators

Each maps to concrete code in the repo. The novelty is the *integration* — no published system (LinkedIn LiRank/JUDE, ZipRecruiter, Indeed) combines all five.

| # | Differentiator | Code mapping |
|---|---|---|
| **D1** | LLM-grounded explanations citing ESCO skill IDs + resume NER entities | `src/models/llm_reranker.py` + `src/nlp/resume_parser.py` + `src/ontology/skill_graph.py` |
| **D2** | Ontology-weighted skill transferability (Django↔Flask = 0.85, not 0) | `src/ontology/skill_graph.py` + `src/ontology/esco_loader.py` + `src/features/ranking_features.py` |
| **D3** | First-class reciprocal recommendation — bilateral score `P(apply) × P(shortlist)` | `src/models/reciprocal.py` (NEW) + `src/pipeline/multi_stage.py` |
| **D4** | 4-stage pipeline including Stage-4 enrichment (salary + skill-gap roadmap + career path) | `src/models/salary_predictor.py` + `src/models/skill_gap.py` + `src/models/career_path.py` |
| **D5** | Auditable per-stage decomposition (Pipeline Inspector) | Streamlit Pipeline Inspector tab + `GET /pipeline/inspect/{user_id}` |

D1, D2, D4, D5 are already covered by Parts I–II. D3 requires the new code below.

---

## 23. Reciprocal Recommendation Upgrade (NEW — for D3)

### 23.1 Why this matters

Industry systems (LinkedIn, Indeed, ZipRecruiter) are job-seeker-centric or recruiter-centric, not both. Reciprocal RS — modeling *both* sides' preferences and combining them — is an active 2024 research frontier (SIGKDD'24 "Revisiting Reciprocal Recommender Systems", AAAI'24, SIGIR'24 MIRROR) but has not landed in production at scale. Without this, our D3 claim is just text in a proposal with no code behind it.

### 23.2 Architecture

```
                      ┌────────────────────────┐
   user features ──▶  │  UserTower             │ ──▶ user_emb
                      └────────────────────────┘
                                                       ╲
                                                        × dot ──▶ s_user_to_job
                                                       ╱
                      ┌────────────────────────┐
   job features ──▶   │  JobTower              │ ──▶ job_emb
                      └────────────────────────┘

                      ┌────────────────────────┐
   job features ──▶   │  JobToUserTower (NEW)  │ ──▶ recruiter_pref_emb
                      └────────────────────────┘
                                                       ╲
                                                        × dot ──▶ s_job_to_user
                                                       ╱
                      ┌────────────────────────┐
   user features ──▶  │  UserAsCandidateTower  │ ──▶ candidate_emb
                      └────────────────────────┘   (reuse UserTower or sibling head)

   bilateral_score = sigmoid(s_user_to_job) × sigmoid(s_job_to_user)
```

The forward direction is the existing two-tower. The inverse direction (`JobToUserTower`) is trained on the **same interaction data flipped** — for each (user, job) positive pair, the recruiter-side model learns "which users would this job's recruiter shortlist."

### 23.3 Training data

- **Positive pairs**: same as forward direction — (user, job) where action ∈ {save, apply}
- **Negative sampling**: in-batch negatives, recruiter-side
- **Loss**: BPR or in-batch softmax on the inverse direction
- **Shortlist signal proxy**: in absence of true recruiter labels, treat `apply` as positive bilateral signal (user wants the job *and* the job's profile fits the user). Future work could use recruiter-response data if available.

### 23.4 Integration into the multi-stage pipeline

- **Stage 1 (Retrieval)**: unchanged — two-tower + FAISS + LightGCN + Mult-VAE produce candidate union.
- **Stage 2 (Ranking)**: LambdaMART feature vector gains 3 new features:
  - `s_user_to_job` (forward two-tower score)
  - `s_job_to_user` (inverse tower score)
  - `bilateral_score` (product)
- **Stage 2 ensemble final score**: `α × LambdaMART + β × DeepFM + γ × bilateral_score`, weights tuned on validation.
- **Stage 3 (LLM re-rank)**: prompt includes both directions' contributions for richer reasoning ("you'd likely apply, and your profile matches their typical hire").

### 23.5 New evaluation metrics (from SIGKDD'24)

Added to `src/evaluation/metrics.py`:

- **`bilateral_coverage`** — fraction of recommended pairs where *both* sides score above their respective median. Penalizes systems that only optimize one side.
- **`balanced_ranking_ratio`** — ratio of NDCG when ranking from user-side vs. recruiter-side. Closer to 1.0 = more balanced.
- **`two_sided_ndcg`** — geometric mean of user-side NDCG@K and recruiter-side NDCG@K.

### 23.6 New files / modifications

| File | Status | Purpose |
|---|---|---|
| `src/models/reciprocal.py` | NEW | `JobToUserTower`, `BilateralScorer`, `ReciprocalTrainer` |
| `src/pipeline/multi_stage.py` | MODIFY | Wire bilateral score into Stage-2 ensemble; pass to Stage-3 LLM prompt |
| `src/evaluation/metrics.py` | MODIFY | Add `bilateral_coverage`, `balanced_ranking_ratio`, `two_sided_ndcg` |
| `src/features/ranking_features.py` | MODIFY | Add `s_user_to_job`, `s_job_to_user`, `bilateral_score` features |
| `tests/test_reciprocal.py` | NEW | Unit tests: forward/inverse score symmetry sanity, bilateral score range, metric correctness on toy data |
| `notebooks/04_multi_stage_evaluation.ipynb` | MODIFY | Add reciprocal-metrics row to V1→V7 comparison; add a V8 with bilateral scoring |
| `models_artifacts/reciprocal_tower.pt` | NEW (build artifact) | Trained inverse tower weights |
| `models_artifacts/bilateral_scorer.pkl` | NEW (build artifact) | Serialized scorer |

### 23.7 Timeline (slots into existing schedule)

| Days | Task |
|---|---|
| 64-65 | Implement `JobToUserTower` + flipped data loader |
| 66 | Train inverse tower; sanity-check with held-out interactions |
| 67 | `BilateralScorer` + integration into LambdaMART feature vector |
| 68 | Add new metrics; rerun V7 evaluation with bilateral; produce V8 row |
| 69 | Tests + documentation + LLM-prompt update for Stage 3 |

Total: ~5-6 days. **New project total: ~69 days.**

---

## 24. Updated Evaluation Strategy (extends Section 19)

| Variant | Retrieval | Ranking | Re-ranking | Enrichment | Reciprocal |
|---|---|---|---|---|---|
| V1–V7 | (as in §19) | | | | — |
| **V8 Full + Reciprocal** | All retrieval sources | LambdaMART + DeepFM + bilateral | LLM + BERT4Rec (bilateral-aware prompt) | Salary + Career + Gaps | **Yes** |

V8 is evaluated on the standard metrics **plus** `bilateral_coverage`, `balanced_ranking_ratio`, `two_sided_ndcg`. Expectation: V8 lifts bilateral metrics meaningfully over V7 with at most a small (<2%) loss on user-side NDCG@10 — the trade-off that makes reciprocal recommendation defensible.

---

## 25. Sections DROPPED from Proposal Document

Recorded here so the plan stays the source of truth. The following sample-template fields were explicitly dropped (user-approved) from `Project_Proposal.md`:

- Required Budget field in header block
- Section 10 Project Budget (entire — Tables 10.1, 11.2, A/B/C/D percentage breakdown)
- External Investigator row in header block
- Table 8.3 Strategic Technology Program Goals (GOAL 1/2/3 mapping)

Phase 4 was also reworded from *"Deep learning model and Probabilistic graphical model"* to *"Multi-Stage Neural Pipeline + Reciprocal Matching"* — no PGM/LDA/PMF padding added; the plan is honest about what's actually built.

---

## 26. Final Verification (extends Section 21)

11. Reciprocal upgrade lands: `pytest tests/test_reciprocal.py` passes
12. `GET /recommend/multi-stage/{user_id}` returns `bilateral_score` per item alongside per-stage scores
13. Evaluation notebook reports `bilateral_coverage`, `balanced_ranking_ratio`, `two_sided_ndcg` for V8 alongside Precision@10/NDCG@10
14. Each of the 5 differentiators (D1–D5) maps to a real code file (verification checklist in `how_we_are_different.md` §5)
15. Every paper cited in `Project_Proposal.md` and `how_we_are_different.md` has a working DOI/arXiv link
