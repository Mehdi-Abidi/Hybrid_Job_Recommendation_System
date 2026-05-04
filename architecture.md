# Architecture — Hybrid Job Recommendation System

This document explains the full architecture of the project: what we built, what technologies and algorithms we used, how the pieces fit together, and *why* we made each choice. Each concept is presented twice — once in **plain English** so anyone can follow, and once in **technical terms** at the level of an intermediate Recommender Systems course.

---

## 1. The Problem We Are Solving

**Plain English.** Job seekers waste hours scrolling past irrelevant listings; recruiters drown in unqualified applicants. The right job exists for the right person — they just can't find each other. We are building a system that takes a person's resume and history of clicks/applies, and returns a short ranked list of jobs that genuinely fit them. The same engine, run in reverse, helps a recruiter find candidates for a role.

**Technical.** A two-sided matching problem with implicit feedback (views, saves, applies — no explicit ratings), heavy cold-start (new users and new jobs every day), sparse user-item interaction matrix (>99% empty), rich side information on both sides (resume text, structured skills, location, experience for users; description, requirements, salary for jobs), and a reciprocal preference structure (a successful match requires both sides to be willing). Output is top-K ranked recommendations optimised for NDCG@K with secondary objectives for coverage, diversity, and bilateral balance.

---

## 2. High-Level System Shape

**Plain English.** The system is a *funnel*. We start with hundreds of thousands of jobs, narrow down to a few hundred good candidates, then carefully rank those, then have an LLM read the top 20 and write human explanations for the final 10. After ranking, we attach extra information (predicted salary, missing skills, career path) before showing them.

**Technical.** Multi-stage retrieval-and-ranking pipeline mirroring industrial RS architectures (YouTube, LinkedIn, Pinterest):

```
Request
  │
  ▼
[Stage 1: Retrieval]  ~350K jobs → ~500–800 candidates
  Two-Tower NN + FAISS ANN, LightGCN, Mult-VAE
  │
  ▼
[Stage 2: Ranking]    candidates → top 50
  LambdaMART (XGBoost rank:ndcg) + DeepFM ensemble
  │
  ▼
[Stage 3: Re-ranking] top 20 → top 10
  BERT4Rec (sequence) + LLM semantic re-rank with explanations
  │
  ▼
[Stage 4: Enrichment] top 10
  Salary prediction · Skill-gap analysis · Career path
  │
  ▼
Final response
```

The funnel exists because each stage trades computational cost for precision: cheap-and-broad retrieval cannot afford rich features; rich rankers cannot afford to score 350K items. Separating concerns also makes the system testable, swappable, and explainable per stage.

---

## 3. The Layered Architecture

**Plain English.** The code is split into independent layers like a stack of pancakes. Data sits at the bottom; on top of that we build features (numerical fingerprints of jobs and people); on top of features we build models; models are served through an API; the API is consumed by a UI. Each layer talks only to the one directly beneath it, so we can replace the UI without touching the models, or swap a model without touching the API.

**Technical.** Classic separation-of-concerns layout: `data → features → models → evaluation → serving (API) → presentation (UI)`. Configuration lives outside the code in YAML and is loaded as typed dataclasses. Trained models and indexes are persisted as artifacts (`.pkl`, `.pt`, FAISS index files), so training and serving are decoupled processes.

---

## 4. Data Layer

**Plain English.** We use real LinkedIn postings (~350K jobs) plus a synthetic interaction generator as a fallback when the data is unavailable. Before any model sees the data, we clean it: strip HTML, normalise locations, parse salary ranges, deduplicate, and convert clicks/saves/applies into numeric "implicit ratings" (view = 1, save = 3, apply = 5).

**Technical.**
- **Source**: LinkedIn job postings dataset + Kaggle resume corpus, mapped to a canonical schema by a real-data adapter.
- **Implicit feedback weighting**: 1/3/5 reflects observed conversion gradients in the literature; explicit ratings do not exist for jobs.
- **Splits**: time-based when timestamps are available (avoids future-leakage); per-user random 80/20 otherwise.
- **Interaction matrix**: stored sparsely (CSR) — densifying would require terabytes.
- **Why a synthetic fallback exists**: ensures CI and local dev work without external API access, and enables cold-start stress testing where we control the user activity distribution.

---

## 5. Feature Engineering

**Plain English.** Models cannot read text or categories — they need numbers. Feature engineering turns each job description into a vector of numbers (its "fingerprint"), and does the same for each user.

**Technical.** Three feature families:

### 5.1 Text features
- **TF-IDF** (5000 dims, sparse): term-frequency × inverse-document-frequency. Cheap, interpretable (we can point at the exact words that matched), good baseline.
- **Sentence embeddings** (`all-MiniLM-L6-v2`, 384 dims, dense): pre-trained Sentence-Transformer. Captures *semantic* similarity — knows "ML engineer" ≈ "machine learning developer" even with no shared tokens.
- We keep both because they answer different questions: TF-IDF answers *"which words matched?"* (good for explanations), embeddings answer *"how close in meaning?"* (good for accuracy).

### 5.2 User profiles
- For users with history: weighted average of the embeddings of jobs they interacted with, weighted by action strength (apply = 5 × weight of view = 1).
- For new users (cold-start): direct embedding of the resume text + skills list.

### 5.3 Structured features
- Skills → multi-hot vector over the known skill vocabulary.
- Location → one-hot of normalised city/state.
- Experience → binned to entry/junior/mid/senior; preserves ordinal structure.

### 5.4 Cross-features (used in ranking)
Built only after retrieval, only for surviving candidates: skill-overlap counts, location-match flag, experience-vs-seniority match, salary-expectation alignment, recency, retrieval-stage scores. These are the "interaction features" that let the ranker reason about *fit*, not just per-side properties.

---

## 6. Stage 1 — Retrieval

**Goal**: from ~350K jobs, recall a few hundred plausible candidates fast. Optimised for **recall**, not precision.

### 6.1 Two-Tower Neural Network (primary retriever)

**Plain English.** Two small neural networks — one for users, one for jobs — both producing 128-number vectors. We train them so that vectors of *people who applied to a job* end up close to that *job's* vector in space. After training, recommending becomes "find the job vectors nearest to this user's vector."

**Technical.**
- `UserTower(user_features) → ℝ¹²⁸`, `JobTower(job_features) → ℝ¹²⁸`.
- Loss: contrastive / in-batch softmax with negatives drawn from the same batch.
- Score = dot product (or cosine).
- Why two towers and not one big model: at serving time we can pre-compute all job vectors *once* and only compute the user vector per request, then run nearest-neighbour search. A monolithic model would need to score every (user, job) pair from scratch.
- Why two-tower beats SVD here: SVD is ID-based (a brand-new user has no row), two-tower is feature-based (a brand-new user with just a resume still gets a meaningful vector). It also captures non-linearities that SVD's linear factorisation cannot.

### 6.2 FAISS (Approximate Nearest Neighbour index)

**Plain English.** Once every job has a 128-number fingerprint, we build a fast "lookup tree" over them. Searching this tree for the closest fingerprints to a user's vector takes milliseconds even with 350K jobs, instead of comparing one-by-one.

**Technical.** `IVFFlat` index — partition embedding space into Voronoi cells via k-means, search only the few nearest cells at query time. Sub-linear retrieval (`O(√N)` typical). We include this even at 1–2 user scale because brute-force cosine over 350K × 128 is slow and the architectural pattern is the point.

### 6.3 LightGCN (Graph Neural Network)

**Plain English.** Treat users, jobs, and skills as dots and the connections between them (applications, "this job needs Python", "this user knows Python") as lines. Each user's "fingerprint" is influenced by the jobs they applied to, which are influenced by similar users, and so on for a few hops. This catches patterns like "people who applied to startups tend to apply to other startups."

**Technical.** Embedding propagation over the user-job(-skill) bipartite/tripartite graph: `e^(k+1) = Σ neighbours / √(deg)`. No non-linearity, no self-connection — the "Light" in LightGCN. Empirically beats heavier GCN/NGCF for collaborative-filtering signals. Provides a **complementary** retrieval source: where two-tower captures attribute similarity, LightGCN captures multi-hop relational signals.

### 6.4 Mult-VAE (auxiliary, for users with rich history)

**Plain English.** A neural network that learns the *distribution* of what users like, then "imagines" plausible items for any specific user. Best for users who have already interacted a lot.

**Technical.** Variational autoencoder over implicit-feedback vectors: encoder maps a user's interaction vector to a latent Gaussian; decoder reconstructs a multinomial distribution over all items. Trained with the multinomial likelihood + KL term (ELBO). State-of-the-art on MovieLens/Netflix benchmarks for implicit CF; complementary inductive bias to discriminative two-tower.

### 6.5 Why three retrievers, not one

Their candidate sets are unioned and deduped. Different inductive biases catch different patterns; the union dominates any single retriever in recall. The downstream ranker is responsible for sorting out which retriever's candidates are best for which user.

### 6.6 Classical hybrid as fallback
For the warmup/baseline pipeline (and as a safety net) we keep the classical hybrid: weighted blend of content-based (cosine over embeddings/TF-IDF), collaborative (Surprise SVD), and popularity (recency-decayed weighted interaction counts). The blend weights adapt to user activity (cold/lukewarm/warm) and the popularity model is the always-on safety net for true-zero-history users.

---

## 7. Stage 2 — Ranking

**Goal**: take the ~500–800 candidates from Stage 1 and order them precisely. Optimised for **precision and NDCG**.

### 7.1 LambdaMART (XGBoost `rank:ndcg`)

**Plain English.** A "team of decision trees" that learns to score each (user, job) pair. Crucially, it's trained to put *the right items at the top* — not just to predict ratings accurately. The order of the list is what matters.

**Technical.** Gradient-boosted trees with the LambdaRank loss: instead of regressing a target, gradients are weighted by how much swapping two items would change NDCG. Pointwise scoring at inference, listwise objective at training. Inputs: the cross-features built in §5.4 plus the retrieval-stage scores. Tree-based ⇒ no scaling/encoding fuss, native handling of missing values, feature-importance for interpretability.

### 7.2 DeepFM (companion ranker)

**Plain English.** A neural ranker that's especially good at learning "combinations" — e.g. "senior user applying to junior role = bad", "user knows React + job category is frontend = good". Used alongside LambdaMART; the two scores are blended.

**Technical.** Combines a Factorisation Machine (explicit 2nd-order feature interactions over embeddings of categorical features) with a deep MLP (higher-order interactions) sharing the same input embeddings. Why ensemble with LambdaMART: trees miss smooth high-order interactions; nets miss tabular sharpness. The Airbnb / Alibaba production stacks ensemble both for the same reason.

### 7.3 Reciprocal scoring

**Plain English.** A normal job recommender asks "would the user like this job?" Ours also asks "would the recruiter shortlist this user?" and multiplies the two probabilities. A great match needs *both* sides to want it.

**Technical.** A `JobToUserTower` is trained on the flipped interaction graph (job → which users) using BPR / in-batch softmax. The bilateral score `σ(s_u→j) × σ(s_j→u)` becomes an additional ranking feature alongside LambdaMART/DeepFM scores. Three new evaluation metrics — `bilateral_coverage`, `balanced_ranking_ratio`, `two_sided_ndcg` — measure whether we are actually balanced (from SIGKDD'24 reciprocal-RS work). This is one of our defensible novelties: industrial systems are typically one-sided.

---

## 8. Stage 3 — Re-Ranking

**Goal**: bring sequence awareness and natural-language reasoning to the final shortlist.

### 8.1 BERT4Rec (sequential model)

**Plain English.** Looks at the *order* of your past actions, not just their average. "Applied to startup → startup → enterprise" tells a story (you may be growing out of startups). A model that averages your history loses that story.

**Technical.** Bidirectional Transformer with masked-item prediction (Cloze objective) over chronological interaction sequences. We mask the last item, predict it, and use the model's score over candidates as an extra ranking signal. Bidirectional > autoregressive (SASRec) on most RS benchmarks because the masked-LM objective sees both past and future context during training.

### 8.2 LLM Re-Ranker (Claude)

**Plain English.** The top 20 candidates and a compact summary of the user's profile are sent to an LLM. The LLM reorders them and writes one-sentence reasons in plain English: *"This role matches your Python/ETL background and Seattle aligns with your stated preference; the 4-year experience requirement fits your 5 years."*

**Technical.**
- Anthropic SDK with **prompt caching** on the system prompt and stable user-profile summary so per-request token cost stays low.
- Bounded budget: only the top 20 are sent.
- Output: structured JSON (re-ordered IDs + per-item reasoning + confidence). Reasoning is grounded by passing ESCO skill IDs and resume-NER entities into the prompt — making it a **traceable** explanation, not free-form hallucination. (This grounding is differentiator D1.)
- Why an LLM at all: classical "explainability" (feature weights) explains the model, not the *match*. An LLM can reason holistically about transferable skills and trade-offs.

---

## 9. Stage 4 — Enrichment

**Goal**: turn a ranked list into a useful product.

- **Salary prediction.** XGBoost regressor on (title embedding, location, required skills, experience, remote flag) → predicted median + interval. Tagged "above / at / below market" relative to the user. Useful even when the posting itself omits salary.
- **Skill-gap analysis.** For each recommendation, compute `required_skills − user_skills`, *weighted by ontology distance* (so wanting React when the user has Vue is a smaller gap than wanting Rust when the user has none). For each missing skill the LLM produces a learning suggestion; common skills are prompt-cached.
- **Career-path prediction.** Markov / small Transformer over title-transition sequences observed in real resumes, augmented with title embeddings to generalise to unseen titles. Predicts likely next 1–3 steps and (optionally) biases retrieval toward those titles.

These are our "differentiator D4" — none of the major incumbents bundle all three on a per-recommendation basis.

---

## 10. Domain Layer (NLP + Ontology)

### 10.1 Resume NER

**Plain English.** A user uploads a PDF resume. A domain-specific entity recogniser pulls out skills, job titles, employers, education, durations, certifications. No more typing skills into a form.

**Technical.** spaCy NER, fine-tuned on a labelled resume corpus, augmented with regex patterns and an LLM-extraction fallback for low-confidence spans. Entity types: `SKILL | JOB_TITLE | COMPANY | EDUCATION | DURATION | CERTIFICATION`. Output feeds (a) the user tower's skill multi-hot, (b) the LTR cross-features, (c) the LLM re-ranker prompt, (d) career-path prediction.

### 10.2 ESCO Skill Ontology

**Plain English.** A pre-built "map of skills" where related skills are connected. Django and Flask are both Python web frameworks, so they're close on the map. PyTorch comes after Deep Learning, which comes after Machine Learning. We use this map so that a job asking for *Flask* still matches a candidate who knows *Django* — instead of treating every skill as an unrelated word.

**Technical.** ESCO (European Skills, Competences, Qualifications and Occupations) loaded into NetworkX. Skill similarity computed via shortest-path / Wu-Palmer over the taxonomy. Used to (a) replace binary skill-overlap with ontology-weighted overlap in LTR features, (b) enrich the LLM re-ranker prompt with relationship context, (c) ground skill-gap analysis in real prerequisite chains. Differentiator D2.

### 10.3 Query Understanding

**Plain English.** User types *"remote senior python jobs in healthcare startups paying >$150k"*. An LLM parses that into structured filters and a semantic search query, which feeds into the pipeline as a pre-filter.

**Technical.** LLM with structured-output / tool-use mode produces a JSON schema (`{remote, seniority, skills[], industry, company_stage, min_salary}`). Hard filters are applied as a candidate pre-filter; the residual free-text portion is embedded for the two-tower retrieval call. Falls back to standard recommendation flow if intent is vague.

---

## 11. Serving Layer — FastAPI

**Plain English.** A small web server that exposes endpoints like "give me recommendations for user 42" and "explain why job X was recommended". The web/UI talks to this; nothing else needs to know how the models work.

**Technical.** FastAPI + Uvicorn. Endpoints:
- `GET /recommend/{user_id}` — full multi-stage pipeline.
- `POST /recommend/new-user` — cold-start path (resume + skills in body).
- `GET /similar-jobs/{job_id}` — job-to-job via FAISS.
- `GET /explain/{user_id}/{job_id}` — explanation bundle (matched skills, similar applied jobs, score breakdown, LLM reasoning).
- `GET /pipeline/inspect/{user_id}` — per-stage decomposition for the auditability differentiator (D5).
- `GET /health` — liveness probe.

Pydantic schemas enforce typed request/response. FastAPI auto-generates Swagger UI, which doubles as live API documentation.

---

## 12. Presentation Layer — Streamlit Demo

**Plain English.** Three tabs: a **Job Seeker** view (upload resume, see top jobs with explanations, salary, skill gaps, career path), a **Recruiter** view (pick a job, see top candidates with LLM-drafted outreach), and a **Pipeline Inspector** that visualises the funnel — showing how scores changed at each stage so the system's decisions are auditable.

**Technical.** Streamlit chosen for Python-native rapid prototyping (no JS toolchain, sufficient for portfolio demos). The Recruiter view re-uses the two-tower in reverse direction over a user-embedding FAISS index; same LTR features with (job, user) flipped. The Pipeline Inspector visualises the V1→V8 funnel and is the user-facing payload for differentiator D5 (auditable per-stage decomposition).

---

## 13. Evaluation Framework

**Plain English.** We measure not just accuracy but also whether we cover the catalogue (vs. always recommending the same popular jobs), whether recommendations are diverse, and — uniquely — whether *both sides* of the match are happy.

**Technical.** Metrics in `src/evaluation/metrics.py`:

| Metric | Measures |
|---|---|
| Precision@K, Recall@K | hit-rate quality / coverage |
| NDCG@K | ranking quality (position-weighted) |
| MAP | average precision across all relevant |
| Coverage | fraction of catalogue ever recommended |
| Diversity | average intra-list dissimilarity |
| `bilateral_coverage` | fraction of recs where both sides score above their median |
| `balanced_ranking_ratio` | user-side NDCG ÷ recruiter-side NDCG (closer to 1 = balanced) |
| `two_sided_ndcg` | geometric mean of both sides' NDCG@K |

Pipeline variants V1–V8 (baseline hybrid → full multi-stage → full + reciprocal) are evaluated side-by-side. Cold-start performance is sliced by user activity buckets (0, 1–4, 5–20, 20+) — the system should *degrade gracefully* rather than collapse for sparse users.

---

## 14. Why This Architecture (Defensive Novelty)

**Plain English.** Most job-recommender projects are a single hybrid model. Industrial systems (LinkedIn, Indeed, ZipRecruiter) are one-sided and don't expose how they decide. Our differentiation is the *combination*: grounded LLM explanations citing real ontology IDs, ontology-weighted skill transferability, true reciprocal (two-sided) scoring, a four-stage pipeline including enrichment, and an auditable per-stage inspector.

**Technical.** The five differentiators (D1–D5) are not individually novel — each maps to recent literature (LLM4Rec 2024, ESCO-grounded matching, SIGKDD'24 reciprocal-RS, multi-stage industrial RS, RecSys explainability). The novelty is **integration**: no published system combines all five end-to-end with code. Each differentiator is implemented in a specific file (`src/models/llm_reranker.py`, `src/ontology/skill_graph.py`, `src/models/reciprocal.py`, `src/models/{salary_predictor, skill_gap, career_path}.py`, the Pipeline Inspector tab + `/pipeline/inspect/{user_id}` endpoint) so the claim is verifiable rather than rhetorical.

---

## 15. Tech Stack Summary

| Layer | Tools | Reason |
|---|---|---|
| Language | Python 3.10+ | ML/data-science standard |
| Data | Pandas, NumPy, SciPy (sparse) | mature, sparse-aware |
| Classical text | scikit-learn (TF-IDF) | speed, interpretability |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | semantic, small, CPU-fine |
| Classical CF | Surprise (SVD) | reference baseline |
| Neural | PyTorch (CUDA-aware) | two-tower, BERT4Rec, DeepFM, Mult-VAE, LightGCN |
| Graph ML | PyTorch Geometric | LightGCN |
| ANN | FAISS (`IVFFlat`) | sub-linear retrieval |
| Ranking | XGBoost (`rank:ndcg`) | LambdaMART gold standard |
| LLM | Anthropic Claude API + prompt caching | re-ranking, explanations, query parsing, gap suggestions |
| NLP | spaCy (custom NER) | resume parsing |
| Ontology | NetworkX over ESCO | skill graph |
| API | FastAPI + Uvicorn + Pydantic | typed, async, auto-docs |
| UI | Streamlit | Python-native demo |
| Config | PyYAML + dataclasses | typed, human-readable |
| Tests | Pytest | standard |

---

## 16. End-to-End Walkthrough

A request `GET /recommend/{user_id}` flows as follows:

1. **Profile load.** Structured profile (NER-extracted skills, normalised to ESCO IDs, with predicted next career steps) is fetched.
2. *(Optional)* **Query parse.** Free-text query → structured filters via LLM.
3. **Retrieval.** Two-tower → FAISS top-500; LightGCN top-500; Mult-VAE top-200 (rich-history users only). Union + dedupe → ~800 candidates. Career-path bias optionally re-weights.
4. **Ranking.** LambdaMART + DeepFM ensemble + reciprocal bilateral score → top 20.
5. **Re-ranking.** BERT4Rec scores the sequence-fit; LLM re-orders the top 20 and writes per-item explanations grounded in ESCO + resume entities → top 10.
6. **Enrichment.** Salary prediction, ontology-weighted skill-gap with LLM learning suggestions, career-path overlay attached to each item.
7. **Response.** JSON with: ranked items, per-stage scores (for the Inspector), bilateral score, salary tag, skill gaps, LLM reasoning.
8. **Logging.** Interaction logged; user profile vector (weighted average of interacted job embeddings) recomputed for next request. (No live online learning in current scope — see `future_considerations.md`.)

That, end to end, is the system.
