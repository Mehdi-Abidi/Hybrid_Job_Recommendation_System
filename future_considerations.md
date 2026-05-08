# Future Considerations

This document catalogs features, techniques, and enhancements that were **researched and evaluated but intentionally deferred** from the current project scope. Each entry includes what it does, why it was considered, why it was deferred, and when it would make sense to revisit.

Scope boundaries for the current project:
- **Algorithmic depth is the priority** — Tiers 1-3 of our enhancement research
- **Not prioritizing**: production utility, UI polish, scalability, MLOps, deployment maturity
- **User scale**: 1-2 users, static datasets, no live traffic patterns

---

## A. Deferred Algorithmic Components

### A.1 Contextual Bandit (LinUCB)

**What it does**: Online learning layer that balances exploration (trying uncertain recommendations) vs exploitation (serving confident recommendations). Updates its statistics on every user interaction in real-time, learning from live feedback that frozen trained models cannot react to.

**Why it was considered**: It is genuinely the only component in the pipeline that *learns from live user interactions in real-time*. The two-tower network, LambdaMART, BERT4Rec, etc. are all frozen after offline training — a user clicking "save" on a job right now cannot influence them until the next retraining cycle. The bandit fills that gap.

**Why deferred**: Scope decision — we want to focus the first iteration on getting the multi-stage retrieval-ranking-reranking pipeline solid before layering in online learning. Bandit is not part of the *core* recommendation algorithm; it is an adaptive layer on top.

**When to add it**: After the multi-stage pipeline is working end-to-end and we want a real feedback loop. Implementation is ~2 days and integrates at the final step of Stage 3 (re-ranking), replacing 2-3 of the top-10 slots with exploratory picks and updating arm statistics per interaction.

**File path (future)**: `src/models/bandit.py` — `LinUCBBandit` class.

---

### A.2 Neural Network Fine-Tuning / Incremental Retraining

**What it does**: Instead of retraining neural models (two-tower, BERT4Rec, DeepFM, LightGCN, Mult-VAE) from scratch when new data arrives, load existing weights and train for a few epochs on new interactions. Takes minutes instead of hours, avoids catastrophic forgetting by mixing old and new data.

**Why it was considered**: Production RS cannot afford full retraining cycles. Fine-tuning is the industry-standard pattern for keeping neural models current without discarding learned representations.

**Why deferred**: Our dataset is static (CareerBuilder dump). There is no stream of new data to fine-tune on. Adding fine-tuning infrastructure without new data is purely architectural demonstration — no functional benefit at current scope.

**When to add it**: When the system ingests live job postings or accumulates real user interactions over time. Requires:
- A checkpointing system for neural models
- A data versioning mechanism (e.g., DVC) to track what data the current weights saw
- A fine-tuning script that warm-starts from the latest checkpoint and trains on the delta
- Regularization to prevent catastrophic forgetting (e.g., elastic weight consolidation, or replaying a sample of old data)

---

## B. Tier 4 — MLOps and Production Maturity (Deferred)

### B.1 MLflow Experiment Tracking + Model Registry
**What it does**: Logs every training run with hyperparameters, metrics, and artifacts. Maintains a versioned registry of production models with stage transitions (Staging → Production → Archived).
**Why deferred**: Pure infrastructure with no algorithmic contribution. Valuable for industry portfolios but adds no depth to the recommendation quality.
**When to revisit**: If converting this into a production system or if presenting to industry employers where MLOps maturity is the signal they want.

### B.2 A/B Testing Framework
**What it does**: Route users to variant A or B based on hash; track per-variant metrics with bootstrap confidence intervals for statistical significance.
**Why deferred**: Requires real user traffic to be meaningful. With 1-2 users, no statistically valid A/B test is possible.
**When to revisit**: Only if the system gets deployed with meaningful user traffic.

### B.3 Model Drift & Data Drift Monitoring
**What it does**: Detects when input feature distributions shift over time, alerts when model quality degrades, can trigger automatic retraining.
**Why deferred**: Static dataset means no drift. Monitoring noise with no signal.
**When to revisit**: When real job postings are continuously ingested.

### B.4 Feedback Loop Infrastructure
**What it does**: Logs clicks, applies, dismisses from the UI into a queryable store. Periodic retraining pipeline consumes new feedback. Closes the loop between user behavior and model updates.
**Why deferred**: Depends on live users and fine-tuning infrastructure (A.2) — neither in scope.
**When to revisit**: Together with A.1 (bandit) and A.2 (fine-tuning) when live deployment happens.

### B.5 Dockerization + docker-compose
**What it does**: Containerize API, UI, vector index, model artifacts. Single-command deployment.
**Why deferred**: Production hygiene, not algorithmic depth. Useful when deploying.
**When to revisit**: When the system is being shipped anywhere beyond local dev.

### B.6 CI/CD (GitHub Actions)
**What it does**: Automated testing, linting, and model evaluation on every pull request.
**Why deferred**: Project-management tooling, not an RS concern.
**When to revisit**: When multiple contributors join or when stability guarantees become important.

### B.7 Pydantic Config + Strict Typing
**What it does**: Hydra or Pydantic-Settings for typed configuration; mypy strict mode across the codebase.
**Why deferred**: Code quality improvement, not a new capability.
**When to revisit**: Before any production handoff.

### B.8 Observability (Prometheus + Grafana)
**What it does**: Metric dashboards for recommendation latency, cache hit rate, LLM token usage, click-through rate.
**Why deferred**: Needs traffic to observe. No meaningful metrics at 1-2 users.
**When to revisit**: Only with real deployment.

---

## C. Tier 5 — Fairness, Ethics, and Beyond-Accuracy (Deferred)

### C.1 Diversity-Aware Re-Ranking (MMR)
**What it does**: Maximal Marginal Relevance algorithm that re-ranks top candidates to balance relevance with intra-list diversity. Avoids showing 10 near-identical jobs.
**Why deferred**: Useful quality-of-life improvement but not algorithmic depth. Easy to add later (~1-2 days).
**When to revisit**: After core pipeline is functional; candidate for a polishing phase.

### C.2 Fairness Metrics + Bias Mitigation
**What it does**: Detects whether the system systematically underserves demographic groups. Mitigation via re-weighting or adversarial debiasing.
**Why deferred**: Requires demographic metadata we don't have in the CareerBuilder dataset; meaningful fairness analysis needs statistical user populations.
**When to revisit**: If working with a dataset that has protected-attribute labels and enough users for statistically valid group comparisons.

### C.3 Popularity Bias Correction
**What it does**: Inverse propensity scoring in evaluation to account for popularity bias in logged data. Long-tail item boosting during ranking.
**Why deferred**: Quality refinement rather than core depth. Easier to add once the baseline is measured.
**When to revisit**: During the evaluation-tuning phase if long-tail coverage is poor.

### C.4 Serendipity Metrics
**What it does**: Measures whether recommendations are surprising-but-relevant rather than obvious.
**Why deferred**: Evaluation metric, not a model. Low priority until we have baseline metrics to compare against.
**When to revisit**: As part of the evaluation suite expansion.

---

## D. Tier 6 — User Experience and Interface (Deferred)

### D.1 Production-Style Frontend (Next.js + FastAPI)
**What it does**: Replace Streamlit with a Next.js/React frontend calling the FastAPI backend. More polished UX, filter interactions, real-time updates.
**Why deferred**: UI work, not RS depth. Streamlit is sufficient for demoing the algorithmic work.
**When to revisit**: If productizing or if presenting to non-technical stakeholders where visual polish matters.

### D.2 User Onboarding Flow
**What it does**: Upload resume → parse via NER → confirm extracted skills → show recommendations. Streamlined cold-start experience.
**Why deferred**: Partially overlaps with the Resume NER feature we're already including in Tier 3. A dedicated onboarding flow would be UX polish on top.
**When to revisit**: When the NER component is done; adding a polished onboarding wrapper is a 1-day task.

### D.3 Conversational Interface (Chatbot)
**What it does**: LLM + tool use where the user converses with the system in natural language and the bot calls recommender functions as tools. "Find me remote Python jobs paying >$120k."
**Why deferred**: Overlaps with the Query Understanding feature in Tier 3 but goes further with full conversational state. Scope expansion.
**When to revisit**: After Query Understanding is implemented; extending it into a full conversational agent would be a 3-day addition.

### D.4 Saved Searches + Email Alerts
**What it does**: Persist user search criteria; mock email digest of new matches via APScheduler.
**Why deferred**: Product feature, not RS depth.
**When to revisit**: For a productized version.

### D.5 Explanation Visualization (SHAP)
**What it does**: SHAP-style visualization of LambdaMART feature contributions per recommendation. Visual skill-match highlighting in job descriptions.
**Why deferred**: LLM explanations already provide interpretability. SHAP would add a second interpretability track, useful but redundant for our scope.
**When to revisit**: For a research/academic presentation where quantitative feature attribution is valued.

---

## E. Tier 7 — Data and Infrastructure (Deferred)

### E.1 Real Data Pipeline (Web Scraping / Live APIs)
**What it does**: Scrape Indeed/LinkedIn (within ToS) or use Adzuna/Remotive APIs for live job data. Daily ingest pipeline with Airflow or Prefect.
**Why deferred**: Data infrastructure, not RS depth. CareerBuilder static dataset is sufficient for demonstrating algorithms.
**When to revisit**: If the system is being deployed or if the Kaggle dataset proves insufficient in diversity.

### E.2 Vector Database Upgrade (Qdrant / Weaviate)
**What it does**: Replace FAISS with a production vector database supporting metadata filtering, hybrid search, and persistence.
**Why deferred**: FAISS is sufficient at 1-2 user scale. Qdrant adds containerization and service overhead without algorithmic benefit.
**When to revisit**: If scaling beyond single-machine deployment.

### E.3 Hybrid Search (Dense + Sparse with RRF)
**What it does**: Combine dense embeddings (semantic) with sparse BM25 (keyword) using reciprocal rank fusion. Elasticsearch or Weaviate for sparse index.
**Why deferred**: Our two-tower + FAISS already covers dense retrieval well. TF-IDF covers sparse in the baseline. Adding a full BM25 index is infrastructure work.
**When to revisit**: If retrieval recall becomes a bottleneck.

### E.4 Data Versioning (DVC)
**What it does**: Version datasets and model artifacts in git alongside code. Reproducibility.
**Why deferred**: Reproducibility tooling, not algorithmic work.
**When to revisit**: When the project matures to a point where data lineage matters.

---

## F. Tier 8 — Research-Grade / Exotic (Deferred)

### F.1 Reinforcement Learning Recommender (DQN / Actor-Critic)
**What it does**: Full RL formulation where state = user+session, action = recommend an item, reward = user engagement. Long-horizon optimization for cumulative user satisfaction rather than per-request relevance.
**Why deferred**: Very hard to do well without large-scale user interaction data. High implementation cost, high risk of not working, low marginal value over multi-stage ranking. More research project than portfolio piece.
**When to revisit**: Only as a dedicated research follow-up, not as an extension of this project.

### F.2 Causal Inference for Evaluation (Off-Policy Evaluation)
**What it does**: Uses inverse propensity scoring and doubly-robust estimators to estimate how a new model *would have* performed on logged data, without deploying it.
**Why deferred**: Requires logged policy probabilities and real deployment data. Nothing to evaluate offline-causally without those.
**When to revisit**: If we deploy with logging and want to evaluate model changes before rolling them out.

### F.3 Multi-Task Learning
**What it does**: Single neural model with a shared encoder and multiple task-specific heads predicting apply probability, save probability, salary expectation, tenure prediction, etc.
**Why deferred**: Architecturally complex; marginal gains over separate specialized models; harder to debug and evaluate.
**When to revisit**: Research extension if the specialized models hit a ceiling.

### F.4 Federated Learning Simulation
**What it does**: Simulates privacy-preserving training where user data stays on device and only model updates are shared.
**Why deferred**: Overkill for this scope. Privacy-preserving ML is a distinct research area.
**When to revisit**: Not in the scope of this project.

### F.5 Differential Privacy
**What it does**: Adds calibrated noise during training to provide mathematical privacy guarantees.
**Why deferred**: Niche; adds noise that reduces model quality without corresponding benefit at our scope.
**When to revisit**: If working with sensitive user data under regulatory requirements.

### F.6 Counterfactual Explanations
**What it does**: Answers "what would have changed my recommendations?" — e.g., "You would have been recommended Job X if you had React experience."
**Why deferred**: Advanced explainability beyond what LLM explanations already provide. High implementation cost.
**When to revisit**: For a research-focused extension on interpretability.

### F.7 Meta-Learning for Cold-Start (MAML-style)
**What it does**: Few-shot adaptation — learns how to quickly specialize to a new user from a handful of interactions.
**Why deferred**: Pure research territory. Two-tower's feature-based cold-start is sufficient for our needs.
**When to revisit**: Not in scope.

---

## G. Deferred Model Architectures (Not in Tier 2 Core)

### G.1 Neural Collaborative Filtering (NCF)
**What it does**: MLP-based alternative to SVD for collaborative filtering. Simpler than two-tower.
**Why deferred**: Largely subsumed by two-tower. NCF is a stepping-stone architecture rather than a complementary one.
**When to revisit**: Useful as a baseline comparison in an evaluation-heavy follow-up.

### G.2 SASRec (Self-Attentive Sequential Recommendation)
**What it does**: Causal (autoregressive) transformer for sequential recommendation. Alternative to BERT4Rec.
**Why deferred**: BERT4Rec was chosen for Tier 2 because bidirectional attention typically outperforms on shorter sequences. SASRec would be a comparison point, not a new capability.
**When to revisit**: For an ablation study comparing sequential architectures.

---

## Summary Table — Priority for Future Work

| Feature | Category | Complexity | Time Estimate | When It Becomes Relevant |
|---------|----------|------------|---------------|-------------------------|
| LinUCB Bandit | Online learning | Medium | 2 days | After core pipeline works |
| NN Fine-tuning | Online learning | Medium | 3 days | With live data stream |
| Diversity (MMR) Re-ranking | Quality | Low | 1-2 days | Polish phase |
| MLflow + Model Registry | MLOps | Low-Medium | 2-3 days | Industry presentation |
| Dockerization | MLOps | Low | 1 day | Deployment |
| Feedback Loop Infra | MLOps | Medium | 2 days | With real users |
| User Onboarding Flow | UX | Low | 1 day | After NER is built |
| Conversational Interface | UX | Medium | 3 days | After Query Understanding |
| SHAP Explanations | Interpretability | Medium | 2 days | Research presentation |
| Real Data Pipeline | Data | Medium | 3-4 days | Productization |
| Vector DB Upgrade | Infra | Medium | 2 days | Scale-out |
| Hybrid Search (BM25) | Retrieval | Medium | 2 days | Recall bottleneck |
| A/B Testing Framework | MLOps | Medium | 2 days | With real traffic |
| Drift Monitoring | MLOps | Medium | 2 days | With live data |
| Fairness Metrics | Ethics | Medium | 2-3 days | With demographic data |
| Causal Evaluation | Research | High | 3-4 days | Production logging in place |
| Multi-Task Learning | Research | High | 4-5 days | Research extension |
| RL Recommender | Research | Very High | 7-10 days | Dedicated research project |
| GNN alternatives beyond LightGCN | Research | Very High | 5-7 days | Research comparison |
| NCF baseline | Research | Medium | 2-3 days | Ablation study |
| SASRec comparison | Research | Medium-High | 3-4 days | Sequential architecture ablation |
| Federated Learning | Research | Very High | N/A | Out of scope |
| Differential Privacy | Research | High | N/A | Regulatory context |
| Meta-Learning (MAML) | Research | Very High | N/A | Out of scope |

---

## Closing Note

Everything in this document was intentionally excluded to keep the current project focused on **algorithmic and domain depth** (Tiers 1-3). The exclusions are not rejections — they are deferrals, most of them with clear triggers for when they would make sense to revisit.

The core principle guiding exclusions: if a feature's value depends on real-world deployment conditions (live traffic, demographic data, continuous data streams, regulatory compliance), it is premature to build it at current scope.
