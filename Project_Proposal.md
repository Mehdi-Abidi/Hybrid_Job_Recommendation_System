# Project Proposal

---

## Header

| Field | Value |
|---|---|
| **Affiliation** | _[Your University / Institution]_ |
| **Specialization** | Artificial Intelligence & Recommender Systems |
| **Department** | _[Department of Computer Science / Software Engineering]_ |
| **Domain** | AI / Machine Learning / Information Retrieval |
| **Proposal Title** | Hybrid Job Recommendation System with Reciprocal Matching, Skill-Ontology Grounding, and LLM-Explained Multi-Stage Ranking |
| **Desired Starting Date** | 2026-05-01 |
| **Proposed Duration (Max months)** | 3 months (~63 working days) |
| **Investigator's Affiliation (Dept. / College)** | _[Your Department / College]_ |
| **Academic Title** | _[Undergraduate / Graduate Researcher]_ |
| **Team Members** | Principal Investigator: _[Your Name]_; Faculty Advisor: _[Advisor Name]_ |

---

## Abstract

Job seekers and recruiters waste enormous time on mismatched search-and-application loops. While platforms like LinkedIn, Indeed, and ZipRecruiter operate hybrid recommender systems at scale, public engineering write-ups (LiRank, ZipRecruiter Tech, Indeed/O'Reilly) and recent academic surveys (Springer JBD 2025, Frontiers AI 2025) reveal persistent unsolved problems: **opaque match scores**, **one-sided modelling** (job-seeker-only or recruiter-only), **flat skill matching** without ontology grounding, and **shallow explanations** that do not reference the user's actual skills or career trajectory.

This project builds a **multi-stage hybrid job recommender** that addresses these gaps simultaneously. The system is organised as a four-stage pipeline — **Retrieval → Ranking → Re-ranking → Enrichment** — and contributes five concrete differentiators: (D1) LLM-grounded explanations citing ESCO skill IDs and resume NER entities; (D2) ontology-weighted skill transferability so "Django" and "Flask" register as related rather than disjoint; (D3) **first-class reciprocal recommendation** computing a bilateral score `P(user applies) × P(recruiter shortlists)`, evaluated with the new metrics from SIGKDD '24; (D4) a Stage-4 enrichment pass producing salary predictions, skill-gap roadmaps, and career-path forecasts per recommendation; (D5) auditable per-stage attribution exposed through a Pipeline Inspector UI. The architecture combines classical baselines (content-based, SVD collaborative filtering, popularity), neural models (Two-Tower, BERT4Rec, DeepFM, LightGCN, Mult-VAE), learning-to-rank (LambdaMART), and an LLM re-ranker over real-world data (LinkedIn postings, resume corpus, Train_rev1 UK salary set).

Expected outcomes: a working FastAPI + Streamlit demo, a reproducible evaluation notebook comparing seven pipeline variants (V1 baseline → V7 full system) on Precision@10, NDCG@10, bilateral coverage, balanced-ranking ratio, coverage, and diversity, and a written report defending each differentiator against current industry and academic state-of-the-art.

---

## Keywords

Recommender Systems, Hybrid Filtering, Two-Tower Neural Networks, Learning-to-Rank, Reciprocal Recommendation, Skill Ontology (ESCO), Large Language Models, Explainable AI, Job Matching, Multi-Stage Pipeline.

---

## 1. Introduction

The mismatch between job seekers and job postings is a structural inefficiency in the global labour market. LinkedIn alone handles over 60 million jobs and 1 billion members; Indeed reports 200 million unique monthly visitors and processes billions of input signals per day. Despite this scale and the heavy ML investment of incumbents, the user-facing experience remains thin: a job seeker is shown a ranked list with a single match score and no actionable reasoning, while a recruiter is shown a parallel ranked list of candidates that the seeker may never realistically be shortlisted for. The two pipelines do not co-optimise for mutual fit.

Public engineering blogs and academic surveys consistently identify the same open problems:
- **Opacity**: even within LinkedIn's LiRank pipeline, decision attribution at the candidate level is non-trivial.
- **One-sidedness**: most production systems are seeker-centric or recruiter-centric, not bilaterally optimised.
- **Flat skill matching**: "Django wanted, user has Flask" is treated as a miss in both string-match and naive embedding approaches.
- **Shallow explanations**: feature weights and SHAP values are technical artefacts, not user-actionable insights.
- **No learning roadmaps**: the recommender stops at "here are jobs" and leaves the user to figure out what skills they would need to acquire to qualify for the better matches.

This project targets these five gaps directly. The deliverable is a working hybrid recommender that demonstrates each fix end-to-end, evaluated on real data, with code, metrics, and a demo UI.

---

## 2. Literature Review

The literature on job recommendation has matured rapidly in the last three years. We organise the review by the gap each line of work addresses.

**Hybrid baselines and surveys.** The Springer Journal of Big Data 2025 survey and Frontiers in AI 2025 article both describe the modern hybrid stack (content + CF + popularity, optionally fused with a learned ranker) and explicitly flag explainability, reciprocity, ontology grounding, and dataset standardisation as open problems. The arXiv 2111.13576 review (2021, still widely cited) catalogues earlier hybrids and highlights cold-start handling.

**Industrial multi-stage architectures.** LinkedIn's LiRank (arXiv 2402.06859) describes a two-tower retrieval stage feeding into a ranking stage, with knowledge-distillation-based quality recovery. ZipRecruiter's tech blog describes a multimodal learning approach over 60+ engineered factors with an in-progress GNN unification. Indeed (O'Reilly) describes a hybrid offline + online pipeline. None of these systems publish per-stage decision attribution to end users.

**Reciprocal recommendation.** The SIGKDD '24 paper *"Revisiting Reciprocal Recommender Systems: Metrics, Formulation, and Method"* introduces five new evaluation metrics covering bilateral coverage, balanced ranking, and two-sided NDCG. AAAI '24 *"Knowledge-Aware Explainable Reciprocal Recommendation"* and SIGIR '24 *"MIRROR: Multi-View Reciprocal Recommender for Online Recruitment"* extend reciprocal modelling with knowledge graphs and multi-view embeddings respectively. This is the most active 2024 research frontier in our space and is **not in production at scale anywhere public**.

**LLM-enhanced job recommendation.** GIRL (arXiv 2307.02157) introduces generative job recommendations with LLM grounding. JobRecoGPT (arXiv 2309.11805) demonstrates LLM-powered explanations. SkillSync (IJERT 2025) extends to skill-gap analysis. The integration of these three directions in a single system remains absent.

**Skill ontologies.** ESCO (3000+ occupations, 13K+ skills) and O*NET are open, well-maintained, and freely usable, yet the 2024 ScienceDirect transformer + O*NET paper notes their grounding remains rare in production-style hybrids despite zero licence cost.

**Identified gaps.** Our project targets the five-way intersection of: per-stage auditable pipelines, LLM-grounded reasoning, ontology-weighted skill matching, reciprocal scoring, and Stage-4 enrichment (salary + skill-gap + career path). We have found no published system that combines all five.

---

## 3. Description (Rationale) of the Proposed Work

### 3.1 Why job recommendation over more common alternatives

Movie and music recommendation are saturated portfolio domains with limited remaining novelty. Job recommendation is a billion-dollar live problem with: a rich feature space (structured + unstructured), a natural cold-start problem (new users and new postings daily), a multi-stakeholder structure that demands reciprocal modelling, and direct alignment with current 2024-2025 research frontiers (LLM4Rec, reciprocal RS, skill-graph reasoning).

### 3.2 Why a multi-stage neural + LLM pipeline over a classical hybrid

A classical hybrid blends three models in one pass and limits both quality and interpretability. A multi-stage pipeline separates concerns: retrieval optimises **recall** (do not lose good candidates), ranking optimises **precision** (order them well), re-ranking adds **reasoning** (LLM context), enrichment adds **decision support** (salary, skill gaps, career path). This mirrors how YouTube, Meta, and LinkedIn actually architect production systems while making each stage individually inspectable.

### 3.3 Why these five differentiators, specifically

Each differentiator targets a published open problem (Section 2) and is implementable within the project's three-month scope. The differentiators are deliberately compounding: the LLM re-ranker (D1) consumes the resume NER and ESCO entities (D2); reciprocal scoring (D3) extends the same two-tower embedding space without additional model families; Stage-4 enrichment (D4) reuses skill-gap features already computed for ranking; per-stage auditability (D5) is essentially free given the multi-stage architecture.

---

## 4. Research Objectives

### Table 4.1 — Objectives and their Approach Mapping

| # | Objective | Approach |
|---|---|---|
| 1 | Build a working classical hybrid baseline | Content-based (sentence-transformer embeddings) + SVD collaborative filtering + popularity baseline + tiered combiner |
| 2 | Implement neural retrieval (D differentiator support) | Two-Tower neural network (PyTorch, in-batch softmax) + FAISS ANN index |
| 3 | Implement learning-to-rank for ranking stage | LambdaMART via XGBoost `rank:ndcg` over 11 engineered cross-features |
| 4 | **Deliver D1: LLM-grounded explanations** | LLM re-ranker (Azure GPT-4o-mini / Anthropic Claude) over the top-20 with structured-JSON output citing ESCO skill IDs and resume NER entities |
| 5 | **Deliver D2: Ontology-weighted skill transferability** | Load ESCO into a NetworkX graph; compute skill match using shortest-path / Wu-Palmer similarity; expose as a ranking feature |
| 6 | **Deliver D3: First-class reciprocal scoring** | Train inverse `JobToUserTower`; compute bilateral score `P(apply) × P(shortlist)`; evaluate with SIGKDD '24 metrics |
| 7 | **Deliver D4: Stage-4 enrichment** | Salary regressor (XGBoost on Train_rev1) + skill-gap analyser + career-path predictor, run on final top-10 |
| 8 | **Deliver D5: Per-stage auditability** | Capture per-stage scores in the pipeline orchestrator; expose via `/pipeline/inspect` endpoint and Streamlit Pipeline Inspector tab |

### Table 4.2 — Objective Phases and Tasks Mapping

| Objective | Phases | Tasks |
|---|---|---|
| 1 | 2, 3 | Data acquisition, preprocessing, classical model fits |
| 2, 3, 7 | 4 | Two-Tower training, FAISS index, LTR training, salary/skill-gap/career models |
| 4, 5, 7 | 4 | LLM re-ranker, ESCO loader, ontology features, enrichment integration |
| 4 | 4 | Reciprocal tower training, bilateral scorer, integration into Stage 2 ensemble |
| 5 & 6 | 5 | Evaluation framework, reciprocal metrics, V1-V7 comparison notebook |
| 7 & 8 | 6 | FastAPI endpoints, Streamlit Pipeline Inspector, demo polish |

---

## 5. Scope

### In scope
- A functional multi-stage recommender producing top-10 recommendations with explanations, salary estimates, skill-gap roadmaps, and career-path suggestions.
- Reciprocal scoring with bilateral metrics.
- Real datasets: LinkedIn postings (~123 K), resume corpus (~2.5 K), Train_rev1 UK salary set (~240 K).
- FastAPI backend + Streamlit demo UI.
- Reproducible evaluation comparing seven pipeline variants (V1 baseline → V7 full).

### Out of scope (deferred)
- Production scalability (the system is built for 1–2 concurrent users, not millions).
- Live data scraping or streaming embedding updates.
- MLOps infrastructure (MLflow, A/B testing, drift monitoring, CI/CD).
- Fairness and bias auditing (acknowledged as critical but beyond timeline).
- Mobile or production frontend.

---

## 6. Research Design

The project executes seven phases.

**6.1.1 Phase 1 — Comprehensive Literature Review**: cover hybrid RS surveys, multi-stage architectures, reciprocal RS, LLM4Rec, and skill ontologies. Output: structured bibliography aligned to the five differentiators.

**6.1.2 Phase 2 — Data Collection**: download and stage three real datasets — LinkedIn postings (Kaggle `arshkon/linkedin-job-postings`), a resume corpus (Kaggle `Resume.csv`), and Train_rev1 (Kaggle `airiddha/trainrev1`). Implement a fallback synthetic generator for offline / classroom-mode runs.

**6.1.3 Phase 3 — Data Pre-processing & Feature Extraction**: clean and deduplicate jobs and users; extract resume entities via spaCy NER + regex fallback; normalise extracted skills against the ESCO ontology; build implicit-rating interactions (view = 1, save = 3, apply = 5); construct sparse interaction matrices and time-based train/test splits.

**6.1.4 Phase 4 — Multi-Stage Neural Pipeline + Reciprocal Matching**:
implements the four-stage architecture: **(Stage 1)** Two-Tower neural retrieval + FAISS ANN + LightGCN + Mult-VAE; **(Stage 2)** LambdaMART + DeepFM ranking ensemble with ontology-weighted skill features; **(Stage 3)** BERT4Rec sequential model + LLM re-ranking (Azure GPT-4o-mini) with skill-ontology-grounded reasoning; **(Stage 4)** salary predictor + skill-gap analyser + career-path forecaster. The **reciprocal layer** trains an inverse `JobToUserTower` and computes `bilateral_score = P(apply) × P(shortlist)`, integrated into the Stage-2 ensemble and surfaced as a primary ranking signal.

**6.1.5 Phase 5 — Evaluation**: implement Precision@K, Recall@K, NDCG@K, MAP, coverage, intra-list diversity, plus the reciprocal metrics from SIGKDD '24 (bilateral coverage, balanced-ranking ratio, two-sided NDCG). Compare seven pipeline variants V1 (baseline hybrid) → V7 (full multi-stage + reciprocal + enrichment).

**6.1.6 Phase 6 — Tool Development**: FastAPI service with `/recommend/multi-stage`, `/pipeline/inspect`, `/explain`, `/salary/predict`, `/skill-gap`, `/recommend/new-user` endpoints; Streamlit UI with four tabs (Job Seeker, Recruiter, Pipeline Inspector, Tools).

**6.1.7 Phase 7 — Writing-up and Documentation**: project report covering literature, methodology, results, and the differentiation defence; README, evaluation notebooks, and runbooks for reproduction.

---

## 7. Research Methodology

### 7.1 Stage 1 — Retrieval (~500 candidates)
- **Two-Tower** (PyTorch): user tower MLP over (multi-hot ESCO skills + experience bin + location one-hot + 384-dim resume embedding) → 128-dim; job tower symmetric over job features → 128-dim. Trained with in-batch softmax contrastive loss. CUDA-accelerated.
- **FAISS** index over the 128-dim job embedding space (`IVFFlat`, fallback `Flat`).
- **LightGCN** on the user-job-skill bipartite graph as an alternative retrieval source; **Mult-VAE** as a generative auxiliary for users with rich history. Union-and-deduplicate yields ~700–900 candidates forwarded to Stage 2.

### 7.2 Stage 2 — Ranking (~20)
- **LambdaMART** (XGBoost `rank:ndcg`) over engineered cross-features: two-tower dot product, content cosine, collab predicted rating, popularity, **ESCO ontology-weighted skill overlap** (D2), location match, experience match, salary alignment, recency, user category apply rate, **bilateral score** (D3).
- **DeepFM** as a parallel ranker; ensemble = 0.6 × LambdaMART + 0.4 × DeepFM (weights tuned on validation).

### 7.3 Stage 3 — Re-ranking (top-10)
- **BERT4Rec** scores candidates against the user's recent interaction sequence for warm users.
- **LLM re-ranker** (Azure GPT-4o-mini): prompt includes user profile summary (NER-extracted entities, normalised ESCO skills, experience, location), candidate job summaries, and an instruction to return ranked job IDs with per-job reasoning citing transferable skills. Prompt-cached system instructions for cost control. Graceful fallback if no API key is configured.

### 7.4 Stage 4 — Enrichment (D4)
- **Salary predictor**: XGBoost regressor trained on Train_rev1 (GBP→USD), features = title embedding + location + required skills + experience.
- **Skill-gap analyser**: `required_skills − user_skills` weighted by ESCO graph distance; LLM-suggested learning resources per gap.
- **Career-path forecaster**: title-transition Markov model + transformer over title sequences extracted from the resume corpus; biases retrieval and surfaces a 1–3 step trajectory.

### 7.5 Reciprocal Layer (D3)
Train `JobToUserTower(job_features) → 128-dim` and `RecruiterPreferenceModel(user_features) → 128-dim` on the same interaction data with roles flipped. Compute:
```
bilateral_score = sigmoid(user_tower · job_tower) × sigmoid(jtu_tower · user_features)
                = P(user applies)              × P(recruiter shortlists)
```
Evaluated with bilateral coverage, balanced-ranking ratio, two-sided NDCG (SIGKDD '24).

### 7.6 Auditability (D5)
The pipeline orchestrator captures per-candidate, per-stage scores. The `/pipeline/inspect/{user_id}` endpoint returns a JSON funnel; the Streamlit Pipeline Inspector tab renders 500 → 50 → 20 → 10 with stage contributions and natural-language explanations.

### 7.7 Tech stack
Python 3.10+, PyTorch (CUDA), FAISS, XGBoost, scikit-learn, sentence-transformers, spaCy, NetworkX, pandas, FastAPI + Uvicorn, Streamlit, pytest, Azure OpenAI / Anthropic SDK.

---

## 8. Management & Working Plans with Team Roles

### 8.1 Management Plan, Research Team, Roles, and Tasks

The project is delivered by a small team. The Principal Investigator owns code, experiments, and the report; the Faculty Advisor reviews milestones and the final report.

#### Table 8.1 — Role and Involvement Duration of Research Team

| Team Member | Role | Duration (months) |
|---|---|---|
| _[Your Name]_ — Principal Investigator | Architecture, implementation, evaluation, writing | 3 |
| _[Advisor Name]_ — Faculty Advisor | Milestone review, methodology guidance, report review | 3 (advisory, ~2 hrs/week) |

#### Table 8.2 — Project Timeline Mapped with Team Roles and Phases

| Days | Phase | Owner | Deliverable |
|---|---|---|---|
| 1–4 | 1, 2 | PI | Lit review draft, datasets staged |
| 5–7 | 3 | PI | Cleaned data, ESCO loaded, NER pipeline working |
| 8–22 | 4 (Stage 1 + 2) | PI | Two-Tower + FAISS + LTR end-to-end |
| 23–28 | 4 (Stage 3) | PI | BERT4Rec + LLM re-ranker integrated |
| 29–35 | 4 (Stage 4) | PI | Salary, skill-gap, career-path enrichment |
| 36–40 | 4 (Reciprocal) | PI | `JobToUserTower`, bilateral scorer, integrated |
| 41–48 | 5 | PI + Advisor review | V1–V7 evaluation, reciprocal metrics |
| 49–55 | 6 | PI | FastAPI + Streamlit Pipeline Inspector |
| 56–63 | 7 | PI + Advisor review | Final report, README, demo run-through |

### 8.2 Project Outcomes and Deliverables

1. **Working code repository** with classical, neural (Tier 1 + Tier 2), domain (Tier 3), and reciprocal models, all CUDA-enabled where applicable.
2. **Reproducible evaluation notebook** comparing V1 baseline through V7 full pipeline on Precision@10, NDCG@10, Recall@50, coverage, diversity, bilateral coverage, balanced-ranking ratio, two-sided NDCG.
3. **FastAPI backend** with documented endpoints (Swagger UI auto-generated).
4. **Streamlit demo UI** with four tabs (Job Seeker, Recruiter, Pipeline Inspector, Tools).
5. **Written project report** including the differentiation defence (`how_we_are_different.md`).
6. **Test suite** (≥60 tests) covering preprocessing, models, metrics, API, reciprocal scoring.

---

## 9. Utilization

The deliverables benefit several stakeholder groups directly:

- **Job seekers** receive recommendations with actionable reasoning, projected salary, missing-skills roadmaps, and career trajectory — turning a passive list into a planning tool.
- **Recruiters** (via the Recruiter tab) get bilaterally-scored candidate lists where matches reflect mutual fit, reducing wasted outreach.
- **University placement offices and career counsellors** can use the explanation + skill-gap output to advise students concretely on how to qualify for target roles.
- **Researchers** get a reference implementation of reciprocal scoring with SIGKDD '24 metrics and ontology-weighted skill matching — both currently underexplored in published systems.
- **Open-source ecosystem**: the codebase is structured for reuse — the ESCO loader, Two-Tower trainer, reciprocal scorer, and Pipeline Inspector are individually adoptable.

---

## References

1. Springer Journal of Big Data (2025). *Job recommender systems: a systematic literature review, applications, open issues, and challenges*. https://link.springer.com/article/10.1186/s40537-025-01173-y
2. Frontiers in Artificial Intelligence (2025). *Explainable person–job matching*. https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1660548/pdf
3. arXiv 2111.13576 (2021). *Job Recommender Systems: A Review*. https://arxiv.org/pdf/2111.13576
4. arXiv 2402.06859 (2024). *LiRank: Industrial Large Scale Ranking Models at LinkedIn*. https://arxiv.org/html/2402.06859v1
5. ZipRecruiter Tech (2024). *Multimodal Learning for Employment Marketplace Recommendation*. https://medium.com/ziprecruiter-tech/multimodal-learning-for-employment-marketplace-recommendation-ee67bdbede53
6. O'Reilly. *Algorithms and architecture for job recommendations* (Indeed). https://www.oreilly.com/content/algorithms-and-architecture-for-job-recommendations/
7. SIGKDD '24. *Revisiting Reciprocal Recommender Systems: Metrics, Formulation, and Method*. https://arxiv.org/html/2408.09748v1
8. SIGIR '24. *MIRROR: A Multi-View Reciprocal Recommender System for Online Recruitment*. https://dl.acm.org/doi/10.1145/3626772.3657776
9. AAAI '24. *Knowledge-Aware Explainable Reciprocal Recommendation*. https://people.ece.ubc.ca/minchen/min_paper/2024/2024-AAAI.pdf
10. arXiv 2409.00720 (2024). *Fair Reciprocal Recommendation in Matching Markets*. https://arxiv.org/abs/2409.00720
11. arXiv 2307.02157. *GIRL: Generative Job Recommendations with Large Language Model*. https://arxiv.org/pdf/2307.02157
12. arXiv 2309.11805. *JobRecoGPT: Explainable Job Recommendations using LLMs*. https://arxiv.org/pdf/2309.11805
13. IJERT (2025). *SkillSync: An Explainable AI Framework for Resume Evaluation, Skill Gap Analysis, and Career Alignment*. https://www.ijert.org/skillsync-an-explainable-ai-framework-for-resume-evaluation-skill-gap-analysis-and-career-alignment-ijertconv14is010027
14. ScienceDirect (2024). *A novel approach for job matching and skill recommendation using transformers and the O*NET database*. https://www.sciencedirect.com/science/article/pii/S2214579625000048
15. ESCO — European Skills, Competences, Qualifications and Occupations. https://esco.ec.europa.eu/
16. O*NET OnLine — Occupational Information Network. https://www.onetonline.org/
17. He, X. et al. (2020). *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*. SIGIR '20.
18. Sun, F. et al. (2019). *BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer*. CIKM '19.
19. Guo, H. et al. (2017). *DeepFM: A Factorization-Machine based Neural Network for CTR Prediction*. IJCAI '17.
20. Liang, D. et al. (2018). *Variational Autoencoders for Collaborative Filtering*. WWW '18.
21. Burges, C. (2010). *From RankNet to LambdaRank to LambdaMART: An Overview*. Microsoft Research.
22. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP '19.
23. Johnson, J., Douze, M., & Jégou, H. (2017). *Billion-scale similarity search with GPUs* (FAISS). arXiv 1702.08734.
