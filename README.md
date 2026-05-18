# SaaS Feature Recommendation System

A two-stage recommender for SaaS feature adoption built on synthetic B2B usage data.

**Goal:** Recommend the right product features to the right customers to reduce churn.

---

## Pipeline

```
generate_data.py → build_interaction_matrix.py → build_evaluation_split.py
                                                          ↓
                                              candidate_generation.py  (Stage 1)
                                                          ↓
                                                    ranker.py          (Stage 2)
                                                          ↓
                                                  evaluation.py
```

## Setup

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
uvicorn api.rec_api:app --reload --port 8000
```

---

## Theory & Derivations

### 1. Implicit Feedback Scoring

Raw event counts are compressed logarithmically to prevent power-user dominance:

$$s_{ui} = \log(1 + c_{ui})$$

where $c_{ui}$ is the raw usage count for user $u$ on feature $i$. This yields a sparse matrix $R \in \mathbb{R}^{U \times I}$ (8,000 × 20, ~76% zeros).

---

### 2. Matrix Factorization via ALS

**Objective:** Decompose $R$ into latent factor matrices $P \in \mathbb{R}^{U \times k}$ (users) and $Q \in \mathbb{R}^{I \times k}$ (items) such that:

$$\hat{R} = PQ^\top \approx R$$

**Loss function** (weighted, to handle implicit feedback):

$$\mathcal{L} = \sum_{u,i} c_{ui}(r_{ui} - p_u q_i^\top)^2 + \lambda(\|P\|_F^2 + \|Q\|_F^2)$$

where $c_{ui} = 1 + \alpha \cdot s_{ui}$ is a confidence weight and $\lambda$ is L2 regularization.

**The ALS trick:** Both $P$ and $Q$ are unknown, making this non-convex — unsolvable by standard OLS. ALS sidesteps this by alternating:

**Fix $Q$, solve for $p_u$:**

$$p_u = (Q^\top C^u Q + \lambda I)^{-1} Q^\top C^u r_u$$

**Fix $P$, solve for $q_i$:**

$$q_i = (P^\top C^i P + \lambda I)^{-1} P^\top C^i r_i$$

Each sub-problem is now a standard **Gauss OLS normal equation** $(X^\top X + \lambda I)\theta = X^\top y$, which has a closed-form solution. Alternating 50 times converges to a local minimum.

---

### 3. LightFM (Hybrid Model)

Extends matrix factorization with side-information. User and item representations are built from feature embeddings:

$$e_u = \sum_{f \in \mathcal{F}_u} e_f, \quad e_i = \sum_{g \in \mathcal{G}_i} e_g$$

Prediction score (WARP loss variant):

$$\hat{r}_{ui} = e_u \cdot e_i + b_u + b_i$$

WARP loss optimizes ranking directly by sampling negatives until a violation is found, then weighting the update by the approximate rank of the violation.

---

### 4. Candidate Merging

Each model produces a ranked list; the union forms the candidate pool:

$$\mathcal{C}_u = \left(\text{ALS}_{50}(u) \cup \text{LightFM}_{50}(u)\right) \setminus \mathcal{H}_u$$

where $\mathcal{H}_u$ is the set of features user $u$ has already adopted.

---

### 5. LightGBM LambdaRank (Stage 2)

LambdaRank defines gradients directly from swapped-pair NDCG gains rather than from a differentiable loss:

$$\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma(\hat{s}_i - \hat{s}_j)}} \cdot |\Delta \text{NDCG}_{ij}|$$

where $|\Delta \text{NDCG}_{ij}|$ is the absolute change in NDCG from swapping items $i$ and $j$. LightGBM fits a gradient-boosted tree ensemble to these $\lambda$ values.

**Key feature: churn signal boost:**

$$b_u^{(i)} = \text{churn\_prob}_u \times \text{churn\_reduction\_score}_i$$

This biases recommendations toward features that are empirically "sticky" for at-risk customers.

---

### 6. Evaluation (Temporal Hold-out)

For each user, the most recently adopted 20% of features are held out. The remaining 80% form the training matrix. Metrics are computed at cutoff $k$:

$$\text{Precision@}k = \frac{|\hat{R}_k \cap T_u|}{k}, \quad \text{Recall@}k = \frac{|\hat{R}_k \cap T_u|}{|T_u|}$$

$$\text{NDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}, \quad \text{DCG@}k = \sum_{j=1}^{k} \frac{\mathbb{1}[r_j \in T_u]}{\log_2(j+1)}$$

where $T_u$ is the held-out ground truth set and $\hat{R}_k$ is the model's top-$k$ predictions.

---

## API

```
GET /recommend/{customer_id}?top_k=10
GET /similar-users/{customer_id}?top_n=5
GET /feature-affinity/{feature_id}?top_n=100
GET /health
GET /metrics/summary
```

---

## Notes

- Cold start triggers at `tenure_days < 30` or `recent_interactions < 3`; falls back to plan-tier popularity ranking.
- ALS: 50 iterations, rank $k=64$, $\lambda=0.01$, $\alpha=40$.
- Ranker hyperparameters tuned via Optuna (40 trials, default).
