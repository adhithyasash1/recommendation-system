# SaaS Feature Recommendation System

A production-style, two-stage recommender engine for SaaS feature adoption, built on synthetic B2B usage data.

**The core objective:** Recommend the right product features to the right customers so they get more value from the software and are less likely to cancel their subscription (churn).

---

## How It Works (System Overview)

The system is a classic **two-stage recommendation pipeline**:

1. **Stage 1 — Candidate Generation (`src/candidate_generation.py`):** Two ML models (ALS + LightFM) independently cast a wide net and each nominate their top 50 feature candidates per customer. Their shortlists are merged and de-duplicated, and any features the customer already uses are removed.
2. **Stage 2 — Ranking (`src/ranker.py`):** A LightGBM LambdaRank model acts as the final judge, re-ranking the merged candidate pool using rich account context, plan tier, adoption patterns, and a key churn-awareness signal: `churn_signal_boost = churn_probability × churn_reduction_score`.

---

## Project Layout

```text
data/          Synthetic data generation, interaction matrix, train/test split
src/           Candidate generation, ranker, cold-start fallback, evaluation
api/           FastAPI app and Pydantic schemas
models/        Trained ALS, LightFM, and LightGBM artifacts
outputs/       Evaluation reports and plots
logs/          API logs
```

---

## Setup

The pinned ML stack is safest on **Python 3.11**. `scikit-learn` uses `1.4.1.post1` because the original `1.4.1` pin is not available for this environment.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

> If your shell has a `python` alias inside the virtualenv, use the exact execution order below. If not, replace `python` with `.venv/bin/python`.

---

## Execution Order

```bash
# 1. Generate synthetic data (customers, features, events)
python data/generate_data.py

# 2. Build the user–feature interaction matrix
python data/build_interaction_matrix.py

# 3. Create the train/test evaluation split
python data/build_evaluation_split.py

# 4. Train Stage 1 candidate retrieval models (ALS + LightFM)
python src/candidate_generation.py

# 5. Train Stage 2 LightGBM ranker
python src/ranker.py

# 6. Evaluate the full pipeline
python src/evaluation.py

# 7. Serve the recommendation API
uvicorn api.rec_api:app --reload --port 8000
```

For faster local iteration of the Optuna tuning stage, set:

```bash
RANKER_OPTUNA_TRIALS=8 python src/ranker.py
```

The default is 40 Optuna trials.

---

## Step-by-Step Pipeline Explanation

### Step 1 — Data Generation (`data/generate_data.py`)

This script builds the entire synthetic dataset that the ML models train on. It simulates a hypothetical B2B SaaS platform and produces three CSV files:

| File | Description |
|---|---|
| `features.csv` | Item catalog — 20 hardcoded SaaS features with metadata |
| `customers.csv` | User base — 8,000 simulated customer accounts |
| `events.csv` | Interaction log — timestamped records of which customers used which features |

#### The 20 SaaS Features

Features are grouped into functional categories that mirror a real-world platform:

| Category | Features |
|---|---|
| **Integrations & Automation** | `slack_integration`, `zapier_integration`, `crm_integration`, `webhook_setup` |
| **Data & Analytics** | `data_export`, `advanced_analytics`, `scheduled_reports`, `ai_insights`, `data_warehouse_sync` |
| **Security & Admin** | `sso_login`, `audit_logs`, `custom_roles` |
| **Productivity & Collaboration** | `team_collaboration`, `mobile_app`, `bulk_operations`, `custom_workflows` |
| **Customization & Support** | `custom_dashboards`, `white_labeling`, `priority_support` |

Each feature carries metadata that drives the simulation logic:

- **`plan_required`** (`starter` / `growth` / `enterprise`): Enforces paywalls — Starter customers cannot adopt Enterprise-only features.
- **`complexity_tier`** (`basic` / `advanced` / `power`): Controls baseline adoption probability. Basic features are used by nearly everyone; Power features are adopted by only a small fraction of advanced users.
- **`churn_reduction_score`**: A simulated "stickiness" metric. Features like `priority_support` and `sso_login` meaningfully reduce a customer's computed churn probability, modeling the real-world idea that "sticky" features retain customers.
- **`avg_adoption_rate`**: The baseline probability weight used when randomly generating usage events.

#### Co-Adoption Rules

A hardcoded `CO_ADOPTION_RULES` dictionary simulates the *"People Also Bought"* effect. If a customer adopts `sso_login` (`f06`), the script applies a large probability multiplier so they are very likely to also adopt `custom_roles` (`f12`) and `audit_logs` (`f10`) — mirroring how enterprise IT administrators actually configure software.

#### How Customers Are Simulated

Each of the 8,000 customers is randomly assigned traits:
- **Subscription plan:** Starter, Growth, or Enterprise
- **Industry, contract type, account age**
- **Churn probability:** Computed from their traits — low-engagement Starter users have high churn risk; active Enterprise users have low churn risk.

#### How Events Are Generated

The `generate_events` function mathematically models usage through four layers:

1. **Paywall check:** A customer is ineligible for features outside their plan tier.
2. **Adoption probability:** Calculated per eligible customer–feature pair, weighted by plan tier, churn risk, and feature complexity.
3. **Co-adoption boost:** If a customer adopts a "trigger" feature, related features get large probability multipliers.
4. **Usage intensity:** Adopted features are assigned a realistic monthly usage count — annual-contract customers use features more heavily; high-churn-risk customers generate sparse usage.

The result is an `events.csv` full of structured patterns: Enterprise customers have dense, high-frequency usage of advanced features, while churning Starter customers show sparse usage of basic features.

---

### Step 2 — Interaction Matrix (`data/build_interaction_matrix.py`)

Converts the raw event log into a **user–feature matrix** that ML models can consume.

**Shape: 8,000 rows (customers) × 20 columns (features)**

Key decisions:

| Decision | Detail |
|---|---|
| **Recency window** | Only events from the last 90 days (`LOOKBACK_DAYS = 90`) are used — recent behavior is a stronger signal than old behavior |
| **Log-scale implicit score** | Raw usage counts are compressed with `log(1 + count)` so power users (1,000 uses → score 6.9) don't dominate over moderate users (10 uses → score 2.4) |
| **Sparsity** | ~76% of cells are zero — most customers use only a handful of features. This is expected and handled by saving as a compressed sparse matrix (`interaction_sparse.npz`) |
| **ID mapping** | `user_item_mappings.json` maps ML model row/column indices back to human-readable Customer IDs and feature names |

#### Adoption Rate Health Check (output of this script)

```
Most adopted feature:  slack_integration  — 60.9% of users
Least adopted feature: white_labeling     —  3.1% of users
```

The calculation uses a clean pandas trick: `(matrix_df > 0).mean(axis=0)` — treating each boolean as 0/1 and averaging to get the exact adoption percentage per column.

---

### Step 3 — Train / Test Split (`data/build_evaluation_split.py`)

Prepares a held-out evaluation set so you can measure how well the recommendation model predicts future behavior.

**Strategy: Temporal Leave-N%-Out**

- For each customer, the script finds the exact first-adoption date of every feature they used.
- It hides the **most recently adopted 20%** of each customer's features (`HOLDOUT_FRACTION = 0.20`).
- This simulates the real-world task: *"Given what a customer has used so far, can we predict the next feature they will adopt?"*

**Outputs:**

| File | Contents |
|---|---|
| `train_interactions.npz` | Full matrix with the held-out 20% erased (zeros) — used for model training |
| `test_interactions.npz` | Sparse matrix containing only the held-out 20% — the ground truth |
| `evaluation_users.csv` | Answer key: Customer ID → list of hidden feature IDs (e.g., `C00123 \| f02\|f18`) |

---

### Step 4 — Candidate Generation (`src/candidate_generation.py`)

Stage 1 of the two-stage pipeline. Trains two fundamentally different ML models and merges their candidate lists.

#### Model 1: ALS (Alternating Least Squares)

A **pure Collaborative Filtering** algorithm. It knows nothing about what a feature does or what a customer's plan tier is — it only sees the interaction matrix.

**How it reasons:** *"Customer A uses `slack_integration` and `data_export`. Customer B uses those same two features plus `api_access`. Therefore, Customer A would probably like `api_access`."*

**The math behind ALS:**

ALS solves the matrix factorization problem: decompose the 8,000 × 20 interaction matrix **R** into two smaller matrices **U** (users, shape 8000 × k) and **V** (features, shape 20 × k) such that `U × Vᵀ ≈ R`.

This is where **Gauss's Ordinary Least Squares** comes in — but with a twist. Since both **U** and **V** are unknown and multiplied together, the problem is non-linear and cannot be solved directly with OLS. ALS's elegant workaround:

1. **Initialize** **V** with random numbers.
2. **Freeze V, solve U:** With **V** treated as a known constant, the equation becomes linear. Apply Gauss's Least Squares exactly to find the optimal **U**.
3. **Freeze U, solve V:** Now treat **U** as the known constant. Apply Gauss's Least Squares exactly to find the optimal **V**.
4. **Repeat** for `ALS_ITERATIONS = 50` rounds until convergence.

Each alternation slightly reduces the squared error. After 50 rounds the model has converged to an accurate mapping of "hidden preferences" for both customers and features.

#### Model 2: LightFM (Hybrid Model)

A **Hybrid Collaborative + Content-Based** model. In addition to the interaction matrix, LightFM receives:
- **Customer features:** MRR, team size, churn risk, plan tier, industry
- **Item features:** Category, complexity tier, tags, description

**How it reasons:** *"This customer is Enterprise-tier with a 100+ team and high churn risk. Historically, large high-risk teams respond well to `audit_logs` and `custom_roles`."*

#### Merging the Candidates

```
ALS top-50 candidates per customer
         +
LightFM top-50 candidates per customer
         ↓
    Pool of ≤100 candidates
         ↓
  Remove already-adopted features
         ↓
  Save to merged_candidates.pkl
```

---

### Step 5 — Ranking (`src/ranker.py`)

Stage 2 of the two-stage pipeline. A **LightGBM LambdaRank** model re-scores the merged candidate list to find the single best recommendation for each customer.

The ranker uses rich feature engineering including:

- Account context: plan tier, MRR, team size, tenure, contract type
- Adoption gaps: how many features in the same category has the customer already adopted?
- Collaborative signals: the raw ALS and LightFM scores from Stage 1
- **Key retention signal:** `churn_signal_boost = churn_probability × churn_reduction_score`

Hyperparameters are tuned with **Optuna** (40 trials by default).

---

### Step 6 — Evaluation (`src/evaluation.py`)

Measures recommendation quality by checking predictions against the `evaluation_users.csv` answer key.

**Outputs:**

| File | Contents |
|---|---|
| `outputs/evaluation_report.csv` | Per-user metrics (precision, recall, NDCG) |
| `outputs/recommendation_distribution.csv` | How often each feature is recommended |
| `outputs/ab_test_results.csv` | Simulated A/B test results |
| `outputs/plots/` | Visualizations |

---

## API

Served via FastAPI. Start with:

```bash
uvicorn api.rec_api:app --reload --port 8000
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/recommend/{customer_id}?top_k=10` | Top-k feature recommendations for a customer |
| `GET` | `/similar-users/{customer_id}?top_n=5` | Customers with similar usage profiles |
| `GET` | `/feature-affinity/{feature_id}?top_n=100` | Customers most likely to adopt a given feature |
| `GET` | `/health` | Service health check |
| `GET` | `/metrics/summary` | Aggregate recommendation metrics |

**Example:**

```bash
curl -s http://127.0.0.1:8000/recommend/C00001?top_k=10
```

---

## System Notes

- The synthetic catalog has exactly **20 SaaS features**.
- The customer set spans **C00001 to C08000** (8,000 customers).
- Ranker training uses all 20 catalog items per eligible user to ensure positive labels are available; online prediction uses the merged Stage 1 candidate list (already-used features excluded).
- **Cold start** triggers when `tenure_days < 30` or the customer has fewer than 3 recent interactions; handled by `src/cold_start.py` using a fallback popularity + plan-tier heuristic.

---

## Glossary

| Term | Definition |
|---|---|
| **Churn** | A customer canceling their subscription |
| **Adoption** | A customer using a feature for the first time |
| **Co-Adoption** | When using feature A naturally leads a customer to also use feature B |
| **Usage Intensity** | How frequently a customer uses a feature (low = once a month; high = many times daily) |
| **Starter / Growth / Enterprise** | Subscription plan tiers — each unlocks a different set of features |
| **Paywall** | A gate that prevents lower-tier customers from accessing higher-tier features |
| **SSO (Single Sign-On)** | Enterprise security feature — employees use one company login across all apps |
| **API / Webhooks** | Programmatic interfaces that allow two software systems to communicate automatically |
| **Audit Logs** | A tamper-proof record of every user action — required by enterprises for compliance |
| **CRM** | Customer Relationship Management software (e.g., Salesforce) |
| **Data Warehouse** | A centralized storage system where companies consolidate data from all their tools |
| **ALS** | Alternating Least Squares — a collaborative filtering algorithm using iterated OLS |
| **LightFM** | A hybrid recommendation model that combines collaborative filtering with content features |
| **LambdaRank** | A learning-to-rank objective that optimizes directly for ranking quality metrics like NDCG |
| **Implicit Feedback** | Learning from usage events (clicks, views) rather than explicit ratings |
| **Matrix Factorization** | Decomposing a large sparse matrix into two smaller dense matrices to infer latent preferences |
| **Sparsity** | The fraction of zero entries in the interaction matrix (~76% in this project) |
| **NDCG** | Normalized Discounted Cumulative Gain — a standard ranking quality metric |
