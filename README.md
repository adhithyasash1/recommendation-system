# SaaS Feature Recommendation System

Production-style two-stage recommender for SaaS feature adoption.

Stage 1 retrieves candidates with implicit ALS and LightFM hybrid modeling. Stage 2 re-ranks the candidate pool with a LightGBM LambdaRank model that uses account context, plan tier, adoption gaps, collaborative signals, and the key retention feature `churn_signal_boost = churn_probability * churn_reduction_score`.

## Project Layout

```text
data/          Synthetic data generation, interaction matrix, train/test split
src/           Candidate generation, ranker, cold-start fallback, evaluation
api/           FastAPI app and Pydantic schemas
dashboard/     Streamlit recommendation demo
models/        Trained ALS, LightFM, and LightGBM artifacts
outputs/       Evaluation reports and plots
logs/          API logs
```

## Setup

The pinned ML stack is safest on Python 3.11. `scikit-learn` uses `1.4.1.post1` because the original `1.4.1` pin is not available for this environment.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

If your shell has a `python` alias inside the virtualenv, use the exact execution order below. If not, replace `python` with `.venv/bin/python`.

## Execution Order

```bash
python data/generate_data.py
python data/build_interaction_matrix.py
python data/build_evaluation_split.py
python src/candidate_generation.py
python src/ranker.py
python src/evaluation.py
uvicorn api.rec_api:app --reload --port 8000
streamlit run dashboard/rec_dashboard.py
```

For faster local iteration of the Optuna stage, set:

```bash
RANKER_OPTUNA_TRIALS=8 python src/ranker.py
```

The default remains 40 trials.

## API

```text
GET /recommend/{customer_id}?top_k=10
GET /similar-users/{customer_id}?top_n=5
GET /feature-affinity/{feature_id}?top_n=100
GET /health
GET /metrics/summary
```

Example:

```bash
curl -s http://127.0.0.1:8000/recommend/C00001?top_k=10
```

## Notes

- The synthetic catalog has exactly 20 SaaS features.
- The customer set follows the requested `C00001` to `C08000` range.
- Ranker training uses all 20 catalog items per eligible user. This keeps positive labels available, while online prediction still uses merged Stage 1 candidates that exclude already-used features.
- Cold start triggers when `tenure_days < 30` or the customer has fewer than 3 recent interactions.
- Evaluation writes `outputs/evaluation_report.csv`, `outputs/recommendation_distribution.csv`, `outputs/ab_test_results.csv`, and plots under `outputs/plots/`.

## Sources

Implementation follows the project specification in this repository prompt and the local source files in this workspace. No external documentation was used while writing the code.
