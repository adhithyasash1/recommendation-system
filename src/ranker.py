"""Stage 2 LightGBM LambdaRank model for SaaS feature recommendations."""

from __future__ import annotations

import json
import os
import pickle
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRanker
from loguru import logger
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

try:
    from src.candidate_generation import churn_risk_tier, load_mappings
except ImportError:
    from candidate_generation import churn_risk_tier, load_mappings

RANDOM_STATE = 42
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"

PLAN_ENCODING = {"starter": 0, "growth": 1, "enterprise": 2}
CONTRACT_ENCODING = {"monthly": 0, "annual": 1}
COMPLEXITY_ENCODING = {"basic": 0, "advanced": 1, "power": 2}
RISK_ENCODING = {"low": 0, "medium": 1, "high": 2}
FEATURE_AGE_MONTHS = 36
SIMILAR_USER_COUNT = 20
DEFAULT_OPTUNA_TRIALS = 40
VALIDATION_FRACTION = 0.20
CTA_BY_COMPLEXITY = {
    "basic": "Start using {feature_name} \u2192",
    "advanced": "Set up {feature_name} (15 min) \u2192",
    "power": "Explore {feature_name} with your team \u2192",
}


def seed_everything() -> None:
    """Seed random generators for reproducible ranking."""
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def _binary_interactions(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    binary = matrix.copy().tocsr()
    binary.data = np.ones_like(binary.data, dtype=np.float32)
    return binary


def _compute_similar_neighbors(user_factors: np.ndarray, n_users: int) -> np.ndarray:
    if user_factors.shape[0] != n_users or n_users <= 1:
        return np.zeros((n_users, 0), dtype=np.int32)
    safe_factors = np.nan_to_num(user_factors, nan=0.0, posinf=0.0, neginf=0.0)
    neighbor_count = min(SIMILAR_USER_COUNT + 1, n_users)
    model = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine")
    model.fit(safe_factors)
    _, indices = model.kneighbors(safe_factors)
    return indices[:, 1:].astype(np.int32)


def _compute_coadoption_matrix(train_interactions: sparse.csr_matrix) -> np.ndarray:
    binary = _binary_interactions(train_interactions)
    item_item = (binary.T @ binary).toarray().astype(np.float32)
    item_counts = np.asarray(binary.sum(axis=0)).ravel().astype(np.float32)
    union = item_counts[:, None] + item_counts[None, :] - item_item
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.divide(item_item, union, out=np.zeros_like(item_item), where=union > 0)
    np.fill_diagonal(jaccard, 0.0)
    return jaccard


def _feature_columns(customers: pd.DataFrame, features: pd.DataFrame) -> list[str]:
    industries = sorted(customers["industry"].unique().tolist())
    categories = sorted(features["category"].unique().tolist())
    columns = [
        "plan_tier_encoded",
        "contract_type_encoded",
        "tenure_days",
        "team_size",
        "mrr",
        "churn_probability",
        "churn_risk_tier",
        "is_in_first_90_days",
    ]
    columns.extend([f"industry_{industry}" for industry in industries])
    columns.extend([f"category_{category}" for category in categories])
    columns.extend(
        [
            "complexity_tier_encoded",
            "plan_required_encoded",
            "churn_reduction_score",
            "avg_adoption_rate",
            "feature_age_months",
            "als_candidate_score",
            "lightfm_candidate_score",
            "plan_allows_feature",
            "similar_users_adoption_rate",
            "co_adoption_signal",
            "churn_signal_boost",
            "adoption_gap_score",
        ]
    )
    return columns


@lru_cache(maxsize=2)
def load_ranker_context(include_model: bool = True) -> dict[str, Any]:
    """Load data, model artifacts, and precomputed matrices used by the ranker."""
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    features = pd.read_csv(DATA_DIR / "feature_metadata.csv")
    mappings = load_mappings()
    train_interactions = sparse.load_npz(DATA_DIR / "train_interactions.npz").tocsr().astype(np.float32)
    full_interactions = sparse.load_npz(DATA_DIR / "interaction_sparse.npz").tocsr().astype(np.float32)
    merged_candidates = _load_pickle(DATA_DIR / "merged_candidates.pkl")
    als_payload = np.load(MODEL_DIR / "als_model.npz")
    user_factors = np.asarray(als_payload["user_factors"], dtype=np.float32)
    similar_neighbors = _compute_similar_neighbors(user_factors, len(mappings["customer_ids"]))
    coadoption = _compute_coadoption_matrix(train_interactions)

    ranker_payload = None
    if include_model and (MODEL_DIR / "lgbm_ranker.joblib").exists():
        ranker_payload = joblib.load(MODEL_DIR / "lgbm_ranker.joblib")

    return {
        "customers": customers,
        "features": features,
        "mappings": mappings,
        "train_interactions": train_interactions,
        "full_interactions": full_interactions,
        "merged_candidates": merged_candidates,
        "user_factors": user_factors,
        "similar_neighbors": similar_neighbors,
        "coadoption": coadoption,
        "ranker_payload": ranker_payload,
        "feature_columns": _feature_columns(customers, features),
    }


def _candidate_score_map(candidates: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        str(candidate["feature_id"]): {
            "als_score": float(candidate.get("als_score", 0.0)),
            "lightfm_score": float(candidate.get("lightfm_score", 0.0)),
        }
        for candidate in candidates
    }


def _plan_allows(plan_tier: str, plan_required: str) -> bool:
    return PLAN_ENCODING[plan_tier] >= PLAN_ENCODING[plan_required]


def _user_category_adoption_rate(
    user_idx: int,
    category: str,
    context: dict[str, Any],
) -> float:
    features = context["features"]
    feature_ids = list(context["mappings"]["feature_ids"])
    feature_to_index = dict(context["mappings"]["feature_to_index"])
    category_feature_ids = features.loc[features["category"] == category, "feature_id"].tolist()
    if not category_feature_ids:
        return 0.0
    used_indices = set(context["train_interactions"][user_idx].indices.tolist())
    category_indices = {int(feature_to_index[feature_id]) for feature_id in category_feature_ids}
    return float(len(used_indices & category_indices) / len(category_indices))


def _similar_users_adoption_rate(user_idx: int, item_idx: int, context: dict[str, Any]) -> float:
    neighbors = context["similar_neighbors"]
    if neighbors.shape[1] == 0:
        return 0.0
    neighbor_indices = neighbors[user_idx]
    adopted = context["train_interactions"][neighbor_indices, item_idx].toarray().ravel() > 0
    return float(adopted.mean()) if len(adopted) else 0.0


def _co_adoption_signal(user_idx: int, item_idx: int, context: dict[str, Any]) -> tuple[float, str | None]:
    used_indices = context["train_interactions"][user_idx].indices.tolist()
    if not used_indices:
        return 0.0, None
    coadoption_scores = context["coadoption"][used_indices, item_idx]
    best_position = int(np.argmax(coadoption_scores))
    best_score = float(coadoption_scores[best_position])
    if best_score <= 0:
        return 0.0, None
    best_feature_idx = int(used_indices[best_position])
    return best_score, str(context["mappings"]["feature_ids"][best_feature_idx])


def build_feature_row(
    customer_id: str,
    feature_id: str,
    context: dict[str, Any],
    candidate_scores: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Build one ranker feature row for a customer-feature pair."""
    customers = context["customers"]
    features = context["features"]
    mappings = context["mappings"]
    customer = customers.loc[customers["customer_id"] == customer_id].iloc[0]
    feature = features.loc[features["feature_id"] == feature_id].iloc[0]
    customer_to_index = dict(mappings["customer_to_index"])
    feature_to_index = dict(mappings["feature_to_index"])
    user_idx = int(customer_to_index[customer_id])
    item_idx = int(feature_to_index[feature_id])
    scores = (candidate_scores or {}).get(feature_id, {"als_score": 0.0, "lightfm_score": 0.0})
    risk_tier = churn_risk_tier(float(customer["churn_probability"]))
    category = str(feature["category"])
    similar_rate = _similar_users_adoption_rate(user_idx, item_idx, context)
    co_signal, _ = _co_adoption_signal(user_idx, item_idx, context)
    adoption_gap = float(feature["avg_adoption_rate"]) - _user_category_adoption_rate(user_idx, category, context)

    row: dict[str, float] = {
        "plan_tier_encoded": float(PLAN_ENCODING[str(customer["plan_tier"])]),
        "contract_type_encoded": float(CONTRACT_ENCODING[str(customer["contract_type"])]),
        "tenure_days": float(customer["tenure_days"]),
        "team_size": float(customer["team_size"]),
        "mrr": float(customer["mrr"]),
        "churn_probability": float(customer["churn_probability"]),
        "churn_risk_tier": float(RISK_ENCODING[risk_tier]),
        "is_in_first_90_days": float(int(float(customer["tenure_days"]) < 90)),
        "complexity_tier_encoded": float(COMPLEXITY_ENCODING[str(feature["complexity_tier"])]),
        "plan_required_encoded": float(PLAN_ENCODING[str(feature["plan_required"])]),
        "churn_reduction_score": float(feature["churn_reduction_score"]),
        "avg_adoption_rate": float(feature["avg_adoption_rate"]),
        "feature_age_months": float(FEATURE_AGE_MONTHS),
        "als_candidate_score": float(scores.get("als_score", 0.0)),
        "lightfm_candidate_score": float(scores.get("lightfm_score", 0.0)),
        "plan_allows_feature": float(int(_plan_allows(str(customer["plan_tier"]), str(feature["plan_required"])))),
        "similar_users_adoption_rate": similar_rate,
        "co_adoption_signal": co_signal,
        "churn_signal_boost": float(customer["churn_probability"]) * float(feature["churn_reduction_score"]),
        "adoption_gap_score": adoption_gap,
    }

    for industry in customers["industry"].unique():
        row[f"industry_{industry}"] = float(int(str(customer["industry"]) == industry))
    for feature_category in features["category"].unique():
        row[f"category_{feature_category}"] = float(int(category == feature_category))

    return row


def _build_grouped_matrix(
    customer_ids: list[str],
    context: dict[str, Any],
    include_all_items: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    feature_ids = list(context["mappings"]["feature_ids"])
    feature_columns = list(context["feature_columns"])
    train = context["train_interactions"]
    customer_to_index = dict(context["mappings"]["customer_to_index"])
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    groups: list[int] = []

    for customer_id in customer_ids:
        user_idx = int(customer_to_index[customer_id])
        if train[user_idx].nnz == 0:
            continue
        candidates = context["merged_candidates"].get(customer_id, [])
        candidate_scores = _candidate_score_map(candidates)
        item_pool = feature_ids if include_all_items else [str(item["feature_id"]) for item in candidates]
        start_len = len(rows)
        for feature_id in item_pool:
            item_idx = int(context["mappings"]["feature_to_index"][feature_id])
            rows.append(build_feature_row(customer_id, feature_id, context, candidate_scores))
            labels.append(int(train[user_idx, item_idx] > 0))
        group_len = len(rows) - start_len
        if group_len:
            groups.append(group_len)

    frame = pd.DataFrame(rows).reindex(columns=feature_columns, fill_value=0.0)
    return frame, np.asarray(labels, dtype=np.int32), groups


def _ndcg_at_k_for_groups(y_true: np.ndarray, y_score: np.ndarray, groups: list[int], k: int = 10) -> float:
    scores: list[float] = []
    offset = 0
    for group_size in groups:
        group_true = y_true[offset : offset + group_size]
        group_score = y_score[offset : offset + group_size]
        order = np.argsort(-group_score)[:k]
        gains = group_true[order]
        discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
        dcg = float(np.sum(gains * discounts))
        ideal = np.sort(group_true)[::-1][:k]
        ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
        idcg = float(np.sum(ideal * ideal_discounts))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
        offset += group_size
    return float(np.mean(scores)) if scores else 0.0


def _split_training_users(context: dict[str, Any]) -> tuple[list[str], list[str]]:
    customer_ids = list(context["mappings"]["customer_ids"])
    train = context["train_interactions"]
    eligible = [customer_id for customer_id in customer_ids if train[int(context["mappings"]["customer_to_index"][customer_id])].nnz > 0]
    rng = np.random.default_rng(RANDOM_STATE)
    shuffled = list(rng.permutation(eligible))
    split_idx = max(1, int(len(shuffled) * (1.0 - VALIDATION_FRACTION)))
    return shuffled[:split_idx], shuffled[split_idx:]


def _objective_factory(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    group_train: list[int],
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    group_val: list[int],
):
    def objective(trial: optuna.Trial) -> float:
        max_depth = trial.suggest_int("max_depth", 3, 8)
        num_leaves = trial.suggest_int("num_leaves", 20, 150)
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": -1,
        }
        model = LGBMRanker(**params)
        model.fit(
            x_train,
            y_train,
            group=group_train,
            eval_set=[(x_val, y_val)],
            eval_group=[group_val],
            eval_at=[10],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        predictions = model.predict(x_val)
        return _ndcg_at_k_for_groups(y_val, predictions, group_val, k=10)

    return objective


def _plot_feature_importance(model: LGBMRanker, feature_columns: list[str]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    gains = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(gains)[-20:]
    selected_features = [feature_columns[idx] for idx in order]
    selected_gains = gains[order]
    colors = ["#ef4444" if feature == "churn_signal_boost" else "#6366f1" for feature in selected_features]

    plt.figure(figsize=(10, 7))
    plt.barh(selected_features, selected_gains, color=colors)
    plt.title("LightGBM Ranker Feature Importance by Gain")
    plt.xlabel("Gain")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ranker_feature_importance.png", dpi=180)
    plt.close()


def train_ranker() -> LGBMRanker:
    """Tune and train the final LightGBM ranker."""
    seed_everything()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    context = load_ranker_context(include_model=False)
    train_users, validation_users = _split_training_users(context)
    logger.info("Building ranker matrices for {} train and {} validation users", len(train_users), len(validation_users))

    x_train, y_train, group_train = _build_grouped_matrix(train_users, context, include_all_items=True)
    x_val, y_val, group_val = _build_grouped_matrix(validation_users, context, include_all_items=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trials = int(os.getenv("RANKER_OPTUNA_TRIALS", str(DEFAULT_OPTUNA_TRIALS)))
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(_objective_factory(x_train, y_train, group_train, x_val, y_val, group_val), n_trials=trials)
    logger.info("Best ranker NDCG@10 {:.4f} with params {}", study.best_value, study.best_params)

    all_users = train_users + validation_users
    x_all, y_all, group_all = _build_grouped_matrix(all_users, context, include_all_items=True)
    best_params = {
        **study.best_params,
        "objective": "lambdarank",
        "metric": "ndcg",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }
    final_ranker = LGBMRanker(**best_params)
    final_ranker.fit(x_all, y_all, group=group_all)

    payload = {"model": final_ranker, "feature_columns": list(context["feature_columns"]), "best_params": best_params}
    joblib.dump(payload, MODEL_DIR / "lgbm_ranker.joblib")
    (MODEL_DIR / "ranker_feature_columns.json").write_text(json.dumps(context["feature_columns"], indent=2))
    _plot_feature_importance(final_ranker, list(context["feature_columns"]))

    print(f"LightGBM ranker trained. Best validation NDCG@10: {study.best_value:.4f}")
    print(f"Saved model: {MODEL_DIR / 'lgbm_ranker.joblib'}")
    return final_ranker


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if max_score > min_score:
        scaled = (scores - min_score) / (max_score - min_score)
        return 0.75 + 0.25 * scaled
    return 0.75 + 0.25 * (1.0 / (1.0 + np.exp(-scores)))


def _format_feature_name(feature_name: str) -> str:
    return feature_name.replace("_", " ").title()


def _reason_for_recommendation(
    feature_name: str,
    plan_tier: str,
    category: str,
    row: dict[str, float],
    co_feature_name: str | None,
) -> str:
    if row["churn_signal_boost"] > 0.30:
        return f"Users with similar churn risk who adopted {feature_name} reduced churn by {row['churn_reduction_score'] * 100:.0f}%"
    if row["similar_users_adoption_rate"] > 0.70:
        return f"{row['similar_users_adoption_rate'] * 100:.0f}% of {plan_tier} customers similar to you have adopted {feature_name}"
    if row["co_adoption_signal"] > 0.50 and co_feature_name:
        return f"Teams using {co_feature_name} also find {feature_name} essential"
    if row["adoption_gap_score"] > 0.30:
        return f"You're behind peers in {category} because {feature_name} closes that gap"
    return "Recommended based on your usage pattern and plan tier"


def rank_features(customer_id: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Rank merged feature candidates for a customer and return explanation-ready recommendations."""
    context = load_ranker_context(include_model=True)
    ranker_payload = context["ranker_payload"]
    if ranker_payload is None:
        raise FileNotFoundError("models/lgbm_ranker.joblib not found. Run python src/ranker.py first.")

    ranker = ranker_payload["model"] if isinstance(ranker_payload, dict) else ranker_payload
    feature_columns = ranker_payload.get("feature_columns", context["feature_columns"]) if isinstance(ranker_payload, dict) else context["feature_columns"]
    customers = context["customers"]
    features = context["features"]
    mappings = context["mappings"]

    if customer_id not in set(customers["customer_id"]):
        raise KeyError(f"Unknown customer_id: {customer_id}")

    candidates = context["merged_candidates"].get(customer_id, [])
    if not candidates:
        user_idx = int(mappings["customer_to_index"][customer_id])
        used = {mappings["feature_ids"][idx] for idx in context["train_interactions"][user_idx].indices}
        candidates = [{"feature_id": feature_id, "als_score": 0.0, "lightfm_score": 0.0} for feature_id in mappings["feature_ids"] if feature_id not in used]

    candidate_scores = _candidate_score_map(candidates)
    rows: list[dict[str, float]] = []
    feature_ids: list[str] = []
    row_metadata: list[dict[str, Any]] = []

    for candidate in candidates:
        feature_id = str(candidate["feature_id"])
        row = build_feature_row(customer_id, feature_id, context, candidate_scores)
        co_signal, co_feature_id = _co_adoption_signal(
            int(mappings["customer_to_index"][customer_id]),
            int(mappings["feature_to_index"][feature_id]),
            context,
        )
        co_feature_name = None
        if co_feature_id:
            raw_name = features.loc[features["feature_id"] == co_feature_id, "feature_name"].iloc[0]
            co_feature_name = _format_feature_name(str(raw_name))
        row["co_adoption_signal"] = co_signal
        rows.append(row)
        feature_ids.append(feature_id)
        row_metadata.append({"co_feature_name": co_feature_name})

    matrix = pd.DataFrame(rows).reindex(columns=feature_columns, fill_value=0.0)
    raw_scores = np.asarray(ranker.predict(matrix), dtype=np.float32)
    normalized_scores = _normalize_scores(raw_scores)
    order = np.argsort(-raw_scores)[:top_k]

    feature_lookup = features.set_index("feature_id")
    customer = customers.loc[customers["customer_id"] == customer_id].iloc[0]
    recommendations: list[dict[str, Any]] = []

    for display_rank, row_idx in enumerate(order, start=1):
        feature_id = feature_ids[int(row_idx)]
        feature = feature_lookup.loc[feature_id]
        feature_name = _format_feature_name(str(feature["feature_name"]))
        feature_row = rows[int(row_idx)]
        reason = _reason_for_recommendation(
            feature_name=feature_name,
            plan_tier=str(customer["plan_tier"]),
            category=str(feature["category"]),
            row=feature_row,
            co_feature_name=row_metadata[int(row_idx)]["co_feature_name"],
        )
        recommendations.append(
            {
                "rank": display_rank,
                "feature_id": feature_id,
                "feature_name": feature_name,
                "feature_category": str(feature["category"]),
                "category": str(feature["category"]),
                "complexity_tier": str(feature["complexity_tier"]),
                "relevance_score": round(float(normalized_scores[int(row_idx)]), 4),
                "churn_reduction_score": round(float(feature["churn_reduction_score"]), 4),
                "adoption_gap": round(float(feature_row["adoption_gap_score"]), 4),
                "churn_signal_boost": round(float(feature_row["churn_signal_boost"]), 4),
                "similar_users_adoption_rate": round(float(feature_row["similar_users_adoption_rate"]), 4),
                "co_adoption_signal": round(float(feature_row["co_adoption_signal"]), 4),
                "reason": reason,
                "cta": CTA_BY_COMPLEXITY[str(feature["complexity_tier"])].format(feature_name=feature_name),
            }
        )

    return recommendations


def main() -> None:
    """Train and save the ranker."""
    train_ranker()


if __name__ == "__main__":
    main()
