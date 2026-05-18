"""Cold-start recommendation fallback for new or low-activity customers."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

try:
    from src.candidate_generation import churn_risk_tier
except ImportError:
    from candidate_generation import churn_risk_tier

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
COLD_START_TENURE_DAYS = 30
COLD_START_MIN_INTERACTIONS = 3
PLAN_ONBOARDING_PATHS = {
    "starter": ["f15", "f01", "f03", "f05", "f16"],
    "growth": ["f04", "f02", "f13", "f08", "f17"],
    "enterprise": ["f06", "f12", "f10", "f09", "f20"],
}
CTA_BY_COMPLEXITY = {
    "basic": "Start using {feature_name} \u2192",
    "advanced": "Set up {feature_name} (15 min) \u2192",
    "power": "Explore {feature_name} with your team \u2192",
}


@lru_cache(maxsize=1)
def _load_context() -> dict[str, Any]:
    mappings = json.loads((DATA_DIR / "user_item_mappings.json").read_text())
    return {
        "customers": pd.read_csv(DATA_DIR / "customers.csv"),
        "features": pd.read_csv(DATA_DIR / "feature_metadata.csv"),
        "interactions": sparse.load_npz(DATA_DIR / "interaction_sparse.npz").tocsr(),
        "mappings": mappings,
    }


def _format_feature_name(feature_name: str) -> str:
    return feature_name.replace("_", " ").title()


def should_use_cold_start(customer_id: str) -> bool:
    """Return True when a customer is new or has too few interactions for warm-start ranking."""
    context = _load_context()
    customers = context["customers"]
    if customer_id not in set(customers["customer_id"]):
        raise KeyError(f"Unknown customer_id: {customer_id}")
    customer = customers.loc[customers["customer_id"] == customer_id].iloc[0]
    user_idx = int(context["mappings"]["customer_to_index"][customer_id])
    total_interactions = int(context["interactions"][user_idx].nnz)
    return int(customer["tenure_days"]) < COLD_START_TENURE_DAYS or total_interactions < COLD_START_MIN_INTERACTIONS


def _same_plan_popularity(plan_tier: str, context: dict[str, Any]) -> dict[str, float]:
    customers = context["customers"]
    interactions = context["interactions"]
    mappings = context["mappings"]
    plan_customer_ids = customers.loc[customers["plan_tier"] == plan_tier, "customer_id"].tolist()
    plan_indices = [int(mappings["customer_to_index"][customer_id]) for customer_id in plan_customer_ids]
    if not plan_indices:
        return {feature_id: 0.0 for feature_id in mappings["feature_ids"]}
    adoption = interactions[plan_indices].copy()
    adoption.data = np.ones_like(adoption.data)
    rates = np.asarray(adoption.mean(axis=0)).ravel()
    return {str(feature_id): float(rates[idx]) for idx, feature_id in enumerate(mappings["feature_ids"])}


def get_cold_start_recommendations(customer_id: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Return onboarding and same-plan popularity recommendations for cold-start customers."""
    context = _load_context()
    customers = context["customers"]
    features = context["features"]
    mappings = context["mappings"]
    customer = customers.loc[customers["customer_id"] == customer_id].iloc[0]
    plan_tier = str(customer["plan_tier"])
    user_idx = int(mappings["customer_to_index"][customer_id])
    used = {str(mappings["feature_ids"][idx]) for idx in context["interactions"][user_idx].indices}
    plan_popularity = _same_plan_popularity(plan_tier, context)
    curated = PLAN_ONBOARDING_PATHS[plan_tier]

    candidate_ids: list[str] = []
    for feature_id in curated:
        if feature_id not in used and feature_id not in candidate_ids:
            candidate_ids.append(feature_id)
    for feature_id, _ in sorted(plan_popularity.items(), key=lambda item: item[1], reverse=True):
        if feature_id not in used and feature_id not in candidate_ids:
            candidate_ids.append(feature_id)

    feature_lookup = features.set_index("feature_id")
    recommendations: list[dict[str, Any]] = []
    max_popularity = max(plan_popularity.values()) if plan_popularity else 1.0
    for rank, feature_id in enumerate(candidate_ids[:top_k], start=1):
        feature = feature_lookup.loc[feature_id]
        raw_name = str(feature["feature_name"])
        feature_name = _format_feature_name(raw_name)
        popularity_score = plan_popularity.get(feature_id, 0.0)
        raw_relevance = popularity_score / max_popularity if max_popularity > 0 else float(feature["avg_adoption_rate"])
        relevance_score = 0.75 + 0.25 * raw_relevance
        churn_boost = float(customer["churn_probability"]) * float(feature["churn_reduction_score"])
        recommendations.append(
            {
                "rank": rank,
                "feature_id": feature_id,
                "feature_name": feature_name,
                "feature_category": str(feature["category"]),
                "category": str(feature["category"]),
                "complexity_tier": str(feature["complexity_tier"]),
                "relevance_score": round(float(np.clip(relevance_score, 0.0, 1.0)), 4),
                "churn_reduction_score": round(float(feature["churn_reduction_score"]), 4),
                "adoption_gap": round(float(feature["avg_adoption_rate"]) - popularity_score, 4),
                "churn_signal_boost": round(churn_boost, 4),
                "similar_users_adoption_rate": round(popularity_score, 4),
                "co_adoption_signal": 0.0,
                "reason": f"Popular among new {plan_tier} customers - great starting point",
                "cta": CTA_BY_COMPLEXITY[str(feature["complexity_tier"])].format(feature_name=feature_name),
                "churn_risk_tier": churn_risk_tier(float(customer["churn_probability"])),
            }
        )

    return recommendations
