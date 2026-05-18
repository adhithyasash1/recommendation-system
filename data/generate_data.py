"""Generate synthetic SaaS customer, feature, and usage event data."""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

RANDOM_STATE = 42
NUM_CUSTOMERS = 8_000
START_DATE = pd.Timestamp("2022-01-01")
END_DATE = pd.Timestamp("2024-12-31")
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

PLAN_ORDER = {"starter": 0, "growth": 1, "enterprise": 2}
PLAN_PROBABILITIES = {"starter": 0.50, "growth": 0.35, "enterprise": 0.15}
CONTRACT_PROBABILITIES = {"monthly": 0.60, "annual": 0.40}
PLAN_TIER_MULTIPLIER = {"starter": 0.70, "growth": 1.00, "enterprise": 1.40}
INDUSTRIES = ["saas", "ecommerce", "fintech", "healthcare", "edtech"]
COMPLEXITY_USAGE_MULTIPLIER = {"basic": 1.10, "advanced": 0.90, "power": 0.78}

FEATURE_ROWS = [
    {
        "feature_id": "f01",
        "feature_name": "slack_integration",
        "category": "integrations",
        "complexity_tier": "basic",
        "plan_required": "starter",
        "avg_adoption_rate": 0.72,
        "churn_reduction_score": 0.35,
        "description": "Connect product notifications and workflow updates to Slack channels.",
        "tags": "integration|slack|notifications|collaboration",
    },
    {
        "feature_id": "f02",
        "feature_name": "api_access",
        "category": "developer",
        "complexity_tier": "advanced",
        "plan_required": "growth",
        "avg_adoption_rate": 0.45,
        "churn_reduction_score": 0.42,
        "description": "Use programmatic APIs to automate product workflows and data access.",
        "tags": "api|developer|automation|integration",
    },
    {
        "feature_id": "f03",
        "feature_name": "data_export",
        "category": "data",
        "complexity_tier": "basic",
        "plan_required": "starter",
        "avg_adoption_rate": 0.68,
        "churn_reduction_score": 0.18,
        "description": "Export account data to CSV files for offline analysis and reporting.",
        "tags": "data|export|csv|reporting",
    },
    {
        "feature_id": "f04",
        "feature_name": "custom_dashboards",
        "category": "analytics",
        "complexity_tier": "advanced",
        "plan_required": "growth",
        "avg_adoption_rate": 0.51,
        "churn_reduction_score": 0.38,
        "description": "Build role-specific dashboards with saved charts and product KPIs.",
        "tags": "analytics|dashboard|reporting|customization",
    },
    {
        "feature_id": "f05",
        "feature_name": "team_collaboration",
        "category": "collaboration",
        "complexity_tier": "basic",
        "plan_required": "starter",
        "avg_adoption_rate": 0.63,
        "churn_reduction_score": 0.44,
        "description": "Invite teammates, assign work, and collaborate inside shared projects.",
        "tags": "team|collaboration|sharing|productivity",
    },
    {
        "feature_id": "f06",
        "feature_name": "sso_login",
        "category": "security",
        "complexity_tier": "advanced",
        "plan_required": "enterprise",
        "avg_adoption_rate": 0.38,
        "churn_reduction_score": 0.55,
        "description": "Centralize identity management with SAML or OIDC single sign-on.",
        "tags": "security|sso|identity|enterprise",
    },
    {
        "feature_id": "f07",
        "feature_name": "webhook_setup",
        "category": "developer",
        "complexity_tier": "advanced",
        "plan_required": "growth",
        "avg_adoption_rate": 0.29,
        "churn_reduction_score": 0.31,
        "description": "Send real-time product events to external systems through webhooks.",
        "tags": "webhook|developer|automation|integration",
    },
    {
        "feature_id": "f08",
        "feature_name": "zapier_integration",
        "category": "integrations",
        "complexity_tier": "basic",
        "plan_required": "starter",
        "avg_adoption_rate": 0.55,
        "churn_reduction_score": 0.28,
        "description": "Connect common no-code workflows through Zapier integrations.",
        "tags": "integration|zapier|automation|no-code",
    },
    {
        "feature_id": "f09",
        "feature_name": "advanced_analytics",
        "category": "analytics",
        "complexity_tier": "power",
        "plan_required": "enterprise",
        "avg_adoption_rate": 0.32,
        "churn_reduction_score": 0.45,
        "description": "Analyze advanced usage cohorts, funnels, and operational performance.",
        "tags": "analytics|cohorts|funnels|enterprise",
    },
    {
        "feature_id": "f10",
        "feature_name": "audit_logs",
        "category": "security",
        "complexity_tier": "advanced",
        "plan_required": "enterprise",
        "avg_adoption_rate": 0.28,
        "churn_reduction_score": 0.41,
        "description": "Track account activity with searchable security and compliance logs.",
        "tags": "security|audit|compliance|enterprise",
    },
    {
        "feature_id": "f11",
        "feature_name": "white_labeling",
        "category": "customization",
        "complexity_tier": "power",
        "plan_required": "enterprise",
        "avg_adoption_rate": 0.21,
        "churn_reduction_score": 0.33,
        "description": "Customize customer-facing product surfaces with your own branding.",
        "tags": "branding|customization|white-label|enterprise",
    },
    {
        "feature_id": "f12",
        "feature_name": "custom_roles",
        "category": "security",
        "complexity_tier": "advanced",
        "plan_required": "enterprise",
        "avg_adoption_rate": 0.35,
        "churn_reduction_score": 0.48,
        "description": "Define granular permissions for operational, finance, and admin roles.",
        "tags": "security|roles|permissions|enterprise",
    },
    {
        "feature_id": "f13",
        "feature_name": "priority_support",
        "category": "support",
        "complexity_tier": "basic",
        "plan_required": "growth",
        "avg_adoption_rate": 0.48,
        "churn_reduction_score": 0.52,
        "description": "Get faster support response times and priority escalation paths.",
        "tags": "support|priority|success|retention",
    },
    {
        "feature_id": "f14",
        "feature_name": "bulk_operations",
        "category": "productivity",
        "complexity_tier": "advanced",
        "plan_required": "growth",
        "avg_adoption_rate": 0.41,
        "churn_reduction_score": 0.29,
        "description": "Apply product changes to many records or users in one operation.",
        "tags": "productivity|bulk|operations|automation",
    },
    {
        "feature_id": "f15",
        "feature_name": "mobile_app",
        "category": "productivity",
        "complexity_tier": "basic",
        "plan_required": "starter",
        "avg_adoption_rate": 0.59,
        "churn_reduction_score": 0.22,
        "description": "Use key product workflows on iOS and Android devices.",
        "tags": "mobile|productivity|access|notifications",
    },
    {
        "feature_id": "f16",
        "feature_name": "scheduled_reports",
        "category": "analytics",
        "complexity_tier": "basic",
        "plan_required": "growth",
        "avg_adoption_rate": 0.44,
        "churn_reduction_score": 0.25,
        "description": "Send scheduled analytics reports to stakeholders by email.",
        "tags": "analytics|reporting|schedule|email",
    },
    {
        "feature_id": "f17",
        "feature_name": "crm_integration",
        "category": "integrations",
        "complexity_tier": "advanced",
        "plan_required": "growth",
        "avg_adoption_rate": 0.36,
        "churn_reduction_score": 0.37,
        "description": "Sync customer and opportunity data with CRM systems.",
        "tags": "integration|crm|sales|sync",
    },
    {
        "feature_id": "f18",
        "feature_name": "custom_workflows",
        "category": "productivity",
        "complexity_tier": "power",
        "plan_required": "enterprise",
        "avg_adoption_rate": 0.27,
        "churn_reduction_score": 0.43,
        "description": "Build custom workflow automations around business-specific processes.",
        "tags": "workflow|automation|productivity|enterprise",
    },
    {
        "feature_id": "f19",
        "feature_name": "ai_insights",
        "category": "analytics",
        "complexity_tier": "power",
        "plan_required": "enterprise",
        "avg_adoption_rate": 0.23,
        "churn_reduction_score": 0.39,
        "description": "Surface AI-generated insights from account behavior and trends.",
        "tags": "ai|analytics|insights|prediction",
    },
    {
        "feature_id": "f20",
        "feature_name": "data_warehouse_sync",
        "category": "data",
        "complexity_tier": "power",
        "plan_required": "enterprise",
        "avg_adoption_rate": 0.19,
        "churn_reduction_score": 0.46,
        "description": "Sync product data into cloud data warehouses for central analysis.",
        "tags": "data|warehouse|sync|enterprise",
    },
]

CO_ADOPTION_RULES = {
    "f01": [("f07", 3.0)],
    "f02": [("f20", 2.0)],
    "f06": [("f12", 2.0), ("f10", 2.0)],
}


def seed_everything() -> np.random.Generator:
    """Seed Python and NumPy random generators for reproducible data."""
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    return np.random.default_rng(RANDOM_STATE)


def build_feature_catalog() -> pd.DataFrame:
    """Create the canonical 20-item SaaS feature catalog."""
    features = pd.DataFrame(FEATURE_ROWS)
    features.to_csv(DATA_DIR / "feature_metadata.csv", index=False)
    logger.info("Wrote feature catalog with {} features", len(features))
    return features


def _sample_date(rng: np.random.Generator) -> pd.Timestamp:
    days = int((END_DATE - START_DATE).days)
    return START_DATE + pd.Timedelta(days=int(rng.integers(0, days + 1)))


def _sample_team_size(plan_tier: str, rng: np.random.Generator) -> int:
    if plan_tier == "starter":
        return int(rng.integers(1, 6))
    if plan_tier == "growth":
        return int(rng.integers(5, 51))
    return int(rng.integers(50, 501))


def _sample_mrr(plan_tier: str, rng: np.random.Generator) -> int:
    if plan_tier == "starter":
        return int(rng.integers(49, 200))
    if plan_tier == "growth":
        return int(rng.integers(299, 1_000))
    return int(rng.integers(1_499, 10_000))


def _compute_churn_probability(
    plan_tier: str,
    contract_type: str,
    tenure_days: int,
    team_size: int,
    rng: np.random.Generator,
) -> float:
    base = 0.09
    contract_effect = 0.11 if contract_type == "monthly" else -0.02
    plan_effect = {"starter": 0.055, "growth": 0.020, "enterprise": -0.030}[plan_tier]
    tenure_effect = 0.13 if tenure_days < 90 else 0.045 if tenure_days < 365 else -0.015
    team_effect = 0.030 if team_size <= 5 else 0.015 if team_size <= 20 else -0.010
    noise = float(rng.normal(0.0, 0.045))
    return float(np.clip(base + contract_effect + plan_effect + tenure_effect + team_effect + noise, 0.02, 0.85))


def generate_customers(rng: np.random.Generator) -> pd.DataFrame:
    """Generate synthetic SaaS customer profiles."""
    plan_names = list(PLAN_PROBABILITIES)
    plan_probs = list(PLAN_PROBABILITIES.values())
    contract_names = list(CONTRACT_PROBABILITIES)
    contract_probs = list(CONTRACT_PROBABILITIES.values())
    records: list[dict[str, object]] = []

    for idx in range(1, NUM_CUSTOMERS + 1):
        plan_tier = str(rng.choice(plan_names, p=plan_probs))
        contract_type = str(rng.choice(contract_names, p=contract_probs))
        signup_date = _sample_date(rng)
        tenure_days = max(1, int((END_DATE - signup_date).days))
        team_size = _sample_team_size(plan_tier, rng)
        mrr = _sample_mrr(plan_tier, rng)
        churn_probability = _compute_churn_probability(plan_tier, contract_type, tenure_days, team_size, rng)

        records.append(
            {
                "customer_id": f"C{idx:05d}",
                "signup_date": signup_date.date().isoformat(),
                "plan_tier": plan_tier,
                "contract_type": contract_type,
                "team_size": team_size,
                "mrr": mrr,
                "tenure_days": tenure_days,
                "industry": str(rng.choice(INDUSTRIES)),
                "churn_probability": round(churn_probability, 4),
            }
        )

    customers = pd.DataFrame(records)
    customers.to_csv(DATA_DIR / "customers.csv", index=False)
    logger.info("Wrote customers.csv with mean churn probability {:.3f}", customers["churn_probability"].mean())
    return customers


def _plan_allows(plan_tier: str, plan_required: str) -> bool:
    return PLAN_ORDER[plan_tier] >= PLAN_ORDER[plan_required]


def _monthly_dates(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    current = pd.Timestamp(start).replace(day=1)
    final = pd.Timestamp(end).replace(day=1)
    while current <= final:
        yield current
        current = current + pd.DateOffset(months=1)


def _adoption_probability(customer: pd.Series, feature: pd.Series, rng: np.random.Generator) -> float:
    plan_multiplier = PLAN_TIER_MULTIPLIER[str(customer["plan_tier"])]
    churn_drag = 1.0 - 0.65 * float(customer["churn_probability"])
    complexity = COMPLEXITY_USAGE_MULTIPLIER[str(feature["complexity_tier"])]
    stochastic_noise = float(rng.lognormal(mean=0.0, sigma=0.16))
    probability = float(feature["avg_adoption_rate"]) * plan_multiplier * churn_drag * complexity * stochastic_noise
    return float(np.clip(probability, 0.01, 0.95))


def _conditional_boost_probability(base_probability: float, multiplier: float) -> float:
    target_probability = min(0.95, base_probability * multiplier)
    if base_probability >= 1.0:
        return 0.0
    return float(max(0.0, (target_probability - base_probability) / (1.0 - base_probability)))


def _usage_count(customer: pd.Series, feature: pd.Series, rng: np.random.Generator) -> int:
    churn_factor = 1.0 - 0.55 * float(customer["churn_probability"])
    contract_factor = 1.12 if customer["contract_type"] == "annual" else 1.0
    complexity_factor = {"basic": 1.15, "advanced": 0.90, "power": 0.72}[str(feature["complexity_tier"])]
    value = rng.gamma(shape=2.0, scale=8.5) * churn_factor * contract_factor * complexity_factor
    return int(np.clip(round(value), 1, 50))


def generate_events(customers: pd.DataFrame, features: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate monthly feature usage events for adopted customer-feature pairs."""
    feature_by_id = features.set_index("feature_id")
    feature_records = features.to_dict("records")
    rows: list[dict[str, object]] = []

    for customer in customers.itertuples(index=False):
        customer_series = pd.Series(customer._asdict())
        eligible = [
            feature
            for feature in feature_records
            if _plan_allows(str(customer_series["plan_tier"]), str(feature["plan_required"]))
        ]
        base_probabilities = {
            str(feature["feature_id"]): _adoption_probability(customer_series, pd.Series(feature), rng)
            for feature in eligible
        }
        adopted: set[str] = {
            feature_id for feature_id, probability in base_probabilities.items() if float(rng.random()) < probability
        }

        for source_feature, targets in CO_ADOPTION_RULES.items():
            if source_feature not in adopted:
                continue
            for target_feature, multiplier in targets:
                if target_feature not in base_probabilities or target_feature in adopted:
                    continue
                boost_probability = _conditional_boost_probability(base_probabilities[target_feature], multiplier)
                if float(rng.random()) < boost_probability:
                    adopted.add(target_feature)

        signup_date = pd.Timestamp(customer_series["signup_date"])
        months = list(_monthly_dates(signup_date, END_DATE))
        if not months:
            continue

        for feature_id in sorted(adopted):
            feature = feature_by_id.loc[feature_id]
            first_month_index = int(rng.integers(0, len(months)))
            for usage_date in months[first_month_index:]:
                rows.append(
                    {
                        "customer_id": str(customer_series["customer_id"]),
                        "feature_id": feature_id,
                        "usage_date": usage_date.date().isoformat(),
                        "usage_count": _usage_count(customer_series, feature, rng),
                    }
                )

    events = pd.DataFrame(rows, columns=["customer_id", "feature_id", "usage_date", "usage_count"])
    events.to_csv(DATA_DIR / "events.csv", index=False)

    total_pairs = len(customers) * len(features)
    adopted_pairs = events[["customer_id", "feature_id"]].drop_duplicates().shape[0]
    sparsity = 1.0 - adopted_pairs / total_pairs
    logger.info("Wrote events.csv with {} rows and {:.1%} customer-feature sparsity", len(events), sparsity)
    return events


def main() -> None:
    """Generate all raw CSV data from scratch."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = seed_everything()
    features = build_feature_catalog()
    customers = generate_customers(rng)
    events = generate_events(customers, features, rng)

    print(f"Generated {len(features)} features, {len(customers)} customers, {len(events):,} usage events")
    print(f"Mean churn probability: {customers['churn_probability'].mean():.3f}")


if __name__ == "__main__":
    main()
