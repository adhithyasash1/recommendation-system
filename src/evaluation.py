"""Offline evaluation suite for retrieval, ranking, diversity, and business metrics."""

from __future__ import annotations

import json
import pickle
import random
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import sparse, stats

try:
    from src.cold_start import get_cold_start_recommendations, should_use_cold_start
    from src.ranker import rank_features
except ImportError:
    from cold_start import get_cold_start_recommendations, should_use_cold_start
    from ranker import rank_features

RANDOM_STATE = 42
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
RANKING_K_VALUES = [1, 3, 5, 10, 15]
TOP_K_BUSINESS = 10
TREATMENT_FRACTION = 0.80
CHURN_REDUCTION_SCALE = 0.02


def seed_everything() -> np.random.Generator:
    """Seed random generators for reproducible evaluation."""
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    return np.random.default_rng(RANDOM_STATE)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def _load_mappings() -> dict[str, Any]:
    return json.loads((DATA_DIR / "user_item_mappings.json").read_text())


def _parse_held_out(value: str) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {item for item in str(value).split("|") if item}


def _precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    top_items = recommended[:k]
    if not top_items:
        return 0.0
    return float(len(set(top_items) & relevant) / len(top_items))


def _recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return float(len(set(recommended[:k]) & relevant) / len(relevant))


def _ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    gains = np.array([1.0 if feature_id in relevant else 0.0 for feature_id in recommended[:k]], dtype=np.float32)
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float(np.sum(gains * discounts))
    ideal_len = min(len(relevant), k)
    if ideal_len == 0:
        return 0.0
    ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_len + 2))
    idcg = float(np.sum(ideal_discounts))
    return dcg / idcg if idcg > 0 else 0.0


def _mrr(recommended: list[str], relevant: set[str]) -> float:
    for idx, feature_id in enumerate(recommended, start=1):
        if feature_id in relevant:
            return 1.0 / idx
    return 0.0


def _hit_rate(recommended: list[str], relevant: set[str], k: int) -> float:
    return float(bool(set(recommended[:k]) & relevant))


def _tag_sets(features: pd.DataFrame) -> dict[str, set[str]]:
    return {
        str(row["feature_id"]): {tag.strip() for tag in str(row["tags"]).split("|") if tag.strip()}
        for _, row in features.iterrows()
    }


def _intra_list_diversity(feature_ids: list[str], tag_lookup: dict[str, set[str]]) -> float:
    pairs = list(combinations(feature_ids, 2))
    if not pairs:
        return 0.0
    distances = []
    for left, right in pairs:
        left_tags = tag_lookup.get(left, set())
        right_tags = tag_lookup.get(right, set())
        union = left_tags | right_tags
        jaccard = len(left_tags & right_tags) / len(union) if union else 0.0
        distances.append(1.0 - jaccard)
    return float(np.mean(distances))


def evaluate_stage1(evaluation_users: pd.DataFrame, merged_candidates: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    """Compute candidate retrieval metrics from held-out features."""
    recall_10: list[float] = []
    recall_50: list[float] = []
    hit_rate: list[float] = []

    for row in evaluation_users.itertuples(index=False):
        relevant = _parse_held_out(str(row.held_out_features))
        if not relevant:
            continue
        candidates = [str(item["feature_id"]) for item in merged_candidates.get(str(row.customer_id), [])]
        recall_10.append(_recall_at_k(candidates, relevant, 10))
        recall_50.append(_recall_at_k(candidates, relevant, 50))
        hit_rate.append(float(bool(set(candidates[:50]) & relevant)))

    return {
        "stage1_recall_at_10": float(np.mean(recall_10)) if recall_10 else 0.0,
        "stage1_recall_at_50": float(np.mean(recall_50)) if recall_50 else 0.0,
        "candidate_hit_rate": float(np.mean(hit_rate)) if hit_rate else 0.0,
    }


def _recommend_for_user(customer_id: str, top_k: int) -> list[dict[str, Any]]:
    if should_use_cold_start(customer_id):
        return get_cold_start_recommendations(customer_id, top_k=top_k)
    return rank_features(customer_id, top_k=top_k)


def _precompute_recommendations(evaluation_users: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    recommendations: dict[str, list[dict[str, Any]]] = {}
    for idx, row in enumerate(evaluation_users.itertuples(index=False), start=1):
        customer_id = str(row.customer_id)
        try:
            recommendations[customer_id] = _recommend_for_user(customer_id, max(RANKING_K_VALUES))
        except Exception as exc:
            logger.warning("Skipping recommendations for {} due to {}", customer_id, exc)
            recommendations[customer_id] = []
        if idx % 1_000 == 0:
            logger.info("Evaluated recommendations for {} users", idx)
    return recommendations


def evaluate_ranking(
    evaluation_users: pd.DataFrame,
    recommendations: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[int, float] | float]:
    """Compute Precision, Recall, NDCG, MRR, and Hit Rate for ranked recommendations."""
    precision = {k: [] for k in RANKING_K_VALUES}
    recall = {k: [] for k in RANKING_K_VALUES}
    ndcg = {k: [] for k in RANKING_K_VALUES}
    hit_rate = {k: [] for k in RANKING_K_VALUES}
    mrr_values: list[float] = []

    for row in evaluation_users.itertuples(index=False):
        customer_id = str(row.customer_id)
        relevant = _parse_held_out(str(row.held_out_features))
        if not relevant:
            continue
        recommended = [str(item["feature_id"]) for item in recommendations.get(customer_id, [])]
        for k in RANKING_K_VALUES:
            precision[k].append(_precision_at_k(recommended, relevant, k))
            recall[k].append(_recall_at_k(recommended, relevant, k))
            ndcg[k].append(_ndcg_at_k(recommended, relevant, k))
            hit_rate[k].append(_hit_rate(recommended, relevant, k))
        mrr_values.append(_mrr(recommended, relevant))

    return {
        "precision": {k: float(np.mean(values)) if values else 0.0 for k, values in precision.items()},
        "recall": {k: float(np.mean(values)) if values else 0.0 for k, values in recall.items()},
        "ndcg": {k: float(np.mean(values)) if values else 0.0 for k, values in ndcg.items()},
        "hit_rate": {k: float(np.mean(values)) if values else 0.0 for k, values in hit_rate.items()},
        "mrr": float(np.mean(mrr_values)) if mrr_values else 0.0,
    }


def evaluate_beyond_accuracy(
    features: pd.DataFrame,
    interactions: sparse.csr_matrix,
    mappings: dict[str, Any],
    recommendations: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Compute catalog coverage, diversity, popularity bias, novelty, and serendipity."""
    feature_ids = list(mappings["feature_ids"])
    tag_lookup = _tag_sets(features)
    feature_lookup = features.set_index("feature_id")
    top10_by_user = {
        customer_id: [str(item["feature_id"]) for item in recs[:TOP_K_BUSINESS]]
        for customer_id, recs in recommendations.items()
    }
    all_top10 = [feature_id for feature_list in top10_by_user.values() for feature_id in feature_list]
    coverage = len(set(all_top10)) / len(feature_ids) if feature_ids else 0.0
    diversity = float(np.mean([_intra_list_diversity(items, tag_lookup) for items in top10_by_user.values() if items]))

    adoption_binary = interactions.copy()
    adoption_binary.data = np.ones_like(adoption_binary.data)
    popularity_values = np.asarray(adoption_binary.mean(axis=0)).ravel()
    popularity = {feature_id: float(max(popularity_values[idx], 1.0 / interactions.shape[0])) for idx, feature_id in enumerate(feature_ids)}
    recommendation_counts = pd.Series(all_top10).value_counts().reindex(feature_ids, fill_value=0)
    adoption_rates = features.set_index("feature_id")["avg_adoption_rate"].reindex(feature_ids)
    popularity_correlation = float(np.corrcoef(recommendation_counts.values, adoption_rates.values)[0, 1])
    if np.isnan(popularity_correlation):
        popularity_correlation = 0.0

    novelty_values: list[float] = []
    serendipity_values: list[float] = []
    feature_points: dict[str, dict[str, float]] = {
        feature_id: {"recommendation_count": 0.0, "novelty_sum": 0.0, "serendipity_sum": 0.0}
        for feature_id in feature_ids
    }

    for recs in recommendations.values():
        for item in recs[:TOP_K_BUSINESS]:
            feature_id = str(item["feature_id"])
            novelty = float(np.log2(1.0 / popularity[feature_id]))
            serendipity = novelty * float(item.get("relevance_score", 0.0))
            novelty_values.append(novelty)
            serendipity_values.append(serendipity)
            feature_points[feature_id]["recommendation_count"] += 1.0
            feature_points[feature_id]["novelty_sum"] += novelty
            feature_points[feature_id]["serendipity_sum"] += serendipity

    distribution_rows: list[dict[str, Any]] = []
    for feature_id in feature_ids:
        count = feature_points[feature_id]["recommendation_count"]
        distribution_rows.append(
            {
                "feature_id": feature_id,
                "feature_name": str(feature_lookup.loc[feature_id, "feature_name"]),
                "recommendation_count": int(count),
                "avg_adoption_rate": float(feature_lookup.loc[feature_id, "avg_adoption_rate"]),
                "novelty": feature_points[feature_id]["novelty_sum"] / count if count else 0.0,
                "serendipity": feature_points[feature_id]["serendipity_sum"] / count if count else 0.0,
            }
        )
    distribution = pd.DataFrame(distribution_rows)
    distribution.to_csv(OUTPUT_DIR / "recommendation_distribution.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.scatter(adoption_rates.values, recommendation_counts.values, color="#6366f1")
    for feature_id, x_value, y_value in zip(feature_ids, adoption_rates.values, recommendation_counts.values, strict=False):
        plt.annotate(feature_id, (x_value, y_value), fontsize=8)
    plt.title("Popularity Bias: Recommendation Frequency vs Adoption Rate")
    plt.xlabel("Average adoption rate")
    plt.ylabel("Top-10 recommendation frequency")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "popularity_bias.png", dpi=180)
    plt.close()

    return (
        {
            "catalog_coverage": coverage,
            "aggregate_diversity": diversity,
            "popularity_bias_correlation": popularity_correlation,
            "novelty": float(np.mean(novelty_values)) if novelty_values else 0.0,
            "serendipity": float(np.mean(serendipity_values)) if serendipity_values else 0.0,
        },
        distribution,
    )


def simulate_ab_test(
    evaluation_users: pd.DataFrame,
    features: pd.DataFrame,
    mappings: dict[str, Any],
    train_interactions: sparse.csr_matrix,
    recommendations: dict[str, list[dict[str, Any]]],
    rng: np.random.Generator,
) -> dict[str, float]:
    """Run a simulated 90-day adoption A/B test for model recommendations."""
    feature_lookup = features.set_index("feature_id")
    customer_ids = evaluation_users["customer_id"].tolist()
    shuffled = list(rng.permutation(customer_ids))
    treatment_count = int(len(shuffled) * TREATMENT_FRACTION)
    treatment_users = set(shuffled[:treatment_count])
    control_users = set(shuffled[treatment_count:])

    treatment_rates: list[float] = []
    control_rates: list[float] = []
    treatment_churn_delta: list[float] = []
    control_churn_delta: list[float] = []
    feature_ids = list(mappings["feature_ids"])

    for customer_id in treatment_users:
        recs = recommendations.get(str(customer_id), [])[:TOP_K_BUSINESS]
        if not recs:
            continue
        adopted_flags = []
        churn_impact = 0.0
        for item in recs:
            feature_id = str(item["feature_id"])
            base_rate = float(feature_lookup.loc[feature_id, "avg_adoption_rate"])
            probability = float(np.clip(base_rate * float(item.get("relevance_score", 0.0)) * 1.3, 0.0, 0.95))
            adopted = float(rng.random() < probability)
            adopted_flags.append(adopted)
            churn_impact += adopted * float(feature_lookup.loc[feature_id, "churn_reduction_score"]) * CHURN_REDUCTION_SCALE
        treatment_rates.append(float(np.mean(adopted_flags)))
        treatment_churn_delta.append(churn_impact)

    for customer_id in control_users:
        user_idx = int(mappings["customer_to_index"][str(customer_id)])
        used = {feature_ids[idx] for idx in train_interactions[user_idx].indices}
        available = [feature_id for feature_id in feature_ids if feature_id not in used]
        if not available:
            continue
        sampled = list(rng.choice(available, size=min(TOP_K_BUSINESS, len(available)), replace=False))
        adopted_flags = []
        churn_impact = 0.0
        for feature_id in sampled:
            probability = float(feature_lookup.loc[feature_id, "avg_adoption_rate"])
            adopted = float(rng.random() < probability)
            adopted_flags.append(adopted)
            churn_impact += adopted * float(feature_lookup.loc[feature_id, "churn_reduction_score"]) * CHURN_REDUCTION_SCALE
        control_rates.append(float(np.mean(adopted_flags)))
        control_churn_delta.append(churn_impact)

    treatment_rate = float(np.mean(treatment_rates)) if treatment_rates else 0.0
    control_rate = float(np.mean(control_rates)) if control_rates else 0.0
    lift = (treatment_rate - control_rate) / control_rate * 100.0 if control_rate > 0 else 0.0
    t_stat, p_value = stats.ttest_ind(treatment_rates, control_rates, equal_var=False) if treatment_rates and control_rates else (0.0, 1.0)
    projected_churn_reduction = (float(np.mean(control_churn_delta)) - float(np.mean(treatment_churn_delta))) * 100.0

    results = {
        "treatment_adoption_rate": treatment_rate,
        "control_adoption_rate": control_rate,
        "adoption_lift_pct": lift,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "projected_churn_reduction_pp": projected_churn_reduction,
    }
    pd.DataFrame([results]).to_csv(OUTPUT_DIR / "ab_test_results.csv", index=False)
    return results


def evaluate_cold_start_comparison(
    evaluation_users: pd.DataFrame,
    recommendations: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    """Compare simple cold-start and warm-start ranking quality on key @5 metrics."""
    rows: list[dict[str, Any]] = []
    for segment_name, predicate in {
        "cold_start": should_use_cold_start,
        "warm_start": lambda customer_id: not should_use_cold_start(customer_id),
    }.items():
        segment = evaluation_users.loc[evaluation_users["customer_id"].map(predicate)]
        if segment.empty:
            rows.extend(
                [
                    {"segment": segment_name, "metric": "Hit Rate@5", "value": 0.0},
                    {"segment": segment_name, "metric": "NDCG@5", "value": 0.0},
                    {"segment": segment_name, "metric": "MRR", "value": 0.0},
                ]
            )
            continue
        hits: list[float] = []
        ndcgs: list[float] = []
        mrrs: list[float] = []
        for row in segment.itertuples(index=False):
            relevant = _parse_held_out(str(row.held_out_features))
            recommended = [str(item["feature_id"]) for item in recommendations.get(str(row.customer_id), [])]
            hits.append(_hit_rate(recommended, relevant, 5))
            ndcgs.append(_ndcg_at_k(recommended, relevant, 5))
            mrrs.append(_mrr(recommended, relevant))
        rows.extend(
            [
                {"segment": segment_name, "metric": "Hit Rate@5", "value": float(np.mean(hits))},
                {"segment": segment_name, "metric": "NDCG@5", "value": float(np.mean(ndcgs))},
                {"segment": segment_name, "metric": "MRR", "value": float(np.mean(mrrs))},
            ]
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT_DIR / "cold_start_comparison.csv", index=False)
    return comparison


def _save_report(
    stage1: dict[str, float],
    ranking: dict[str, dict[int, float] | float],
    beyond_accuracy: dict[str, float],
    ab_results: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric_name, value in stage1.items():
        rows.append({"metric": metric_name, "k": "", "value": value})
    for metric_name in ["precision", "recall", "ndcg", "hit_rate"]:
        metric_values = ranking[metric_name]
        assert isinstance(metric_values, dict)
        for k, value in metric_values.items():
            rows.append({"metric": metric_name, "k": k, "value": value})
    rows.append({"metric": "mrr", "k": "", "value": float(ranking["mrr"])})
    for metric_name, value in beyond_accuracy.items():
        rows.append({"metric": metric_name, "k": TOP_K_BUSINESS, "value": value})
    for metric_name, value in ab_results.items():
        rows.append({"metric": metric_name, "k": TOP_K_BUSINESS, "value": value})

    report = pd.DataFrame(rows)
    report.to_csv(OUTPUT_DIR / "evaluation_report.csv", index=False)
    return report


def _print_summary(ranking: dict[str, dict[int, float] | float], beyond: dict[str, float], ab: dict[str, float]) -> None:
    precision = ranking["precision"]
    recall = ranking["recall"]
    ndcg = ranking["ndcg"]
    hit_rate = ranking["hit_rate"]
    assert isinstance(precision, dict) and isinstance(recall, dict) and isinstance(ndcg, dict) and isinstance(hit_rate, dict)

    print("📊 A/B Test Results:")
    print(f"  Treatment adoption rate: {ab['treatment_adoption_rate'] * 100:.1f}%")
    print(f"  Control adoption rate: {ab['control_adoption_rate'] * 100:.1f}%")
    significance = "significant" if ab["p_value"] < 0.05 else "not significant"
    print(f"  Adoption lift: {ab['adoption_lift_pct']:+.1f}% (p={ab['p_value']:.4f}, {significance})")
    print(f"  Projected churn reduction (treatment vs control): {ab['projected_churn_reduction_pp']:+.2f} pp")
    print()
    print("| Metric       | @5     | @10    |")
    print("|--------------|--------|--------|")
    print(f"| Precision    | {precision[5]:.3f}  | {precision[10]:.3f}  |")
    print(f"| Recall       | {recall[5]:.3f}  | {recall[10]:.3f}  |")
    print(f"| NDCG         | {ndcg[5]:.3f}  | {ndcg[10]:.3f}  |")
    print(f"| Hit Rate     | {hit_rate[5]:.3f}  | {hit_rate[10]:.3f}  |")
    print(f"| MRR          | {float(ranking['mrr']):.3f}  | -      |")
    print(f"| Coverage     | -      | {beyond['catalog_coverage'] * 100:.1f}%  |")
    print(f"| Novelty      | -      | {beyond['novelty']:.2f}   |")
    print(f"| Serendipity  | -      | {beyond['serendipity']:.2f}   |")
    print(f"| A/B Lift     | -      | {ab['adoption_lift_pct']:+.1f}% |")


def main() -> None:
    """Run the full offline evaluation suite and persist reports."""
    rng = seed_everything()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(DATA_DIR / "feature_metadata.csv")
    evaluation_users = pd.read_csv(DATA_DIR / "evaluation_users.csv")
    mappings = _load_mappings()
    interactions = sparse.load_npz(DATA_DIR / "interaction_sparse.npz").tocsr()
    train_interactions = sparse.load_npz(DATA_DIR / "train_interactions.npz").tocsr()
    merged_candidates = _load_pickle(DATA_DIR / "merged_candidates.pkl")

    stage1 = evaluate_stage1(evaluation_users, merged_candidates)
    recommendations = _precompute_recommendations(evaluation_users)
    ranking = evaluate_ranking(evaluation_users, recommendations)
    beyond, _ = evaluate_beyond_accuracy(features, interactions, mappings, recommendations)
    ab_results = simulate_ab_test(evaluation_users, features, mappings, train_interactions, recommendations, rng)
    evaluate_cold_start_comparison(evaluation_users, recommendations)
    _save_report(stage1, ranking, beyond, ab_results)
    _print_summary(ranking, beyond, ab_results)


if __name__ == "__main__":
    main()
