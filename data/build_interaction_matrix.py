"""Build dense and sparse customer-feature interaction matrices."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import sparse

RANDOM_STATE = 42
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
END_DATE = pd.Timestamp("2024-12-31")
LOOKBACK_DAYS = 90


def seed_everything() -> None:
    """Seed random generators for reproducible matrix construction."""
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


def build_interaction_matrix() -> tuple[pd.DataFrame, sparse.csr_matrix]:
    """Aggregate recent feature usage into dense and sparse interaction matrices."""
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    features = pd.read_csv(DATA_DIR / "feature_metadata.csv")
    events = pd.read_csv(DATA_DIR / "events.csv", parse_dates=["usage_date"])

    customer_ids = customers["customer_id"].tolist()
    feature_ids = features["feature_id"].tolist()
    lookback_start = END_DATE - pd.Timedelta(days=LOOKBACK_DAYS)

    recent_events = events.loc[events["usage_date"] >= lookback_start].copy()
    grouped = (
        recent_events.groupby(["customer_id", "feature_id"], as_index=False)["usage_count"].sum()
        if not recent_events.empty
        else pd.DataFrame(columns=["customer_id", "feature_id", "usage_count"])
    )
    grouped["implicit_score"] = np.log1p(grouped["usage_count"].astype(float))

    matrix_df = pd.DataFrame(0.0, index=customer_ids, columns=feature_ids)
    for row in grouped.itertuples(index=False):
        matrix_df.at[str(row.customer_id), str(row.feature_id)] = float(row.implicit_score)

    interaction_sparse = sparse.csr_matrix(matrix_df.values, dtype=np.float32)

    matrix_df.index.name = "customer_id"
    matrix_df.to_csv(DATA_DIR / "interaction_matrix.csv")
    sparse.save_npz(DATA_DIR / "interaction_sparse.npz", interaction_sparse)

    mappings = {
        "customer_ids": customer_ids,
        "feature_ids": feature_ids,
        "customer_to_index": {customer_id: idx for idx, customer_id in enumerate(customer_ids)},
        "feature_to_index": {feature_id: idx for idx, feature_id in enumerate(feature_ids)},
    }
    (DATA_DIR / "user_item_mappings.json").write_text(json.dumps(mappings, indent=2))

    adoption_rates = (matrix_df > 0).mean(axis=0)
    feature_names = features.set_index("feature_id")["feature_name"].to_dict()
    most_feature_id = str(adoption_rates.idxmax())
    least_feature_id = str(adoption_rates.idxmin())
    sparsity = float((matrix_df.values == 0).mean() * 100)

    logger.info("Saved dense and sparse matrices with shape {}", matrix_df.shape)
    print(f"Interaction matrix: {matrix_df.shape[0]} total users x {matrix_df.shape[1]} items")
    print(f"Sparsity: {sparsity:.1f}% zeros")
    print(f"Most adopted feature: {feature_names[most_feature_id]} ({adoption_rates[most_feature_id] * 100:.1f}% of users)")
    print(f"Least adopted feature: {feature_names[least_feature_id]} ({adoption_rates[least_feature_id] * 100:.1f}% of users)")

    return matrix_df, interaction_sparse


def main() -> None:
    """Run interaction matrix construction."""
    seed_everything()
    build_interaction_matrix()


if __name__ == "__main__":
    main()
