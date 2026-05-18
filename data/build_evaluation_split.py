"""Build time-based train and test interaction splits."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from loguru import logger
from scipy import sparse

RANDOM_STATE = 42
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
HOLDOUT_FRACTION = 0.20
RECENT_HOLDOUT_DAYS = 90


def seed_everything() -> None:
    """Seed random generators for reproducible train/test splitting."""
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


def _load_mappings() -> dict[str, object]:
    return json.loads((DATA_DIR / "user_item_mappings.json").read_text())


def build_evaluation_split() -> tuple[sparse.csr_matrix, sparse.csr_matrix, pd.DataFrame]:
    """Hold out each user's latest adopted features for offline evaluation."""
    mappings = _load_mappings()
    customer_ids = list(mappings["customer_ids"])
    feature_ids = list(mappings["feature_ids"])
    customer_to_index = dict(mappings["customer_to_index"])
    feature_to_index = dict(mappings["feature_to_index"])

    interactions = sparse.load_npz(DATA_DIR / "interaction_sparse.npz").tocsr().astype(np.float32)
    events = pd.read_csv(DATA_DIR / "events.csv", parse_dates=["usage_date"])

    first_adoptions = (
        events.groupby(["customer_id", "feature_id"], as_index=False)["usage_date"].min()
        if not events.empty
        else pd.DataFrame(columns=["customer_id", "feature_id", "usage_date"])
    )

    train = interactions.copy().tolil()
    test = sparse.lil_matrix(interactions.shape, dtype=np.float32)
    evaluation_rows: list[dict[str, str]] = []

    for customer_id in customer_ids:
        user_adoptions = first_adoptions.loc[first_adoptions["customer_id"] == customer_id].copy()
        if user_adoptions.empty:
            continue

        user_adoptions = user_adoptions.sort_values("usage_date")
        holdout_count = max(1, int(math.ceil(len(user_adoptions) * HOLDOUT_FRACTION)))
        latest_first_use = user_adoptions["usage_date"].max()
        recent_cutoff = latest_first_use - pd.Timedelta(days=RECENT_HOLDOUT_DAYS)
        recent = user_adoptions.loc[user_adoptions["usage_date"] >= recent_cutoff]

        selected = recent.tail(holdout_count)
        if len(selected) < holdout_count:
            fill = user_adoptions.loc[~user_adoptions["feature_id"].isin(selected["feature_id"])]
            selected = pd.concat([fill.tail(holdout_count - len(selected)), selected], ignore_index=True)

        held_out_features = [str(feature_id) for feature_id in selected["feature_id"].tolist()]
        user_index = int(customer_to_index[customer_id])
        retained_held_out: list[str] = []

        for feature_id in held_out_features:
            item_index = int(feature_to_index[feature_id])
            value = float(interactions[user_index, item_index])
            if value <= 0:
                value = 1.0
            train[user_index, item_index] = 0.0
            test[user_index, item_index] = value
            retained_held_out.append(feature_id)

        if retained_held_out:
            evaluation_rows.append(
                {
                    "customer_id": customer_id,
                    "held_out_features": "|".join(retained_held_out),
                }
            )

    train_csr = train.tocsr()
    test_csr = test.tocsr()
    evaluation_users = pd.DataFrame(evaluation_rows)

    sparse.save_npz(DATA_DIR / "train_interactions.npz", train_csr)
    sparse.save_npz(DATA_DIR / "test_interactions.npz", test_csr)
    evaluation_users.to_csv(DATA_DIR / "evaluation_users.csv", index=False)

    logger.info(
        "Saved train/test split: {} train nnz, {} test nnz, {} evaluation users",
        train_csr.nnz,
        test_csr.nnz,
        len(evaluation_users),
    )
    print(f"Train interactions: {train_csr.nnz:,}")
    print(f"Test interactions: {test_csr.nnz:,}")
    print(f"Evaluation users: {len(evaluation_users):,}")

    return train_csr, test_csr, evaluation_users


def main() -> None:
    """Run train/test split construction."""
    seed_everything()
    build_evaluation_split()


if __name__ == "__main__":
    main()
