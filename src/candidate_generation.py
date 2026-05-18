"""Stage 1 candidate generation with ALS and LightFM hybrid recommenders."""

from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from implicit import als
from lightfm import LightFM
from lightfm.data import Dataset
from loguru import logger
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

RANDOM_STATE = 42
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

ALS_FACTORS = 128
ALS_REGULARIZATION = 0.05
ALS_ITERATIONS = 50
ALS_ALPHA = 40.0
LIGHTFM_COMPONENTS = 128
LIGHTFM_EPOCHS = 50
LIGHTFM_THREADS = 4
MAX_CANDIDATES = 50

PLAN_ORDER = {"starter": 0, "growth": 1, "enterprise": 2}
TENURE_BUCKETS = [(0, 90, "0-90d"), (91, 365, "91-365d"), (366, 730, "1-2yr"), (731, 100_000, "2yr+")]
TEAM_BUCKETS = [(1, 5, "1-5"), (6, 20, "6-20"), (21, 100, "21-100"), (101, 100_000, "100+")]
MRR_BUCKETS = [(0, 199, "<200"), (200, 999, "200-999"), (1_000, 1_000_000, "1000+")]


def seed_everything() -> None:
    """Seed random generators for reproducible candidate generation."""
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


def load_mappings() -> dict[str, Any]:
    """Load stable customer and feature index mappings."""
    return json.loads((DATA_DIR / "user_item_mappings.json").read_text())


def _bucket(value: float, buckets: list[tuple[int, int, str]]) -> str:
    for low, high, name in buckets:
        if low <= value <= high:
            return name
    return buckets[-1][2]


def churn_risk_tier(churn_probability: float) -> str:
    """Convert churn probability into a stable categorical risk tier."""
    if churn_probability < 0.20:
        return "low"
    if churn_probability <= 0.50:
        return "medium"
    return "high"


def _churn_reduction_bucket(score: float) -> str:
    if score < 0.30:
        return "low"
    if score < 0.45:
        return "med"
    return "high"


def _customer_feature_tokens(customer: pd.Series) -> list[str]:
    return [
        f"plan_tier:{customer['plan_tier']}",
        f"contract_type:{customer['contract_type']}",
        f"tenure_bucket:{_bucket(float(customer['tenure_days']), TENURE_BUCKETS)}",
        f"team_size_bucket:{_bucket(float(customer['team_size']), TEAM_BUCKETS)}",
        f"churn_risk_tier:{churn_risk_tier(float(customer['churn_probability']))}",
        f"mrr_bucket:{_bucket(float(customer['mrr']), MRR_BUCKETS)}",
    ]


def _build_item_feature_tokens(features: pd.DataFrame) -> tuple[list[str], dict[str, dict[str, float]]]:
    tag_text = features["tags"].str.replace("|", " ", regex=False).fillna("")
    vectorizer = TfidfVectorizer(max_features=20, token_pattern=r"(?u)\b[\w-]+\b")
    tag_matrix = vectorizer.fit_transform(tag_text)
    tag_names = [f"tag:{token}" for token in vectorizer.get_feature_names_out()]

    item_features: dict[str, dict[str, float]] = {}
    for row_idx, feature in features.reset_index(drop=True).iterrows():
        tokens: dict[str, float] = {
            f"category:{feature['category']}": 1.0,
            f"complexity_tier:{feature['complexity_tier']}": 1.0,
            f"plan_required:{feature['plan_required']}": 1.0,
            f"churn_reduction:{_churn_reduction_bucket(float(feature['churn_reduction_score']))}": 1.0,
        }
        row = tag_matrix.getrow(row_idx)
        for value, tag_idx in zip(row.data, row.indices, strict=False):
            tokens[tag_names[int(tag_idx)]] = float(value)
        item_features[str(feature["feature_id"])] = tokens

    base_feature_names = sorted({token for tokens in item_features.values() for token in tokens})
    return base_feature_names + tag_names, item_features


def train_als(train_interactions: sparse.csr_matrix, mappings: dict[str, Any], features: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """Train implicit ALS and return top-N candidates for every customer."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_scores = train_interactions.tocsr().astype(np.float32)
    confidence = train_scores.copy()
    confidence.data = 1.0 + ALS_ALPHA * confidence.data
    item_user_matrix = confidence.T.tocsr()

    model = als.AlternatingLeastSquares(
        factors=ALS_FACTORS,
        regularization=ALS_REGULARIZATION,
        iterations=ALS_ITERATIONS,
        random_state=RANDOM_STATE,
        use_gpu=False,
    )
    model.fit(item_user_matrix)

    user_factors = np.asarray(model.item_factors, dtype=np.float32)
    item_factors = np.asarray(model.user_factors, dtype=np.float32)
    np.savez_compressed(
        MODEL_DIR / "als_model.npz",
        user_factors=user_factors,
        item_factors=item_factors,
    )

    customer_ids = list(mappings["customer_ids"])
    feature_ids = list(mappings["feature_ids"])
    feature_names = features.set_index("feature_id")["feature_name"].to_dict()
    als_candidates: dict[str, list[tuple[str, float]]] = {}

    for user_idx, customer_id in enumerate(customer_ids):
        scores = item_factors @ user_factors[user_idx]
        used_items = set(train_scores[user_idx].indices.tolist())
        item_ids = [int(item_idx) for item_idx in np.argsort(-scores) if int(item_idx) not in used_items]
        als_candidates[str(customer_id)] = [
            (feature_ids[int(item_id)], float(scores[int(item_id)]))
            for item_id in item_ids[:MAX_CANDIDATES]
            if int(item_id) < len(feature_ids)
        ]

    with (DATA_DIR / "als_candidates.pkl").open("wb") as file:
        pickle.dump(als_candidates, file)

    sample_names = [feature_names[feature_id] for feature_id, _ in als_candidates.get("C00001", [])[:10]]
    print(f"ALS trained. User embedding shape: {user_factors.shape}")
    print(f"Sample recommendations for user C00001: {sample_names}")
    logger.info("Saved ALS model and candidates")
    return als_candidates


def build_lightfm_dataset(
    customers: pd.DataFrame,
    features: pd.DataFrame,
    train_interactions: sparse.csr_matrix,
    mappings: dict[str, Any],
) -> tuple[Dataset, sparse.coo_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    """Build LightFM interactions plus sparse user and item feature matrices."""
    customer_ids = list(mappings["customer_ids"])
    feature_ids = list(mappings["feature_ids"])

    user_token_map = {
        str(customer["customer_id"]): _customer_feature_tokens(customer)
        for _, customer in customers.iterrows()
    }
    user_feature_names = sorted({token for tokens in user_token_map.values() for token in tokens})
    item_feature_names, item_token_map = _build_item_feature_tokens(features)

    dataset = Dataset()
    dataset.fit(
        users=customer_ids,
        items=feature_ids,
        user_features=user_feature_names,
        item_features=item_feature_names,
    )

    user_features_matrix = dataset.build_user_features(
        ((customer_id, user_token_map[str(customer_id)]) for customer_id in customer_ids),
        normalize=False,
    ).tocsr()
    item_features_matrix = dataset.build_item_features(
        ((feature_id, item_token_map[str(feature_id)]) for feature_id in feature_ids),
        normalize=False,
    ).tocsr()

    interactions_coo = train_interactions.tocoo().astype(np.float32)
    return dataset, interactions_coo, user_features_matrix, item_features_matrix


def train_lightfm(
    customers: pd.DataFrame,
    features: pd.DataFrame,
    train_interactions: sparse.csr_matrix,
    mappings: dict[str, Any],
) -> dict[str, list[tuple[str, float]]]:
    """Train a hybrid LightFM WARP model and return per-customer candidates."""
    dataset, interactions, user_features_matrix, item_features_matrix = build_lightfm_dataset(
        customers, features, train_interactions, mappings
    )

    lightfm_model = LightFM(
        no_components=LIGHTFM_COMPONENTS,
        loss="warp",
        learning_rate=0.05,
        item_alpha=1e-6,
        user_alpha=1e-6,
        random_state=RANDOM_STATE,
    )
    lightfm_model.fit(
        interactions,
        user_features=user_features_matrix,
        item_features=item_features_matrix,
        epochs=LIGHTFM_EPOCHS,
        num_threads=LIGHTFM_THREADS,
        verbose=True,
    )

    customer_ids = list(mappings["customer_ids"])
    feature_ids = list(mappings["feature_ids"])
    all_item_indices = np.arange(len(feature_ids), dtype=np.int32)
    train_binary = train_interactions.tocsr()
    lightfm_candidates: dict[str, list[tuple[str, float]]] = {}

    for user_idx, customer_id in enumerate(customer_ids):
        scores = lightfm_model.predict(
            user_idx,
            all_item_indices,
            user_features=user_features_matrix,
            item_features=item_features_matrix,
            num_threads=LIGHTFM_THREADS,
        )
        used = set(train_binary[user_idx].indices.tolist())
        ranked_indices = [
            int(item_idx)
            for item_idx in np.argsort(-scores)
            if int(item_idx) not in used
        ][:MAX_CANDIDATES]
        lightfm_candidates[str(customer_id)] = [
            (feature_ids[item_idx], float(scores[item_idx])) for item_idx in ranked_indices
        ]

    model_payload = {
        "model": lightfm_model,
        "dataset_mapping": dataset.mapping(),
        "user_features": user_features_matrix,
        "item_features": item_features_matrix,
        "customer_ids": customer_ids,
        "feature_ids": feature_ids,
    }
    joblib.dump(model_payload, MODEL_DIR / "lightfm_model.pkl")
    with (DATA_DIR / "lightfm_candidates.pkl").open("wb") as file:
        pickle.dump(lightfm_candidates, file)

    logger.info("Saved LightFM model and candidates")
    return lightfm_candidates


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def merge_candidates(
    customer_id: str,
    top_k: int = MAX_CANDIDATES,
    als_candidates: dict[str, list[tuple[str, float]]] | None = None,
    lightfm_candidates: dict[str, list[tuple[str, float]]] | None = None,
    train_interactions: sparse.csr_matrix | None = None,
    mappings: dict[str, Any] | None = None,
) -> list[dict[str, float | str]]:
    """Union ALS and LightFM candidates, deduplicate, and exclude already-used features."""
    loaded_mappings = mappings if mappings is not None else load_mappings()
    loaded_train = train_interactions if train_interactions is not None else sparse.load_npz(DATA_DIR / "train_interactions.npz").tocsr()
    loaded_als = als_candidates if als_candidates is not None else _load_pickle(DATA_DIR / "als_candidates.pkl")
    loaded_lightfm = lightfm_candidates if lightfm_candidates is not None else _load_pickle(DATA_DIR / "lightfm_candidates.pkl")

    feature_ids = list(loaded_mappings["feature_ids"])
    customer_to_index = dict(loaded_mappings["customer_to_index"])
    user_idx = int(customer_to_index[customer_id])
    already_used = {feature_ids[int(item_idx)] for item_idx in loaded_train[user_idx].indices}

    merged: dict[str, dict[str, float | str]] = {}
    for feature_id, score in loaded_als.get(customer_id, []):
        if feature_id in already_used:
            continue
        merged.setdefault(
            feature_id,
            {"feature_id": feature_id, "als_score": 0.0, "lightfm_score": 0.0, "candidate_score": 0.0},
        )
        merged[feature_id]["als_score"] = max(float(merged[feature_id]["als_score"]), float(score))

    for feature_id, score in loaded_lightfm.get(customer_id, []):
        if feature_id in already_used:
            continue
        merged.setdefault(
            feature_id,
            {"feature_id": feature_id, "als_score": 0.0, "lightfm_score": 0.0, "candidate_score": 0.0},
        )
        merged[feature_id]["lightfm_score"] = max(float(merged[feature_id]["lightfm_score"]), float(score))

    for payload in merged.values():
        payload["candidate_score"] = max(float(payload["als_score"]), float(payload["lightfm_score"]))

    return sorted(merged.values(), key=lambda item: float(item["candidate_score"]), reverse=True)[:top_k]


def build_merged_candidates(
    als_candidates: dict[str, list[tuple[str, float]]],
    lightfm_candidates: dict[str, list[tuple[str, float]]],
    train_interactions: sparse.csr_matrix,
    mappings: dict[str, Any],
) -> dict[str, list[dict[str, float | str]]]:
    """Create and persist merged candidate pools for every customer."""
    customer_ids = list(mappings["customer_ids"])
    merged_candidates = {
        str(customer_id): merge_candidates(
            str(customer_id),
            top_k=MAX_CANDIDATES,
            als_candidates=als_candidates,
            lightfm_candidates=lightfm_candidates,
            train_interactions=train_interactions,
            mappings=mappings,
        )
        for customer_id in customer_ids
    }
    with (DATA_DIR / "merged_candidates.pkl").open("wb") as file:
        pickle.dump(merged_candidates, file)

    lengths = np.array([len(items) for items in merged_candidates.values()])
    print(f"Avg candidates per user: {lengths.mean():.1f} | Users with <10 candidates: {int((lengths < 10).sum())}")
    logger.info("Saved merged candidates")
    return merged_candidates


def main() -> None:
    """Train both candidate generators and persist merged candidate pools."""
    seed_everything()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    features = pd.read_csv(DATA_DIR / "feature_metadata.csv")
    train_interactions = sparse.load_npz(DATA_DIR / "train_interactions.npz").tocsr().astype(np.float32)
    mappings = load_mappings()

    als_candidates = train_als(train_interactions, mappings, features)
    lightfm_candidates = train_lightfm(customers, features, train_interactions, mappings)
    build_merged_candidates(als_candidates, lightfm_candidates, train_interactions, mappings)


if __name__ == "__main__":
    main()
