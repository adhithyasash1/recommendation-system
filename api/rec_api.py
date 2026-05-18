"""FastAPI serving layer for the SaaS feature recommendation system."""

from __future__ import annotations

import json
import pickle
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from scipy import sparse

from api.schemas import FeatureAffinityResponse, RecommendationResponse, SimilarUsersResponse
from src.candidate_generation import churn_risk_tier
from src.cold_start import get_cold_start_recommendations, should_use_cold_start
from src.ranker import rank_features

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
LOG_DIR = ROOT_DIR / "logs"
MODEL_VERSION = "1.0.0"

LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(LOG_DIR / "api.log", rotation="10 MB", retention="14 days", serialize=True, enqueue=True)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def _load_mappings() -> dict[str, Any]:
    return json.loads((DATA_DIR / "user_item_mappings.json").read_text())


def _customer_dict(customers: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {str(row["customer_id"]): row.to_dict() for _, row in customers.iterrows()}


def _feature_dict(features: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {str(row["feature_id"]): row.to_dict() for _, row in features.iterrows()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and data artifacts into memory at application startup."""
    ranker_payload = joblib.load(MODEL_DIR / "lgbm_ranker.joblib")
    als_npz = np.load(MODEL_DIR / "als_model.npz")
    lightfm_payload = joblib.load(MODEL_DIR / "lightfm_model.pkl")
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    features = pd.read_csv(DATA_DIR / "feature_metadata.csv")

    app.state.ranker = ranker_payload
    app.state.als_factors = {
        "user_factors": np.asarray(als_npz["user_factors"], dtype=np.float32),
        "item_factors": np.asarray(als_npz["item_factors"], dtype=np.float32),
    }
    als_npz.close()
    app.state.lightfm = lightfm_payload
    app.state.customers_frame = customers
    app.state.features_frame = features
    app.state.customers = _customer_dict(customers)
    app.state.features = _feature_dict(features)
    app.state.candidates = _load_pickle(DATA_DIR / "merged_candidates.pkl")
    app.state.mappings = _load_mappings()
    app.state.interactions = sparse.load_npz(DATA_DIR / "interaction_sparse.npz").tocsr()
    app.state.start_time = datetime.utcnow()
    app.state.total_requests = 0
    app.state.total_latency_ms = 0.0
    logger.info("recommendation_api_started")
    yield
    logger.info("recommendation_api_stopped")


app = FastAPI(title="SaaS Feature Recommendation API", version=MODEL_VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """Track request latency and increment health counters."""
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000.0
    request.app.state.total_requests += 1
    request.app.state.total_latency_ms += latency_ms
    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
    logger.info(
        "request_completed method={} path={} status_code={} latency_ms={:.2f}",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response


@app.get("/recommend/{customer_id}", response_model=RecommendationResponse)
def recommend(customer_id: str, top_k: int = Query(10, ge=1, le=15)) -> RecommendationResponse:
    """Return top feature recommendations for a customer."""
    if customer_id not in app.state.customers:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    customer = app.state.customers[customer_id]
    start = time.perf_counter()
    cold_start_used = should_use_cold_start(customer_id)
    recommendations = (
        get_cold_start_recommendations(customer_id, top_k=top_k)
        if cold_start_used
        else rank_features(customer_id, top_k=top_k)
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "recommendation_generated customer_id={} plan_tier={} top_k={} churn_probability={} cold_start_used={} latency_ms={:.2f}",
        customer_id,
        customer["plan_tier"],
        top_k,
        customer["churn_probability"],
        cold_start_used,
        latency_ms,
    )

    return RecommendationResponse(
        customer_id=customer_id,
        plan_tier=str(customer["plan_tier"]),
        churn_probability=float(customer["churn_probability"]),
        churn_risk_tier=churn_risk_tier(float(customer["churn_probability"])),
        recommendations=recommendations,
        cold_start_used=cold_start_used,
        candidate_pool_size=len(app.state.candidates.get(customer_id, [])),
        model_version=MODEL_VERSION,
        generated_at=f"{datetime.utcnow().isoformat()}Z",
    )


@app.get("/similar-users/{customer_id}", response_model=SimilarUsersResponse)
def similar_users(customer_id: str, top_n: int = Query(5, ge=1, le=50)) -> SimilarUsersResponse:
    """Return ALS-nearest customers and their shared adopted features."""
    if customer_id not in app.state.customers:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    mappings = app.state.mappings
    user_idx = int(mappings["customer_to_index"][customer_id])
    factors = app.state.als_factors["user_factors"]
    query = factors[user_idx]
    norms = np.linalg.norm(factors, axis=1) * max(float(np.linalg.norm(query)), 1e-9)
    similarities = np.divide(factors @ query, norms, out=np.zeros(factors.shape[0], dtype=np.float32), where=norms > 0)
    similarities[user_idx] = -np.inf
    top_indices = np.argsort(-similarities)[:top_n]

    feature_ids = list(mappings["feature_ids"])
    feature_names = app.state.features_frame.set_index("feature_id")["feature_name"].to_dict()
    user_features = set(app.state.interactions[user_idx].indices.tolist())
    payload = []
    for neighbor_idx in top_indices:
        neighbor_features = set(app.state.interactions[int(neighbor_idx)].indices.tolist())
        shared_features = [
            feature_names[feature_ids[idx]].replace("_", " ").title()
            for idx in sorted(user_features & neighbor_features)
        ]
        payload.append(
            {
                "customer_id": mappings["customer_ids"][int(neighbor_idx)],
                "similarity_score": round(float(similarities[int(neighbor_idx)]), 4),
                "shared_features": shared_features,
            }
        )

    return SimilarUsersResponse(customer_id=customer_id, similar_users=payload)


@app.get("/feature-affinity/{feature_id}", response_model=FeatureAffinityResponse)
def feature_affinity(feature_id: str, top_n: int = Query(100, ge=1, le=500)) -> FeatureAffinityResponse:
    """Return users most likely to adopt a given feature."""
    if feature_id not in app.state.features:
        raise HTTPException(status_code=404, detail=f"Feature {feature_id} not found")

    mappings = app.state.mappings
    item_idx = int(mappings["feature_to_index"][feature_id])
    model = app.state.lightfm["model"]
    user_features = app.state.lightfm["user_features"]
    item_features = app.state.lightfm["item_features"]
    user_indices = np.arange(len(mappings["customer_ids"]), dtype=np.int32)
    item_indices = np.full(len(user_indices), item_idx, dtype=np.int32)
    scores = model.predict(
        user_indices,
        item_indices,
        user_features=user_features,
        item_features=item_features,
        num_threads=4,
    )
    adopted_mask = app.state.interactions[:, item_idx].toarray().ravel() > 0
    scores = np.where(adopted_mask, -np.inf, scores)
    top_indices = np.argsort(-scores)[:top_n]

    top_customers = []
    for user_idx in top_indices:
        if not np.isfinite(scores[int(user_idx)]):
            continue
        customer_id = mappings["customer_ids"][int(user_idx)]
        customer = app.state.customers[customer_id]
        top_customers.append(
            {
                "customer_id": customer_id,
                "predicted_score": round(float(scores[int(user_idx)]), 4),
                "plan_tier": customer["plan_tier"],
                "churn_probability": float(customer["churn_probability"]),
            }
        )

    feature_name = str(app.state.features[feature_id]["feature_name"]).replace("_", " ").title()
    return FeatureAffinityResponse(feature_id=feature_id, feature_name=feature_name, top_customers=top_customers)


@app.get("/health")
def health() -> dict[str, Any]:
    """Return service health and latency counters."""
    uptime_seconds = (datetime.utcnow() - app.state.start_time).total_seconds()
    total_requests = int(app.state.total_requests)
    avg_latency_ms = app.state.total_latency_ms / total_requests if total_requests else 0.0
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "uptime_seconds": round(uptime_seconds, 2),
        "total_requests": total_requests,
        "avg_latency_ms": round(avg_latency_ms, 2),
    }


def _metric_value(report: pd.DataFrame, metric: str, k: int | None = None) -> float:
    rows = report.loc[report["metric"] == metric]
    if k is not None:
        rows = rows.loc[pd.to_numeric(rows["k"], errors="coerce") == float(k)]
    if rows.empty:
        return 0.0
    return float(rows.iloc[0]["value"])


@app.get("/metrics/summary")
def metrics_summary() -> dict[str, float]:
    """Return cached headline evaluation metrics."""
    report_path = OUTPUT_DIR / "evaluation_report.csv"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Evaluation report not found. Run python src/evaluation.py first.")
    report = pd.read_csv(report_path)
    return {
        "ndcg_at_10": _metric_value(report, "ndcg", 10),
        "mrr": _metric_value(report, "mrr"),
        "hit_rate_at_10": _metric_value(report, "hit_rate", 10),
        "catalog_coverage": _metric_value(report, "catalog_coverage", 10),
        "adoption_lift_pct": _metric_value(report, "adoption_lift_pct", 10),
    }
