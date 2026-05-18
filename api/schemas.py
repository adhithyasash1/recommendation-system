"""Pydantic response schemas for the recommendation API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RecommendationItem(BaseModel):
    """One feature recommendation with ranking score and explanation."""

    rank: int
    feature_id: str
    feature_name: str
    feature_category: str
    complexity_tier: str
    relevance_score: float = Field(ge=0, le=1)
    churn_reduction_score: float
    reason: str
    cta: str


class RecommendationResponse(BaseModel):
    """Recommendation API response for one customer."""

    model_config = ConfigDict(protected_namespaces=())

    customer_id: str
    plan_tier: str
    churn_probability: float
    churn_risk_tier: str
    recommendations: list[RecommendationItem]
    cold_start_used: bool
    candidate_pool_size: int
    model_version: str = "1.0.0"
    generated_at: str


class SimilarUsersResponse(BaseModel):
    """Nearest ALS-neighbor response for one customer."""

    customer_id: str
    similar_users: list[dict]


class FeatureAffinityResponse(BaseModel):
    """Top customers likely to adopt a feature."""

    feature_id: str
    feature_name: str
    top_customers: list[dict]
