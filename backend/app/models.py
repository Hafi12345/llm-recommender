from typing import List, Optional
from pydantic import BaseModel, Field


class WeightsInput(BaseModel):
    """User-supplied importance weights. They don't need to sum to anything
    in particular -- the API normalizes them internally."""
    performance: float = Field(0.25, ge=0, le=10, description="How much performance/quality matters")
    price: float = Field(0.25, ge=0, le=10, description="How much low cost matters")
    privacy: float = Field(0.25, ge=0, le=10, description="How much data privacy / openness matters")
    speed: float = Field(0.25, ge=0, le=10, description="How much latency/throughput matters")
    top_n: int = Field(5, ge=1, le=20, description="Number of recommendations to return")


class ModelRecommendation(BaseModel):
    model_name: str
    provider: str
    match_score: float
    performance_score: float
    blended_price_per_mtok: float
    privacy_score: float
    speed_score: float
    open_weight: bool
    context_window_k: float
    notes: Optional[str] = None


class RecommendationResponse(BaseModel):
    results: List[ModelRecommendation]
