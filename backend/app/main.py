from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.data_loader import load_clean
from app.recommender import Weights, recommend
from app.models import WeightsInput, RecommendationResponse, ModelRecommendation

app = FastAPI(
    title="LLM Recommender API",
    description="Recommends LLM models based on price, privacy, performance, and speed preferences.",
    version="1.0.0",
)

# Tighten allow_origins to your deployed Streamlit URL once you have it,
# rather than leaving it wide open in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_DATA = load_clean()


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(_DATA)}


@app.get("/models")
def list_models():
    """Returns the full cleaned dataset -- handy for debugging or building
    a richer frontend table later."""
    return _DATA.to_dict(orient="records")


@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendation(payload: WeightsInput):
    try:
        weights = Weights(
            performance=payload.performance,
            price=payload.price,
            privacy=payload.privacy,
            speed=payload.speed,
        )
        top = recommend(_DATA, weights, top_n=payload.top_n)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    results = [ModelRecommendation(**row) for row in top.to_dict(orient="records")]
    return RecommendationResponse(results=results)
