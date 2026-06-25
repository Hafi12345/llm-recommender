import pandas as pd
import pytest

from app.recommender import Weights, score_models, recommend
from app.data_loader import clean, REQUIRED_COLUMNS


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            "model_name": "CheapFast", "provider": "TestCo",
            "input_price_per_mtok": 0.1, "output_price_per_mtok": 0.4,
            "context_window_k": 128, "performance_score": 60,
            "privacy_score": 5, "speed_score": 95, "open_weight": "true",
            "notes": "",
        },
        {
            "model_name": "PricyPowerful", "provider": "TestCo",
            "input_price_per_mtok": 10, "output_price_per_mtok": 40,
            "context_window_k": 1000, "performance_score": 98,
            "privacy_score": 8, "speed_score": 40, "open_weight": "false",
            "notes": "",
        },
    ])


def test_clean_adds_blended_price(sample_df):
    cleaned = clean(sample_df)
    assert "blended_price_per_mtok" in cleaned.columns
    cheap_row = cleaned.loc[cleaned.model_name == "CheapFast", "blended_price_per_mtok"].iloc[0]
    assert cheap_row == pytest.approx(0.25 * 0.1 + 0.75 * 0.4)


def test_clean_normalizes_open_weight_bool(sample_df):
    cleaned = clean(sample_df)
    assert bool(cleaned.loc[cleaned.model_name == "CheapFast", "open_weight"].iloc[0]) is True
    assert bool(cleaned.loc[cleaned.model_name == "PricyPowerful", "open_weight"].iloc[0]) is False


def test_weights_normalize_to_one():
    w = Weights(performance=2, price=2, privacy=0, speed=0)
    norm = w.normalized()
    assert norm["performance"] == pytest.approx(0.5)
    assert norm["price"] == pytest.approx(0.5)
    assert norm["privacy"] == 0
    assert sum(norm.values()) == pytest.approx(1.0)


def test_all_zero_weights_raises():
    w = Weights(performance=0, price=0, privacy=0, speed=0)
    with pytest.raises(ValueError):
        w.normalized()


def test_price_weight_favors_cheap_model(sample_df):
    cleaned = clean(sample_df)
    w = Weights(performance=0, price=1, privacy=0, speed=0)
    top = recommend(cleaned, w, top_n=1)
    assert top.iloc[0]["model_name"] == "CheapFast"


def test_performance_weight_favors_powerful_model(sample_df):
    cleaned = clean(sample_df)
    w = Weights(performance=1, price=0, privacy=0, speed=0)
    top = recommend(cleaned, w, top_n=1)
    assert top.iloc[0]["model_name"] == "PricyPowerful"


def test_match_score_bounded_0_to_100(sample_df):
    cleaned = clean(sample_df)
    scored = score_models(cleaned, Weights())
    assert scored["match_score"].between(0, 100).all()


def test_required_columns_present():
    assert "model_name" in REQUIRED_COLUMNS
    assert "performance_score" in REQUIRED_COLUMNS
