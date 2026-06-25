"""
data_loader.py
Loads and cleans the curated LLM metrics dataset (backend/data/llm_models.csv).

NOTE ON DATA: performance_score, privacy_score, and speed_score are curated
composite estimates assembled from public pricing pages and benchmark
leaderboards (snapshot: June 2026) -- not official vendor-published single
numbers. Re-running `python -m app.data_loader` after editing the CSV is a
good sanity check whenever you refresh the numbers.
"""
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "llm_models.csv"

REQUIRED_COLUMNS = {
    "model_name", "provider", "input_price_per_mtok", "output_price_per_mtok",
    "context_window_k", "performance_score", "privacy_score", "speed_score",
    "open_weight",
}

NUMERIC_COLUMNS = [
    "input_price_per_mtok", "output_price_per_mtok", "context_window_k",
    "performance_score", "privacy_score", "speed_score",
]


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans a raw LLM dataframe: dedupes, coerces types, clips out-of-range
    scores, and derives a blended price-per-million-tokens column."""
    df = df.copy()

    # Drop exact duplicate model entries (keep first occurrence)
    df = df.drop_duplicates(subset="model_name")

    # Coerce numeric columns; anything unparseable becomes NaN and gets dropped
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=NUMERIC_COLUMNS)
    dropped = before - len(df)
    if dropped:
        print(f"[data_loader] Dropped {dropped} row(s) with unparseable numeric values.")

    # Clip scores defensively in case future data edits go out of range
    df["performance_score"] = df["performance_score"].clip(0, 100)
    df["speed_score"] = df["speed_score"].clip(0, 100)
    df["privacy_score"] = df["privacy_score"].clip(1, 10)
    df["input_price_per_mtok"] = df["input_price_per_mtok"].clip(lower=0)
    df["output_price_per_mtok"] = df["output_price_per_mtok"].clip(lower=0)

    # Blended price assumes a 1:3 input:output token ratio -- a reasonable
    # default for typical chat/agent workloads where replies run longer than
    # prompts. Adjust the 0.25/0.75 split if your workload skews differently.
    df["blended_price_per_mtok"] = (
        0.25 * df["input_price_per_mtok"] + 0.75 * df["output_price_per_mtok"]
    )

    # Normalize the open_weight column to a real boolean regardless of how
    # it was written in the source CSV (true/True/1/yes, etc.)
    df["open_weight"] = (
        df["open_weight"].astype(str).str.lower().isin(["true", "1", "yes"])
    )

    return df.reset_index(drop=True)


def load_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    """Convenience one-shot: load + clean in a single call."""
    return clean(load_raw(path))


if __name__ == "__main__":
    data = load_clean()
    print(data.head(10).to_string(index=False))
    print(f"\nLoaded and cleaned {len(data)} models.")
