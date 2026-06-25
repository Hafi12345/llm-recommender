import os
import requests
import streamlit as st
import pandas as pd

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="LLM Recommender", page_icon="🤖", layout="centered")

st.title("🤖 LLM Model Recommender")
st.write(
    "Tell us what matters to you, and we'll rank LLMs from our curated dataset "
    "based on your priorities."
)

st.subheader("How much do you care about each factor?")
st.caption("Drag each slider — they don't need to add up to anything, we normalize them for you.")

col_a, col_b = st.columns(2)
with col_a:
    performance = st.slider("🧠 Performance (benchmark quality)", 0, 10, 5)
    price = st.slider("💰 Price (cost per token)", 0, 10, 5)
with col_b:
    privacy = st.slider("🔒 Privacy (data handling / open-weight)", 0, 10, 5)
    speed = st.slider("⚡ Speed (latency / throughput)", 0, 10, 5)

top_n = st.number_input("How many recommendations?", min_value=1, max_value=16, value=5)

if st.button("Get Recommendations", type="primary", use_container_width=True):
    payload = {
        "performance": performance,
        "price": price,
        "privacy": privacy,
        "speed": speed,
        "top_n": int(top_n),
    }
    try:
        resp = requests.post(f"{BACKEND_URL}/recommend", json=payload, timeout=15)
        resp.raise_for_status()
        results = resp.json()["results"]
    except requests.exceptions.RequestException as e:
        st.error(f"Couldn't reach the recommendation API at {BACKEND_URL}: {e}")
    else:
        if not results:
            st.warning("No models matched. Try different weights.")
        else:
            df = pd.DataFrame(results)
            df["match_score"] = df["match_score"].round(1)
            df["blended_price_per_mtok"] = df["blended_price_per_mtok"].round(2)

            st.subheader("Your Top Matches")
            for i, row in df.iterrows():
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        badge = "🟢 Open-weight" if row["open_weight"] else "🔒 Closed/API-only"
                        st.markdown(f"**#{i + 1}. {row['model_name']}** — {row['provider']} · {badge}")
                        if row.get("notes"):
                            st.caption(row["notes"])
                    with col2:
                        st.metric("Match", f"{row['match_score']}%")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Performance", f"{row['performance_score']:.0f}/100")
                    c2.metric("Price /Mtok", f"${row['blended_price_per_mtok']}")
                    c3.metric("Privacy", f"{row['privacy_score']:.0f}/10")
                    c4.metric("Speed", f"{row['speed_score']:.0f}/100")

            st.divider()
            st.bar_chart(df.set_index("model_name")["match_score"])

st.divider()
with st.expander("ℹ️ About this dataset"):
    st.write(
        "Pricing, performance, privacy, and speed figures are a curated snapshot "
        "compiled from public pricing pages and benchmark leaderboards (June 2026). "
        "They are estimates for comparison purposes, not official vendor figures — "
        "always confirm exact pricing on the provider's site before budgeting."
    )
