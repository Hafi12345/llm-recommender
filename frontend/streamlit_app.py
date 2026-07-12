import os
import requests
import streamlit as st
import pandas as pd

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="LLM Recommender",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Design tokens & global CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500&family=JetBrains+Mono:wght@500;600&display=swap');

/* ── Tokens ── */
:root {
  --bg:           #070B14;
  --surface:      #0D1424;
  --surface-hi:   #111B2E;
  --border:       #1C2B45;
  --accent:       #06B6D4;
  --accent-dim:   rgba(6,182,212,0.10);
  --accent-glow:  rgba(6,182,212,0.28);
  --t1:           #E8EDF5;
  --t2:           #7B8FAF;
  --t3:           #3A4F6A;
  --green:        #34D399;
  --amber:        #FBBF24;
  --r:            10px;
  --font-d: 'Space Grotesk', sans-serif;
  --font-b: 'Inter', sans-serif;
  --font-m: 'JetBrains Mono', monospace;
}

/* ── Base ── */
.stApp, [data-testid="stAppViewContainer"] {
  background-color: var(--bg) !important;
  font-family: var(--font-b) !important;
}
[data-testid="block-container"] {
  padding-top: 1.5rem !important;
  padding-bottom: 5rem !important;
  max-width: 760px !important;
}
#MainMenu, footer, header { visibility: hidden; }
hr { border-color: var(--border) !important; opacity: 1 !important; }

/* ── Hero ── */
.hero {
  text-align: center;
  padding: 2.25rem 1rem 1.75rem;
}
.hero-eyebrow {
  font-family: var(--font-m);
  font-size: 0.68rem;
  color: var(--accent);
  letter-spacing: 0.2em;
  text-transform: uppercase;
  margin-bottom: 0.85rem;
}
.hero h1 {
  font-family: var(--font-d);
  font-size: 2.3rem;
  font-weight: 700;
  color: var(--t1);
  line-height: 1.15;
  letter-spacing: -0.025em;
  margin: 0 0 0.8rem;
}
.hero h1 em { font-style: normal; color: var(--accent); }
.hero p {
  font-size: 0.95rem;
  color: var(--t2);
  max-width: 430px;
  margin: 0 auto;
  line-height: 1.65;
}

/* ── Section labels ── */
.section-label {
  font-family: var(--font-m);
  font-size: 0.65rem;
  color: var(--t3);
  letter-spacing: 0.16em;
  text-transform: uppercase;
  margin: 1.75rem 0 0.9rem;
  padding-bottom: 0.6rem;
  border-bottom: 1px solid var(--border);
}

/* ── Slider labels ── */
div[data-testid="stSlider"] > label,
.stSlider > label {
  font-family: var(--font-b) !important;
  font-size: 0.875rem !important;
  font-weight: 500 !important;
  color: var(--t1) !important;
}

/* ── Number input ── */
div[data-testid="stNumberInput"] label {
  font-family: var(--font-b) !important;
  font-size: 0.875rem !important;
  color: var(--t2) !important;
}
div[data-testid="stNumberInput"] input {
  background: var(--surface-hi) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  color: var(--t1) !important;
  font-family: var(--font-m) !important;
  font-size: 0.9rem !important;
}

/* ── Primary CTA button ── */
.stButton > button[kind="primary"] {
  background: var(--accent) !important;
  color: #070B14 !important;
  font-family: var(--font-d) !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  border: none !important;
  border-radius: var(--r) !important;
  height: 3rem !important;
  letter-spacing: 0.015em !important;
  transition: box-shadow 0.2s ease, opacity 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
  opacity: 0.88 !important;
  box-shadow: 0 0 28px var(--accent-glow) !important;
}

/* ── Result cards ── */
.model-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 1.35rem 1.5rem 1.2rem;
  margin-bottom: 0.8rem;
  transition: border-color 0.2s ease;
}
.model-card:hover     { border-color: rgba(6,182,212,0.3); }
.model-card.top-pick  { border-color: rgba(6,182,212,0.45);
                         background: linear-gradient(135deg,#0D1424 0%,#0E1C30 100%); }

/* card header row */
.card-header {
  display: flex;
  align-items: flex-start;
  gap: 0.9rem;
  margin-bottom: 0.95rem;
}
.rank-num {
  font-family: var(--font-m);
  font-size: 1rem;
  font-weight: 600;
  color: var(--t3);
  min-width: 1.8rem;
  padding-top: 3px;
}
.top-pick .rank-num { color: var(--accent); }
.card-info  { flex: 1; min-width: 0; }
.card-name  {
  font-family: var(--font-d);
  font-size: 1rem;
  font-weight: 600;
  color: var(--t1);
  margin-bottom: 0.25rem;
}
.card-meta {
  font-size: 0.78rem;
  color: var(--t2);
  display: flex;
  align-items: center;
  gap: 0.45rem;
  flex-wrap: wrap;
}
.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.2rem;
  font-family: var(--font-m);
  font-size: 0.6rem;
  font-weight: 600;
  letter-spacing: 0.07em;
  padding: 0.18rem 0.45rem;
  border-radius: 4px;
  text-transform: uppercase;
}
.badge-open   { background: rgba(52,211,153,0.10); color: var(--green); }
.badge-closed { background: rgba(251,191,36,0.10);  color: var(--amber); }

/* match score (top right) */
.score-block        { text-align: right; }
.score-pct          {
  font-family: var(--font-m);
  font-size: 1.35rem;
  font-weight: 600;
  color: var(--accent);
  line-height: 1;
}
.score-sublabel {
  font-size: 0.62rem;
  color: var(--t3);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-top: 0.2rem;
}

/* ── SIGNATURE ELEMENT: glowing match-score bar ── */
.bar-track {
  height: 3px;
  background: var(--border);
  border-radius: 99px;
  overflow: hidden;
  margin-bottom: 1rem;
}
.bar-fill {
  height: 100%;
  border-radius: 99px;
  background: linear-gradient(90deg, #0891B2 0%, #06B6D4 100%);
  box-shadow: 0 0 10px var(--accent-glow);
}

/* metrics grid */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(4,1fr);
  gap: 0.45rem;
}
.metric-cell {
  background: var(--surface-hi);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.55rem 0.7rem;
}
.metric-val {
  font-family: var(--font-m);
  font-size: 0.88rem;
  font-weight: 600;
  color: var(--t1);
  margin-bottom: 0.1rem;
}
.metric-key {
  font-size: 0.6rem;
  color: var(--t3);
  text-transform: uppercase;
  letter-spacing: 0.09em;
}

/* card note */
.card-note {
  font-size: 0.76rem;
  color: var(--t2);
  margin-top: 0.7rem;
  padding-top: 0.7rem;
  border-top: 1px solid var(--border);
  line-height: 1.55;
}

/* results header */
.results-heading {
  font-family: var(--font-d);
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--t1);
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin: 1.75rem 0 0.9rem;
}
.results-pill {
  font-family: var(--font-m);
  font-size: 0.68rem;
  color: var(--accent);
  background: var(--accent-dim);
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
}

/* comparison chart */
.chart-row {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  margin-bottom: 0.5rem;
}
.chart-label {
  font-size: 0.73rem;
  color: var(--t2);
  width: 150px;
  flex-shrink: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-family: var(--font-b);
}
.chart-track {
  flex: 1;
  height: 5px;
  background: var(--border);
  border-radius: 99px;
  overflow: hidden;
}
.chart-fill {
  height: 100%;
  border-radius: 99px;
  background: linear-gradient(90deg, #0891B2, var(--accent));
}
.chart-val {
  font-family: var(--font-m);
  font-size: 0.68rem;
  color: var(--t2);
  width: 36px;
  text-align: right;
}

/* expander */
div[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  background: var(--surface) !important;
}
div[data-testid="stExpander"] summary {
  color: var(--t2) !important;
  font-size: 0.82rem !important;
}
div[data-testid="stExpander"] p {
  color: var(--t2) !important;
  font-size: 0.82rem !important;
  line-height: 1.6 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">⚡ AI Model Selector</div>
  <h1>Find the right <em>LLM</em><br>for your use case</h1>
  <p>Set your priorities — we score every major model against them
     and rank the best fit for you.</p>
</div>
""", unsafe_allow_html=True)


# ── Controls ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">01 — Set your priorities</div>', unsafe_allow_html=True)
st.caption("Slide each dial from 0 (don't care) to 10 (critical). We normalize the weights automatically.")

col_a, col_b = st.columns(2)
with col_a:
    performance = st.slider("🧠 Performance", 0, 10, 5,
                             help="Benchmark quality — reasoning, coding, instruction-following")
    price = st.slider("💰 Price", 0, 10, 5,
                       help="Cost per million tokens (blended input/output)")
with col_b:
    privacy = st.slider("🔒 Privacy", 0, 10, 5,
                         help="Data-handling posture and open-weight availability")
    speed = st.slider("⚡ Speed", 0, 10, 5,
                       help="Latency and throughput")

st.markdown('<div class="section-label">02 — How many results?</div>', unsafe_allow_html=True)
top_n = st.number_input("Models to show", min_value=1, max_value=16, value=5, label_visibility="collapsed")

st.markdown("<div style='margin-top:1.25rem'></div>", unsafe_allow_html=True)

run = st.button("Get Recommendations →", type="primary", use_container_width=True)


# ── Results ────────────────────────────────────────────────────────────────────
if run:
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
        st.error(f"Could not reach the API at {BACKEND_URL}. Make sure the backend is running.\n\n{e}")
    else:
        if not results:
            st.warning("No models matched. Try adjusting your weights.")
        else:
            df = pd.DataFrame(results)
            df["match_score"]           = df["match_score"].round(1)
            df["blended_price_per_mtok"] = df["blended_price_per_mtok"].round(3)

            # results header
            st.markdown(
                f'<div class="results-heading">'
                f'Top matches <span class="results-pill">{len(df)} models</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Model cards ──
            for i, row in df.iterrows():
                is_top    = i == 0
                card_cls  = "model-card top-pick" if is_top else "model-card"
                badge_cls = "badge badge-open" if row["open_weight"] else "badge badge-closed"
                badge_txt = "Open-weight" if row["open_weight"] else "API-only"
                bar_w     = f"{row['match_score']:.1f}%"
                price_str = f"${row['blended_price_per_mtok']}/Mtok"
                note_html = (
                    f'<div class="card-note">{row["notes"]}</div>'
                    if row.get("notes") else ""
                )

                st.markdown(f"""
                <div class="{card_cls}">
                  <div class="card-header">
                    <div class="rank-num">#{i+1}</div>
                    <div class="card-info">
                      <div class="card-name">{row['model_name']}</div>
                      <div class="card-meta">
                        {row['provider']}
                        <span class="{badge_cls}">{badge_txt}</span>
                      </div>
                    </div>
                    <div class="score-block">
                      <div class="score-pct">{row['match_score']:.1f}%</div>
                      <div class="score-sublabel">match</div>
                    </div>
                  </div>

                  <div class="bar-track">
                    <div class="bar-fill" style="width:{bar_w}"></div>
                  </div>

                  <div class="metrics-grid">
                    <div class="metric-cell">
                      <div class="metric-val">{row['performance_score']:.0f}<span style="font-size:0.6rem;color:var(--t3)">/100</span></div>
                      <div class="metric-key">Performance</div>
                    </div>
                    <div class="metric-cell">
                      <div class="metric-val">{price_str}</div>
                      <div class="metric-key">Price</div>
                    </div>
                    <div class="metric-cell">
                      <div class="metric-val">{row['privacy_score']:.0f}<span style="font-size:0.6rem;color:var(--t3)">/10</span></div>
                      <div class="metric-key">Privacy</div>
                    </div>
                    <div class="metric-cell">
                      <div class="metric-val">{row['speed_score']:.0f}<span style="font-size:0.6rem;color:var(--t3)">/100</span></div>
                      <div class="metric-key">Speed</div>
                    </div>
                  </div>
                  {note_html}
                </div>
                """, unsafe_allow_html=True)

            # ── Comparison chart ──
            st.markdown('<div class="section-label" style="margin-top:1.75rem">03 — Score comparison</div>',
                        unsafe_allow_html=True)
            max_score = df["match_score"].max()
            chart_rows = ""
            for _, row in df.iterrows():
                fill_w = f"{(row['match_score'] / max_score * 100):.1f}%"
                chart_rows += f"""
                <div class="chart-row">
                  <div class="chart-label">{row['model_name']}</div>
                  <div class="chart-track"><div class="chart-fill" style="width:{fill_w}"></div></div>
                  <div class="chart-val">{row['match_score']:.1f}</div>
                </div>"""
            st.markdown(f'<div style="margin-top:0.5rem">{chart_rows}</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:2.5rem'></div>", unsafe_allow_html=True)
with st.expander("ℹ️ About this dataset"):
    st.write(
        "Pricing, performance, privacy, and speed figures are a curated snapshot "
        "compiled from public pricing pages and benchmark leaderboards (June 2026). "
        "They are estimates for comparison purposes — always confirm exact pricing "
        "on the provider's site before budgeting."
    )
