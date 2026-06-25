# 🤖 LLM Model Recommender

A weighted multi-criteria recommendation engine that suggests an LLM based on
how much you care about **performance**, **price**, **privacy**, and **speed**.

## How it works (the "ML" part)

There's no labeled dataset of "the correct model for situation X" to train a
classifier on, so this isn't a supervised-learning project — it's a **ranking
problem**, solved with a transparent **Weighted Sum Model (WSM)**:

1. Each model's 4 raw metrics are min-max normalized to a 0–1 scale.
2. Price is inverted (cheaper = higher normalized score).
3. Your 4 slider values are normalized to importance weights that sum to 1.
4. Each model's `match_score = Σ(weight_i × normalized_metric_i) × 100`.

This is intentionally simple and explainable — you can always tell a user
*why* a model ranked where it did. See `backend/app/recommender.py` for a
note on upgrading to TOPSIS if you want a more scale-robust method later.

## Project structure

```
llm-recommender/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app & routes
│   │   ├── models.py        # Pydantic request/response schemas
│   │   ├── recommender.py   # Weighted scoring engine
│   │   └── data_loader.py   # CSV loading + cleaning
│   ├── data/
│   │   └── llm_models.csv   # Curated dataset (see "About the data" below)
│   ├── tests/
│   │   └── test_recommender.py
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── Dockerfile
├── frontend/
│   ├── streamlit_app.py     # UI: sliders in, ranked cards + chart out
│   ├── requirements.txt
│   └── Dockerfile
├── .github/workflows/ci.yml # Lint -> test -> docker build -> gated deploy
├── render.yaml               # Render Blueprint (both services as code)
└── README.md
```

## Running it locally

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Visit `http://localhost:8000/docs` for the interactive API docs.

**Frontend** (in a second terminal):
```bash
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Visit `http://localhost:8501`. It talks to `http://localhost:8000` by default
(override with the `BACKEND_URL` environment variable).

**Run tests:**
```bash
cd backend
pip install -r requirements-dev.txt
pytest -v
ruff check app
```

## Deploying to Render

Both services run as independent Docker web services on Render, wired
together with `render.yaml` (a "Blueprint" — infrastructure as code).

1. Push this repo to GitHub.
2. On Render: **New > Blueprint**, point it at your repo. It will read
   `render.yaml` and create both services.
3. Once the backend deploys, copy its public URL
   (e.g. `https://llm-recommender-backend.onrender.com`).
4. On the **frontend** service → Environment, set `BACKEND_URL` to that URL.
5. Re-deploy the frontend.
6. On **each** service → Settings → Deploy Hook, copy the unique URL Render
   gives you.
7. In your GitHub repo → Settings → Secrets and variables → Actions, add:
   - `RENDER_BACKEND_DEPLOY_HOOK`
   - `RENDER_FRONTEND_DEPLOY_HOOK`

From now on, pushing to `main` triggers the GitHub Actions pipeline
(lint → test → Docker build validation), and **only if everything passes**
does it call those deploy hooks to actually ship to Render. `autoDeploy` is
set to `false` in `render.yaml` specifically so Render never deploys broken
code on its own — GitHub Actions is the sole gatekeeper.

## About the data

`backend/data/llm_models.csv` is a curated snapshot (June 2026) assembled
from public pricing pages and benchmark leaderboards. `performance_score`,
`privacy_score`, and `speed_score` are composite estimates for comparison
purposes — not single official numbers from any one vendor. LLM pricing and
benchmarks change fast; treat this as a demo dataset and refresh the CSV
periodically (the cleaning pipeline in `data_loader.py` will validate
whatever you put in it).

## Possible next steps

- Swap the WSM for TOPSIS for more scale-robust ranking.
- Add a natural-language-to-weights feature (LLM call that parses "I'm
  building a HIPAA app on a budget" into slider values).
- Add a `/models` filter UI so users can exclude providers entirely.
- Move the dataset to a small Postgres table instead of a CSV once it needs
  to be edited via an admin UI rather than a file edit + redeploy.
