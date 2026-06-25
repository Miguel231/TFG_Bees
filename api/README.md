# Swarm Risk API + Dashboard

Production system wrapping the models deployed in the TFG:

- **Swarm (3-day):** LSTM Unidirectional from `01_swarm_multihorizon` — G-Mean = 0.866, 13/14 events detected, 136 false alarms
- **Swarm (14-day):** LSTM Bidirectional from `01_swarm_multihorizon` — G-Mean = 0.815, 13/14 events detected, 55 false alarms
- **Honey supers (7-day):** XGBoost bio_no_tmp — G-Mean = 0.757, 16/20 events detected
- **Honey supers (14-day):** XGBoost hive_19 — G-Mean = 0.741, 16/20 events detected

---

## 1. Model artefacts

All model files are committed to the repo under `api/models/`. No extra export step is needed.

| File | Description |
|---|---|
| `lstm_uni_01_3d.pt` | Swarm LSTM Uni weights — 3-day horizon (NB01) |
| `lstm_bi_01_14d.pt` | Swarm LSTM Bi weights — 14-day horizon (NB01) |
| `scaler_01.pkl` | StandardScaler fitted on NB01 training split |
| `median_fill_01.json` | Training-split medians for NaN imputation |
| `model_meta_01_3d.json` | Architecture params for the 3d model |
| `model_meta_01_14d.json` | Architecture params for the 14d model |
| `lgbm_alza_solo_hive_7d.pkl` | XGBoost honey super model — 7-day horizon |
| `lgbm_alza_solo_hive_14d.pkl` | XGBoost honey super model — 14-day horizon |
| `lgbm_alza_meta.json` | Alza model metadata (feature lists, thresholds) |

To retrain the alza models with updated data:

```bash
cd api
python train_alza_features.py   # grid search, saves best 7d + 14d to api/models/
```

---

## 2. Install and start

```bash
cd api
pip install -r requirements.txt
playwright install chromium    # for the weekly xlsx scraper

# Terminal 1 — FastAPI backend
uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Streamlit dashboard
streamlit run dashboard.py --server.port 8501
```

Health check: `http://localhost:8000/health` → `{"status": "ok"}`  
Dashboard: `http://localhost:8501`

---

## 3. Dashboard tabs

| Tab | Content |
|---|---|
| Swarm Risk | LSTM probabilities per hive at 3d and 14d, risk level cards, last run date |
| Honey Supers | XGBoost 7d/14d recommendations per hive |
| Sensor Data | Latest raw readings table |
| Risk Trends | Historical swarm probability over time |
| Models | Walk-forward CV table, feature importance per deployed model |
| History | Log of all weekly runs |

---

## 4. Weekly automation (n8n)

Import `n8n_workflow.json` into a local n8n instance (`n8n start`, then open `http://localhost:5678`). The workflow triggers every **Monday at 9am** and runs two branches in parallel:

- **Swarm branch:** `POST /weekly-run?days=60` — downloads last 60 days for 8 swarm hives via Playwright, runs both LSTMs (3d + 14d), saves results, sends alert if any hive is HIGH risk at 3d
- **Alza branch:** `POST /weekly-alza-update?days=30` — downloads last 30 days for all 17 alza hives, rebuilds `daily_features_prod.csv` (~7 min)

Both endpoints are also callable manually via `http://localhost:8000/docs`.

The n8n email nodes require a Gmail SMTP credential (`Settings → Credentials → Add → SMTP`, host `smtp.gmail.com`, port 587, with a Google App Password).

---

## 5. Manual prediction

```bash
curl -X POST "http://localhost:8000/predict?box_id=3" \
     -F "file=@ultimos_45_dias_box3.csv"
```

The CSV must have the standard sensor columns (`Hive name`, `Time`, `Weight`, `Frequency`, `Volume`, `Temperature heart`, `Humidity heart`, `Temperature scale`, `Humidity scale`) with at least 35 days of history.

Example response:

```json
{
  "box_id": 3,
  "date": "2026-06-20",
  "risk_level": "MEDIO",
  "swarm_risk_probability": 0.34,
  "horizon_days": 3,
  "risk_14d": {
    "probability": 0.61,
    "risk_level": "ALTO",
    "horizon_days": 14
  },
  "message": "Riesgo 3d: MEDIO (34.0%)  |  Riesgo 14d: ALTO (61.0%)"
}
```

`risk_level`: `BAJO` (<20%), `MEDIO` (20–50%), `ALTO` (≥50%).

---

## Known limitations

- **Swarm history:** `days_since_last_swarm` and `n_swarms_this_season` are set to sentinel values (999 and 0) in production — the API has no confirmed-swarm database. If one is added, `inference.py` should be updated to query it.
- **Feature pipeline:** `inference.py` replicates the NB01 feature engineering manually. If features change in the notebook, `inference.py` must be updated in sync.
- **No caching:** each `/predict` call recomputes all rolling features from scratch. For high-frequency polling, cache the computed feature table and only append new rows.
