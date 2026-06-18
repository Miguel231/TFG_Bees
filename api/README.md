# Swarm Risk API + Dashboard

Production system wrapping the best TFG models:
- **Swarm:** LSTM Unidirectional from `03_swarm_night_enhanced` — AUC = 0.887, 13/14 events detected, 218 false alarms
- **Honey super (alzas):** LightGBM with G-Mean threshold — 7-day and 14-day horizons, 17 hives

---

## 1. Generate model artefacts

### Swarm model

Open `notebooks/models/03_swarm_night_enhanced.ipynb` and run through **Section 7.3 "Exportar el modelo para inferencia"**. This creates `api_export/` with 5 files. Copy them to `api/models/`:

```bash
mkdir api/models
cp api_export/* api/models/
```

### Honey super model

Open `notebooks/models/04_honey_super.ipynb` and run through the export section. This creates `alza_best_7d.pkl` and `alza_best_14d.pkl`. Copy them to `api/models/` as well.

---

## 2. Install and start

```bash
cd api
pip install -r requirements.txt
python -m playwright install chromium    # for the weekly xlsx scraper

# Terminal 1 — FastAPI backend
uvicorn main:app --reload --port 8000

# Terminal 2 — Streamlit dashboard
streamlit run dashboard.py --server.port 8501
```

Health check: `http://localhost:8000/health` → `{"status": "ok"}`  
Dashboard: `http://localhost:8501`

---

## 3. Dashboard tabs

| Tab | Content |
|---|---|
| Swarm Risk | LSTM probabilities per hive, risk level cards, last run date |
| Honey Supers | LightGBM 7d/14d recommendations per hive (only hives with registered alzas) |
| Sensor Data | Latest raw readings table |
| Risk Trends | Historical swarm probability over time |
| Models | Walk-forward CV table, feature importance per deployed model |
| History | Log of all weekly runs |

---

## 4. Weekly automation (n8n)

Import `n8n_workflow.json` into a local n8n instance. The workflow triggers every **Monday at 9am** and runs two branches in parallel:

- **Swarm branch:** `POST /weekly-run?days=60` — downloads last 60 days for 8 swarm hives via Playwright, runs the LSTM, saves results, sends alert if any hive is HIGH risk
- **Alza branch:** `POST /weekly-alza-update?days=30` — downloads last 30 days for all 17 alza hives, merges into the master raw CSV, rebuilds `daily_features_final.csv` from scratch (~7 min)

Both endpoints are also callable manually via the FastAPI docs at `http://localhost:8000/docs`.

---

## 5. Manual prediction

```bash
curl -X POST "http://localhost:8000/predict?box_id=3" \
     -F "file=@ultimos_45_dias_box3.csv"
```

The CSV must have the standard sensor columns (`Hive name`, `Time`, `Weight`, `Frequency`, `Volume`, `Temperature heart`, `Humidity heart`, `Temperature scale`, `Humidity scale`) with **at least ~35 days** of history up to the most recent reading.

Example response:

```json
{
  "box_id": 3,
  "date": "2026-06-16",
  "swarm_risk_probability": 0.34,
  "risk_level": "MEDIO",
  "horizon_days": 3,
  "message": "Riesgo de enjambrazon en los proximos 3 dias: MEDIO (34.0%)"
}
```

`risk_level`: `BAJO` (<20%), `MEDIO` (20–50%), `ALTO` (≥50%).

---

## Known limitations

- **`days_since_swarm` / `has_prior_swarm`:** set to sentinel values (999 / 0) in production — the API has no confirmed-swarm database. If one is added, `inference.py` should be updated to query it.
- **Feature pipeline in `inference.py`** replicates the notebook manually — if features change in the notebook, `inference.py` must be updated in sync.
- **No caching:** each `/predict` call recomputes all rolling features from scratch. For high-frequency polling, cache the computed `feat_day` and only append new rows.
