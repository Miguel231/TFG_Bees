"""
FastAPI backend for swarm-risk prediction.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

n8n (or anything else) calls POST /predict with the recent raw sensor CSV
for one hive; the API replicates the notebook's feature pipeline and returns
the LSTM's swarm-risk probability for the next 3 days.

See README.md for the model artifacts you need to drop into api/models/
before this will start successfully.
"""
import asyncio
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import SwarmPredictor
from data_fetcher import download_excel, xlsx_to_df

RESULTS_FILE  = Path(__file__).parent / "latest_results.json"
HISTORY_FILE  = Path(__file__).parent / "run_history.json"
MAX_HISTORY   = 52  # keep ~1 year of weekly runs

_executor = ThreadPoolExecutor(max_workers=4)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("swarm-api")

app = FastAPI(
    title="Swarm Risk Prediction API",
    description="LSTM-based early swarming detection for UAB apiary hives (TFG Miguel Arpa).",
    version="1.0.0",
)

# Allow a local dashboard (Streamlit, etc.) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: Optional[SwarmPredictor] = None


@app.on_event("startup")
def _load_model():
    global predictor
    try:
        predictor = SwarmPredictor()
        logger.info("Modelo cargado correctamente desde %s", predictor.model_dir)
    except FileNotFoundError as e:
        # Don't crash the whole app — let /health report the problem clearly
        # instead of the API failing to start with a confusing traceback.
        logger.error("No se pudo cargar el modelo: %s", e)
        predictor = None


class PredictRequest(BaseModel):
    box_id: int


@app.get("/health")
def health():
    return {
        "status": "ok" if predictor is not None else "model_not_loaded",
        "model_dir": predictor.model_dir if predictor else None,
    }


@app.post("/predict")
async def predict(box_id: int, file: UploadFile = File(...)):
    """
    box_id: hive identifier (matches 'Hive name' in the raw CSV).
    file: CSV with the unified-dataset columns for that hive, covering at
          least ~35 days up to the most recent reading (more history -> more
          reliable rolling features).
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no cargado. Revisa api/models/ (ver README.md) y reinicia la API.",
        )

    try:
        raw_bytes = await file.read()
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV invalido: {e}")

    if "Hive name" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="El CSV debe tener las columnas del dataset unificado (incluye 'Hive name').",
        )

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, predictor.predict, df, box_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Error en inferencia")
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")

    return result


@app.post("/weekly-run")
async def weekly_run(days: int = 60):
    """
    Full automated pipeline — called by n8n weekly scheduler (or manually).

    1. Opens beehivemonitoring.com via Playwright and downloads the xlsx
       for all 8 swarm hives (last `days` days).
    2. Calls /predict internally for each hive.
    3. Returns a JSON summary with all predictions and any ALTO-risk alerts.

    No file upload needed — the scraper handles auth and download automatically.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no cargado. Revisa api/models/ y reinicia la API.",
        )

    try:
        logger.info("weekly-run: downloading xlsx (last %d days)...", days)
        xlsx_bytes = await download_excel(days=days)
    except Exception as e:
        logger.exception("weekly-run: download failed")
        raise HTTPException(status_code=502, detail=f"Error descargando datos: {e}")

    try:
        df = await asyncio.get_event_loop().run_in_executor(_executor, xlsx_to_df, xlsx_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parseando xlsx: {e}")

    hives_found = sorted(df["Hive name"].unique().tolist())
    logger.info("weekly-run: %d rows, hives=%s", len(df), hives_found)

    results = []
    loop = asyncio.get_event_loop()
    for box_id in hives_found:
        try:
            res = await loop.run_in_executor(_executor, predictor.predict, df, box_id)
            results.append(res)
        except ValueError as e:
            results.append({"box_id": box_id, "risk_level": "SIN_DATOS", "error": str(e)})
        except Exception as e:
            results.append({"box_id": box_id, "risk_level": "ERROR", "error": str(e)})

    output = {
        "status": "ok",
        "run_date": datetime.today().strftime("%Y-%m-%d"),
        "run_timestamp": datetime.today().isoformat(timespec="seconds"),
        "hives": hives_found,
        "predictions": results,
        "alerts": [r for r in results if r.get("risk_level") == "ALTO"],
        "alert_count": sum(1 for r in results if r.get("risk_level") == "ALTO"),
    }
    try:
        RESULTS_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    except Exception as exc:
        logger.warning("Could not persist results: %s", exc)

    # Append compact entry to history (newest first, capped at MAX_HISTORY)
    try:
        history = json.loads(HISTORY_FILE.read_text()) if HISTORY_FILE.exists() else []
        history.insert(0, {
            "run_date":      output["run_date"],
            "run_timestamp": output["run_timestamp"],
            "alert_count":   output["alert_count"],
            "alerts":        [a.get("box_id") for a in output["alerts"]],
            "predictions": [
                {
                    "box_id":                  p.get("box_id"),
                    "risk_level":              p.get("risk_level"),
                    "swarm_risk_probability":  p.get("swarm_risk_probability"),
                    "date":                    p.get("date"),
                }
                for p in output["predictions"]
            ],
        })
        HISTORY_FILE.write_text(
            json.dumps(history[:MAX_HISTORY], ensure_ascii=False, indent=2, default=str)
        )
    except Exception as exc:
        logger.warning("Could not update history: %s", exc)

    logger.info("weekly-run done: %d hives, %d alerts", len(hives_found), output["alert_count"])
    return output


@app.get("/history")
def get_history():
    """Returns all saved weekly-run entries (newest first) for the history tab."""
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo historial: {e}")


@app.get("/results")
def get_results():
    """Returns the latest saved weekly-run results for the dashboard."""
    if not RESULTS_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Sin datos todavía. El análisis automático se ejecuta cada lunes a las 9h via n8n.",
        )
    try:
        return json.loads(RESULTS_FILE.read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo resultados: {e}")


@app.post("/weekly-alza-update")
async def weekly_alza_update(days: int = 30):
    """
    Weekly honey-super feature update — called by n8n alongside /weekly-run.

    Downloads last `days` days from beehivemonitoring.com for all 17 alza hives,
    merges with the master raw CSV, rebuilds daily_features_final.csv from scratch,
    and returns a JSON status summary.

    Takes ~7–10 minutes.  Set n8n HTTP-Request timeout to 720000 ms (12 min).
    """
    script = Path(__file__).parent / "update_alza_features.py"
    cmd    = ["python", str(script), f"--days={days}"]
    logger.info("weekly-alza-update: starting (days=%d)", days)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=900)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504,
                            detail="weekly-alza-update timed out after 15 min")
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"Failed to launch update script: {exc}")

    stderr_str = stderr.decode("utf-8", errors="replace")
    stdout_str = stdout.decode("utf-8", errors="replace").strip()
    for line in stderr_str.splitlines():
        logger.info("[alza] %s", line)

    if proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"update_alza_features.py exited {proc.returncode}: {stderr_str[-500:]}",
        )

    try:
        result = json.loads(stdout_str.split("\n")[-1])
    except Exception:
        result = {"status": "ok", "raw_output": stdout_str[-500:]}

    logger.info("weekly-alza-update done: %s", result.get("status"))
    return result


@app.get("/")
def root():
    return {
        "service": "Swarm Risk Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict?box_id=<int>  (multipart form: file=<csv>)",
        "weekly_run": "POST /weekly-run?days=60  (swarm automated pipeline, called by n8n)",
        "weekly_alza_update": "POST /weekly-alza-update?days=30  (alza feature CSV update, called by n8n)",
    }
