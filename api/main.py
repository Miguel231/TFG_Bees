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
import io
import logging

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import SwarmPredictor

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

predictor: SwarmPredictor | None = None


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
        result = predictor.predict(df, box_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Error en inferencia")
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")

    return result


@app.get("/")
def root():
    return {
        "service": "Swarm Risk Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict?box_id=<int>  (multipart form: file=<csv>)",
    }
