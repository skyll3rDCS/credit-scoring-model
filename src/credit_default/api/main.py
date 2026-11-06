"""FastAPI-сервис PD-модели"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from credit_default.api.schemas import (
    BatchPrediction,
    BatchRequest,
    ClientFeatures,
    Health,
    Prediction,
)
from credit_default.api.service import load_model, predict_frame


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model = load_model()
    except FileNotFoundError as exc:
        app.state.model = None
        app.state.model_error = str(exc)
    else:
        app.state.model_error = None
    yield


app = FastAPI(
    title="Credit card default PD",
    description="Вероятность дефолта клиента в следующем месяце.",
    version="0.1.0",
    lifespan=lifespan,
)


def _require_model():
    model = getattr(app.state, "model", None)
    if model is None:
        detail = getattr(app.state, "model_error", None) or "Модель не загружена"
        raise HTTPException(status_code=503, detail=detail)
    return model


@app.get("/health", response_model=Health)
def health() -> Health:
    loaded = getattr(app.state, "model", None) is not None
    return Health(status="ok" if loaded else "model_missing", model_loaded=loaded)


@app.post("/predict", response_model=Prediction)
def predict(payload: ClientFeatures) -> Prediction:
    model = _require_model()
    frame = pd.DataFrame([payload.model_dump()])
    result = predict_frame(model, frame)[0]
    return Prediction(**result)


@app.post("/predict-batch", response_model=BatchPrediction)
def predict_batch(payload: BatchRequest) -> BatchPrediction:
    model = _require_model()
    frame = pd.DataFrame([record.model_dump() for record in payload.records])
    predictions = [Prediction(**item) for item in predict_frame(model, frame)]
    return BatchPrediction(predictions=predictions)
