"""Загрузка модели и расчет предсказания по сырым признакам клиента"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd

from credit_default.config import load_params, project_root
from credit_default.data.features import RAW_FEATURES, prepare_features


def model_path() -> Path:
    override = os.environ.get("MODEL_PATH")
    if override:
        return Path(override)
    return project_root() / "models" / "model.joblib"


def decision_threshold() -> float:
    try:
        return float(load_params()["train"]["threshold"])
    except (OSError, KeyError, TypeError, ValueError):
        return 0.5


def load_model(path: Path | None = None):
    path = path or model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {path}. Сначала выполните make train."
        )
    return joblib.load(path)


def predict_frame(
    model, frame: pd.DataFrame, threshold: float | None = None
) -> list[dict]:
    """Принимает сырые колонки клиента и возвращает класс и вероятность"""
    missing = [column for column in RAW_FEATURES if column not in frame.columns]
    if missing:
        raise KeyError(f"Во входных данных нет колонок: {missing}")
    threshold = decision_threshold() if threshold is None else threshold
    features = prepare_features(frame)
    probabilities = model.predict_proba(features)[:, 1]
    return [
        {"prediction": int(probability >= threshold), "probability": float(probability)}
        for probability in probabilities
    ]
