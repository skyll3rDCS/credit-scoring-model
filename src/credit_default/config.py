"""Пути проекта и чтение params.yaml"""

from __future__ import annotations

from pathlib import Path

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_params() -> dict:
    path = project_root() / "params.yaml"
    with path.open(encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    if not isinstance(params, dict):
        raise ValueError(f"Некорректный файл параметров: {path}")
    return params
