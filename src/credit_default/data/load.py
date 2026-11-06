"""Загрузка датасета Default of Credit Card Clients"""

from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from credit_default.config import load_params, project_root

UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)


def raw_csv_path() -> Path:
    params = load_params()
    return project_root() / "data" / "raw" / params["data"]["raw_filename"]


def _looks_like_dataset(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False
    header = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    return "LIMIT_BAL" in header and "PAY_0" in header


def _fallback_copies() -> list[Path]:
    root = project_root()
    return [root.parent / "UCI_Credit_Card.csv"]


def _frame_from_xls(payload: bytes) -> pd.DataFrame:
    # В файле UCI две строки заголовка: коды X1/Y и человекочитаемые имена.
    frame = pd.read_excel(io.BytesIO(payload), header=1, engine="xlrd")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _download_uci_zip(destination: Path) -> None:
    request = urllib.request.Request(
        UCI_ZIP_URL,
        headers={"User-Agent": "credit-default-pd/0.1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = next(
            item for item in archive.namelist() if item.lower().endswith(".xls")
        )
        frame = _frame_from_xls(archive.read(name))
    frame.to_csv(destination, index=False)


def ensure_raw_data(destination: Path | None = None) -> Path:
    """Кладет CSV в data/raw: локальная копия или официальный архив UCI"""
    destination = destination or raw_csv_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _looks_like_dataset(destination):
        return destination

    for candidate in _fallback_copies():
        if _looks_like_dataset(candidate):
            shutil.copyfile(candidate, destination)
            return destination

    try:
        _download_uci_zip(destination)
    except Exception as exc:
        if destination.exists():
            destination.unlink()
        raise FileNotFoundError(
            "Не удалось получить UCI Credit Card Default. "
            f"Положите CSV в {destination}. Причина: {exc}"
        ) from exc

    if not _looks_like_dataset(destination):
        destination.unlink(missing_ok=True)
        raise FileNotFoundError(
            f"Архив UCI скачан, но таблица не похожа на датасет: {destination}"
        )
    return destination


def main() -> None:
    path = ensure_raw_data()
    print(f"Датасет готов: {path}")


if __name__ == "__main__":
    main()
