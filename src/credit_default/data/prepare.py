"""Стадия prepare: валидация, признаки, разбиение на train и test"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_default.config import load_params, project_root
from credit_default.data.features import (
    TARGET,
    add_features,
    normalize_demographics,
    standardize_columns,
)
from credit_default.data.load import ensure_raw_data
from credit_default.validation.suite import DataValidationError, validate_raw_dataframe


def prepare_dataset(
    raw: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    standardized = standardize_columns(raw)
    validation = validate_raw_dataframe(standardized, already_standardized=True)
    if not validation.success:
        raise DataValidationError(validation)
    featured = add_features(normalize_demographics(standardized))
    train_df, test_df = train_test_split(
        featured,
        test_size=test_size,
        random_state=random_state,
        stratify=featured[TARGET],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def run_prepare() -> tuple[Path, Path]:
    params = load_params()
    raw_path = ensure_raw_data()
    raw = pd.read_csv(raw_path)
    train_df, test_df = prepare_dataset(
        raw,
        test_size=float(params["prepare"]["test_size"]),
        random_state=int(params["prepare"]["random_state"]),
    )
    processed = project_root() / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    train_path = processed / "train.csv"
    test_path = processed / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    train_rate = float(train_df[TARGET].mean())
    test_rate = float(test_df[TARGET].mean())
    print(f"train: {train_df.shape} -> {train_path}")
    print(f"test: {test_df.shape} -> {test_path}")
    print(f"default rate train={train_rate:.4f} test={test_rate:.4f}")
    return train_path, test_path


def main() -> None:
    run_prepare()


if __name__ == "__main__":
    main()
