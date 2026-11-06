"""Очистка сырых данных и признаки для PD-модели"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "default_payment_next_month"
TARGET_ALIASES = {
    "default.payment.next.month": TARGET,
    "default payment next month": TARGET,
    "DEFAULT_PAYMENT_NEXT_MONTH": TARGET,
    "Default_Payment_Next_Month": TARGET,
    "Y": TARGET,
}

PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

RAW_FEATURES = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    *PAY_STATUS_COLS,
    *BILL_COLS,
    *PAY_AMT_COLS,
]

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE", "age_bin"]
NUMERIC_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    *PAY_STATUS_COLS,
    *BILL_COLS,
    *PAY_AMT_COLS,
    "pay_status_mean",
    "pay_status_max",
    "n_late_payments",
    "bill_amt_mean",
    "bill_amt_std",
    "pay_amt_mean",
    "utilization",
    "payment_ratio",
    "bill_trend",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_RAW_COLUMNS = RAW_FEATURES + [TARGET]

AGE_BINS = [0, 25, 35, 45, 55, 200]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55+"]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит имена колонок к канону и переводит поля в числа"""
    frame = df.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.rename(columns=TARGET_ALIASES)
    if "ID" in frame.columns:
        frame = frame.drop(columns=["ID"])
    for column in REQUIRED_RAW_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def normalize_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Схлопывает редкие коды EDUCATION и MARRIAGE в категорию «прочее»"""
    frame = df.copy()
    education = frame["EDUCATION"]
    frame["EDUCATION"] = education.where(education.isin([1, 2, 3]), 4)
    frame["MARRIAGE"] = frame["MARRIAGE"].replace({0: 3})
    return frame


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет агрегаты истории платежей, утилизацию лимита и бин возраста"""
    frame = df.copy()
    pay = frame[PAY_STATUS_COLS]
    bills = frame[BILL_COLS]
    payments = frame[PAY_AMT_COLS]

    frame["pay_status_mean"] = pay.mean(axis=1)
    frame["pay_status_max"] = pay.max(axis=1)
    frame["n_late_payments"] = (pay > 0).sum(axis=1).astype(int)
    frame["bill_amt_mean"] = bills.mean(axis=1)
    frame["bill_amt_std"] = bills.std(axis=1, ddof=0).fillna(0.0)
    frame["pay_amt_mean"] = payments.mean(axis=1)

    limit = frame["LIMIT_BAL"].replace(0, np.nan)
    utilization = frame["BILL_AMT1"] / limit
    frame["utilization"] = utilization.replace([np.inf, -np.inf], np.nan)

    previous_bill = frame["BILL_AMT2"].clip(lower=0)
    ratio = frame["PAY_AMT1"] / previous_bill.where(previous_bill > 0, np.nan)
    frame["payment_ratio"] = ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame["bill_trend"] = frame["BILL_AMT1"] - frame["BILL_AMT6"]
    frame["age_bin"] = pd.cut(
        frame["AGE"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
    ).astype(object)
    return frame


def _codes_to_str(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    def _one(value: float):
        if pd.isna(value):
            return pd.NA
        return str(int(value))

    return numeric.map(_one).astype(object)


def model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Собирает матрицу признаков в порядке, который ждет sklearn Pipeline"""
    frame = df.copy()
    for column in ("SEX", "EDUCATION", "MARRIAGE"):
        frame[column] = _codes_to_str(frame[column])
    frame["age_bin"] = frame["age_bin"].astype(object)
    missing = [column for column in MODEL_FEATURES if column not in frame.columns]
    if missing:
        raise KeyError(f"Нет признаков для модели: {missing}")
    return frame[MODEL_FEATURES]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Полный путь от сырой таблицы до матрицы модели"""
    standardized = standardize_columns(df)
    normalized = normalize_demographics(standardized)
    return model_matrix(add_features(normalized))
