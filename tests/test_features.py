import pandas as pd

from credit_default.data.features import (
    RAW_FEATURES,
    TARGET,
    add_features,
    normalize_demographics,
    standardize_columns,
)


def _raw_row(**overrides) -> pd.DataFrame:
    data = {column: 0 for column in RAW_FEATURES}
    data.update(
        {
            "LIMIT_BAL": 10000,
            "SEX": 1,
            "EDUCATION": 2,
            "MARRIAGE": 1,
            "AGE": 30,
            "BILL_AMT1": 5000,
            "BILL_AMT2": 4000,
            "PAY_AMT1": 1000,
            TARGET: 0,
        }
    )
    data.update(overrides)
    return pd.DataFrame([data])


def test_standardize_renames_kaggle_target_and_drops_id():
    frame = _raw_row()
    frame = frame.rename(columns={TARGET: "default.payment.next.month"})
    frame.insert(0, "ID", 7)
    standardized = standardize_columns(frame)
    assert TARGET in standardized.columns
    assert "ID" not in standardized.columns
    assert "default.payment.next.month" not in standardized.columns


def test_normalize_collapses_rare_codes():
    frame = _raw_row(EDUCATION=5, MARRIAGE=0)
    normalized = normalize_demographics(frame)
    assert int(normalized.loc[0, "EDUCATION"]) == 4
    assert int(normalized.loc[0, "MARRIAGE"]) == 3


def test_age_bins_and_payment_aggregates():
    frame = _raw_row(
        AGE=24,
        PAY_0=2,
        PAY_2=1,
        PAY_3=0,
        PAY_4=-1,
        PAY_5=-2,
        PAY_6=0,
        BILL_AMT2=0,
        PAY_AMT1=500,
    )
    featured = add_features(frame)
    assert featured.loc[0, "age_bin"] == "<25"
    assert int(featured.loc[0, "n_late_payments"]) == 2
    assert featured.loc[0, "pay_status_max"] == 2
    assert featured.loc[0, "utilization"] == 0.5
    assert featured.loc[0, "payment_ratio"] == 0.0

    older = add_features(_raw_row(AGE=25))
    assert older.loc[0, "age_bin"] == "25-34"
    senior = add_features(_raw_row(AGE=55))
    assert senior.loc[0, "age_bin"] == "55+"


def test_payment_ratio_uses_previous_bill():
    featured = add_features(_raw_row(BILL_AMT2=2000, PAY_AMT1=500))
    assert featured.loc[0, "payment_ratio"] == 0.25
