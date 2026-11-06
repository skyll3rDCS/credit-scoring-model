from pathlib import Path

import numpy as np
import pandas as pd

from credit_default.data.features import RAW_FEATURES, TARGET
from credit_default.validation.suite import validate_raw_dataframe

FIXTURE = Path(__file__).parent / "fixtures" / "credit_sample.csv"


def _valid_frame(n: int = 5) -> pd.DataFrame:
    data = {column: [0] * n for column in RAW_FEATURES}
    data["LIMIT_BAL"] = [20000] * n
    data["SEX"] = [1, 2, 1, 2, 1][:n]
    data["EDUCATION"] = [1, 2, 3, 0, 4][:n]
    data["MARRIAGE"] = [1, 2, 3, 0, 1][:n]
    data["AGE"] = [24, 35, 45, 55, 22][:n]
    data["PAY_0"] = [0, -1, 2, 0, -2][:n]
    data[TARGET] = [0, 1, 0, 0, 1][:n]
    return pd.DataFrame(data)


def test_valid_batch_passes():
    report = validate_raw_dataframe(_valid_frame())
    assert report.success, report.summary()


def test_anomalies_fail_validation():
    frame = _valid_frame()
    frame.loc[0, "AGE"] = -5
    frame.loc[1, "LIMIT_BAL"] = np.nan
    frame.loc[2, TARGET] = 9
    report = validate_raw_dataframe(frame)
    assert not report.success
    assert len(report.failed) >= 2


def test_checked_in_sample_passes():
    frame = pd.read_csv(FIXTURE)
    report = validate_raw_dataframe(frame)
    assert report.success, report.summary()
