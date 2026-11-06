import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from credit_default.api.service import predict_frame
from credit_default.data.features import (
    RAW_FEATURES,
    add_features,
    model_matrix,
    normalize_demographics,
)
from credit_default.models.pipeline import build_pipeline


def test_predict_frame_returns_class_and_probability():
    rng = np.random.default_rng(3)
    n = 30
    raw = pd.DataFrame({column: rng.integers(0, 5, n) for column in RAW_FEATURES})
    raw["LIMIT_BAL"] = rng.integers(10000, 50000, n)
    raw["AGE"] = rng.integers(22, 60, n)
    raw["SEX"] = rng.choice([1, 2], n)
    raw["EDUCATION"] = rng.choice([1, 2, 3], n)
    raw["MARRIAGE"] = rng.choice([1, 2], n)
    target = np.array([index % 2 for index in range(n)])
    features = model_matrix(add_features(normalize_demographics(raw)))
    model = build_pipeline(LogisticRegression(max_iter=300))
    model.fit(features, target)

    results = predict_frame(model, raw.iloc[:2], threshold=0.5)
    assert len(results) == 2
    assert set(results[0]) == {"prediction", "probability"}
    assert results[0]["prediction"] in (0, 1)
    assert 0.0 <= results[0]["probability"] <= 1.0
