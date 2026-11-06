import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from credit_default.data.features import (
    RAW_FEATURES,
    TARGET,
    add_features,
    model_matrix,
    normalize_demographics,
)
from credit_default.models.pipeline import build_pipeline


def _training_frame(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {column: rng.normal(size=n) for column in RAW_FEATURES}
    data["LIMIT_BAL"] = rng.integers(10000, 80000, n)
    data["SEX"] = rng.choice([1, 2], n)
    data["EDUCATION"] = rng.choice([1, 2, 3], n)
    data["MARRIAGE"] = rng.choice([1, 2, 3], n)
    data["AGE"] = rng.integers(21, 70, n)
    for column in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"):
        data[column] = rng.integers(-2, 3, n)
    frame = pd.DataFrame(data)
    frame[TARGET] = np.array([index % 2 for index in range(n)])
    return model_matrix(add_features(normalize_demographics(frame)))


def test_pipeline_has_preprocessing_and_model_steps():
    pipeline = build_pipeline(LogisticRegression(max_iter=200))
    assert set(pipeline.named_steps) == {"preprocess", "model"}


def test_pipeline_predicts_probabilities_for_unknown_category():
    features = _training_frame()
    target = np.array([index % 2 for index in range(len(features))])
    pipeline = build_pipeline(LogisticRegression(max_iter=300))
    pipeline.fit(features, target)
    mutated = features.copy()
    mutated.loc[mutated.index[0], "SEX"] = "9"
    probabilities = pipeline.predict_proba(mutated)
    assert probabilities.shape == (len(features), 2)
    assert np.isfinite(probabilities).all()
