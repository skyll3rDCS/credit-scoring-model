import numpy as np

from credit_default.models.evaluate import classification_metrics


def test_perfect_ranking_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = classification_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["roc_auc"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_threshold_changes_precision_and_recall():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.4, 0.6, 0.7, 0.2])
    loose = classification_metrics(y_true, y_prob, threshold=0.3)
    strict = classification_metrics(y_true, y_prob, threshold=0.65)
    assert loose["recall"] == 1.0
    assert loose["precision"] == 2 / 3
    assert strict["precision"] == 1.0
    assert strict["recall"] == 0.5
    assert strict["f1"] == 2 / 3
