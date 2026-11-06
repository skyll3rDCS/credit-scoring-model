"""Population Stability Index для числовых распределений"""

from __future__ import annotations

import numpy as np


def population_stability_index(
    expected,
    actual,
    bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """PSI между эталонной и новой выборкой.

    Бинны строятся по квантилям эталона. Значения около 0 - распределения
    совпадают; выше 0.25 обычно считают существенным сдвигом.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if expected.size == 0 or actual.size == 0:
        raise ValueError("Для PSI нужны непустые конечные выборки")

    quantiles = np.linspace(0, 1, bins + 1)
    breaks = np.unique(np.quantile(expected, quantiles))
    if breaks.size < 3:
        return 0.0
    breaks[0] = min(breaks[0], actual.min()) - 1e-9
    breaks[-1] = max(breaks[-1], actual.max()) + 1e-9

    expected_counts = np.histogram(expected, bins=breaks)[0].astype(float)
    actual_counts = np.histogram(actual, bins=breaks)[0].astype(float)
    expected_perc = expected_counts / expected_counts.sum()
    actual_perc = actual_counts / actual_counts.sum()
    expected_perc = np.clip(expected_perc, epsilon, None)
    actual_perc = np.clip(actual_perc, epsilon, None)
    expected_perc = expected_perc / expected_perc.sum()
    actual_perc = actual_perc / actual_perc.sum()
    psi = (actual_perc - expected_perc) * np.log(actual_perc / expected_perc)
    return float(np.sum(psi))


def psi_band(value: float, warn: float = 0.1, alert: float = 0.25) -> str:
    if value < warn:
        return "stable"
    if value < alert:
        return "moderate"
    return "shift"
