import numpy as np

from credit_default.monitoring.psi import population_stability_index, psi_band


def test_identical_samples_have_near_zero_psi():
    rng = np.random.default_rng(1)
    expected = rng.normal(size=2000)
    actual = expected.copy()
    assert population_stability_index(expected, actual) < 0.01
    assert psi_band(0.01) == "stable"


def test_shifted_sample_has_material_psi():
    rng = np.random.default_rng(2)
    expected = rng.normal(loc=0, scale=1, size=3000)
    actual = rng.normal(loc=3, scale=1, size=3000)
    value = population_stability_index(expected, actual)
    assert value > 0.25
    assert psi_band(value) == "shift"
