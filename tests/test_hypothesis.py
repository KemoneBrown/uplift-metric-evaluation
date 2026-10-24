"""Unit tests for the bootstrap CI, the H2 / H2a decision logic, and the
pre-registered interquartile-range reporting."""
import numpy as np
import pandas as pd

from src import config
from src.hypothesis import bootstrap_mean_ci, h2, h2a, descriptives


def test_bootstrap_constant_has_zero_width_ci():
    mean, low, high = bootstrap_mean_ci([0.5] * 50, 1000, 0.95, 42)
    assert mean == 0.5 and low == 0.5 and high == 0.5


def test_bootstrap_determinism():
    v = list(np.linspace(0, 1, 50))
    assert bootstrap_mean_ci(v, 2000, 0.95, 7) == bootstrap_mean_ci(v, 2000, 0.95, 7)


def test_h2_not_supported_when_tau_high():
    r = h2(pd.DataFrame({"kendall_tau": [0.99] * 50}))
    assert "not supported" in r["verdict"]


def test_h2_supported_when_tau_low():
    r = h2(pd.DataFrame({"kendall_tau": [0.50] * 50}))
    assert r["verdict"].startswith("supported")


def test_h2a_supported_when_qini_clearly_lower():
    within = pd.DataFrame({"qini_stability": [0.5] * 20, "auuc_stability": [1.0] * 20})
    r = h2a(within)
    assert r["verdict"].startswith("supported")


def test_h2a_not_supported_when_identical():
    within = pd.DataFrame({"qini_stability": [1.0] * 20, "auuc_stability": [1.0] * 20})
    assert "not supported" in h2a(within)["verdict"]


def test_h2a_reports_median_and_iqr():
    within = pd.DataFrame({"qini_stability": [0.5, 1.0, 0.5, 1.0],
                           "auuc_stability": [1.0, 1.0, 0.5, 0.5]})
    r = h2a(within)
    assert "median_diff" in r and "iqr_diff" in r


def test_descriptives_includes_iqr():
    rows = [{"subsample": i, "model": m, "qini": j + i * 0.1, "auuc": j - i * 0.1}
            for i in range(10) for j, m in enumerate(config.MODELS)]
    out = descriptives(pd.DataFrame(rows))
    assert {"qini_iqr", "auuc_iqr"}.issubset(out.columns)