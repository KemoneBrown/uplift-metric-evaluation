"""Unit tests for the rank-correlation analysis, on constructed rankings with
known Kendall tau values."""
import pandas as pd

from src import config
from src.correlations import cross_metric_correlations, within_metric_stability

_MODELS = list(config.MODELS)


def _sub(i, qini, auuc):
    return [{"subsample": i, "model": m, "qini": q, "auuc": a}
            for m, q, a in zip(_MODELS, qini, auuc)]


def _reference(qini, auuc):
    return pd.DataFrame([{"model": m, "qini": q, "auuc": a}
                         for m, q, a in zip(_MODELS, qini, auuc)])


def test_identical_ordering_gives_tau_one():
    df = pd.DataFrame(_sub(0, [4, 3, 2, 1], [4, 3, 2, 1]))
    cross = cross_metric_correlations(df)
    assert cross.loc[0, "kendall_tau"] == 1.0
    assert cross.loc[0, "spearman_rho"] == 1.0


def test_reversed_ordering_gives_tau_minus_one():
    df = pd.DataFrame(_sub(0, [4, 3, 2, 1], [1, 2, 3, 4]))
    cross = cross_metric_correlations(df)
    assert cross.loc[0, "kendall_tau"] == -1.0


def test_within_stability_matches_reference():
    df = pd.DataFrame(_sub(0, [4, 3, 2, 1], [4, 3, 2, 1]) + _sub(1, [4, 3, 2, 1], [1, 2, 3, 4]))
    ref = _reference([4, 3, 2, 1], [4, 3, 2, 1])
    within = within_metric_stability(df, ref)
    assert within.loc[0, "qini_stability"] == 1.0     # matches reference exactly
    assert within.loc[1, "auuc_stability"] == -1.0    # reversed from reference


def test_output_shape():
    df = pd.DataFrame(_sub(0, [4, 3, 2, 1], [4, 3, 2, 1]) + _sub(1, [3, 4, 1, 2], [3, 4, 1, 2]))
    cross = cross_metric_correlations(df)
    assert set(cross.columns) == {"subsample", "kendall_tau", "spearman_rho"}
    assert len(cross) == 2