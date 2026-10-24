"""Unit tests for the Qini and AUUC metrics.

Two cases are computed by hand so the implementation is verified against known
values, and one test cross-validates the metrics against causalml's reference
implementation by confirming identical model rankings (pre-registration plan
item: metric modules cross-validated against reference implementations).
"""
import numpy as np
import pandas as pd

from src.metrics import qini_coefficient, auuc, evaluate


def test_qini_exact():
    score = np.array([4, 3, 2, 1]); w = np.array([1, 0, 1, 0]); y = np.array([1, 0, 0, 0])
    assert abs(qini_coefficient(score, w, y) - 0.375) < 1e-9


def test_auuc_exact():
    score = np.array([4, 3, 2, 1]); w = np.array([1, 0, 1, 0]); y = np.array([1, 0, 0, 0])
    assert abs(auuc(score, w, y) - 1.375) < 1e-9


def test_evaluate_matches_individual_functions():
    rng = np.random.RandomState(0)
    score = rng.normal(size=500); w = rng.binomial(1, 0.5, 500); y = rng.binomial(1, 0.3, 500)
    res = evaluate(score, w, y)
    assert abs(res["qini"] - qini_coefficient(score, w, y)) < 1e-9
    assert abs(res["auuc"] - auuc(score, w, y)) < 1e-9


def test_better_ranking_scores_higher():
    w = np.array([1, 1, 1, 0, 0, 0]); y = np.array([1, 0, 0, 0, 0, 0])
    good = np.array([6, 5, 4, 3, 2, 1]); bad = np.array([1, 2, 3, 4, 5, 6])
    assert qini_coefficient(good, w, y) > qini_coefficient(bad, w, y)
    assert auuc(good, w, y) > auuc(bad, w, y)


def test_determinism():
    rng = np.random.RandomState(1)
    score = rng.normal(size=300); w = rng.binomial(1, 0.5, 300); y = rng.binomial(1, 0.3, 300)
    assert qini_coefficient(score, w, y) == qini_coefficient(score, w, y)
    assert auuc(score, w, y) == auuc(score, w, y)


def test_rankings_match_causalml_reference():
    """Our Qini and AUUC must rank a set of models in the same order as
    causalml's reference implementation (values differ by normalisation, so we
    compare orderings, which is what the study uses)."""
    from causalml.metrics import qini_score, auuc_score

    rng = np.random.RandomState(0)
    n = 20000
    w = rng.binomial(1, 0.5, n)
    x = rng.normal(size=n)
    p_control = 1.0 / (1.0 + np.exp(-(x - 1.0)))
    effect = 0.3 * (x > 0).astype(float)          # heterogeneous positive effect
    y = rng.binomial(1, np.clip(p_control + w * effect, 0.0, 1.0))
    models = {"aligned": x, "noise": rng.normal(size=n), "reversed": -x}

    our_qini_order = sorted(models, key=lambda m: qini_coefficient(models[m], w, y))
    our_auuc_order = sorted(models, key=lambda m: auuc(models[m], w, y))

    df = pd.DataFrame({"w": w, "y": y, **models})
    cml_qini = qini_score(df, outcome_col="y", treatment_col="w")
    cml_auuc = auuc_score(df, outcome_col="y", treatment_col="w")
    cml_qini_order = [m for m in cml_qini.sort_values().index if m in models]
    cml_auuc_order = [m for m in cml_auuc.sort_values().index if m in models]

    assert our_qini_order == cml_qini_order
    assert our_auuc_order == cml_auuc_order