"""Unit tests for subsample generation and per-subsample evaluation."""
import numpy as np
import pandas as pd

from src.analysis import generate_subsamples, evaluate_subsamples


def _td(n=20000, seed=0):
    rng = np.random.RandomState(seed)
    treatment = rng.binomial(1, 0.85, n)
    conversion = rng.binomial(1, 0.10, n)
    return treatment, conversion


def test_subsample_size_exact():
    t, c = _td()
    subs = generate_subsamples(t, c, n_subsamples=5, subsample_size=4000, seed=42)
    assert len(subs) == 5
    assert all(len(s) == 4000 for s in subs)


def test_no_within_duplicates():
    t, c = _td()
    for s in generate_subsamples(t, c, 3, 4000, 42):
        assert len(np.unique(s)) == len(s)


def test_determinism():
    t, c = _td()
    a = generate_subsamples(t, c, 4, 3000, 7)
    b = generate_subsamples(t, c, 4, 3000, 7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_stratification_preserved():
    t, c = _td()
    stratum = t.astype(int) * 2 + c.astype(int)
    full = pd.Series(stratum).value_counts(normalize=True).sort_index()
    sub = generate_subsamples(t, c, 1, 4000, 1)[0]
    sub_frac = pd.Series(stratum[sub]).value_counts(normalize=True).sort_index()
    for k in full.index:
        assert abs(full.get(k, 0) - sub_frac.get(k, 0)) < 0.02


def test_evaluate_subsamples_shape():
    t, c = _td()
    subs = generate_subsamples(t, c, 3, 4000, 42)
    scores = {"m1": np.random.RandomState(0).normal(size=len(t)),
              "m2": np.random.RandomState(1).normal(size=len(t))}
    df = evaluate_subsamples(scores, t, c, subs)
    assert len(df) == 3 * 2
    assert set(df.columns) == {"subsample", "model", "qini", "auuc"}