"""Unit tests for the four uplift models, on small synthetic data with a known
positive treatment effect. These run in seconds and need no real data.
"""
import numpy as np
import pandas as pd

from src.models import SLearner, TLearner, KLTree, NoInformationBaseline


def _xy(n=5000, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(12)})
    t = rng.binomial(1, 0.5, n)
    base = 1 / (1 + np.exp(-X["f0"].to_numpy()))
    p = np.clip(base + 0.1 * t, 0, 1)          # inject a +0.1 treatment effect
    y = rng.binomial(1, p)
    return X, t, y


def _small_tree():
    # smaller leaves so a tree can actually be grown on 5,000 synthetic rows
    return KLTree(max_depth=3, min_samples_leaf=100, min_samples_treatment=20)


# --- S-learner ---
def test_s_predict_shape():
    X, t, y = _xy()
    assert SLearner().fit(X, t, y).predict_uplift(X).shape == (len(X),)


def test_s_uplift_is_finite():
    X, t, y = _xy()
    assert np.all(np.isfinite(SLearner().fit(X, t, y).predict_uplift(X)))


def test_s_detects_positive_effect():
    X, t, y = _xy()
    assert SLearner().fit(X, t, y).predict_uplift(X).mean() > 0


def test_s_determinism():
    X, t, y = _xy()
    u1 = SLearner().fit(X, t, y).predict_uplift(X)
    u2 = SLearner().fit(X, t, y).predict_uplift(X)
    assert np.allclose(u1, u2)


# --- T-learner ---
def test_t_predict_shape():
    X, t, y = _xy()
    assert TLearner().fit(X, t, y).predict_uplift(X).shape == (len(X),)


def test_t_uplift_is_finite():
    X, t, y = _xy()
    assert np.all(np.isfinite(TLearner().fit(X, t, y).predict_uplift(X)))


def test_t_detects_positive_effect():
    X, t, y = _xy()
    assert TLearner().fit(X, t, y).predict_uplift(X).mean() > 0


def test_t_determinism():
    X, t, y = _xy()
    u1 = TLearner().fit(X, t, y).predict_uplift(X)
    u2 = TLearner().fit(X, t, y).predict_uplift(X)
    assert np.allclose(u1, u2)


# --- KL-divergence tree ---
def test_kl_predict_shape():
    X, t, y = _xy()
    assert _small_tree().fit(X, t, y).predict_uplift(X).shape == (len(X),)


def test_kl_uplift_is_finite():
    X, t, y = _xy()
    assert np.all(np.isfinite(_small_tree().fit(X, t, y).predict_uplift(X)))


def test_kl_determinism():
    X, t, y = _xy()
    u1 = _small_tree().fit(X, t, y).predict_uplift(X)
    u2 = _small_tree().fit(X, t, y).predict_uplift(X)
    assert np.allclose(u1, u2)


# --- No-information baseline ---
def test_b_predict_shape():
    X, t, y = _xy()
    assert NoInformationBaseline().fit(X, t, y).predict_uplift(X).shape == (len(X),)


def test_b_uplift_is_finite():
    X, t, y = _xy()
    assert np.all(np.isfinite(NoInformationBaseline().fit(X, t, y).predict_uplift(X)))


def test_b_is_constant():
    X, t, y = _xy()
    u = NoInformationBaseline().fit(X, t, y).predict_uplift(X)
    assert np.ptp(u) == 0          # every individual gets the identical score


def test_b_determinism():
    X, t, y = _xy()
    u1 = NoInformationBaseline().fit(X, t, y).predict_uplift(X)
    u2 = NoInformationBaseline().fit(X, t, y).predict_uplift(X)
    assert np.allclose(u1, u2)