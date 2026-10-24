"""Unit tests for the partition logic, on small synthetic data.

These do not need the Criteo download, so they run in seconds. They check
the three properties the pre-registration relies on: exact sizes, disjoint
partitions, and preserved strata.
"""
import numpy as np
import pandas as pd

from src import data_processing as dp


def _synthetic(n=20000, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({f"f{i}": rng.normal(size=n).astype("float32") for i in range(12)})
    df[dp.TREATMENT_COL] = rng.binomial(1, 0.85, n).astype("int8")
    df[dp.CONVERSION_COL] = rng.binomial(1, 0.03, n).astype("int8")
    return df


def test_allocate_sums_exactly():
    frac = pd.Series({0: 0.5, 1: 0.3, 2: 0.15, 3: 0.05})
    assert sum(dp._allocate(1000, frac).values()) == 1000


def test_sizes_exact():
    train, test = dp.make_partitions(_synthetic(), train_size=4000, test_size=4000)
    assert len(train) == 4000 and len(test) == 4000


def test_partitions_disjoint():
    train, test = dp.make_partitions(_synthetic(), train_size=4000, test_size=4000)
    combined = pd.concat([train, test], ignore_index=True)
    assert not combined.duplicated().any()


def test_determinism():
    df = _synthetic()
    a1, a2 = dp.make_partitions(df, train_size=3000, test_size=3000)
    b1, b2 = dp.make_partitions(df, train_size=3000, test_size=3000)
    pd.testing.assert_frame_equal(a1, b1)
    pd.testing.assert_frame_equal(a2, b2)


def test_strata_preserved():
    df = _synthetic()
    train, _ = dp.make_partitions(df, train_size=5000, test_size=5000)
    assert abs(train[dp.TREATMENT_COL].mean() - df[dp.TREATMENT_COL].mean()) < 0.03