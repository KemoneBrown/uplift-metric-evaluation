"""Data ingestion and preprocessing for the Criteo Uplift dataset.

Loads the raw Criteo v2.1 file, validates completeness, and draws two
disjoint partitions of config.TRAIN_SIZE and config.TEST_SIZE rows, each
stratified on the (treatment, conversion) joint distribution so the
randomised treatment balance and the class balance of the full data are
preserved (OSF 10.17605/OSF.IO/5AYKJ, Sampling Plan 3.3).

Treatment is used as randomised (intention-to-treat); the post-treatment
'exposure' column is not used. Per the pre-registration (5.2, 5.4), no
feature standardisation and no value-based exclusion are applied: the only
reduction from the full dataset is the stratified sampling, with listwise
exclusion of any record missing a used field.

All randomness is driven by config.RANDOM_SEED, so the output is
deterministic given the same input file.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src import config

RAW_PATH = os.path.join("data", "criteo-research-uplift-v2.1.csv.gz")
FEATURE_COLS = [f"f{i}" for i in range(12)]
TREATMENT_COL = "treatment"
CONVERSION_COL = "conversion"
USED_COLS = FEATURE_COLS + [TREATMENT_COL, CONVERSION_COL]

_READ_DTYPES = {c: "float32" for c in FEATURE_COLS}
_READ_DTYPES[TREATMENT_COL] = "int8"
_READ_DTYPES[CONVERSION_COL] = "int8"

TRAIN_OUT = os.path.join("outputs", "train_partition.csv.gz")
TEST_OUT = os.path.join("outputs", "test_partition.csv.gz")
_GZIP = {"method": "gzip", "mtime": 0}  # fixed mtime for byte-stable output


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Load only the columns the study uses, with compact dtypes."""
    return pd.read_csv(path, usecols=USED_COLS, dtype=_READ_DTYPES)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Listwise exclusion of records missing any used field (pre-reg 5.4/5.5).

    No value-based exclusion and no de-duplication are performed, so the only
    reduction from the full dataset is this completeness check followed by the
    stratified sampling in make_partitions.
    """
    before = len(df)
    df = df.dropna(subset=USED_COLS).reset_index(drop=True)
    dropped = before - len(df)
    print(f"Validation: read {before:,}, dropped {dropped:,} incomplete, retained {len(df):,}.")
    return df


def _allocate(total: int, fractions: pd.Series) -> dict:
    """Split `total` across strata in proportion to `fractions`, summing exactly."""
    raw = fractions * total
    counts = np.floor(raw).astype(int)
    shortfall = total - int(counts.sum())
    order = (raw - np.floor(raw)).sort_values(ascending=False).index
    for i in range(shortfall):
        counts[order[i]] += 1
    return counts.to_dict()


def make_partitions(df: pd.DataFrame, train_size=None, test_size=None):
    """Draw disjoint train and test partitions, each stratified on
    (treatment, conversion). Within each stratum the eligible rows are
    shuffled once with the fixed seed; the first block becomes train and
    the next, non-overlapping block becomes test."""
    train_size = config.TRAIN_SIZE if train_size is None else train_size
    test_size = config.TEST_SIZE if test_size is None else test_size

    stratum = df[TREATMENT_COL].astype(int) * 2 + df[CONVERSION_COL].astype(int)
    fractions = stratum.value_counts(normalize=True).sort_index()

    n_train = _allocate(train_size, fractions)
    n_test = _allocate(test_size, fractions)

    rng = np.random.RandomState(config.RANDOM_SEED)
    train_idx, test_idx = [], []
    for s in sorted(fractions.index):
        rows = df.index[stratum == s].to_numpy()
        rng.shuffle(rows)
        need = n_train[s] + n_test[s]
        if len(rows) < need:
            raise ValueError(
                f"Stratum {s}: need {need:,} rows, only {len(rows):,} available."
            )
        train_idx.append(rows[:n_train[s]])
        test_idx.append(rows[n_train[s]:need])

    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    train = df.loc[train_idx, USED_COLS].reset_index(drop=True)
    test = df.loc[test_idx, USED_COLS].reset_index(drop=True)
    assert set(train_idx).isdisjoint(test_idx), "Partitions overlap."
    return train, test


def _report(name: str, part: pd.DataFrame) -> None:
    print(f"{name}: {len(part):,} rows, "
          f"treatment share {part[TREATMENT_COL].mean():.4f}, "
          f"conversion rate {part[CONVERSION_COL].mean():.5f}")


def main() -> None:
    os.makedirs("outputs", exist_ok=True)
    df = validate(load_raw())
    train, test = make_partitions(df)
    train.to_csv(TRAIN_OUT, index=False, compression=_GZIP)
    test.to_csv(TEST_OUT, index=False, compression=_GZIP)
    print(f"Wrote {TRAIN_OUT} and {TEST_OUT}.")
    _report("source", df)
    _report("train", train)
    _report("test", test)


if __name__ == "__main__":
    main()