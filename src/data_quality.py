"""One-off data-quality checks on the written partitions.

Reports exact-duplicate rows within each partition and any content-level
overlap between train and test. make_partitions already guarantees the two
are disjoint by index; this confirms there is no leakage through identical
feature rows. Traceable to OSF 10.17605/OSF.IO/5AYKJ.
"""
from __future__ import annotations

import pandas as pd

from src import data_processing as dp


def main() -> None:
    train = pd.read_csv(dp.TRAIN_OUT, dtype=dp._READ_DTYPES)
    test = pd.read_csv(dp.TEST_OUT, dtype=dp._READ_DTYPES)

    dup_train = int(train.duplicated(subset=dp.USED_COLS).sum())
    dup_test = int(test.duplicated(subset=dp.USED_COLS).sum())

    train_keys = train[dp.USED_COLS].drop_duplicates()
    test_keys = test[dp.USED_COLS].drop_duplicates()
    overlap = int(len(train_keys.merge(test_keys, on=dp.USED_COLS, how="inner")))

    print(f"Train rows {len(train):,} | exact-duplicate rows within train: {dup_train:,}")
    print(f"Test rows  {len(test):,} | exact-duplicate rows within test:  {dup_test:,}")
    print(f"Distinct rows appearing in BOTH partitions (content overlap): {overlap:,}")


if __name__ == "__main__":
    main()