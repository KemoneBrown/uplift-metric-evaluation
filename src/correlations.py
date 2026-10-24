"""Rank-correlation analysis (statistical analysis component, part 2).

Reads the per-subsample metric table written by analysis.py and computes:

H2 (primary): for each subsample, Kendall's tau-b and Spearman's rho between the
four models' Qini ranking and their AUUC ranking. The mean Kendall's tau-b
across subsamples is the primary test statistic (its bootstrap CI and the 0.7
decision rule are applied in the next step).

H2a (secondary): for each metric, the within-metric stability of each
subsample's model ranking against the reference ranking on the full test
partition, by Kendall's tau-b, giving 50 stability values per metric for the
paired Wilcoxon test.

Tau-b is used throughout so tied model scores are handled consistently
(OSF 10.17605/OSF.IO/5AYKJ, 4.2). Reads saved tables only; no models refit.
"""
from __future__ import annotations

import os
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from src import config
from src.analysis import SUBSAMPLE_METRICS_OUT, REFERENCE_METRICS_OUT

CROSS_METRIC_OUT = os.path.join("outputs", "cross_metric_correlations.csv")
WITHIN_METRIC_OUT = os.path.join("outputs", "within_metric_stability.csv")

_MODEL_ORDER = list(config.MODELS)


def _ordered(df_subset: pd.DataFrame, column: str):
    """Metric values for the four models in a fixed order."""
    return df_subset.set_index("model").loc[_MODEL_ORDER, column].to_numpy()


def cross_metric_correlations(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Per subsample: Kendall tau-b and Spearman rho between the Qini and AUUC
    rankings of the four models."""
    rows = []
    for i in sorted(metrics_df["subsample"].unique()):
        sub = metrics_df[metrics_df["subsample"] == i]
        q, a = _ordered(sub, "qini"), _ordered(sub, "auuc")
        rows.append({"subsample": int(i),
                     "kendall_tau": kendalltau(q, a, variant="b").statistic,
                     "spearman_rho": spearmanr(q, a).statistic})
    return pd.DataFrame(rows)


def within_metric_stability(metrics_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """Per subsample and metric: Kendall tau-b between the subsample ranking and
    the full-partition reference ranking under the same metric."""
    ref_q, ref_a = _ordered(reference_df, "qini"), _ordered(reference_df, "auuc")
    rows = []
    for i in sorted(metrics_df["subsample"].unique()):
        sub = metrics_df[metrics_df["subsample"] == i]
        rows.append({"subsample": int(i),
                     "qini_stability": kendalltau(_ordered(sub, "qini"), ref_q, variant="b").statistic,
                     "auuc_stability": kendalltau(_ordered(sub, "auuc"), ref_a, variant="b").statistic})
    return pd.DataFrame(rows)


def main() -> None:
    metrics_df = pd.read_csv(SUBSAMPLE_METRICS_OUT)
    reference_df = pd.read_csv(REFERENCE_METRICS_OUT)

    cross = cross_metric_correlations(metrics_df)
    within = within_metric_stability(metrics_df, reference_df)
    cross.to_csv(CROSS_METRIC_OUT, index=False)
    within.to_csv(WITHIN_METRIC_OUT, index=False)
    print(f"Wrote {CROSS_METRIC_OUT} and {WITHIN_METRIC_OUT} ({len(cross)} subsamples).")

    print("\nH2 (cross-metric agreement, Qini vs AUUC rankings):")
    print(f"  mean Kendall tau-b : {cross['kendall_tau'].mean():.4f}  "
          f"(min {cross['kendall_tau'].min():.3f}, max {cross['kendall_tau'].max():.3f})")
    print(f"  mean Spearman rho  : {cross['spearman_rho'].mean():.4f}")
    print(f"  pre-registered H2 threshold: mean tau below {config.TAU_THRESHOLD}")

    print("\nH2a (within-metric stability vs full-partition reference):")
    print(f"  Qini stability  median {within['qini_stability'].median():.4f}  "
          f"IQR [{within['qini_stability'].quantile(0.25):.3f}, {within['qini_stability'].quantile(0.75):.3f}]")
    print(f"  AUUC stability  median {within['auuc_stability'].median():.4f}  "
          f"IQR [{within['auuc_stability'].quantile(0.25):.3f}, {within['auuc_stability'].quantile(0.75):.3f}]")


if __name__ == "__main__":
    main()