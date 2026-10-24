"""Confirmatory hypothesis tests (statistical analysis component, part 3).

H2: 95% percentile bootstrap CI for the mean Kendall's tau-b across subsamples
(config.BOOTSTRAP_RESAMPLES resamples), decided against the pre-registered 0.7
threshold by the three-way rule in OSF 10.17605/OSF.IO/5AYKJ, 5.3.

H2a: one-sided Wilcoxon signed-rank test that Qini within-metric stability is
lower than AUUC stability (alpha = config.H2A_ALPHA), with the paired difference
summarised by its median and interquartile range (pre-reg 4.3).

Also reports descriptive statistics (mean, standard deviation and interquartile
range) of the Qini and AUUC scores per model across subsamples (pre-reg 4.3).
Reads saved tables only. Seeds fixed (NFR1).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src import config
from src.analysis import SUBSAMPLE_METRICS_OUT
from src.correlations import CROSS_METRIC_OUT, WITHIN_METRIC_OUT

DESCRIPTIVES_OUT = os.path.join("outputs", "descriptive_stats.csv")
HYPOTHESIS_SUMMARY_OUT = os.path.join("outputs", "hypothesis_summary.csv")


def bootstrap_mean_ci(values, n_resamples, ci_level, seed):
    """Percentile bootstrap CI for the mean of `values` (deterministic seed)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = values[idx].mean(axis=1)
    alpha = 1.0 - ci_level
    low, high = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(low), float(high)


def h2(cross_df):
    mean_tau, low, high = bootstrap_mean_ci(
        cross_df["kendall_tau"], config.BOOTSTRAP_RESAMPLES, config.CI_LEVEL, config.RANDOM_SEED)
    thr = config.TAU_THRESHOLD
    if high < thr:
        verdict = "supported (95% CI lies entirely below 0.7)"
    elif low > thr:
        verdict = "not supported (95% CI lies entirely above 0.7)"
    else:
        verdict = "indeterminate; contingency triggers re-run at 100 subsamples"
    return {"mean_tau": mean_tau, "ci_low": low, "ci_high": high, "verdict": verdict}


def h2a(within_df):
    q = within_df["qini_stability"].to_numpy(dtype=float)
    a = within_df["auuc_stability"].to_numpy(dtype=float)
    diff = a - q  # AUUC minus Qini; positive means AUUC more stable
    median_diff = float(np.median(diff))
    iqr_diff = float(np.quantile(diff, 0.75) - np.quantile(diff, 0.25))
    if np.all(diff == 0):
        return {"statistic": float("nan"), "pvalue": 1.0,
                "median_diff": median_diff, "iqr_diff": iqr_diff,
                "verdict": "not supported (Qini and AUUC stability identical on every subsample)"}
    res = wilcoxon(q, a, alternative="less", zero_method="wilcox")
    supported = res.pvalue < config.H2A_ALPHA
    verdict = ("supported (Qini stability significantly lower, p < 0.05)" if supported
               else "not supported (no significant difference in the predicted direction)")
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue),
            "median_diff": median_diff, "iqr_diff": iqr_diff, "verdict": verdict}


def descriptives(metrics_df):
    g = metrics_df.groupby("model")
    out = pd.DataFrame({
        "qini_mean": g["qini"].mean(),
        "qini_sd": g["qini"].std(),
        "qini_iqr": g["qini"].quantile(0.75) - g["qini"].quantile(0.25),
        "auuc_mean": g["auuc"].mean(),
        "auuc_sd": g["auuc"].std(),
        "auuc_iqr": g["auuc"].quantile(0.75) - g["auuc"].quantile(0.25),
    }).reindex(list(config.MODELS)).reset_index()
    return out


def main() -> None:
    metrics_df = pd.read_csv(SUBSAMPLE_METRICS_OUT)
    cross_df = pd.read_csv(CROSS_METRIC_OUT)
    within_df = pd.read_csv(WITHIN_METRIC_OUT)

    r2 = h2(cross_df)
    r2a = h2a(within_df)
    descriptives(metrics_df).to_csv(DESCRIPTIVES_OUT, index=False)

    pd.DataFrame([
        {"hypothesis": "H2", "statistic": r2["mean_tau"], "ci_low": r2["ci_low"],
         "ci_high": r2["ci_high"], "pvalue": "", "median_diff": "", "iqr_diff": "",
         "verdict": r2["verdict"]},
        {"hypothesis": "H2a", "statistic": r2a["statistic"], "ci_low": "", "ci_high": "",
         "pvalue": r2a["pvalue"], "median_diff": r2a["median_diff"], "iqr_diff": r2a["iqr_diff"],
         "verdict": r2a["verdict"]},
    ]).to_csv(HYPOTHESIS_SUMMARY_OUT, index=False)

    print("H2 (cross-metric ranking inconsistency):")
    print(f"  mean Kendall tau-b {r2['mean_tau']:.4f}  "
          f"95% bootstrap CI [{r2['ci_low']:.4f}, {r2['ci_high']:.4f}]  (threshold {config.TAU_THRESHOLD})")
    print(f"  Verdict: H2 {r2['verdict']}")
    print()
    print("H2a (Qini less stable than AUUC):")
    print(f"  Wilcoxon statistic {r2a['statistic']}  p-value {r2a['pvalue']:.4f}")
    print(f"  paired stability difference (AUUC - Qini): "
          f"median {r2a['median_diff']:.4f}, IQR {r2a['iqr_diff']:.4f}")
    print(f"  Verdict: H2a {r2a['verdict']}")
    print()
    print(f"Wrote {DESCRIPTIVES_OUT} and {HYPOTHESIS_SUMMARY_OUT}.")


if __name__ == "__main__":
    main()