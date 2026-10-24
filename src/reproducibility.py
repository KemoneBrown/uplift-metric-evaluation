"""Reproducibility validation (NFR1; OSF 10.17605/OSF.IO/5AYKJ, 6.1 and 3.6).

Runs the stochastic analytical pipeline (model fitting, subsampling, metric
evaluation, rank correlation, bootstrap and hypothesis tests) several times and
checks that every output file is byte-identical across runs. Any difference is
reported as a failure, matching the pre-registered operational halt rule: if any
re-run produces non-identical output, progression halts until the cause is
diagnosed.

Partition determinism is covered separately by the data_processing unit tests;
this validates the parts of the pipeline that fit models and draw subsamples.
"""
from __future__ import annotations

import hashlib
import os

from src import analysis, correlations, hypothesis

_OUTPUTS = [
    analysis.SUBSAMPLE_METRICS_OUT,
    analysis.REFERENCE_METRICS_OUT,
    correlations.CROSS_METRIC_OUT,
    correlations.WITHIN_METRIC_OUT,
    hypothesis.DESCRIPTIVES_OUT,
    hypothesis.HYPOTHESIS_SUMMARY_OUT,
]


def file_hash(path: str) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_pipeline() -> None:
    """One full analytical pass, regenerating every output file."""
    analysis.main()
    correlations.main()
    hypothesis.main()


def main(n_runs: int = 3) -> None:
    print(f"Reproducibility validation: {n_runs} independent re-runs of the pipeline.\n")
    hashes = []
    for r in range(n_runs):
        print(f"=== Run {r + 1} of {n_runs} ===")
        run_pipeline()
        hashes.append({p: file_hash(p) for p in _OUTPUTS})
        print()

    print("Byte-identity against run 1:")
    all_ok = True
    for p in _OUTPUTS:
        identical = all(h[p] == hashes[0][p] for h in hashes)
        all_ok = all_ok and identical
        status = "IDENTICAL" if identical else "DIFFERENT"
        print(f"  {status:10s} {os.path.basename(p)}  sha256 {hashes[0][p][:16]}")

    print()
    if all_ok:
        print(f"PASS: every output byte-identical across all {n_runs} runs. NFR1 satisfied.")
    else:
        print("FAIL: outputs differ across runs. Pipeline is not reproducible; halt and diagnose.")


if __name__ == "__main__":
    main()