"""Statistical analysis component: subsample generation and per-subsample
model evaluation.

Fits the four models once on the training partition, scores them on the full
test partition, then draws config.N_SUBSAMPLES stratified subsamples of
config.SUBSAMPLE_SIZE from the TEST partition only (never the training data)
and records the Qini coefficient and AUUC per model per subsample. These
per-subsample metrics feed the rank-correlation analysis.

Pre-registration (OSF 10.17605/OSF.IO/5AYKJ, 3.3): each subsample is drawn
without replacement within itself (no observation appears twice in one
subsample) and from an independent fixed-seed stream (so subsamples may overlap
with one another), stratified on (treatment, conversion). All seeds are fixed
(NFR1), so the whole procedure is deterministic.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from src import config
from src import data_processing as dp
from src import models as m
from src.metrics import evaluate

SUBSAMPLE_METRICS_OUT = os.path.join("outputs", "subsample_metrics.csv")
REFERENCE_METRICS_OUT = os.path.join("outputs", "reference_metrics.csv")

_MODEL_FACTORIES = {
    "s_learner": m.SLearner,
    "t_learner": m.TLearner,
    "kl_tree": m.KLTree,
    "no_information_baseline": m.NoInformationBaseline,
}


def fit_models(X, treatment, y) -> dict:
    """Fit every model once on the training partition."""
    return {name: factory().fit(X, treatment, y)
            for name, factory in _MODEL_FACTORIES.items()}


def predict_all(fitted: dict, X) -> dict:
    """Score every fitted model on X once, returning a dict of uplift arrays."""
    return {name: np.asarray(model.predict_uplift(X)) for name, model in fitted.items()}


def _stratum(treatment, conversion):
    return np.asarray(treatment).astype(int) * 2 + np.asarray(conversion).astype(int)


def generate_subsamples(treatment, conversion, n_subsamples, subsample_size, seed):
    """Return a list of index arrays, one per subsample.

    Each subsample is stratified on (treatment, conversion) and drawn without
    replacement within itself, from an independent fixed-seed stream.
    """
    stratum = _stratum(treatment, conversion)
    fractions = pd.Series(stratum).value_counts(normalize=True).sort_index()
    allocation = dp._allocate(subsample_size, fractions)
    pools = {s: np.where(stratum == s)[0] for s in allocation}

    child_seeds = np.random.SeedSequence(seed).spawn(n_subsamples)
    subsamples = []
    for cs in child_seeds:
        rng = np.random.default_rng(cs)
        parts = [rng.choice(pools[s], size=allocation[s], replace=False)
                 for s in sorted(allocation)]
        idx = np.concatenate(parts)
        rng.shuffle(idx)
        subsamples.append(idx)
    return subsamples


def evaluate_subsamples(scores: dict, treatment, conversion, subsamples) -> pd.DataFrame:
    """Qini and AUUC for every model on every subsample."""
    treatment = np.asarray(treatment)
    conversion = np.asarray(conversion)
    rows = []
    for i, idx in enumerate(subsamples):
        w, y = treatment[idx], conversion[idx]
        for name, s in scores.items():
            res = evaluate(s[idx], w, y)
            rows.append({"subsample": i, "model": name,
                         "qini": res["qini"], "auuc": res["auuc"]})
    return pd.DataFrame(rows)


def reference_metrics(scores: dict, treatment, conversion) -> pd.DataFrame:
    """Qini and AUUC for every model on the full test partition (H2a reference)."""
    treatment = np.asarray(treatment)
    conversion = np.asarray(conversion)
    rows = [{"model": name, **evaluate(s, treatment, conversion)}
            for name, s in scores.items()]
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs("outputs", exist_ok=True)
    train = pd.read_csv(dp.TRAIN_OUT, dtype=dp._READ_DTYPES)
    test = pd.read_csv(dp.TEST_OUT, dtype=dp._READ_DTYPES)

    X_tr, t_tr, y_tr = train[dp.FEATURE_COLS], train[dp.TREATMENT_COL].to_numpy(), train[dp.CONVERSION_COL].to_numpy()
    X_te, t_te, y_te = test[dp.FEATURE_COLS], test[dp.TREATMENT_COL].to_numpy(), test[dp.CONVERSION_COL].to_numpy()

    print("Fitting four models on the training partition (the tree takes a few minutes)...")
    fitted = fit_models(X_tr, t_tr, y_tr)
    scores = predict_all(fitted, X_te)

    print(f"Drawing {config.N_SUBSAMPLES} subsamples of {config.SUBSAMPLE_SIZE:,} from the test partition...")
    subsamples = generate_subsamples(t_te, y_te, config.N_SUBSAMPLES, config.SUBSAMPLE_SIZE, config.RANDOM_SEED)

    metrics_df = evaluate_subsamples(scores, t_te, y_te, subsamples)
    ref_df = reference_metrics(scores, t_te, y_te)

    metrics_df.to_csv(SUBSAMPLE_METRICS_OUT, index=False)
    ref_df.to_csv(REFERENCE_METRICS_OUT, index=False)
    print(f"Wrote {SUBSAMPLE_METRICS_OUT} ({len(metrics_df)} rows) and {REFERENCE_METRICS_OUT} ({len(ref_df)} rows).")

    agree = 0
    for i in range(config.N_SUBSAMPLES):
        sub = metrics_df[metrics_df["subsample"] == i]
        top_qini = sub.loc[sub["qini"].idxmax(), "model"]
        top_auuc = sub.loc[sub["auuc"].idxmax(), "model"]
        agree += int(top_qini == top_auuc)
    print(f"Subsamples where Qini and AUUC agree on the single best model: {agree}/{config.N_SUBSAMPLES}")


if __name__ == "__main__":
    main()