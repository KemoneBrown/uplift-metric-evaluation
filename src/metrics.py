"""Uplift evaluation metrics: the Qini coefficient and the Area Under the
Uplift Curve (AUUC).

Both rank individuals by predicted uplift (descending) and accumulate treated
and control responses down the ranking. They differ in how they combine those
counts, which is the structural basis for the ranking disagreement this study
examines (OSF 10.17605/OSF.IO/5AYKJ, 1.2):

- The Qini curve scales control responders by the treated/control count ratio,
  so it tracks the absolute number of incremental conversions.
- The uplift curve takes the per-group response rates and multiplies their
  difference by the number targeted, so it tracks relative rate differences.

Implemented in numpy per pre-registration 6.1 (causalml is confined to the
tree estimator). Deterministic: ties in the score are broken by original order
via a stable sort. Areas are normalised by the number of individuals so values
are comparable across the equally sized subsamples; this scaling is constant
within a subsample and so does not affect the model ranking.
"""
from __future__ import annotations

import numpy as np


def _cumulative(score, treatment, outcome):
    """Rank by score descending (stable), then accumulate counts and responses."""
    score = np.asarray(score, dtype=float)
    w = np.asarray(treatment, dtype=float)
    y = np.asarray(outcome, dtype=float)
    order = np.argsort(-score, kind="mergesort")   # descending; ties keep order
    w, y = w[order], y[order]
    n_t = np.cumsum(w)
    n_c = np.cumsum(1.0 - w)
    r_t = np.cumsum(w * y)
    r_c = np.cumsum((1.0 - w) * y)
    return n_t, n_c, r_t, r_c


def _area(curve):
    """Trapezoidal area under the curve, with the origin (0, 0) prepended."""
    n = len(curve)
    x = np.concatenate(([0.0], np.arange(1, n + 1, dtype=float)))
    y = np.concatenate(([0.0], np.asarray(curve, dtype=float)))
    return float(np.trapz(y, x))


def qini_curve(score, treatment, outcome):
    n_t, n_c, r_t, r_c = _cumulative(score, treatment, outcome)
    ratio = np.divide(n_t, n_c, out=np.zeros_like(n_t), where=n_c > 0)
    return r_t - r_c * ratio


def uplift_curve(score, treatment, outcome):
    n_t, n_c, r_t, r_c = _cumulative(score, treatment, outcome)
    rate_t = np.divide(r_t, n_t, out=np.zeros_like(r_t), where=n_t > 0)
    rate_c = np.divide(r_c, n_c, out=np.zeros_like(r_c), where=n_c > 0)
    return (rate_t - rate_c) * (n_t + n_c)


def qini_coefficient(score, treatment, outcome):
    """Area between the Qini curve and the random-targeting diagonal, normalised
    by the number of individuals. Zero is no better than random; larger is
    better. Not normalised by an optimal-model curve, which is unnecessary for
    ranking and is recorded as an implementation choice."""
    q = qini_curve(score, treatment, outcome)
    n = len(q)
    area_random = 0.5 * n * q[-1]
    return (_area(q) - area_random) / n


def auuc(score, treatment, outcome):
    """Area under the uplift curve, normalised by the number of individuals.
    Larger is better; the no-information baseline traces the near-linear
    random-targeting reference."""
    u = uplift_curve(score, treatment, outcome)
    return _area(u) / len(u)


def evaluate(score, treatment, outcome) -> dict:
    """Both metrics from a single ranking pass. Returns {'qini':.., 'auuc':..}."""
    n_t, n_c, r_t, r_c = _cumulative(score, treatment, outcome)
    n = len(n_t)
    ratio = np.divide(n_t, n_c, out=np.zeros_like(n_t), where=n_c > 0)
    q = r_t - r_c * ratio
    rate_t = np.divide(r_t, n_t, out=np.zeros_like(r_t), where=n_t > 0)
    rate_c = np.divide(r_c, n_c, out=np.zeros_like(r_c), where=n_c > 0)
    u = (rate_t - rate_c) * (n_t + n_c)
    return {"qini": (_area(q) - 0.5 * n * q[-1]) / n, "auuc": _area(u) / n}


def main() -> None:
    """Reference evaluation: fit all four models, score the full test partition,
    and rank the models under each metric. Fitting the tree takes a few minutes."""
    import pandas as pd
    from src import data_processing as dp
    from src import models as m

    train = pd.read_csv(dp.TRAIN_OUT, dtype=dp._READ_DTYPES)
    test = pd.read_csv(dp.TEST_OUT, dtype=dp._READ_DTYPES)
    X_tr, t_tr, y_tr = train[dp.FEATURE_COLS], train[dp.TREATMENT_COL].to_numpy(), train[dp.CONVERSION_COL].to_numpy()
    X_te, t_te, y_te = test[dp.FEATURE_COLS], test[dp.TREATMENT_COL].to_numpy(), test[dp.CONVERSION_COL].to_numpy()

    print("Fitting four models (the tree takes a few minutes)...")
    fitted = {
        "s_learner": m.SLearner().fit(X_tr, t_tr, y_tr),
        "t_learner": m.TLearner().fit(X_tr, t_tr, y_tr),
        "kl_tree": m.KLTree().fit(X_tr, t_tr, y_tr),
        "no_information_baseline": m.NoInformationBaseline().fit(X_tr, t_tr, y_tr),
    }

    rows = {}
    print("Reference evaluation on the full test partition:")
    for name, model in fitted.items():
        rows[name] = evaluate(model.predict_uplift(X_te), t_te, y_te)
        print(f"  {name:24s} qini {rows[name]['qini']:+.5f} | auuc {rows[name]['auuc']:+.5f}")

    qini_rank = sorted(rows, key=lambda k: rows[k]["qini"], reverse=True)
    auuc_rank = sorted(rows, key=lambda k: rows[k]["auuc"], reverse=True)
    print(f"  Qini ranking (best first): {qini_rank}")
    print(f"  AUUC ranking (best first): {auuc_rank}")
    print(f"  Rankings agree: {qini_rank == auuc_rank}")


if __name__ == "__main__":
    main()