"""Figure 4.3: Qini and uplift curves for the four models on the full test
partition, generated from the project's own framework so the curves match the
reported reference metrics exactly.

Run from the project root inside the same container used for the pipeline:

    docker run --rm -v "${PWD}:/project" -w /project uplift-eval:dev python figures.py

It reads outputs/train_partition.csv.gz and outputs/test_partition.csv.gz,
fits the four models with src.models, scores the full test partition, and draws
both curves with src.metrics. It writes outputs/fig_curves.png. The Qini and
AUUC values it prints should match outputs/reference_metrics.csv.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src import data_processing as dp
from src import metrics
from src import models as m

DISPLAY = {
    "s_learner": "S-learner",
    "t_learner": "T-learner",
    "kl_tree": "KL-divergence tree",
    "no_information_baseline": "No-information baseline",
}


def main():
    train = pd.read_csv(dp.TRAIN_OUT, dtype=dp._READ_DTYPES)
    test = pd.read_csv(dp.TEST_OUT, dtype=dp._READ_DTYPES)
    X_tr, t_tr, y_tr = (train[dp.FEATURE_COLS], train[dp.TREATMENT_COL].to_numpy(),
                        train[dp.CONVERSION_COL].to_numpy())
    X_te, t_te, y_te = (test[dp.FEATURE_COLS], test[dp.TREATMENT_COL].to_numpy(),
                        test[dp.CONVERSION_COL].to_numpy())

    print("Fitting four models (the tree takes a few minutes)...")
    fitted = {
        "s_learner": m.SLearner().fit(X_tr, t_tr, y_tr),
        "t_learner": m.TLearner().fit(X_tr, t_tr, y_tr),
        "kl_tree": m.KLTree().fit(X_tr, t_tr, y_tr),
        "no_information_baseline": m.NoInformationBaseline().fit(X_tr, t_tr, y_tr),
    }
    scores = {name: model.predict_uplift(X_te) for name, model in fitted.items()}

    n = len(y_te)
    idx = np.linspace(0, n - 1, 1000).astype(int)
    xpct = (idx + 1) / n * 100.0

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for name in config.MODELS:
        q = metrics.qini_curve(scores[name], t_te, y_te)
        u = metrics.uplift_curve(scores[name], t_te, y_te)
        ev = metrics.evaluate(scores[name], t_te, y_te)
        print(f"  {name:24s} qini {ev['qini']:+.5f} | auuc {ev['auuc']:+.5f}")
        ax[0].plot(xpct, q[idx], label=DISPLAY[name])
        ax[1].plot(xpct, u[idx], label=DISPLAY[name])

    ax[0].set_title("Qini curves")
    ax[0].set_ylabel("Cumulative incremental conversions")
    ax[1].set_title("Uplift curves")
    ax[1].set_ylabel("Cumulative uplift")
    for a in ax:
        a.set_xlabel("Percentage of population targeted")
        a.axhline(0, color="0.6", lw=0.8)
        a.grid(alpha=0.3)
        a.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig("outputs/fig_curves.png", dpi=150, bbox_inches="tight")
    print("Wrote outputs/fig_curves.png")


if __name__ == "__main__":
    main()
