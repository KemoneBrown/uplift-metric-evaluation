"""Uplift model implementations.

Each model exposes the same interface: fit(X, treatment, y) and
predict_uplift(X), returning per-row uplift scores (the estimated change
in conversion probability caused by treatment).

Implements all four models: the S-learner, T-learner, KL-divergence tree and
no-information baseline.

Per pre-registration 5.2, no feature standardisation is applied. Traceable
to OSF 10.17605/OSF.IO/5AYKJ. Deterministic given config.RANDOM_SEED.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from causalml.inference.tree import UpliftTreeClassifier

from src import config

# KL-divergence uplift tree hyperparameters (not fixed by the pre-registration,
# chosen here and documented). Conversion is rare (~0.3%), so the defaults were
# loosened from an initial over-conservative setting that produced no splits:
# smaller leaves, a little more depth and lighter regularisation, so the tree
# grows and yields varying uplift scores that the metrics can rank.
KL_TREE_MAX_DEPTH = 5
KL_TREE_MIN_SAMPLES_LEAF = 2_000
KL_TREE_MIN_SAMPLES_TREATMENT = 200
KL_TREE_N_REG = 5


def _make_base_learner() -> LogisticRegression:
    """Shared base learner for the S- and T-learners.

    Plain logistic regression on the raw features: convex, deterministic and
    reproducible (NFR1). Per pre-registration 5.2 no feature standardisation
    is applied; the solver iteration limit is a numerical setting, not a
    feature transformation. No class_weight is used: uplift is a difference of
    probabilities and needs calibrated probabilities, which re-balancing would
    distort.
    """
    return LogisticRegression(solver="lbfgs", max_iter=2000,
                              random_state=config.RANDOM_SEED)


class SLearner:
    """Single-model meta-learner.

    Fits one model on the features together with the treatment indicator,
    then estimates uplift as the difference in predicted conversion
    probability with the treatment flag set to 1 versus 0.
    """

    def __init__(self):
        self.model = _make_base_learner()
        self._columns = None

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> "SLearner":
        design = X.copy()
        design["treatment"] = np.asarray(treatment)
        self._columns = design.columns
        self.model.fit(design, np.asarray(y))
        return self

    def predict_outcome(self, X: pd.DataFrame, treatment) -> np.ndarray:
        """Predicted P(conversion=1) for a given treatment assignment."""
        design = X.copy()
        design["treatment"] = np.asarray(treatment)
        design = design[self._columns]
        return self.model.predict_proba(design)[:, 1]

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        p_treated = self.predict_outcome(X, np.ones(len(X), dtype=int))
        p_control = self.predict_outcome(X, np.zeros(len(X), dtype=int))
        return p_treated - p_control


class TLearner:
    """Two-model meta-learner.

    Fits one base model on the treated rows and a separate base model on the
    control rows, both on the features alone. Uplift is the treated model's
    predicted conversion probability minus the control model's, for the same
    person. Unlike the S-learner, this can yield uplift of either sign.
    """

    def __init__(self):
        self.model_treated = _make_base_learner()
        self.model_control = _make_base_learner()
        self._columns = None

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> "TLearner":
        treatment = np.asarray(treatment).astype(bool)
        y = np.asarray(y)
        self._columns = X.columns
        self.model_treated.fit(X[treatment], y[treatment])
        self.model_control.fit(X[~treatment], y[~treatment])
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self._columns]
        p_treated = self.model_treated.predict_proba(X)[:, 1]
        p_control = self.model_control.predict_proba(X)[:, 1]
        return p_treated - p_control


class KLTree:
    """Uplift decision tree using the KL-divergence splitting criterion
    (causalml). Each split is chosen to maximise the Kullback-Leibler
    divergence between the treated and control outcome distributions, so the
    tree partitions people by how differently they respond to treatment.
    Can yield uplift of either sign.
    """
    _CONTROL = "control"
    _TREATMENT = "treatment"

    def __init__(self, max_depth=None, min_samples_leaf=None,
                 min_samples_treatment=None, n_reg=None):
        self.model = UpliftTreeClassifier(
            control_name=self._CONTROL,
            evaluationFunction="KL",
            max_depth=KL_TREE_MAX_DEPTH if max_depth is None else max_depth,
            min_samples_leaf=(KL_TREE_MIN_SAMPLES_LEAF
                              if min_samples_leaf is None else min_samples_leaf),
            min_samples_treatment=(KL_TREE_MIN_SAMPLES_TREATMENT
                                   if min_samples_treatment is None else min_samples_treatment),
            n_reg=KL_TREE_N_REG if n_reg is None else n_reg,
            random_state=config.RANDOM_SEED,
        )
        self._columns = None

    def _labels(self, treatment) -> np.ndarray:
        treatment = np.asarray(treatment).astype(int)
        return np.where(treatment == 1, self._TREATMENT, self._CONTROL)

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> "KLTree":
        self._columns = X.columns
        self.model.fit(X.values, self._labels(treatment), np.asarray(y).astype(int))
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.asarray(self.model.predict(X[self._columns].values))
        if preds.ndim == 2:
            classes = [str(c) for c in getattr(self.model, "classes_", [])]
            if self._TREATMENT in classes:
                preds = preds[:, classes.index(self._TREATMENT)]
            else:
                preds = preds[:, -1]
        return preds.astype(float)


class NoInformationBaseline:
    """RQ1 benchmark (pre-reg 2.3). Assigns the same predicted uplift to every
    individual - the overall average treatment effect estimated on the training
    data - so it carries no information to discriminate responders and produces
    a flat expected uplift curve. A uniform predicted-uplift comparator, not a
    re-randomisation of treatment.
    """

    def __init__(self):
        self._value = 0.0
        self._columns = None

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> "NoInformationBaseline":
        treatment = np.asarray(treatment).astype(bool)
        y = np.asarray(y).astype(float)
        self._columns = X.columns
        self._value = float(y[treatment].mean() - y[~treatment].mean())
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._value, dtype=float)


def main() -> None:
    """Fit all four models on the train partition and report diagnostics."""
    from sklearn.metrics import roc_auc_score
    from src import data_processing as dp

    train = pd.read_csv(dp.TRAIN_OUT, dtype=dp._READ_DTYPES)
    test = pd.read_csv(dp.TEST_OUT, dtype=dp._READ_DTYPES)

    X_tr, t_tr, y_tr = train[dp.FEATURE_COLS], train[dp.TREATMENT_COL].to_numpy(), train[dp.CONVERSION_COL].to_numpy()
    X_te, t_te, y_te = test[dp.FEATURE_COLS], test[dp.TREATMENT_COL].to_numpy(), test[dp.CONVERSION_COL].to_numpy()

    # --- S-learner ---
    s = SLearner().fit(X_tr, t_tr, y_tr)
    s_auc_tr = roc_auc_score(y_tr, s.predict_outcome(X_tr, t_tr))
    s_auc_te = roc_auc_score(y_te, s.predict_outcome(X_te, t_te))
    s_up = s.predict_uplift(X_te)
    print("S-learner")
    print(f"  outcome AUC  train {s_auc_tr:.4f} | test {s_auc_te:.4f} | gap {s_auc_tr - s_auc_te:+.4f}")
    print(f"  uplift  mean {s_up.mean():.6f} | min {s_up.min():.6f} | max {s_up.max():.6f} | negative share {(s_up < 0).mean():.4f}")

    # --- T-learner ---
    t = TLearner().fit(X_tr, t_tr, y_tr)
    tr_b, te_b = t_tr.astype(bool), t_te.astype(bool)
    auc_tr_tr = roc_auc_score(y_tr[tr_b], t.model_treated.predict_proba(X_tr[tr_b])[:, 1])
    auc_te_tr = roc_auc_score(y_te[te_b], t.model_treated.predict_proba(X_te[te_b])[:, 1])
    auc_tr_co = roc_auc_score(y_tr[~tr_b], t.model_control.predict_proba(X_tr[~tr_b])[:, 1])
    auc_te_co = roc_auc_score(y_te[~te_b], t.model_control.predict_proba(X_te[~te_b])[:, 1])
    t_up = t.predict_uplift(X_te)
    print("T-learner")
    print(f"  treated model AUC  train {auc_tr_tr:.4f} | test {auc_te_tr:.4f} | gap {auc_tr_tr - auc_te_tr:+.4f}")
    print(f"  control model AUC  train {auc_tr_co:.4f} | test {auc_te_co:.4f} | gap {auc_tr_co - auc_te_co:+.4f}")
    print(f"  uplift  mean {t_up.mean():.6f} | min {t_up.min():.6f} | max {t_up.max():.6f} | negative share {(t_up < 0).mean():.4f}")

    # --- KL-divergence tree ---
    print("KL-divergence tree (fitting; this can take several minutes)")
    k = KLTree().fit(X_tr, t_tr, y_tr)
    k_up = k.predict_uplift(X_te)
    print(f"  uplift  mean {k_up.mean():.6f} | min {k_up.min():.6f} | max {k_up.max():.6f} | negative share {(k_up < 0).mean():.4f}")

    # --- No-information baseline ---
    b = NoInformationBaseline().fit(X_tr, t_tr, y_tr)
    b_up = b.predict_uplift(X_te)
    print("No-information baseline")
    print(f"  uplift  constant {b_up[0]:.6f} for all {len(b_up):,} rows (flat expected uplift curve)")


if __name__ == "__main__":
    main()