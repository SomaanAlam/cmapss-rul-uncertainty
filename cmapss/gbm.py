"""Gradient-boosted quantile regression for RUL, with engine-grouped cross-fitting."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np

from .data import RUL_CAP

QUANTILES = (0.05, 0.5, 0.95)  # 90% central interval + median

PARAMS = dict(
    n_estimators=400,
    learning_rate=0.04,
    num_leaves=15,
    min_child_samples=100,   # rows within an engine are near-duplicates; keep leaves broad
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    n_jobs=4,
    verbose=-1,
)


class QuantileGBM:
    """One LightGBM quantile model started from an explicit interior value.

    Why not LightGBM's default start: it initialises a quantile model at the
    alpha-quantile of the labels. Here ~30% of labels sit exactly on the 125
    cap, so for alpha = 0.95 that start is 125 and the first tree finds no
    split at all - the model stays a constant 125 for every engine, however
    close to failure. Starting every model from the median of the labels
    avoids the dead start. Outputs are clipped to the target's support
    [0, RUL_CAP], which cannot hurt coverage.
    """

    def __init__(self, alpha, seed=0):
        self.model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, random_state=seed, **PARAMS)

    def fit(self, X, y):
        self.init = float(np.median(y))
        self.model.fit(X, y, init_score=np.full(len(y), self.init))
        return self

    def predict(self, X):
        return np.clip(self.model.predict(X, raw_score=True) + self.init, 0, RUL_CAP)


def fit_models(X, y, seed=0):
    return {q: QuantileGBM(q, seed).fit(X, y) for q in QUANTILES}


def predict(models, X):
    """Return an (n, 3) array of [q05, q50, q95], sorted to avoid crossing."""
    out = np.column_stack([models[q].predict(X) for q in QUANTILES])
    return np.sort(out, axis=1)


def cross_fit(X, y, fold_of_row, seed=0):
    """Out-of-fold predictions for every training row + the per-fold models.

    ``fold_of_row`` assigns every row to a fold by *engine*, so no engine is
    ever scored by a model that saw any of its cycles.
    """
    oof = np.zeros((len(X), len(QUANTILES)))
    models = []
    for f in np.unique(fold_of_row):
        tr, va = fold_of_row != f, fold_of_row == f
        m = fit_models(X[tr], y[tr], seed=seed + int(f))
        oof[va] = predict(m, X[va])
        models.append(m)
    return oof, models


def predict_ensemble(models, X):
    """Average the fold models' quantiles (cross-conformal style)."""
    return np.sort(np.mean([predict(m, X) for m in models], axis=0), axis=1)


def make_folds(units, k=5, seed=0):
    """Deterministic engine -> fold map."""
    rng = np.random.default_rng(seed)
    u = np.array(sorted(set(units)))
    rng.shuffle(u)
    return {int(unit): i % k for i, unit in enumerate(u)}
