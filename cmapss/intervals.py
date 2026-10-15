"""Conformal calibration of prediction intervals, plus evaluation metrics."""
from __future__ import annotations

import numpy as np

from .data import RUL_CAP


# --------------------------------------------------------------------------- #
# Conformalized quantile regression (Romano, Patterson & Candes, 2019)
# --------------------------------------------------------------------------- #
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))/n empirical quantile (finite-sample correct)."""
    n = len(scores)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(scores)[k - 1])


class CQR:
    """Widen/narrow a [lo, hi] interval so it reaches 1-alpha coverage.

    ``bin_edges=None`` gives one global correction. With ``bin_edges`` the
    correction is computed separately inside each bin of ``bin_var`` (a
    Mondrian / group-conditional version): coverage then holds within each
    bin, not just on average, which matters because RUL errors are much
    smaller near end of life than early in life.
    """

    def __init__(self, alpha: float = 0.1, bin_edges=None, min_bin: int = 200):
        self.alpha, self.bin_edges, self.min_bin = alpha, bin_edges, min_bin

    def _bin(self, v):
        return np.clip(np.digitize(v, self.bin_edges[1:-1]), 0, len(self.bin_edges) - 2)

    def fit(self, lo, hi, y, bin_var=None):
        score = np.maximum(lo - y, y - hi)
        self.global_q = conformal_quantile(score, self.alpha)
        self.q = None
        if self.bin_edges is not None:
            b = self._bin(bin_var)
            self.q = np.full(len(self.bin_edges) - 1, self.global_q)
            for k in range(len(self.q)):
                m = b == k
                if m.sum() >= self.min_bin:
                    self.q[k] = conformal_quantile(score[m], self.alpha)
        return self

    def predict(self, lo, hi, bin_var=None):
        q = self.global_q if self.q is None else self.q[self._bin(bin_var)]
        return np.clip(lo - q, 0, RUL_CAP), np.clip(hi + q, 0, RUL_CAP)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def mae(pred, true):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def nasa_score(pred, true):
    """Asymmetric PHM08 score: late predictions are penalised more than early."""
    d = np.asarray(pred, float) - np.asarray(true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def interval_score(y, lo, hi, alpha=0.1):
    """Winkler / interval score (lower is better); a proper scoring rule that
    rewards narrow intervals and penalises misses in proportion to the miss."""
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((hi - lo)
                         + (2 / alpha) * np.maximum(lo - y, 0)
                         + (2 / alpha) * np.maximum(y - hi, 0)))


def coverage(y, lo, hi):
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def unit_bootstrap(units, num, den=None, n_boot=2000, seed=0, root=False):
    """95% CI for sum(num)/sum(den) resampling *engines* (not rows), because
    rows from one engine are strongly dependent."""
    rng = np.random.default_rng(seed)
    codes, inv = np.unique(units, return_inverse=True)
    n = np.bincount(inv, weights=np.asarray(num, float), minlength=len(codes))
    d = (np.bincount(inv, minlength=len(codes)).astype(float) if den is None
         else np.bincount(inv, weights=np.asarray(den, float), minlength=len(codes)))
    idx = rng.integers(0, len(codes), size=(n_boot, len(codes)))
    ratio = n[idx].sum(1) / d[idx].sum(1)
    if root:
        ratio = np.sqrt(ratio)
    return float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))
