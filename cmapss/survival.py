"""Survival analysis: Kaplan-Meier curves and a landmark Cox model.

Why a *landmark* Cox model
--------------------------
A classical Cox model has one hazard per engine. Sensor readings change
continuously, so the covariates are time-varying, and the survival function of
a time-varying Cox model cannot be forecast without also forecasting the
sensors. The landmark approach (van Houwelingen, 2007) sidesteps that: at a
set of landmark cycles t0 it records each still-running engine's current
covariates and models the *residual* life T - t0 given them. Stacking all
landmarks gives one "super model" in which ``cycle`` (= t0) is itself a
covariate. For any engine at any cycle we then get a full conditional survival
curve S(u | x(t0)) = P(RUL > u), hence a median RUL and a prediction interval
read directly off the curve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from lifelines.utils import median_survival_times

from .data import MIN_CYCLE, RUL_CAP

TIME_GRID = np.arange(1, 601)  # residual-life grid, in cycles
PROBS = (0.05, 0.5, 0.95)


# --------------------------------------------------------------------------- #
# Kaplan-Meier
# --------------------------------------------------------------------------- #
def engine_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """One row per engine with the life information we would really have.

    ``obs_time``/``observed``: what a fleet operator sees today. Train engines
    ran to failure (event = 1); test engines are still running at their last
    cycle (right-censored, event = 0).
    ``true_time``: the true failure time (from the RUL file) - used only to
    check the censored analysis against the truth.
    """
    a = train.groupby("unit").agg(obs_time=("fail_time", "first"), true_time=("fail_time", "first"))
    a["observed"], a["set"] = 1, "train"
    b = test.groupby("unit").agg(obs_time=("last_cycle", "first"), true_time=("fail_time", "first"))
    b["observed"], b["set"] = 0, "test"
    b.index = b.index + 100_000  # keep train and test unit ids distinct
    return pd.concat([a, b])


def fleet_km(eng: pd.DataFrame) -> dict[str, KaplanMeierFitter]:
    """Three Kaplan-Meier fits of engine life."""
    out = {}
    tr = eng[eng.set == "train"]
    out["Train engines (all failed)"] = KaplanMeierFitter().fit(
        tr.obs_time, tr.observed, label="Train engines (all failed)")
    out["Train + test, test censored"] = KaplanMeierFitter().fit(
        eng.obs_time, eng.observed, label="Train + test, test censored")
    out["Train + test, true failure times"] = KaplanMeierFitter().fit(
        eng.true_time, np.ones(len(eng)), label="Train + test, true failure times")
    return out


def km_median_table(kms: dict[str, KaplanMeierFitter]) -> pd.DataFrame:
    rows = []
    for name, k in kms.items():
        ci = median_survival_times(k.confidence_interval_)
        rows.append({"curve": name, "median_life": float(k.median_survival_time_),
                     "ci_low": float(ci.iloc[0, 0]), "ci_high": float(ci.iloc[0, 1])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Landmark Cox model
# --------------------------------------------------------------------------- #
def survival_quantiles(S: np.ndarray, probs=PROBS, times=TIME_GRID) -> np.ndarray:
    """Quantiles of RUL from survival curves. ``S`` is (n_times, n_engines).

    The p-quantile is the first u with P(RUL > u) <= 1 - p. If the curve never
    gets that low on the grid, the grid maximum is returned.
    """
    out = np.empty((S.shape[1], len(probs)))
    for j, p in enumerate(probs):
        hit = S <= (1 - p)
        out[:, j] = times[np.where(hit.any(0), hit.argmax(0), len(times) - 1)]
    return out


class LandmarkCox:
    def __init__(self, cols, penalizer=0.1, step=5):
        self.cols, self.penalizer, self.step = list(cols), penalizer, step

    # -- training data ------------------------------------------------------ #
    def landmarks(self, df):
        m = (df.cycle >= MIN_CYCLE) & ((df.cycle - MIN_CYCLE) % self.step == 0)
        return df[m]

    def _z(self, df):
        return (df[self.cols] - self.mu) / self.sd

    def fit(self, df):
        lm = self.landmarks(df)
        X = lm[self.cols]
        self.mu, self.sd = X.mean(), X.std().replace(0, 1.0)
        d = self._z(lm).assign(duration=lm["rul"].to_numpy(), event=1, unit=lm["unit"].to_numpy())
        self.train_frame_ = d
        self.cph = CoxPHFitter(penalizer=self.penalizer)
        # robust (sandwich) errors clustered on engine: the stacked landmark
        # rows of one engine are not independent observations
        self.cph.fit(d, duration_col="duration", event_col="event", cluster_col="unit", robust=True)
        return self

    # -- prediction --------------------------------------------------------- #
    def survival(self, df) -> np.ndarray:
        return self.cph.predict_survival_function(self._z(df), times=TIME_GRID).to_numpy()

    def quantiles(self, df, probs=PROBS, chunk=4000) -> np.ndarray:
        parts = [survival_quantiles(self.survival(df.iloc[i:i + chunk]), probs)
                 for i in range(0, len(df), chunk)]
        return np.vstack(parts)

    def risk(self, df) -> np.ndarray:
        """Linear predictor x'beta (higher = failing sooner)."""
        return self.cph.predict_log_partial_hazard(self._z(df)).to_numpy()

    # -- reporting ---------------------------------------------------------- #
    def coef_table(self) -> pd.DataFrame:
        s = self.cph.summary
        return pd.DataFrame({
            "hazard_ratio_per_sd": s["exp(coef)"],
            "ci_low": s["exp(coef) lower 95%"],
            "ci_high": s["exp(coef) upper 95%"],
            "p": s["p"],
        }).sort_values("hazard_ratio_per_sd", ascending=False)

    def ph_check(self) -> pd.DataFrame | None:
        """Schoenfeld-residual test of proportional hazards (rank time)."""
        try:
            r = proportional_hazard_test(self.cph, self.train_frame_, time_transform="rank")
            return r.summary[["test_statistic", "p"]]
        except Exception as exc:  # pragma: no cover - diagnostic only
            print("PH test failed:", exc)
            return None


def choose_penalizer(df, fold_of_row, cols, grid=(0.01, 0.03, 0.1, 0.3, 1.0)):
    """Pick the Cox ridge penalty by engine-grouped CV on held-out partial
    log-likelihood."""
    rows = []
    for pen in grid:
        scores = []
        for f in np.unique(fold_of_row):
            m = LandmarkCox(cols, penalizer=pen).fit(df[fold_of_row != f])
            va = m.landmarks(df[fold_of_row == f])
            v = m._z(va).assign(duration=va["rul"].to_numpy(), event=1)
            scores.append(m.cph.score(v, scoring_method="log_likelihood"))
        rows.append({"penalizer": pen, "cv_partial_loglik": float(np.mean(scores))})
    t = pd.DataFrame(rows)
    return float(t.loc[t.cv_partial_loglik.idxmax(), "penalizer"]), t


def cox_cross_fit(df, fold_of_row, cols, penalizer):
    """Out-of-fold RUL quantiles for every training row + the fold models."""
    oof = np.zeros((len(df), len(PROBS)))
    models = []
    for f in np.unique(fold_of_row):
        va = fold_of_row == f
        m = LandmarkCox(cols, penalizer=penalizer).fit(df[~va])
        oof[va] = m.quantiles(df[va])
        models.append(m)
    return oof, models


def cox_predict_ensemble(models, df) -> np.ndarray:
    return np.sort(np.mean([m.quantiles(df) for m in models], axis=0), axis=1)


# --------------------------------------------------------------------------- #
# Held-out risk groups
# --------------------------------------------------------------------------- #
def risk_group_km(model: LandmarkCox, test: pd.DataFrame, landmark: int, n_groups=3):
    """Split test engines still running at ``landmark`` into risk groups by the
    Cox linear predictor and Kaplan-Meier their *true* remaining lives."""
    rows = test[test.cycle == landmark].copy()
    rows["risk"] = model.risk(rows)
    labels = ["Low risk", "Medium risk", "High risk"][:n_groups]
    rows["group"] = pd.qcut(rows["risk"], n_groups, labels=labels)
    fits = {}
    for g in labels:
        r = rows[rows.group == g]
        fits[g] = KaplanMeierFitter().fit(r["rul"], np.ones(len(r)), label=f"{g} (n={len(r)})")
    lr = multivariate_logrank_test(rows["rul"], rows["group"], np.ones(len(rows)))
    return fits, float(lr.p_value), rows
