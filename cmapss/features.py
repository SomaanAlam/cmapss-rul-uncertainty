"""Causal per-engine features (each row only uses that engine's past)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SENSORS

BASE_N = 10  # cycles used to estimate an engine's own healthy baseline


def _causal_baseline(v: pd.Series, n: int) -> pd.Series:
    """Running mean of the first ``n`` cycles, frozen afterwards."""
    e = v.expanding().mean()
    if len(e) > n:
        e.iloc[n:] = e.iloc[n - 1]
    return e


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append features to ``df`` (sorted by unit, cycle).

    Per sensor:
      ``_dev``    5-cycle rolling mean minus the engine's own early baseline
      ``_dev15``  15-cycle rolling mean minus the baseline
      ``_slope``  change of the 5-cycle mean over the last 20 cycles
      ``_sd``     15-cycle rolling standard deviation
    plus ``cycle`` (time in service).
    """
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    g = df.groupby("unit", sort=False)
    new = {}
    for s in SENSORS:
        base = g[s].transform(lambda v: _causal_baseline(v, BASE_N))
        sm5 = g[s].transform(lambda v: v.rolling(5, min_periods=1).mean())
        sm15 = g[s].transform(lambda v: v.rolling(15, min_periods=1).mean())
        lag = sm5.groupby(df["unit"]).shift(20)
        new[f"{s}_dev"] = sm5 - base
        new[f"{s}_dev15"] = sm15 - base
        new[f"{s}_slope"] = (sm5 - lag).fillna(0.0)
        new[f"{s}_sd"] = g[s].transform(lambda v: v.rolling(15, min_periods=2).std()).fillna(0.0)
    return pd.concat([df, pd.DataFrame(new)], axis=1)


def gbm_columns() -> list[str]:
    cols = ["cycle"]
    for s in SENSORS:
        cols += [f"{s}_dev", f"{s}_dev15", f"{s}_slope", f"{s}_sd"]
    return cols


def cox_columns() -> list[str]:
    """Compact covariate set for the (linear) Cox model."""
    return ["cycle"] + [f"{s}_dev" for s in SENSORS]
