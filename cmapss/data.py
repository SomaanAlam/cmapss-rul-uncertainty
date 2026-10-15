"""Loading and labelling the NASA C-MAPSS turbofan data.

Conventions used throughout the project
---------------------------------------
* ``rul`` is the true remaining useful life in cycles at the row's cycle:
  ``failure_time - cycle``. Train engines run to failure, so their failure
  time is their last recorded cycle. Test engines are truncated before
  failure; ``RUL_FDxxx.txt`` gives the true RUL at the last observed cycle,
  which also gives the true RUL at every earlier cycle of that engine.
* Rows with ``rul == 0`` (the failure cycle itself) are dropped: there is
  nothing left to predict there.
* ``rul_cap`` is ``min(rul, RUL_CAP)``. Early in life the degradation signal
  is indistinguishable from noise, so the usual practice (and ours) is to
  model the capped target.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RUL_CAP = 125
MIN_CYCLE = 30  # feature warm-up: the shortest test engine has 31 cycles

OPS = ["op1", "op2", "op3"]
COLS = ["unit", "cycle"] + OPS + [f"s{i}" for i in range(1, 22)]

# Sensors that carry degradation signal in FD001-FD004. The remaining seven
# (s1, s5, s6, s10, s16, s18, s19) are constant or near-constant.
SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11",
           "s12", "s13", "s14", "s15", "s17", "s20", "s21"]

N_CONDITIONS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS)


def load_subset(subset: str = "FD001", raw_dir: Path = RAW):
    """Return ``(train, test)`` frames with ``rul``/``rul_cap``/``fail_time``."""
    train = _read(raw_dir / f"train_{subset}.txt")
    test = _read(raw_dir / f"test_{subset}.txt")
    rul_last = pd.read_csv(raw_dir / f"RUL_{subset}.txt", header=None)[0].to_numpy()

    units = np.sort(test["unit"].unique())
    if len(units) != len(rul_last):
        raise ValueError("RUL file does not match the number of test engines")
    final_rul = pd.Series(rul_last, index=units)

    train["last_cycle"] = train.groupby("unit")["cycle"].transform("max")
    train["fail_time"] = train["last_cycle"]
    test["last_cycle"] = test.groupby("unit")["cycle"].transform("max")
    test["fail_time"] = test["last_cycle"] + test["unit"].map(final_rul)

    for df in (train, test):
        df["rul"] = df["fail_time"] - df["cycle"]
    train = train[train["rul"] > 0].reset_index(drop=True)
    test = test[test["rul"] > 0].reset_index(drop=True)
    for df in (train, test):
        df["rul_cap"] = np.minimum(df["rul"], RUL_CAP)
        df["observed_end"] = df["cycle"] == df["last_cycle"]
    return train, test


def normalise_conditions(train: pd.DataFrame, test: pd.DataFrame, n_conditions: int):
    """Standardise sensors within each operating condition (FD002/FD004).

    With six operating regimes the raw sensor values mostly reflect the regime,
    not the health of the engine. We cluster the three operational settings and
    z-score each sensor within its cluster, using training statistics only.
    """
    if n_conditions <= 1:
        return train, test
    from sklearn.cluster import KMeans

    train, test = train.copy(), test.copy()
    km = KMeans(n_clusters=n_conditions, n_init=10, random_state=0).fit(train[OPS])
    for df in (train, test):
        df["cond"] = km.predict(df[OPS])
    stats = train.groupby("cond")[SENSORS].agg(["mean", "std"])
    for df in (train, test):
        for s in SENSORS:
            mu = df["cond"].map(stats[(s, "mean")])
            sd = df["cond"].map(stats[(s, "std")]).replace(0, 1.0)
            df[s] = (df[s] - mu) / sd
    return train, test


def load_prepared(subset: str = "FD001"):
    train, test = load_subset(subset)
    return normalise_conditions(train, test, N_CONDITIONS[subset])
