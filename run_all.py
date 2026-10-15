"""End-to-end run: features -> RUL models -> conformal intervals -> survival -> report.

    python run_all.py                 # FD001 (default)
    python run_all.py --subset FD002  # six operating conditions
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from cmapss import gbm, intervals as iv, survival as sv
from cmapss.data import MIN_CYCLE, RUL_CAP, load_prepared
from cmapss.features import build_features, cox_columns, gbm_columns

warnings.filterwarnings("ignore")

ALPHA = 0.1  # 90% prediction intervals
MONDRIAN_BINS = [0, 25, 50, 75, 100, RUL_CAP + 1]  # bins of the *predicted* median RUL
TRUE_BUCKETS = [(1, 15), (15, 30), (30, 60), (60, 100), (100, 125), (125, 126)]
BUCKET_LABELS = ["<15", "15-30", "30-60", "60-100", "100-124", "125+ (capped)"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def evaluate_interval(name, y, lo, hi, units, last, pred_med, y_unc):
    """Interval metrics against the (capped) true RUL; ``y_unc`` is the
    uncapped truth, used only for one sensitivity number."""
    cov = ((y >= lo) & (y <= hi)).astype(float)
    wid = hi - lo
    r = {"method": name}
    # benchmark convention: one prediction per test engine, at its last cycle
    r["last_coverage"] = float(cov[last].mean())
    r["last_coverage_ci"] = iv.unit_bootstrap(units[last], cov[last])
    # intervals live on [0, 125], so engines whose true RUL exceeds 125 can never be covered
    r["last_coverage_vs_uncapped_truth"] = float(((y_unc >= lo) & (y_unc <= hi))[last].mean())
    r["last_n_true_rul_above_cap"] = int((y_unc[last] > RUL_CAP).sum())
    r["last_width"] = float(wid[last].mean())
    r["last_interval_score"] = iv.interval_score(y[last], lo[last], hi[last], ALPHA)
    # every test cycle (>= MIN_CYCLE): the true RUL of each is known
    r["all_coverage"] = float(cov.mean())
    r["all_coverage_ci"] = iv.unit_bootstrap(units, cov)
    r["all_width"] = float(wid.mean())
    r["all_width_ci"] = iv.unit_bootstrap(units, wid)
    r["all_interval_score"] = iv.interval_score(y, lo, hi, ALPHA)
    r["coverage_by_true_rul"] = {}
    for (a, b), lab in zip(TRUE_BUCKETS, BUCKET_LABELS):
        m = (y >= a) & (y < b)
        r["coverage_by_true_rul"][lab] = {
            "n": int(m.sum()), "coverage": float(cov[m].mean()), "width": float(wid[m].mean())}
    # coverage within bins of the *predicted* median - what the Mondrian
    # calibration targets (the true RUL is unknown at prediction time)
    r["coverage_by_pred_bin"] = {}
    for a, b in zip(MONDRIAN_BINS[:-1], MONDRIAN_BINS[1:]):
        m = (pred_med >= a) & (pred_med < b)
        r["coverage_by_pred_bin"][f"{a}-{min(b, RUL_CAP)}"] = {
            "n": int(m.sum()), "coverage": float(cov[m].mean()), "width": float(wid[m].mean())}
    return r


def main(subset: str, out_root: Path, quick: bool = False):
    out = out_root / subset
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    log(f"loading {subset}")
    train, test = load_prepared(subset)
    eng = sv.engine_table(train, test)
    train, test = build_features(train), build_features(test)
    train = train[train.cycle >= MIN_CYCLE].reset_index(drop=True)
    test = test[test.cycle >= MIN_CYCLE].reset_index(drop=True)
    gcols, ccols = gbm_columns(), cox_columns()
    fold = train.unit.map(gbm.make_folds(train.unit, k=5, seed=0)).to_numpy()
    y_tr = train.rul_cap.to_numpy(float)
    y_te = test.rul_cap.to_numpy(float)
    last = test.observed_end.to_numpy()
    units = test.unit.to_numpy()
    n_train, n_test = train.unit.nunique(), test.unit.nunique()
    log(f"train: {n_train} engines / {len(train)} rows; test: {n_test} engines / {len(test)} rows")

    # ------------------------------------------------- 1. gradient boosting
    log("LightGBM quantile models (5-fold, grouped by engine)")
    oof_g, models_g = gbm.cross_fit(train[gcols].to_numpy(), y_tr, fold)
    te_g = gbm.predict_ensemble(models_g, test[gcols].to_numpy())

    # ------------------------------------------------ 2. landmark Cox model
    log("Landmark Cox model: choosing ridge penalty by grouped CV")
    pen, pen_table = sv.choose_penalizer(train, fold, ccols)
    log(f"  penalizer = {pen}")
    log("Landmark Cox model: cross-fitting")
    oof_c, models_c = sv.cox_cross_fit(train, fold, ccols, pen)
    te_c = sv.cox_predict_ensemble(models_c, test)
    # RUL is capped at 125 in the target; quantiles of min(R, 125) are min(q, 125)
    oof_c, te_c = np.minimum(oof_c, RUL_CAP), np.minimum(te_c, RUL_CAP)
    cox_full = sv.LandmarkCox(ccols, penalizer=pen).fit(train)

    # --------------------------------------------- 3. conformal calibration
    log("Conformal calibration (CQR, global and Mondrian)")
    results, preds = [], {}
    for name, oof, tp in [("LightGBM", oof_g, te_g), ("Cox landmark", oof_c, te_c)]:
        variants = {f"{name} raw": (tp[:, 0], tp[:, 2])}
        glob = iv.CQR(ALPHA).fit(oof[:, 0], oof[:, 2], y_tr)
        variants[f"{name} + CQR"] = glob.predict(tp[:, 0], tp[:, 2])
        mond = iv.CQR(ALPHA, bin_edges=MONDRIAN_BINS).fit(oof[:, 0], oof[:, 2], y_tr, bin_var=oof[:, 1])
        variants[f"{name} + Mondrian CQR"] = mond.predict(tp[:, 0], tp[:, 2], bin_var=tp[:, 1])
        for vname, (lo, hi) in variants.items():
            preds[vname] = (lo, hi)
            results.append(evaluate_interval(vname, y_te, lo, hi, units, last, tp[:, 1],
                                             test.rul.to_numpy(float)))
        results[-3]["oof_coverage"] = iv.coverage(y_tr, oof[:, 0], oof[:, 2])
        results[-1]["mondrian_q"] = [float(v) for v in mond.q]
        results[-2]["global_q"] = float(glob.global_q)

    # ------------------------------------------------------- 4. point metrics
    pt = []
    for name, tp in [("LightGBM", te_g), ("Cox landmark", te_c)]:
        med = tp[:, 1]
        true_unc = test.rul.to_numpy(float)
        pt.append({
            "model": name,
            "last_rmse_vs_true_rul": iv.rmse(med[last], true_unc[last]),
            "last_mae_vs_true_rul": iv.mae(med[last], true_unc[last]),
            "last_nasa_score": iv.nasa_score(med[last], true_unc[last]),
            "last_rmse_vs_capped_rul": iv.rmse(med[last], y_te[last]),
            "last_c_index": float(concordance_index(true_unc[last], med[last])),
            "all_rmse_vs_capped_rul": iv.rmse(med, y_te),
            "all_rmse_ci": iv.unit_bootstrap(units, (med - y_te) ** 2, root=True),
            "oof_rmse_vs_capped_rul": iv.rmse((oof_g if name == "LightGBM" else oof_c)[:, 1], y_tr),
        })

    # ------------------------------------------------- 5. survival analysis
    log("Kaplan-Meier curves")
    kms = sv.fleet_km(eng)
    km_med = sv.km_median_table(kms)
    n_alive = {t0: int((test.cycle == t0).sum()) for t0 in (60, 80, 100, 120)}
    landmark = 100 if n_alive[100] >= 40 else 80
    risk_fits, risk_p, risk_rows = sv.risk_group_km(cox_full, test, landmark)
    coef = cox_full.coef_table()
    ph = cox_full.ph_check()

    # --------------------------------------------------------------- outputs
    log("writing outputs")
    summary = {
        "subset": subset, "alpha": ALPHA, "n_train_engines": n_train, "n_test_engines": n_test,
        "cox_penalizer": pen, "cox_penalizer_cv": pen_table.to_dict("records"),
        "point_metrics": pt, "interval_results": results,
        "km_medians": km_med.to_dict("records"),
        "risk_group_landmark": landmark, "risk_group_logrank_p": risk_p,
        "n_test_engines_alive_at": n_alive,
        "ph_test": None if ph is None else ph.round(4).reset_index().to_dict("records"),
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    coef.round(4).to_csv(out / "cox_hazard_ratios.csv")
    km_med.to_csv(out / "km_medians.csv", index=False)

    pr = test[["unit", "cycle", "rul", "rul_cap", "observed_end"]].copy()
    pr["lgbm_q50"] = te_g[:, 1]
    pr["cox_q50"] = te_c[:, 1]
    for vname, (lo, hi) in preds.items():
        key = vname.replace(" + ", "_").replace(" ", "_").lower()
        pr[f"{key}_lo"], pr[f"{key}_hi"] = lo, hi
    pr.to_csv(out / "test_predictions.csv", index=False)

    ctx = dict(train=train, test=test, eng=eng, kms=kms, km_med=km_med, risk_fits=risk_fits,
               risk_p=risk_p, landmark=landmark, cox_full=cox_full, preds=preds,
               te_g=te_g, te_c=te_c, results=results, pt=pt, y_te=y_te, last=last, coef=coef,
               out=out, alpha=ALPHA)
    if not quick:
        from cmapss import plots
        plots.make_all(ctx)
    return ctx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--out", default=str(Path(__file__).parent / "outputs"))
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    ctx = main(a.subset, Path(a.out), quick=a.no_plots)

    # ---- console summary
    print("\nPOINT PREDICTIONS (test engines, last observed cycle)")
    print(pd.DataFrame(ctx["pt"]).set_index("model")[
        ["last_rmse_vs_true_rul", "last_mae_vs_true_rul", "last_nasa_score", "last_c_index"]].round(2))
    print("\n90% PREDICTION INTERVALS (truth = RUL capped at 125)")
    rows = []
    for r in ctx["results"]:
        rows.append({"method": r["method"],
                     "cov@last": f'{r["last_coverage"]:.2f}', "width@last": f'{r["last_width"]:.1f}',
                     "cov@all": f'{r["all_coverage"]:.3f}', "width@all": f'{r["all_width"]:.1f}',
                     "int.score@all": f'{r["all_interval_score"]:.1f}'})
    print(pd.DataFrame(rows).set_index("method").to_string())
    print("\nKM medians\n", ctx["km_med"].round(1).to_string(index=False))
    print(f"\nrisk-group log-rank p at landmark {ctx['landmark']}: {ctx['risk_p']:.2e}")
