# Turbofan remaining useful life with uncertainty (NASA C-MAPSS)

Predict how many cycles a jet engine has left, and say how sure we are: every
prediction is a **90% prediction interval**, not a single number. Two model
families are compared:

| Family | What it gives | Uncertainty |
|---|---|---|
| **LightGBM quantile regression** on engineered sensor features | sharp point estimate (median) | 5%/95% quantile models, then **conformal calibration** (CQR) |
| **Survival analysis**: Kaplan–Meier + a **landmark Cox model** | a full survival curve P(RUL > u) per engine | interval read off the curve, optionally conformalised |

Everything runs with `python run_all.py` (about 40 s for FD001, 100 s for FD002).

## Results (FD001, 100 held-out test engines)

**Point prediction at each engine's last observed cycle**

| Model | RMSE | MAE | NASA score | c-index |
|---|---|---|---|---|
| LightGBM (median) | 12.9 | 9.9 | 240 | 0.87 |
| Cox landmark (median) | 15.2 | 11.7 | 433 | 0.86 |

**90% prediction intervals** (truth = RUL capped at 125; width in cycles)

| Method | Coverage, last cycle (n=100) | Width | Coverage, all test cycles | Width | Interval score (lower = better) |
|---|---|---|---|---|---|
| LightGBM raw quantiles | 0.73 | 29 | 0.66 | 26 | 46.7 |
| LightGBM + CQR | 0.89 | 40 | 0.93 | 34 | 41.2 |
| LightGBM + Mondrian CQR | 0.89 | 40 | 0.93 | 33 | 41.2 |
| Cox landmark (raw) | 0.96 | 57 | 0.99 | 63 | 63.1 |
| Cox landmark + Mondrian CQR | 0.96 | 56 | 0.99 | 62 | 62.7 |

What to take from this:

* **Raw quantile regression is overconfident**: its "90%" intervals cover 73% of
  last-cycle truths (66% over all test cycles). Conformal calibration repairs that
  with only ~10 cycles of extra width. That is the main reason to calibrate.
* LightGBM + conformal gives the best intervals: about as many hits as nominal,
  and narrower than Cox. The Cox model is **conservative** (96% at last cycle, much
  wider), and conformal cannot tighten it, because ~30% of calibration targets sit
  exactly on the 125 cap where the conformity score is exactly 0.
* The last-cycle coverage of 0.89 comes from only 100 engines: its 95% bootstrap
  interval is **0.83–0.95**. The all-cycles figure (10,196 rows, bootstrap over
  engines) is 0.93 for the plain CQR variant (0.90–0.96); it exceeds 90% partly because about half
  of those rows are early-life rows with RUL = 125 that are easy to cover.
* Coverage conditional on the *true* RUL is not uniform (Mondrian CQR, all test
  cycles): 1.00 for RUL < 15, 0.93 for 15–30, 0.90 for 30–60, **0.86 for 60–100**,
  0.90 for 100–124, 0.96 at the cap. Conformal prediction guarantees marginal (and,
  with Mondrian bins, predicted-bin) coverage, not coverage for every true RUL.

### Survival analysis findings

* **Kaplan–Meier** (`fig_km.png`, left). Median engine life is 199 cycles (95% CI
  192–207) from the 100 run-to-failure train engines. Treating the 100 test engines
  as right-censored at their last observed cycle and pooling gives 202 (198–213),
  close to the 199 (195–206) obtained with the true failure times from the RUL file,
  so the censored analysis recovers the truth within its interval.
* **Cox landmark model.** Hazard ratios per standard deviation of sensor drift are
  largest for s11 (1.81, CI 1.35–2.43), s4 (1.59), s15 (1.42) and s9 (1.41); see
  `outputs/FD001/cox_hazard_ratios.csv`. Standard errors are cluster-robust by engine.
  Grouping held-out test engines by their Cox risk at cycle 100 separates their true
  remaining-life curves clearly (`fig_km.png`, right; log-rank p = 9e-11, n = 70).
* **Proportional hazards**: the Schoenfeld test flags only `cycle` (p = 0.009);
  the sensor covariates pass. The model is therefore an approximation in how it
  treats time in service.

### Second dataset (FD002, six operating conditions)

Same code, `python run_all.py --subset FD002` (sensors are z-scored within each
operating condition first). It is a much harder problem: last-cycle RMSE 23.6
(LightGBM) and 25.0 (Cox). The calibration story holds: raw LightGBM intervals
cover 82% at last cycle and 71% over all cycles, LightGBM + Mondrian CQR covers
92% and 93%. Outputs are in `outputs/FD002/`.

## Method

1. **Data.** Train engines run to failure; test engines stop early and the RUL file
   gives the true RUL at the last cycle, hence at every earlier cycle of that
   engine. This lets us evaluate on every test cycle, not just the last one. The
   target is capped at 125 cycles (standard practice); rows at the failure cycle
   are dropped; rows before cycle 30 are used only as feature warm-up.
2. **Features** (causal, per engine, 14 informative sensors): 5- and 15-cycle
   rolling means minus the engine's own early baseline, 20-cycle slope, rolling
   standard deviation, and cycle count.
3. **LightGBM**: separate 5%, 50%, 95% quantile models, 5-fold cross-fitting grouped
   **by engine** so no engine is scored by a model that saw its own cycles.
4. **Conformalized quantile regression** (Romano et al., 2019) calibrated on the
   out-of-fold predictions for all 100 training engines (cross-conformal; test
   predictions average the five fold models). The Mondrian variant calibrates
   separately within bins of predicted median RUL.
5. **Survival.** A classical Cox model cannot forecast survival when sensors keep
   changing, so I use a *landmark* Cox model: at every 5th cycle each running
   engine contributes its current covariates and its residual life; stacking these
   gives one model whose `cycle` covariate is the landmark time. For any engine
   the model returns S(u) = P(RUL > u); the 5%/50%/95% crossing points are the
   interval and median. Ridge penalty chosen by engine-grouped CV.
6. **Evaluation**: coverage and width with bootstrap CIs resampling *engines*,
   interval (Winkler) score, RMSE, NASA asymmetric score, c-index.

## Limitations (please read before trusting a number)

* **Intervals live on [0, 125].** True RULs above the cap can never be covered.
  11 of the 100 FD001 test engines have true RUL > 125 at their last cycle, so
  against the *uncapped* truth, last-cycle coverage is 80% for LightGBM + CQR
  (85–87% for Cox), not 89%. The tables use the capped truth throughout.
* Conformal coverage assumes calibration and test rows are exchangeable. Test
  engines are truncated at random points, which differs from calibration rows
  that span whole lives; the marginal numbers above are measured, not guaranteed.
* 100 test engines is a small sample for coverage claims (±6 points at last cycle).
* C-MAPSS is **simulated** data, and FD001 has one fault mode and one operating
  condition. Real fleets add censoring, maintenance interventions and sensor drift.
* Model hyperparameters were checked only on out-of-fold training predictions
  (never on the test set); I did not tune further.
* A bug worth knowing about: LightGBM's default start for a 95% quantile model
  sits on the 125 cap here, so the first tree finds no split and the model is a
  constant 125. `cmapss/gbm.py` starts every quantile model from the label median
  instead and explains why.

## Layout

```
run_all.py            one command: train, calibrate, evaluate, plot
cmapss/data.py        loading, RUL labels, operating-condition normalisation
cmapss/features.py    causal rolling features
cmapss/gbm.py         LightGBM quantile models + grouped cross-fitting
cmapss/intervals.py   CQR / Mondrian CQR, metrics, engine-level bootstrap
cmapss/survival.py    Kaplan–Meier, landmark Cox, risk groups, PH check
cmapss/plots.py       figures
data/raw/             FD001 and FD002 files (see below)
outputs/FD001/        results.json, hazard ratios, KM medians, test predictions, figures
```

Figures: `fig_km.png`, `fig_intervals_examples.png`, `fig_last_cycle.png`,
`fig_coverage.png`, `fig_survival_curves.png`.

## Setup

```
pip install -r requirements.txt
python run_all.py
```

On Python 3.13 with a Debian-patched setuptools, `lifelines` may fail to build its
`autograd-gamma` dependency; installing inside a fresh virtual environment
(`python -m venv .venv`, then upgrade `setuptools`) fixed it for me.

**Data provenance.** The files are NASA's Turbofan Engine Degradation Simulation
Data Set (C-MAPSS) from the NASA Prognostics Center of Excellence repository. My
workspace could not reach NASA's host, so I copied FD001 and FD002 from a public
GitHub mirror (`hankroark/Turbofan-Engine-Degradation`). Row counts match the
published sizes (FD001 train/test: 20,631 / 13,096 rows; FD002: 53,759 / 33,991).
If you want to be certain, replace `data/raw/*.txt` with the files from NASA's zip.
