"""Figures. Palette: three categorical slots (blue, orange, aqua) validated
all-pairs for colour-vision deficiency; every series is also direct-labelled
or in a legend, so identity never relies on colour alone."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import RUL_CAP

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE_BAND, ORANGE_BAND = "#b7d3f6", "#f8cdb8"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": ["DejaVu Sans"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
    "axes.titleweight": "regular", "axes.titlelocation": "left", "legend.frameon": False,
    "lines.linewidth": 2,
})


def _save(fig, out, name):
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_km(ctx):
    kms, out = ctx["kms"], ctx["out"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))

    a = ax[0]
    spec = [("Train engines (all failed)", BLUE, "-"),
            ("Train + test, test censored", ORANGE, "-"),
            ("Train + test, true failure times", INK2, (0, (4, 3)))]
    for name, col, ls in spec:
        k = kms[name]
        k.plot_survival_function(ax=a, ci_show=(name != "Train + test, true failure times"),
                                 color=col, linestyle=ls, ci_alpha=0.15,
                                 show_censors=(name == "Train + test, test censored"),
                                 censor_styles={"ms": 5, "marker": "|", "mew": 1})
    a.set_title("Fleet survival: Kaplan–Meier with censored engines")
    a.set_xlabel("Cycles in service"); a.set_ylabel("P(engine still running)")
    a.set_ylim(0, 1.02)
    a.legend(loc="lower left", fontsize=9)
    a.text(0.98, 0.97, "ticks = engines still running\nat their last observed cycle (censored)",
           transform=a.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)

    b = ax[1]
    cols = {"Low": AQUA, "Medium": BLUE, "High": ORANGE}
    for lab, k in ctx["risk_fits"].items():
        c = cols[lab.split(" ")[0]]
        k.plot_survival_function(ax=b, ci_show=True, ci_alpha=0.12, color=c)
    b.set_title(f"Held-out test engines grouped by Cox risk at cycle {ctx['landmark']}")
    b.set_xlabel(f"Remaining cycles after cycle {ctx['landmark']}")
    b.set_ylabel("P(still running)"); b.set_ylim(0, 1.02)
    b.legend(loc="upper right", fontsize=9)
    p = ctx["risk_p"]
    b.text(0.02, 0.05, f"log-rank p = {p:.1e}", transform=b.transAxes, fontsize=9, color=INK2)
    fig.tight_layout()
    _save(fig, out, "fig_km.png")


# --------------------------------------------------------------------------- #
def fig_examples(ctx):
    """True RUL vs predicted median and 90% interval across an engine's life."""
    test, preds, out = ctx["test"], ctx["preds"], ctx["out"]
    glob = test.groupby("unit")["last_cycle"].first().sort_values()
    picks = [int(glob.index[int(len(glob) * q)]) for q in (0.1, 0.4, 0.7, 0.95)]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), sharey=True)
    rows = [("LightGBM + Mondrian CQR", "LightGBM + Mondrian CQR", ctx["te_g"][:, 1], BLUE, BLUE_BAND),
            ("Cox landmark + Mondrian CQR", "Cox landmark + Mondrian CQR", ctx["te_c"][:, 1], ORANGE, ORANGE_BAND)]
    for r, (title, key, med, col, band) in enumerate(rows):
        lo, hi = preds[key]
        for c, u in enumerate(picks):
            a = axes[r, c]
            m = (test.unit == u).to_numpy()
            cyc = test.cycle.to_numpy()[m]
            a.fill_between(cyc, lo[m], hi[m], color=band, alpha=0.9, lw=0, label="90% interval")
            a.plot(cyc, med[m], color=col, lw=1.8, label="Predicted median")
            a.plot(cyc, test.rul_cap.to_numpy()[m], color=INK, lw=1.3, ls=(0, (4, 2)), label="True RUL (cap 125)")
            a.set_ylim(0, RUL_CAP + 5)
            if r == 0:
                a.set_title(f"Engine {u} ({int(glob[u])} cycles seen)", fontsize=10)
            if r == 1:
                a.set_xlabel("Cycle")
            if c == 0:
                a.set_ylabel(f"{title.split(' + ')[0]}\nRUL (cycles)")
    axes[0, 0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Prediction intervals tighten as failure approaches (test engines, truncated before failure)",
                 x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    _save(fig, out, "fig_intervals_examples.png")


# --------------------------------------------------------------------------- #
def fig_last_cycle(ctx):
    """One interval per test engine at its last observed cycle, sorted by truth."""
    test, preds, out, last = ctx["test"], ctx["preds"], ctx["out"], ctx["last"]
    t = test[last].reset_index(drop=True)
    order = np.argsort(t.rul_cap.to_numpy(), kind="stable")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for a, (key, med, col, band) in zip(ax, [
            ("LightGBM + Mondrian CQR", ctx["te_g"][last, 1], BLUE, BLUE_BAND),
            ("Cox landmark + Mondrian CQR", ctx["te_c"][last, 1], ORANGE, ORANGE_BAND)]):
        lo, hi = preds[key][0][last], preds[key][1][last]
        x = np.arange(len(t))
        truth = t.rul_cap.to_numpy()[order]  # same capped truth the metrics use
        inside = (truth >= lo[order]) & (truth <= hi[order])
        a.vlines(x, lo[order], hi[order], color=band, lw=3.2, zorder=1)
        a.scatter(x, med[order], s=12, color=col, zorder=3, label="Predicted median")
        a.scatter(x[inside], truth[inside], s=14, facecolor=SURF, edgecolor=INK, lw=0.9, zorder=4, label="True RUL, inside interval")
        a.scatter(x[~inside], truth[~inside], s=30, marker="X", color=INK, zorder=5, label="True RUL, missed")
        a.axhline(RUL_CAP, color=MUTED, lw=0.8, ls=":")
        a.set_title(f"{key}: covers {inside.mean():.0%} of {len(t)} test engines")
        a.set_xlabel("Test engines, sorted by true RUL")
        a.set_xticks([])
    ax[0].set_ylabel("RUL at last observed cycle (true RUL capped at 125)")
    ax[0].legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    _save(fig, out, "fig_last_cycle.png")


# --------------------------------------------------------------------------- #
def fig_coverage(ctx):
    """Coverage and width by predicted-RUL bin, raw vs conformal."""
    res = {r["method"]: r for r in ctx["results"]}
    out = ctx["out"]
    bins = list(next(iter(res.values()))["coverage_by_pred_bin"].keys())
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    spec = [("LightGBM raw", BLUE, "--"), ("LightGBM + Mondrian CQR", BLUE, "-"),
            ("Cox landmark raw", ORANGE, "--"), ("Cox landmark + Mondrian CQR", ORANGE, "-")]
    x = np.arange(len(bins))
    for name, col, ls in spec:
        r = res[name]["coverage_by_pred_bin"]
        ax[0].plot(x, [r[b]["coverage"] for b in bins], color=col, ls=ls, marker="o", ms=5, label=name)
        ax[1].plot(x, [r[b]["width"] for b in bins], color=col, ls=ls, marker="o", ms=5, label=name)
    ax[0].axhline(1 - ctx["alpha"], color=INK, lw=1, ls=":")
    ax[0].text(2.5, 1 - ctx["alpha"] - 0.008, "target 90%", ha="center", va="top", fontsize=8.5, color=INK2)
    ax[0].set_ylim(0.4, 1.02)
    ax[0].set_title("Coverage by predicted median RUL (all test cycles)")
    ax[1].set_title("Mean interval width")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(bins)
        a.set_xlabel("Predicted median RUL bin (cycles)")
    ax[0].set_ylabel("Share of true RUL inside interval"); ax[1].set_ylabel("Width (cycles)")
    ax[0].legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    _save(fig, out, "fig_coverage.png")


# --------------------------------------------------------------------------- #
def fig_survival_curves(ctx):
    """Predicted survival function S(u) = P(RUL > u) for three test engines at their last cycle."""
    from . import survival as sv
    test, out, last, m = ctx["test"], ctx["out"], ctx["last"], ctx["cox_full"]
    t = test[last].sort_values("rul").reset_index(drop=True)
    sel = t.iloc[[int(len(t) * q) for q in (0.08, 0.5, 0.9)]]
    S = m.survival(sel)
    q = sv.survival_quantiles(S)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    for i, (a, (_, row)) in enumerate(zip(ax, sel.iterrows())):
        a.plot(sv.TIME_GRID, S[:, i], color=BLUE)
        a.axvspan(q[i, 0], q[i, 2], color=BLUE_BAND, alpha=0.6, lw=0)
        a.axvline(q[i, 1], color=BLUE, lw=1, ls=":")
        a.axvline(row["rul"], color=INK, lw=1.3, ls=(0, (4, 2)))
        a.set_xlim(0, max(q[i, 2] * 1.4, row["rul"] * 1.4, 40))
        a.set_title(f"Engine {int(row['unit'])} at cycle {int(row['cycle'])}")
        a.set_xlabel("Remaining cycles u")
        a.text(0.97, 0.95, f"true RUL {int(row['rul'])}\nmedian {q[i,1]:.0f}\n90% interval {q[i,0]:.0f}–{q[i,2]:.0f}",
               transform=a.transAxes, ha="right", va="top", fontsize=9, color=INK2)
    ax[0].set_ylabel("P(RUL > u)")
    fig.suptitle("Cox landmark model: each engine gets a full survival curve; dashed line = true RUL, shaded = 5%–95% span",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    _save(fig, out, "fig_survival_curves.png")


def make_all(ctx):
    for f in (fig_km, fig_examples, fig_last_cycle, fig_coverage, fig_survival_curves):
        f(ctx)
