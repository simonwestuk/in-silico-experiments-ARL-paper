"""Generate publication figures from results/*.json.

Writes vector PDFs to ../figures (the paper's figure directory) and PNG copies to
./figures. Colour palette is the Okabe-Ito colourblind-safe set.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS_DIR, FIG_DIR, ROOT

# ROOT is the code/ directory; the paper's figure directory is its sibling.
PAPER_FIG_DIR = os.path.join(os.path.dirname(ROOT), "figures")
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

# Okabe-Ito palette.
C = {"ARL": "#0072B2", "B1": "#D55E00", "B2": "#E69F00", "B3": "#009E73",
     "B5": "#CC79A7", "ARL_noAO": "#56B4E9"}
MK = {"ARL": "o", "B1": "s", "B2": "^", "B3": "D", "B5": "v", "ARL_noAO": "P"}
LBL = {"ARL": "ARL", "B1": "B1 direct-ML", "B2": "B2 rule-ITS", "B3": "B3 OLM-only",
       "B5": "B5 smoothed", "ARL_noAO": "ARL -- anti-osc"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 140})


def _load(name):
    with open(os.path.join(RESULTS_DIR, name)) as fh:
        return json.load(fh)


def _save(fig, stem):
    fig.tight_layout()
    fig.savefig(os.path.join(PAPER_FIG_DIR, stem + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, stem + ".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)


def fig_oscillation():
    d = _load("study1_oscillation.json")
    rows = d["rows"]
    sig = [r["sigma"] for r in rows]
    pols = ["ARL", "B1", "B2", "B5", "ARL_noAO"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    ax = axes[0]
    for p in pols:
        y = [r[f"{p}_mor"] for r in rows]
        e = [r[f"{p}_mor_ci"] for r in rows]
        ax.errorbar(sig, y, yerr=e, marker=MK[p], color=C[p], capsize=2, label=LBL[p], ms=5)
    ax.set_xlabel("perception noise $\\sigma$"); ax.set_ylabel("mode oscillation rate (per hour)")
    ax.set_title("(a) Oscillation"); ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for p in pols:
        y = [r[f"{p}_ipi"] for r in rows]
        ax.plot(sig, y, marker=MK[p], color=C[p], label=LBL[p], ms=5)
    ax.set_xlabel("perception noise $\\sigma$"); ax.set_ylabel("predictability IPI (1 = perfect)")
    ax.set_ylim(-0.03, 1.05); ax.set_title("(b) Predictability"); ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    for p in pols:
        y = [r[f"{p}_dens"] for r in rows]
        ax.plot(sig, y, marker=MK[p], color=C[p], label=LBL[p], ms=5)
    budget = d["design"]["budget_interventions"]
    ax.axhline(budget * 2, ls=":", color="grey",
               label=f"budget cap ({budget}/{d['design']['budget_reset_minutes']:.0f}min)")
    ax.set_xlabel("perception noise $\\sigma$"); ax.set_ylabel("intervention density (per hour)")
    ax.set_title("(c) Intervention density"); ax.legend(frameon=False, fontsize=8)
    _save(fig, "fig6_oscillation")


import matplotlib.ticker as mticker

def fig_tradeoff():
    d = _load("study3_sensitivity.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    # Panel A: cooldown-scale sweep -> boundedness/adaptivity trade-off
    ax = axes[0]
    sw = d["arl_sweep"]["cooldown_scale"]
    x = [r["value"] for r in sw]
    us = [r["under_support_rate"] for r in sw]
    dens = [r["density_per_h"] for r in sw]
    ln1 = ax.plot(x, us, marker="o", color="#D55E00", label="under-support rate")
    ax.set_xscale("log")
    ax.set_xlabel("cooldown scale ($\\times$ default)")
    # --- fix: explicit ticks at the sweep points, no minor labels ---
    ax.xaxis.set_major_locator(mticker.FixedLocator(x))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([f"{v:g}$\\times$" for v in x]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    # ---------------------------------------------------------------
    ax.set_ylabel("under-support rate", color="#D55E00"); ax.set_ylim(-0.03, 1.0)
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    ln2 = ax2.plot(x, dens, marker="s", color="#0072B2", label="intervention density")
    ax2.set_ylabel("intervention density (per hour)", color="#0072B2")
    ax.set_title("(a) ARL boundedness vs adaptivity")
    lns = ln1 + ln2
    ax.legend(lns, [l.get_label() for l in lns], frameon=False, fontsize=8, loc="center right")
    # Panel B: policy comparison (density vs under-support)
    ax = axes[1]
    pc = d["policy_comparison"]
    for p in ["ARL", "B1", "B2", "B5", "ARL_noAO"]:
        ax.scatter(pc[p]["density_per_h"], pc[p]["under_support_rate"],
                   color=C[p], marker=MK[p], s=70, label=LBL[p])
    ax.set_xlabel("intervention density (per hour)")
    ax.set_ylabel("under-support rate")
    ax.set_xscale("symlog")
    ax.set_title("(b) Policies under repeated need")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "fig10_tradeoff")


def fig_audit():
    d = _load("study2_audit.json")
    props = ["policy_basis", "decision_basis", "boundary", "non_action", "replay"]
    labels = ["policy\nbasis", "decision\nbasis", "boundary", "non-\naction", "replay"]
    suf = d["sufficiency_all_decisions"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    import numpy as np
    x = np.arange(len(props)); w = 0.2
    for i, p in enumerate(("ARL", "B1", "B2", "B3")):
        vals = [suf[p][prop] for prop in props]
        ax.bar(x + (i - 1.5) * w, vals, w, color=C[p], label=p)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("reconstructability (0--1)"); ax.set_ylim(0, 1.08)
    # Title on top, then the 4-entry legend as one row above the axes, so the
    # legend never overlaps the (full-height) bars.
    ax.set_title("(a) Per-property audit sufficiency (all decisions)", pad=26)
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), columnspacing=1.6, handletextpad=0.5)
    # Panel B: records-to-diagnose
    ax = axes[1]
    loc = d["localization"]
    pol = ["ARL", "B2", "B1"]
    vals = [loc[p]["mean_records_to_diagnose"] for p in pol]
    cols = [C[p] for p in pol]
    bars = ax.bar(pol, vals, color=cols)
    ax.set_ylim(0, max(vals) * 1.32)
    for b, p in zip(bars, pol):
        rk = loc[p]["reason_known_rate"]
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.02,
                f"{b.get_height():.0f} rec\nreason: {'yes' if rk > 0.5 else 'no'}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("records to diagnose an intervention")
    ax.set_title("(b) Localisation cost")
    _save(fig, "fig7_audit")


def fig_latency():
    d = _load("study5_latency_fairness.json")["latency"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    xc = [r["n_candidates"] for r in d["by_candidates"]]
    yc = [r["us_per_decision"] for r in d["by_candidates"]]
    ax.plot(xc, yc, marker="o", color=C["ARL"])
    ax.set_xlabel("number of candidate actions"); ax.set_ylabel("latency per decision ($\\mu$s)")
    ax.set_title("(a) Latency vs candidate-set size")
    ax = axes[1]
    xr = [r["n_routines"] for r in d["by_routines"]]
    yr = [r["us_per_decision"] for r in d["by_routines"]]
    ax.plot(xr, yr, marker="s", color=C["B2"])
    ax.set_ylim(0, max(yr) * 1.6)
    ax.set_xlabel("number of routines"); ax.set_ylabel("latency per decision ($\\mu$s)")
    ax.set_title("(b) Latency vs routine-set size")
    _save(fig, "fig8_latency")


def fig_fairness():
    d = _load("study5_latency_fairness.json")["fairness"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    import numpy as np
    groups = ["A (lower need)", "B (higher need)"]
    x = np.arange(2); w = 0.35
    arl = [d["ARL"]["A"]["density_mean"], d["ARL"]["B"]["density_mean"]]
    b1 = [d["B1"]["A"]["density_mean"], d["B1"]["B"]["density_mean"]]
    arl_w = [d["ARL"]["A"]["max_per_window"], d["ARL"]["B"]["max_per_window"]]
    b1_w = [d["B1"]["A"]["max_per_window"], d["B1"]["B"]["max_per_window"]]
    ba = ax.bar(x - w / 2, arl, w, color=C["ARL"], label="ARL")
    bb = ax.bar(x + w / 2, b1, w, color=C["B1"], label="B1 direct-ML")
    ax.set_ylim(0, max(b1) * 1.28)
    for b, mw in zip(ba, arl_w):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, f"max/win\n{mw}",
                ha="center", va="bottom", fontsize=8)
    for b, mw in zip(bb, b1_w):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, f"max/win\n{mw}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("intervention density (per hour)")
    cap = ax.axhline(d["budget_per_window"], ls=":", color="grey")
    # Legend as a single row of three (ARL, B1, budget cap), placed above the
    # axes so it never overlaps the bars (the figure caption serves as the title).
    ax.legend([ba, bb, cap],
              ["ARL", "B1 direct-ML", f"budget cap ({d['budget_per_window']}/window)"],
              frameon=False, fontsize=9, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), columnspacing=1.6, handletextpad=0.5)
    _save(fig, "fig9_fairness")


def main():
    fig_oscillation()
    fig_audit()
    fig_latency()
    fig_fairness()
    fig_tradeoff()
    print("Figures written to", PAPER_FIG_DIR, "and", FIG_DIR)


if __name__ == "__main__":
    main()
