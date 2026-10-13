"""
plot_results.py
================
Draws one figure per experiment from the CSVs in results/, with one panel
per dataset (mean line, ±1 std band).

Usage: python plot_results.py
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from datasets import DATASETS, RESULTS_DIR

COLORS = {"low": "#eb6834", "high": "#2a78d6", "fair": "#1baf7a"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "font.size": 10})


def read(name):
    path = os.path.join(RESULTS_DIR, f"{name}.csv")
    with open(path) as f:
        return list(csv.DictReader(f))


def panels(title):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig, axes.ravel()


def line_figure(exp, series, ylabel, title):
    fig, axes = panels(title)
    for ax, name in zip(axes, DATASETS):
        rows = read(f"{exp}_{name}")
        eps = sorted({float(r["epsilon"]) for r in rows})
        for col, (key, label) in series.items():
            vals = np.array([[float(r[col]) for r in rows if float(r["epsilon"]) == e] for e in eps])
            m, s = vals.mean(axis=1), vals.std(axis=1)
            ax.plot(eps, m, marker="o", markersize=5, linewidth=2, color=COLORS[key], label=label)
            ax.fill_between(eps, m - s, m + s, color=COLORS[key], alpha=0.15, linewidth=0)
        ax.set_title(name)
        ax.set_xlabel("ε")
        ax.set_ylabel(ylabel)
    axes[0].legend(frameon=False)
    out = os.path.join(RESULTS_DIR, f"{exp}.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


def bar_figure():
    fig, axes = panels("No attack: L1 error of the LASSO/OLS estimate (lower is better)")
    for ax, name in zip(axes, DATASETS):
        rows = read(f"no_attack_{name}")
        for i, strat in enumerate(["standard", "fair"]):
            vals = [float(r["l1"]) for r in rows if r["strategy"] == strat]
            ax.bar(i, np.mean(vals), yerr=np.std(vals), width=0.6, capsize=4,
                   color=COLORS["high" if strat == "standard" else "fair"])
        ax.set_xticks([0, 1], ["Standard", "Fair"])
        ax.set_title(name)
        ax.set_ylabel("L1 distance")
    out = os.path.join(RESULTS_DIR, "no_attack.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


def hist_figure():
    fig, axes = panels("Per-value BIA success under one random seed set (ε = 10, h = 8)")
    for ax, name in zip(axes, DATASETS):
        asr = [float(r["asr"]) for r in read(f"vulnerability_{name}")]
        ax.hist(asr, bins=25, color=COLORS["high"], edgecolor="white", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("ASR of a value")
        ax.set_ylabel("values")
    out = os.path.join(RESULTS_DIR, "vulnerability.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    line_figure("bia", {"asr_low": ("low", "Lowest-PIS cohort"), "asr_high": ("high", "Highest-PIS cohort")},
                "Attack success rate", "BIA: attack success on the lowest- vs highest-PIS cohort")
    line_figure("mga", {"uloss_low": ("low", "Standard, lowest-PIS cohort"),
                        "uloss_high": ("high", "Standard, highest-PIS cohort"),
                        "uloss_fair": ("fair", "Fair RAPPOR, mean of cohorts")},
                "Utility loss", "MGA: utility loss from 1% fake users")
    bar_figure()
    hist_figure()
