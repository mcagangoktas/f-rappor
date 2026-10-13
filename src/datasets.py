"""
datasets.py
============
Loading the six datasets (one integer-coded value per line, one user each,
see datasets/README.md) and saving / summarising experiment results.
"""

import csv
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_DIR = os.path.join(ROOT, "datasets")
RESULTS_DIR = os.path.join(ROOT, "results")

DATASETS = {
    "adult":    "adult.csv",
    "bms":      "BMS.csv",
    "kosarak":  "kosarak.csv",
    "gaussian": "gaussian_100000.csv",
    "uniform":  "uniform.csv",
    "power":    "power.csv",
}


def load(name):
    """Returns (domain, idx): sorted unique values and each user's index into it."""
    values = np.loadtxt(os.path.join(DATA_DIR, DATASETS[name]), dtype=np.int64)
    return np.unique(values, return_inverse=True)


def selected():
    """Dataset names given on the command line, or all of them."""
    return sys.argv[1:] or list(DATASETS)



def save_rows(name, rows):
    """Writes a list of dicts to results/<name>.csv and returns the path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def report(rows, keys, by="epsilon"):
    """Prints mean ± std of `keys` for every value of `by`."""
    print(f"{by:>8}" + "".join(f"{k:>22}" for k in keys))
    for g in dict.fromkeys(r[by] for r in rows):
        sub = [r for r in rows if r[by] == g]
        cells = [np.array([r[k] for r in sub]) for k in keys]
        print(f"{g:>8}" + "".join(f"{c.mean():>13.5f}±{c.std():<8.5f}" for c in cells))
