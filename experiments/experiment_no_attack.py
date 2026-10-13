"""
experiment_no_attack.py
========================
Cost of fairness with no attacker: standard vs Fair RAPPOR seeds, both
decoded with the full LASSO + OLS pipeline. Reports estimation accuracy
(L1 distance to the true distribution), how many values were detected,
and the PIS spread across cohorts.

Usage: python experiment_no_attack.py [dataset ...]   (default: all datasets)
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from datasets import load, report, save_rows, selected
from rappor_client import bloom_matrix, cohort_counts, derive_pq
from rappor_onetime_aggregator import rappor_onetime_aggregate
from seed_selection import fair_seeds, pis, standard_seeds

K_SIZE      = 64
NUM_HASH    = 2
NUM_COHORTS = 4
EPSILON     = math.log(3)
N_RUNS      = 10
BASE_SEED   = 42


def run(name):
    domain, idx = load(name)
    hist = np.array([np.bincount(idx[j::NUM_COHORTS], minlength=len(domain)) for j in range(NUM_COHORTS)])
    sizes = hist.sum(axis=1)
    true_freq = hist.sum(axis=0) / sizes.sum()
    p, q = derive_pq(EPSILON, NUM_HASH)

    rows = []
    for r in range(N_RUNS):
        rng = np.random.default_rng(BASE_SEED + r)
        strategies = {
            "standard": standard_seeds(NUM_COHORTS, NUM_HASH, rng),
            "fair": fair_seeds(domain, NUM_COHORTS, NUM_HASH, K_SIZE, rng),
        }
        for strategy, seeds in strategies.items():
            blooms = [bloom_matrix(domain, s, K_SIZE) for s in seeds]
            counts = cohort_counts(hist, blooms, p, q, rng)
            found = rappor_onetime_aggregate(counts, sizes, blooms, EPSILON, NUM_HASH)

            est = np.zeros(len(domain))
            est[list(found)] = list(found.values())
            scores = [pis(B) for B in blooms]
            rows.append({
                "run": r, "strategy": strategy,
                "l1": float(np.abs(true_freq - np.maximum(est, 0) / sizes.sum()).sum()),
                "detected": len(found),
                "pis_mean": float(np.mean(scores)),
                "pis_range": max(scores) - min(scores),
            })

    print(f"\n{name}: |D| = {len(domain)}, N = {len(idx):,}")
    report(rows, ["l1", "detected", "pis_mean", "pis_range"], by="strategy")
    print("saved", save_rows(f"no_attack_{name}", rows))


if __name__ == "__main__":
    for name in selected():
        run(name)
