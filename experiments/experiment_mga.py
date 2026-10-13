"""
experiment_mga.py
==================
Manipulation-Guided Attack (MGA): fake users (ATTACK_FRAC of the population)
join one cohort and send crafted reports with every bit of NUM_TARGETS target
values set to 1 (other bits are ordinary noise, 1 with probability p). The
damage is the Utility Loss

    ULoss = sum_v |f_attacked(v) - f_honest(v)| / |D|

measured for three cases, across several epsilons:
  - standard RAPPOR, fakes in the naturally lowest-PIS cohort
  - standard RAPPOR, fakes in the naturally highest-PIS cohort
  - Fair RAPPOR, averaged over fakes going into each of its cohorts

Usage: python experiment_mga.py [dataset ...]   (default: all datasets)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from datasets import load, report, save_rows, selected
from rappor_client import bloom_matrix, cohort_counts, derive_pq
from seed_selection import fair_seeds, pis, standard_seeds

K_SIZE      = 128
NUM_HASH    = 4
NUM_COHORTS = 64
NUM_TARGETS = 5
ATTACK_FRAC = 0.01
N_RUNS      = 10
BASE_SEED   = 10
EPSILONS    = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def estimate(counts, sizes, blooms, p, q):
    """Frequency of each value: mean debiased count over its bits, summed over cohorts."""
    t = (counts - p * sizes[:, None]) / (q - p)
    est = sum((B @ t[j]) / B.sum(axis=1) for j, B in enumerate(blooms))
    return est / sizes.sum()


def uloss(j, targets, counts, sizes, blooms, p, q, rng):
    """ULoss when n_attack fake reports join cohort j (honest reports unchanged)."""
    n_attack = int(sizes.sum() * ATTACK_FRAC)
    hit = blooms[j][targets].max(axis=0) == 1
    fake = np.where(hit, n_attack, rng.binomial(n_attack, p, size=hit.size))

    att_counts, att_sizes = counts.copy(), sizes.copy()
    att_counts[j] += fake
    att_sizes[j] += n_attack

    honest = estimate(counts, sizes, blooms, p, q)
    attacked = estimate(att_counts, att_sizes, blooms, p, q)
    return float(np.mean(np.abs(attacked - honest)))


def run(name):
    domain, idx = load(name)
    D = len(domain)
    hist = np.array([np.bincount(idx[j::NUM_COHORTS], minlength=D) for j in range(NUM_COHORTS)])
    sizes = hist.sum(axis=1).astype(float)

    rows = []
    for r in range(N_RUNS):
        rng = np.random.default_rng(BASE_SEED + r)
        std = [bloom_matrix(domain, s, K_SIZE) for s in standard_seeds(NUM_COHORTS, NUM_HASH, rng)]
        fair = [bloom_matrix(domain, s, K_SIZE) for s in fair_seeds(domain, NUM_COHORTS, NUM_HASH, K_SIZE, rng)]
        scores = [pis(B) for B in std]
        lo, hi = int(np.argmin(scores)), int(np.argmax(scores))
        targets = rng.choice(D, NUM_TARGETS, replace=False)

        for eps in EPSILONS:
            p, q = derive_pq(eps, NUM_HASH)
            c_std = cohort_counts(hist, std, p, q, rng)
            c_fair = cohort_counts(hist, fair, p, q, rng)
            rows.append({
                "run": r, "epsilon": eps,
                "uloss_low": uloss(lo, targets, c_std, sizes, std, p, q, rng),
                "uloss_high": uloss(hi, targets, c_std, sizes, std, p, q, rng),
                "uloss_fair": np.mean([
                    uloss(j, targets, c_fair, sizes, fair, p, q, rng) for j in range(NUM_COHORTS)
                ]),
            })

    print(f"\n{name}: |D| = {D}, N = {len(idx):,}")
    report(rows, ["uloss_low", "uloss_high", "uloss_fair"])
    print("saved", save_rows(f"mga_{name}", rows))


if __name__ == "__main__":
    for name in selected():
        run(name)
