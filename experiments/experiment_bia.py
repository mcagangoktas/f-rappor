"""
experiment_bia.py
==================
Bayesian Inference Attack (BIA): the attacker sees one report and knows the
cohort's seeds, and guesses the value that maximises the likelihood of that
report (uniform prior).

Out of NUM_COHORTS standard-RAPPOR cohorts, the naturally lowest- and
highest-PIS ones are attacked, across several epsilons. A clear ASR gap
between them would mean cohort assignment alone makes some users easier to
de-anonymise: a fairness problem.

Usage: python experiment_bia.py [dataset ...]   (default: all datasets)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from datasets import load, report, save_rows, selected
from rappor_client import bloom_matrix, derive_pq, perturb
from seed_selection import pis, standard_seeds

K_SIZE      = 128
NUM_HASH    = 4
NUM_COHORTS = 64
N_RUNS      = 10
N_USERS     = 5000
BASE_SEED   = 10
EPSILONS    = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def bia_guess(reports, B, rng):
    """
    Maximum-likelihood value for each report. With q = 1 - p every matching
    bit contributes log q and every mismatch log p, so the most likely value
    is the one whose Bloom row is closest in Hamming distance. Ties are
    broken uniformly at random by a sub-integer jitter.
    """
    dist = reports @ (1 - B).T.astype(np.int32) + (1 - reports) @ B.T.astype(np.int32)
    return np.argmin(dist + 0.5 * rng.random(dist.shape), axis=1)


def asr(B, users, epsilon, rng):
    p, q = derive_pq(epsilon, NUM_HASH)
    reports = perturb(B[users], p, q, rng).astype(np.int32)
    return float(np.mean(bia_guess(reports, B, rng) == users))


def run(name):
    domain, idx = load(name)
    rows = []
    for r in range(N_RUNS):
        rng = np.random.default_rng(BASE_SEED + r)
        blooms = [bloom_matrix(domain, s, K_SIZE) for s in standard_seeds(NUM_COHORTS, NUM_HASH, rng)]
        scores = [pis(B) for B in blooms]
        lo, hi = int(np.argmin(scores)), int(np.argmax(scores))

        for eps in EPSILONS:
            users = rng.choice(idx, size=N_USERS)
            a_lo, a_hi = asr(blooms[lo], users, eps, rng), asr(blooms[hi], users, eps, rng)
            rows.append({
                "run": r, "epsilon": eps, "pis_low": scores[lo], "pis_high": scores[hi],
                "asr_low": a_lo, "asr_high": a_hi, "gap": a_lo - a_hi,
            })

    print(f"\n{name}: |D| = {len(domain)}, random guess = {1 / len(domain):.4f}")
    report(rows, ["pis_low", "pis_high", "asr_low", "asr_high", "gap"])
    print("saved", save_rows(f"bia_{name}", rows))


if __name__ == "__main__":
    for name in selected():
        run(name)
