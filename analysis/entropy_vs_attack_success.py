"""
entropy_vs_attack_success.py
==============================
Is seed entropy (how evenly a seed set spreads the domain over the bits)
a useful predictor of attack success and utility, as an alternative to PIS?
Compares the 10th, 50th and 90th entropy percentile out of N_CANDIDATES
random seed sets.

Usage: python entropy_vs_attack_success.py [dataset ...]   (default: all datasets)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))

from datasets import load, selected
from experiment_bia import bia_guess
from rappor_client import bloom_matrix, cohort_counts, derive_pq, perturb
from seed_selection import random_seeds, seed_entropy

K_SIZE       = 128
NUM_HASH     = 4
EPSILON      = 2.0
N_CANDIDATES = 2000
N_USERS      = 5000
N_RUNS       = 10


def asr_and_uloss(B, idx, p, q, rng):
    """BIA success rate and single-cohort estimation error on N_USERS sampled users."""
    users = rng.choice(idx, size=N_USERS)
    reports = perturb(B[users], p, q, rng).astype(np.int32)
    asr = np.mean(bia_guess(reports, B, rng) == users)

    hist = np.bincount(users, minlength=len(B))[None, :]
    t = (cohort_counts(hist, [B], p, q, rng)[0] - p * N_USERS) / (q - p)
    est = (B @ t) / B.sum(axis=1) / N_USERS
    return asr, np.mean(np.abs(est - hist[0] / N_USERS))


if __name__ == "__main__":
    p, q = derive_pq(EPSILON, NUM_HASH)
    for name in selected():
        domain, idx = load(name)
        rng = np.random.default_rng(42)
        pool = sorted((random_seeds(NUM_HASH, rng) for _ in range(N_CANDIDATES)),
                      key=lambda s: seed_entropy(domain, s, K_SIZE))

        print(f"\n{name}: |D| = {len(domain)}, max entropy = {np.log2(K_SIZE):.4f}")
        print(f"{'percentile':>10} {'entropy':>9} {'ASR':>17} {'ULoss':>17}")
        for pct in (10, 50, 90):
            seeds = pool[N_CANDIDATES * pct // 100]
            B = bloom_matrix(domain, seeds, K_SIZE)
            res = np.array([asr_and_uloss(B, idx, p, q, rng) for _ in range(N_RUNS)])
            m, s = res.mean(axis=0), res.std(axis=0)
            print(f"{pct:>10} {seed_entropy(domain, seeds, K_SIZE):>9.4f} "
                  f"{m[0]:>9.4f}±{s[0]:.4f} {m[1]:>9.5f}±{s[1]:.5f}")
