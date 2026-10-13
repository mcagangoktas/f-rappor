"""
collision_depth_distribution.py
=================================
Sanity check: share of value pairs that have 0..h Bloom bits in common,
when all h hash functions use the SAME seed (pathological) versus
independent random seeds (the normal case).

With one shared seed every value sets a single bit, so fewer pairs overlap
at all, but every overlap is a complete collision: the two values become
indistinguishable. Independent seeds give more partial overlaps and almost
no complete ones.

Usage: python collision_depth_distribution.py [dataset ...]   (default: all datasets)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from datasets import load, selected
from rappor_client import bloom_matrix
from seed_selection import random_seeds

K_SIZE   = 128
NUM_HASH = 4


def depth_distribution(B):
    """Percentage of value pairs sharing exactly d bits, for d = 0..NUM_HASH."""
    B = B.astype(np.int32)
    shared = (B @ B.T)[np.triu_indices(len(B), k=1)]
    return 100 * np.bincount(shared, minlength=NUM_HASH + 1) / shared.size


if __name__ == "__main__":
    rng = np.random.default_rng(12345)
    for name in selected():
        domain, _ = load(name)
        same = depth_distribution(bloom_matrix(domain, [12345] * NUM_HASH, K_SIZE))
        indep = depth_distribution(bloom_matrix(domain, random_seeds(NUM_HASH, rng), K_SIZE))
        print(f"\n{name}: |D| = {len(domain)}")
        print(f"{'depth':>5} {'same seed %':>12} {'indep. %':>10}")
        for d in range(NUM_HASH + 1):
            print(f"{d:>5} {same[d]:>12.2f} {indep[d]:>10.2f}")
