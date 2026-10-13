"""
pis_definitions.py
===================
Why PIS counts *any* shared bit.

v1 (rejected): two values interfere only if all h hashed positions match
exactly. That happens with probability (1/k)^h per pair, so v1 is ~0 for
every seed set and cannot tell cohorts apart.

v2 (used, seed_selection.pis): two values interfere if they share at least
one bit position.

Usage: python pis_definitions.py [dataset ...]   (default: all datasets)
"""

import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from datasets import load, selected
from rappor_client import bit_indices, bloom_matrix
from seed_selection import pis, random_seeds

K_SIZE = 128


def pis_v1(domain, seeds, k):
    """Average number of other values with exactly the same ordered bit tuple."""
    groups = Counter(map(tuple, bit_indices(domain, seeds, k)))
    return sum(n * (n - 1) for n in groups.values()) / len(domain)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    for name in selected():
        domain, _ = load(name)
        print(f"\n{name}: |D| = {len(domain)}")
        print(f"{'h':>3} {'v1 expected':>12} {'v1 observed':>12} {'v2 observed':>12}")
        for h in [1, 2, 4, 8]:
            seeds = random_seeds(h, rng)
            print(f"{h:>3} {(len(domain) - 1) / K_SIZE**h:>12.2e} "
                  f"{pis_v1(domain, seeds, K_SIZE):>12.4f} {pis(bloom_matrix(domain, seeds, K_SIZE)):>12.2f}")
