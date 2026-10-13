"""
seed_selection.py
==================
How cohort hash seeds are chosen, and the two metrics used to ask whether
that choice makes some cohorts structurally easier to attack than others.
"""

import numpy as np

from rappor_client import bit_indices, bloom_matrix


def random_seeds(h, rng):
    return [int(s) for s in rng.integers(0, 2**32, size=h)]


def seed_entropy(domain, seeds, k):
    """
    Shannon entropy (bits) of how the (value, seed) hashes spread over the
    k positions. The maximum, log2(k), means perfectly even spreading.
    """
    counts = np.bincount(bit_indices(domain, seeds, k).ravel(), minlength=k)
    probs = counts[counts > 0] / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def pis(B):
    """
    Preimage Interference Score of one cohort, from its Bloom matrix B:
    for each value, the number of *other* values sharing at least one bit
    with it, averaged over the domain.

    Low PIS: values sit on more exclusive bits, a report is more identifying.
    High PIS: more bit sharing, more ambiguity for an attacker.
    """
    B = B.astype(np.int32)
    shares = (B @ B.T) > 0
    return float(shares.sum(axis=1).mean() - 1)


def standard_seeds(m, h, rng):
    """Standard RAPPOR: every cohort gets independent random seeds."""
    return [random_seeds(h, rng) for _ in range(m)]


def fair_seeds(domain, m, h, k, rng, pool_size=1000, band=(10, 90)):
    """
    Candidate Fair RAPPOR: draw `pool_size` random seed sets, keep those
    whose PIS lies within the `band` percentiles of the pool, and spread the
    m cohorts evenly across that band so no cohort sits at either extreme.
    """
    pool = [random_seeds(h, rng) for _ in range(pool_size)]
    pool.sort(key=lambda s: pis(bloom_matrix(domain, s, k)))

    lo, hi = (pool_size * b // 100 for b in band)
    kept = pool[lo:hi] if hi - lo >= m else pool
    return [kept[i] for i in np.linspace(0, len(kept) - 1, m).astype(int)]
