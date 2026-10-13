"""
rappor_client.py
=================
Client-side primitives of basic one-time RAPPOR (Bloom filter + per-bit
randomized response, no permanent secret).

A value is encoded as a row of a Bloom matrix B (shape |domain| x k): B[v, i]
is 1 iff value v hashes to bit i under one of the cohort's seeds. Everything
downstream (perturbation, attacks, estimation) works on these matrices, so
each value is hashed once per seed set instead of once per report.
"""

import math

import numpy as np
import xxhash


def derive_pq(epsilon, h):
    """
    p = P(report 1 | bit 0), q = P(report 1 | bit 1) = 1 - p.
    A value flips h bits, so epsilon = 2h * ln(q / p).
    """
    p = 1 / (1 + math.exp(epsilon / (2 * h)))
    return p, 1 - p


def bit_indices(domain, seeds, k):
    """(|domain|, h) array: the bit each seed maps each value to."""
    return np.array([
        [xxhash.xxh32(str(v).encode(), seed=int(s)).intdigest() % k for s in seeds]
        for v in domain
    ])


def bloom_matrix(domain, seeds, k):
    """(|domain|, k) 0/1 matrix of every value's true Bloom filter."""
    idx = bit_indices(domain, seeds, k)
    B = np.zeros((len(domain), k), dtype=np.int8)
    np.put_along_axis(B, idx, 1, axis=1)
    return B


def perturb(rows, p, q, rng):
    """Randomized response on true Bloom rows -> reports actually sent."""
    probs = np.where(rows == 1, q, p)
    return (rng.random(rows.shape) < probs).astype(np.int8)


def cohort_counts(hist, blooms, p, q, rng):
    """
    Aggregate per-cohort bit counts without materialising every report.

    hist[j, v] is the number of users in cohort j holding value v. Each bit
    of each report is an independent Bernoulli draw, so the count of 1s on
    bit i of cohort j is Binomial(n1, q) + Binomial(N_j - n1, p), where n1
    is the number of those users whose true bit i is 1. This is the same
    distribution as summing perturb() over all reports.
    """
    counts = np.empty((len(blooms), blooms[0].shape[1]))
    for j, B in enumerate(blooms):
        n1 = hist[j] @ B
        counts[j] = rng.binomial(n1, q) + rng.binomial(hist[j].sum() - n1, p)
    return counts
