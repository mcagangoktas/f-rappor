"""
rappor_onetime_aggregator.py
=============================
RAPPOR decoding for one-time reports, following the original paper:

  1. Debias per-cohort bit counts with p/q derived from epsilon.
  2. Design matrix: column s marks the bits candidate s sets in every cohort.
  3. LASSO (non-negative) selects candidate strings.
  4. OLS on the selected strings gives counts, standard errors and p-values;
     a string is reported if it passes a Bonferroni-corrected test.
"""

import numpy as np
from scipy import stats
from sklearn.linear_model import LassoCV

from rappor_client import derive_pq


def rappor_onetime_aggregate(counts, cohort_sizes, blooms, epsilon, h, alpha=0.05, cv=5):
    """
    counts[j, i]   : number of reports in cohort j with bit i set
    cohort_sizes[j]: number of reports in cohort j
    blooms[j]      : Bloom matrix of the domain under cohort j's seeds

    Returns {domain index: estimated total count} for significant strings.
    Assumes cohorts of (roughly) equal size, as in the RAPPOR paper.
    """
    p, q = derive_pq(epsilon, h)
    M = blooms[0].shape[0]

    # Negative debiased counts are pure noise; left in, they push the
    # non-negative LASSO towards selecting nothing.
    t = np.maximum((counts - p * cohort_sizes[:, None]) / (q - p), 0.0)

    active = cohort_sizes > 0
    Y = t[active].ravel()
    X = np.vstack([B.T for B, a in zip(blooms, active) if a]).astype(float)

    lasso = LassoCV(cv=cv, fit_intercept=False, positive=True, max_iter=10_000, n_jobs=-1)
    sel = np.flatnonzero(lasso.fit(X, Y).coef_ > 0)
    if sel.size == 0:
        return {}

    Xs = X[:, sel]
    beta = np.linalg.lstsq(Xs, Y, rcond=None)[0]
    sigma2 = np.sum((Y - Xs @ beta) ** 2) / max(len(Y) - sel.size, 1)
    se = np.sqrt(np.maximum(np.diag(sigma2 * np.linalg.pinv(Xs.T @ Xs)), 0.0))

    z = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    pvals = 2 * stats.norm.sf(np.abs(z))

    # beta is a per-cohort count; the total is that times the active cohorts.
    m = active.sum()
    return {int(s): float(b * m) for s, b, pv in zip(sel, beta, pvals) if pv < alpha / M}
