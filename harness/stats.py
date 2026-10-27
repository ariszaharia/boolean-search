"""Paired comparisons of two systems, as fixed in hypotheses.md ("Setting", Statistics):

  * paired per-query differences (system A minus system B, on exactly the same queries);
  * 95% bootstrap confidence interval of the mean difference, 10,000 resamples (percentile method);
  * a two-sided bootstrap p-value, used only for the Holm correction over the confirmatory tests;
  * Holm correction;
  * the equivalence check of H7: the whole interval must lie inside plus or minus a margin M.

The p-value is the share of bootstrap means that lie at least as far from the observed mean as the
observed mean lies from 0 (the bootstrap distribution shifted to the null hypothesis "mean difference = 0").
The +1 in numerator and denominator keeps it above 0 (Efron and Tibshirani, An Introduction to the
Bootstrap, 1993, chapter 16).
"""
import numpy as np

N_RESAMPLES = 10_000
SEED = 20261015   # fixed, so every rerun gives the same interval


def paired_bootstrap(a, b, n_resamples=N_RESAMPLES, seed=SEED, confidence=0.95):
    """a, b: {query_id: value} for the two systems. Both must hold exactly the same queries.
    Returns {'n', 'mean_a', 'mean_b', 'mean_diff', 'ci_low', 'ci_high', 'p_value'}, diff = a - b."""
    if set(a) != set(b):
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        raise ValueError(f'the two systems are scored on different queries '
                         f'(only in A: {only_a[:5]}, only in B: {only_b[:5]})')
    if not a:
        raise ValueError('no queries to compare')

    queries = sorted(a)
    diffs = np.array([a[q] - b[q] for q in queries], dtype=float)
    n = len(diffs)
    observed = float(diffs.mean())

    rng = np.random.default_rng(seed)
    # Each row is one resample: n queries drawn with replacement. Their mean is one bootstrap mean.
    boot_means = diffs[rng.integers(0, n, size=(n_resamples, n))].mean(axis=1)

    tail = (1 - confidence) / 2
    ci_low, ci_high = np.quantile(boot_means, [tail, 1 - tail])
    at_least_as_far = np.sum(np.abs(boot_means - observed) >= abs(observed))
    p_value = (1 + at_least_as_far) / (1 + n_resamples)

    return {'n': n,
            'mean_a': float(np.mean([a[q] for q in queries])),
            'mean_b': float(np.mean([b[q] for q in queries])),
            'mean_diff': observed,
            'ci_low': float(ci_low), 'ci_high': float(ci_high),
            'p_value': float(p_value)}


def holm(p_values, alpha=0.05):
    """p_values: {test_name: p}. Returns {test_name: {'p_adjusted', 'reject'}}.
    Sort the m p-values from smallest; the i-th (i = 0, 1, ...) is multiplied by (m - i). An adjusted value is
    never smaller than the one before it, and never above 1. A test is rejected if its adjusted p <= alpha."""
    m = len(p_values)
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    result, running_max = {}, 0.0
    for i, (name, p) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (m - i) * p))
        result[name] = {'p_adjusted': running_max, 'reject': running_max <= alpha}
    return result


def equivalent(comparison, margin):
    """H7: True if the whole confidence interval of the difference lies inside [-margin, +margin]."""
    return -margin <= comparison['ci_low'] and comparison['ci_high'] <= margin
