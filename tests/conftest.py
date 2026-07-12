"""Naive reference implementations, translated literally from the R sources.

These follow the double loop of ``src/strat_cpp.cpp`` and the formulas of
``R/utils.R`` with plain Python loops, independently of the vectorized
package code.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


def naive_wtd_rank(x, weights):
    """Weighted midranks: r_k = cumsum(freqs)_k - 0.5 * (freqs_k - 1)."""
    x = [float(v) for v in x]
    w = [float(v) for v in weights]
    scale = len(x) / sum(w)  # normwt = TRUE
    w = [v * scale for v in w]
    uniq = sorted(set(x))
    freqs = {u: 0.0 for u in uniq}
    for xi, wi in zip(x, w, strict=True):
        freqs[xi] += wi
    ranks = {}
    csum = 0.0
    for u in uniq:
        csum += freqs[u]
        ranks[u] = csum - 0.5 * (freqs[u] - 1.0)
    return np.array([ranks[xi] for xi in x])


def naive_pair_sums(y, r, w):
    """Literal translation of strat_cpp: all pairs i < j of r-sorted input."""
    n = len(y)
    deno = nume = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if r[j] <= r[i] or y[j] == y[i]:
                continue
            wij = w[i] * w[j]
            deno += wij
            nume += math.copysign(1.0, y[j] - y[i]) * wij
    return deno, nume


def naive_pair_sums_by(y, r, w, c, n_groups):
    """Literal translation of strat_cpp_by."""
    n = len(y)
    deno_by = [0.0] * n_groups
    nume_by = [0.0] * n_groups
    deno_between = nume_between = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if r[j] <= r[i] or y[j] == y[i]:
                continue
            wij = w[i] * w[j]
            s = math.copysign(1.0, y[j] - y[i]) * wij
            if c[i] == c[j]:
                deno_by[c[i]] += wij
                nume_by[c[i]] += s
            else:
                deno_between += wij
                nume_between += s
    return {
        "deno_by": np.array(deno_by),
        "nume_by": np.array(nume_by),
        "deno_within": sum(deno_by),
        "nume_within": sum(nume_by),
        "deno_between": deno_between,
        "nume_between": nume_between,
    }


def naive_strat(outcome, strata, weights=None, ordered=False, group=None):
    """End-to-end naive pipeline; returns (strat, std_error)."""
    outcome = np.asarray(outcome, dtype=float)
    n = len(outcome)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum() * n
    prank = naive_wtd_rank(outcome, w) / n

    levels = sorted(set(strata))
    codes = np.array([levels.index(s) for s in strata])
    s_prank = np.array(
        [
            sum(p * wi for p, wi, c in zip(prank, w, codes, strict=True) if c == k)
            / sum(wi for wi, c in zip(w, codes, strict=True) if c == k)
            for k in range(len(levels))
        ]
    )
    sort_by = codes.astype(float) if ordered else s_prank[codes]
    order = np.argsort(sort_by, kind="stable")
    y, r, ws = prank[order], sort_by[order], w[order]

    if group is None:
        deno, nume = naive_pair_sums(y, r, ws)
    else:
        g_levels = sorted(set(group))
        g_codes = np.array([g_levels.index(g) for g in group])[order]
        sums = naive_pair_sums_by(y, r, ws, g_codes, len(g_levels))
        deno = sums["deno_within"] + sums["deno_between"]
        nume = sums["nume_within"] + sums["nume_between"]
    index = nume / deno
    se = math.sqrt((1.0 - index**2) * n / deno)
    return index, se


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(20260712)
