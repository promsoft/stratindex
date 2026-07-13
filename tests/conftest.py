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


def blocked_pair_sums(y, r, w, block=2048):
    """O(n²) blocked NumPy reference (the 0.1.0 kernel), kept for cross-checks."""
    n = y.size
    deno = 0.0
    nume = 0.0
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        yi = y[i0:i1, None]
        ri = r[i0:i1, None]
        wi = w[i0:i1, None]
        for j0 in range(i0, n, block):
            j1 = min(j0 + block, n)
            yj = y[None, j0:j1]
            rj = r[None, j0:j1]
            valid = (rj > ri) & (yj != yi)
            if j0 == i0:
                valid &= np.arange(j0, j1)[None, :] > np.arange(i0, i1)[:, None]
            wij = wi * w[None, j0:j1] * valid
            deno += wij.sum()
            nume += (np.sign(yj - yi) * wij).sum()
    return deno, nume


def blocked_pair_sums_by(y, r, w, c, n_groups, block=2048):
    """O(n²) blocked NumPy reference for the group decomposition."""
    n = y.size
    deno_by = np.zeros(n_groups)
    nume_by = np.zeros(n_groups)
    deno_between = 0.0
    nume_between = 0.0
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        yi = y[i0:i1, None]
        ri = r[i0:i1, None]
        wi = w[i0:i1, None]
        ci = c[i0:i1]
        for j0 in range(i0, n, block):
            j1 = min(j0 + block, n)
            yj = y[None, j0:j1]
            rj = r[None, j0:j1]
            valid = (rj > ri) & (yj != yi)
            if j0 == i0:
                valid &= np.arange(j0, j1)[None, :] > np.arange(i0, i1)[:, None]
            wij = wi * w[None, j0:j1] * valid
            s = np.sign(yj - yi) * wij
            same = c[None, j0:j1] == ci[:, None]
            w_same = np.where(same, wij, 0.0)
            s_same = np.where(same, s, 0.0)
            deno_by += np.bincount(ci, weights=w_same.sum(axis=1), minlength=n_groups)
            nume_by += np.bincount(ci, weights=s_same.sum(axis=1), minlength=n_groups)
            deno_between += (wij - w_same).sum()
            nume_between += (s - s_same).sum()

    deno_within = deno_by.sum()
    nume_within = nume_by.sum()
    return {
        "deno_by": deno_by,
        "nume_by": nume_by,
        "deno_within": deno_within,
        "nume_within": nume_within,
        "deno_between": deno_between,
        "nume_between": nume_between,
    }


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(20260712)


@pytest.fixture(scope="session")
def cps():
    from stratindex import load_cpsmarch2015

    return load_cpsmarch2015()
