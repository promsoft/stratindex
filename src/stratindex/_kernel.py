"""Pairwise-comparison kernels.

O(n log n) ports of ``strat_cpp`` and ``strat_cpp_by`` from
``src/strat_cpp.cpp`` in the R package. Inputs must be sorted ascending by
``r``. A pair (i, j), i < j, contributes ``w_i * w_j`` to the denominator and
``sign(y_j - y_i) * w_i * w_j`` to the numerator, unless ``r_j == r_i`` (same
stratum position) or ``y_j == y_i`` (tied outcome), in which case it is
skipped.

The denominator is computed by inclusion-exclusion over tie blocks
(``T - A - B + AB``); the numerator is a weighted concordant-minus-discordant
count over pairs with distinct ``r``, computed blockwise (equal-``r`` block
by block, vectorized over y ranks) or, when there are very many blocks, by a
Fenwick tree.
"""

from __future__ import annotations

import itertools

import numpy as np


def _tie_weight(codes: np.ndarray, w: np.ndarray) -> float:
    """Total pair weight within tie blocks: sum over blocks of (W² - Σw²) / 2."""
    wb = np.bincount(codes, weights=w)
    qb = np.bincount(codes, weights=w * w)
    return 0.5 * float((wb * wb - qb).sum())


def _signed_pair_sum_vec(y_codes, w, bounds, n_y) -> float:
    """Blockwise numerator: one bincount + cumsum of inserted weights per r-block."""
    acc = np.zeros(n_y)
    cum = None
    nume = 0.0
    for i, j in itertools.pairwise(bounds):
        yc = y_codes[i:j]
        if i:
            if cum is None:
                cum = np.cumsum(acc)
            below = np.where(yc > 0, cum[yc - 1], 0.0)
            above = cum[-1] - cum[yc]
            nume += float((w[i:j] * (below - above)).sum())
        acc += np.bincount(yc, weights=w[i:j], minlength=n_y)
        cum = None
    return nume


def _signed_pair_sum_bit(y_codes, w, bounds, n_y) -> float:
    """Fenwick-tree numerator: O(n log n) regardless of the number of r-blocks."""
    tree = [0.0] * (n_y + 1)
    y_list = y_codes.tolist()
    w_list = w.tolist()

    def add(i: int, v: float) -> None:
        i += 1
        while i <= n_y:
            tree[i] += v
            i += i & -i

    def prefix(i: int) -> float:  # sum of weights with y code < i
        s = 0.0
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s

    nume = 0.0
    added = 0.0
    for i, j in itertools.pairwise(bounds):
        for k in range(i, j):
            below = prefix(y_list[k])
            above = added - prefix(y_list[k] + 1)
            nume += w_list[k] * (below - above)
        for k in range(i, j):
            add(y_list[k], w_list[k])
            added += w_list[k]
    return nume


def _signed_pair_sum(y_codes: np.ndarray, r_codes: np.ndarray, w: np.ndarray, n_y: int) -> float:
    """Weighted concordant-minus-discordant sum over pairs with distinct r.

    Only pairs from different r-blocks count; ties in ``y`` contribute zero,
    matching ``sign() == 0`` in the reference implementation.
    """
    n = len(y_codes)
    if n == 0:
        return 0.0
    boundaries = np.flatnonzero(r_codes[1:] != r_codes[:-1]) + 1
    bounds = [0, *boundaries.tolist(), n]
    k = len(bounds) - 1
    # the blockwise pass costs O(k * n_y) numpy element-ops, the Fenwick loop
    # O(n log n) python-ops (roughly 50x more expensive each); in practice k
    # (the number of strata) is small and the blockwise pass wins by ~100x
    if k * n_y <= 50 * n * max(1.0, np.log2(n)):
        return _signed_pair_sum_vec(y_codes, w, bounds, n_y)
    return _signed_pair_sum_bit(y_codes, w, bounds, n_y)


def pair_sums(y: np.ndarray, r: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Return ``(deno, nume)`` summed over all valid pairs."""
    w = np.asarray(w, dtype=float)
    _, y_codes = np.unique(y, return_inverse=True)
    _, r_codes = np.unique(r, return_inverse=True)  # r sorted -> codes ascending
    n_y = int(y_codes.max()) + 1 if len(w) else 0

    s = w.sum()
    q = (w * w).sum()
    total = 0.5 * float(s * s - q)
    same_r = _tie_weight(r_codes, w)
    same_y = _tie_weight(y_codes, w)
    # densify the (r, y) cell key so bincount memory stays O(n)
    _, cell_codes = np.unique(r_codes.astype(np.int64) * n_y + y_codes, return_inverse=True)
    same_both = _tie_weight(cell_codes, w)

    deno = total - same_r - same_y + same_both
    nume = _signed_pair_sum(y_codes, r_codes, w, n_y)
    return deno, nume


def pair_sums_by(
    y: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    c: np.ndarray,
    n_groups: int,
) -> dict[str, np.ndarray | float]:
    """Group decomposition of the pairwise sums.

    Within-group sums are computed independently per group (subsetting keeps
    the r order); between-group sums are the total minus the within part.
    """
    deno_by = np.zeros(n_groups)
    nume_by = np.zeros(n_groups)
    for g in range(n_groups):
        idx = np.flatnonzero(c == g)
        if idx.size:
            deno_by[g], nume_by[g] = pair_sums(y[idx], r[idx], w[idx])

    deno_total, nume_total = pair_sums(y, r, w)
    deno_within = float(deno_by.sum())
    nume_within = float(nume_by.sum())
    return {
        "deno_by": deno_by,
        "nume_by": nume_by,
        "deno_within": deno_within,
        "nume_within": nume_within,
        "deno_between": deno_total - deno_within,
        "nume_between": nume_total - nume_within,
    }
