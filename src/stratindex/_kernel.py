"""Pairwise-comparison kernels.

Ports of ``strat_cpp`` and ``strat_cpp_by`` from ``src/strat_cpp.cpp`` in the
R package. Inputs must be sorted ascending by ``r``. A pair (i, j), i < j,
contributes ``w_i * w_j`` to the denominator and ``sign(y_j - y_i) * w_i *
w_j`` to the numerator, unless ``r_j == r_i`` (same stratum position) or
``y_j == y_i`` (tied outcome), in which case it is skipped.

Two implementations:

- ``pair_sums`` / ``pair_sums_by`` — O(n log n): the denominator by
  inclusion-exclusion over tie blocks, the numerator by weighted
  inversion counting with a Fenwick tree. Used by the public API.
- ``pair_sums_blocked`` / ``pair_sums_by_blocked`` — the original O(n²)
  blocked, vectorized NumPy port, kept as an independent reference for
  the test suite.
"""

from __future__ import annotations

import numpy as np

_BLOCK = 2048


def _tie_weight(codes: np.ndarray, w: np.ndarray) -> float:
    """Total pair weight within tie blocks: sum over blocks of (W² - Σw²) / 2."""
    wb = np.bincount(codes, weights=w)
    qb = np.bincount(codes, weights=w * w)
    return 0.5 * float((wb * wb - qb).sum())


def _signed_pair_sum(y_codes: np.ndarray, r_codes: np.ndarray, w: np.ndarray) -> float:
    """Weighted concordant-minus-discordant sum over pairs with distinct r.

    Processes blocks of equal ``r`` in ascending order, maintaining a Fenwick
    tree of inserted weights indexed by y rank: for every item, previously
    inserted items all have strictly smaller ``r``, so those with smaller
    (larger) ``y`` form concordant (discordant) pairs. Ties in ``y``
    contribute zero, matching sign() == 0 in the reference implementation.
    """
    n = len(y_codes)
    n_y = int(y_codes.max()) + 1 if n else 0
    tree = [0.0] * (n_y + 1)

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
    i = 0
    while i < n:
        j = i
        while j < n and r_codes[j] == r_codes[i]:
            j += 1
        for k in range(i, j):
            below = prefix(int(y_codes[k]))
            above = added - prefix(int(y_codes[k]) + 1)
            nume += w[k] * (below - above)
        for k in range(i, j):
            add(int(y_codes[k]), float(w[k]))
            added += w[k]
        i = j
    return nume


def pair_sums(y: np.ndarray, r: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Return ``(deno, nume)`` summed over all valid pairs in O(n log n)."""
    w = np.asarray(w, dtype=float)
    _, y_codes = np.unique(y, return_inverse=True)
    _, r_codes = np.unique(r, return_inverse=True)  # r sorted -> codes ascending

    s = w.sum()
    q = (w * w).sum()
    total = 0.5 * float(s * s - q)
    same_r = _tie_weight(r_codes, w)
    same_y = _tie_weight(y_codes, w)
    n_y = int(y_codes.max()) + 1 if len(w) else 0
    same_both = _tie_weight(r_codes.astype(np.int64) * n_y + y_codes, w)

    deno = total - same_r - same_y + same_both
    nume = _signed_pair_sum(y_codes, r_codes, w)
    # np.float64 so that downstream 0/0 yields NaN (as in R), not ZeroDivisionError
    return np.float64(deno), np.float64(nume)


def pair_sums_by(
    y: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    c: np.ndarray,
    n_groups: int,
) -> dict[str, np.ndarray | float]:
    """Group decomposition of the pairwise sums in O(n log n).

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
    deno_within = deno_by.sum()  # np.float64: 0/0 must yield NaN downstream
    nume_within = nume_by.sum()
    return {
        "deno_by": deno_by,
        "nume_by": nume_by,
        "deno_within": deno_within,
        "nume_within": nume_within,
        "deno_between": deno_total - deno_within,
        "nume_between": nume_total - nume_within,
    }


def pair_sums_blocked(
    y: np.ndarray, r: np.ndarray, w: np.ndarray, block: int = _BLOCK
) -> tuple[float, float]:
    """O(n²) reference: return ``(deno, nume)`` summed over all valid pairs."""
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


def pair_sums_by_blocked(
    y: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    c: np.ndarray,
    n_groups: int,
    block: int = _BLOCK,
) -> dict[str, np.ndarray | float]:
    """O(n²) reference: group decomposition of the pairwise sums."""
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
            # element-wise differences are exact where same-group entries cancel
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
