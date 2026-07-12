"""Pairwise-comparison kernels.

Blocked NumPy ports of ``strat_cpp`` and ``strat_cpp_by`` from
``src/strat_cpp.cpp`` in the R package. Inputs must be sorted ascending by
``r``. A pair (i, j), i < j, contributes ``w_i * w_j`` to the denominator and
``sign(y_j - y_i) * w_i * w_j`` to the numerator, unless ``r_j == r_i`` (same
stratum position) or ``y_j == y_i`` (tied outcome), in which case it is
skipped.
"""

from __future__ import annotations

import numpy as np

_BLOCK = 2048


def pair_sums(
    y: np.ndarray, r: np.ndarray, w: np.ndarray, block: int = _BLOCK
) -> tuple[float, float]:
    """Return ``(deno, nume)`` summed over all valid pairs."""
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


def pair_sums_by(
    y: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    c: np.ndarray,
    n_groups: int,
    block: int = _BLOCK,
) -> dict[str, np.ndarray | float]:
    """Group decomposition of the pairwise sums.

    Pairs within the same group accumulate into per-group ``deno_by`` /
    ``nume_by`` (indexed by the group code of both members); pairs across
    groups accumulate into the between-group sums.
    """
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
            w_same_row = np.where(same, wij, 0.0).sum(axis=1)
            s_same_row = np.where(same, s, 0.0).sum(axis=1)
            deno_by += np.bincount(ci, weights=w_same_row, minlength=n_groups)
            nume_by += np.bincount(ci, weights=s_same_row, minlength=n_groups)
            deno_between += wij.sum() - w_same_row.sum()
            nume_between += s.sum() - s_same_row.sum()

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
