"""Input cleaning and weighted percentile ranks.

Ports the internal helpers of the R package ``strat`` (``R/utils.R``):
``wtd_rank`` (a thin wrapper around ``Hmisc::wtd.rank``) and ``clean``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def wtd_rank(x: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted midranks of ``x``.

    Equivalent to ``Hmisc::wtd.rank(x, weights, normwt = TRUE)``: weights are
    normalized to sum to ``len(x)``, tied values share the midrank of their
    weight block.
    """
    x = np.asarray(x, dtype=float)
    if weights is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(weights, dtype=float)
    w = w / w.sum() * x.size  # normwt = TRUE
    _, inverse = np.unique(x, return_inverse=True)
    freqs = np.bincount(inverse, weights=w)
    ranks = np.cumsum(freqs) - 0.5 * (freqs - 1.0)
    return ranks[inverse]


def _is_na(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == "f":
        return np.isnan(arr)
    if arr.dtype.kind == "O":
        return np.array(
            [v is None or (isinstance(v, float) and math.isnan(v)) for v in arr],
            dtype=bool,
        )
    return np.zeros(arr.shape, dtype=bool)


@dataclass
class CleanData:
    """Complete cases of all inputs, ready for the pairwise kernel.

    Mirrors the data frame returned by ``clean()`` in the R package: ``prank``
    is the weighted percentile rank in (0, 1], ``weights`` are normalized to
    sum to ``n``, strata and group are stored as integer codes plus levels
    (sorted unique values, as ``factor()`` does).
    """

    prank: np.ndarray
    strata_codes: np.ndarray
    strata_levels: np.ndarray
    weights: np.ndarray
    group_codes: np.ndarray | None
    group_levels: np.ndarray | None
    n: int


def clean(
    outcome,
    strata,
    weights=None,
    group=None,
) -> CleanData:
    """Validate inputs, drop incomplete cases, compute weighted percentile ranks."""
    outcome = np.asarray(outcome)
    if (
        outcome.ndim != 1
        or outcome.size == 0
        or not np.issubdtype(outcome.dtype, np.number)
        or outcome.dtype.kind == "b"
    ):
        raise ValueError("outcome has to be a non-empty numeric 1-d array")
    outcome = outcome.astype(float)

    strata = np.asarray(strata)
    if strata.ndim != 1 or len(strata) != len(outcome):
        raise ValueError("outcome and strata have to be of equal length")

    if weights is None:
        weights_arr = np.ones(outcome.size)
    else:
        weights_arr = np.asarray(weights)
        if weights_arr.ndim != 1 or not np.issubdtype(weights_arr.dtype, np.number):
            raise ValueError("weights has to be a numeric 1-d array")
        if len(weights_arr) != len(outcome):
            raise ValueError("outcome and weights have to be of equal length")
        weights_arr = weights_arr.astype(float)

    group_arr: np.ndarray | None = None
    if group is not None:
        group_arr = np.asarray(group)
        if group_arr.ndim != 1 or len(group_arr) != len(outcome):
            raise ValueError("outcome and group have to be of equal length")
        # R's strat() rejects missing values in group up front
        if _is_na(group_arr).any():
            raise ValueError("group contains missing values")

    ok = ~(_is_na(outcome) | _is_na(strata) | _is_na(weights_arr))
    n = int(ok.sum())
    if n == 0:
        raise ValueError("no complete cases!")

    outcome = outcome[ok]
    weights_arr = weights_arr[ok]
    weights_arr = weights_arr / weights_arr.sum() * n
    strata_levels, strata_codes = np.unique(strata[ok], return_inverse=True)

    group_codes = group_levels = None
    if group_arr is not None:
        group_levels, group_codes = np.unique(group_arr[ok], return_inverse=True)

    prank = wtd_rank(outcome, weights_arr) / n

    return CleanData(
        prank=prank,
        strata_codes=strata_codes,
        strata_levels=strata_levels,
        weights=weights_arr,
        group_codes=group_codes,
        group_levels=group_levels,
        n=n,
    )
