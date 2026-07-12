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


def _as_categorical(values) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(categories, codes)`` for pandas-categorical-like input, else None.

    Accepts a ``pandas.Categorical`` (or anything exposing ``categories`` and
    ``codes``) as well as a pandas Series of dtype ``category``.
    """
    if str(getattr(values, "dtype", "")) == "category" and hasattr(values, "values"):
        values = values.values  # Series[category] -> Categorical
    if hasattr(values, "categories") and hasattr(values, "codes"):
        return np.asarray(values.categories), np.asarray(values.codes)
    return None


class _Factor:
    """Level encoding of strata/group values.

    Mirrors R's ``factor()``: for plain arrays, levels are the sorted unique
    values; for pandas categoricals, the category order is respected and
    unused categories are dropped (keeping the order). Missing values are
    exposed via ``na`` before encoding.
    """

    def __init__(self, values):
        cat = _as_categorical(values)
        if cat is not None:
            self._categories, self._codes = cat
            self._values = None
            self.ndim = self._codes.ndim
            self.na = self._codes == -1
        else:
            self._categories = None
            self._values = np.asarray(values)
            self.ndim = self._values.ndim
            self.na = _is_na(self._values)

    def __len__(self) -> int:
        return len(self.na)

    def encode(self, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(levels, codes)`` for the rows selected by ``ok``."""
        if self._categories is None:
            return np.unique(self._values[ok], return_inverse=True)
        kept = self._codes[ok]
        present = np.unique(kept)
        remap = np.zeros(len(self._categories), dtype=np.intp)
        remap[present] = np.arange(len(present))
        return self._categories[present], remap[kept]


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

    strata_f = _Factor(strata)
    if strata_f.ndim != 1 or len(strata_f) != len(outcome):
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

    group_f: _Factor | None = None
    if group is not None:
        group_f = _Factor(group)
        if group_f.ndim != 1 or len(group_f) != len(outcome):
            raise ValueError("outcome and group have to be of equal length")
        # R's strat() rejects missing values in group up front
        if group_f.na.any():
            raise ValueError("group contains missing values")

    ok = ~(_is_na(outcome) | strata_f.na | _is_na(weights_arr))
    n = int(ok.sum())
    if n == 0:
        raise ValueError("no complete cases!")

    outcome = outcome[ok]
    weights_arr = weights_arr[ok]
    weights_arr = weights_arr / weights_arr.sum() * n
    strata_levels, strata_codes = strata_f.encode(ok)

    group_codes = group_levels = None
    if group_f is not None:
        group_levels, group_codes = group_f.encode(ok)

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
