"""Public API: :func:`srank` and :func:`strat`.

Port of ``R/srank.R`` and ``R/strat.R`` from the R package ``strat``
(Xiang Zhou, https://cran.r-project.org/package=strat).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ._kernel import pair_sums, pair_sums_by
from ._utils import CleanData, clean
from .results import SrankResult, StratResult

__all__ = ["srank", "strat"]


def _resolve_inputs(fn_name, args, outcome, strata, weights, group, group_name):
    """Support both call styles: arrays positionally, or a data source first.

    ``fn(outcome, strata[, weights], ...)`` with array-likes, or
    ``fn(data, outcome="col", strata="col", weights="col", group="col")``
    where ``data`` is a mapping / DataFrame and names refer to its columns.
    """
    if args and (isinstance(args[0], Mapping) or hasattr(args[0], "columns")):
        data = args[0]
        if len(args) > 1:
            raise TypeError(
                f"{fn_name}() with a data argument takes columns as keywords: "
                f'{fn_name}(data, outcome="...", strata="...")'
            )
        if outcome is None or strata is None:
            raise TypeError(f"{fn_name}() with a data argument requires outcome= and strata=")

        def col(value, what):
            if value is None or not isinstance(value, str):
                return value
            try:
                return data[value]
            except KeyError:
                raise ValueError(f"{what} column {value!r} not found in data") from None

        if group_name is None and isinstance(group, str):
            group_name = group
        return (
            col(outcome, "outcome"),
            col(strata, "strata"),
            col(weights, "weights"),
            col(group, "group"),
            group_name,
        )

    if len(args) > 3:
        raise TypeError(
            f"{fn_name}() takes at most 3 positional arguments (outcome, strata, weights)"
        )
    positional = list(args) + [None] * (3 - len(args))
    for name, pos, kw in zip(
        ("outcome", "strata", "weights"), positional, (outcome, strata, weights), strict=True
    ):
        if pos is not None and kw is not None:
            raise TypeError(f"{fn_name}() got multiple values for argument '{name}'")
    outcome = outcome if outcome is not None else positional[0]
    strata = strata if strata is not None else positional[1]
    weights = weights if weights is not None else positional[2]
    if outcome is None or strata is None:
        raise TypeError(f"{fn_name}() requires outcome and strata")
    return outcome, strata, weights, group, group_name


def _summarize(cd: CleanData) -> dict[str, np.ndarray]:
    """Per-stratum population share and average percentile rank."""
    k = len(cd.strata_levels)
    w_by_stratum = np.bincount(cd.strata_codes, weights=cd.weights, minlength=k)
    share = w_by_stratum / cd.weights.sum()
    s_prank = (
        np.bincount(cd.strata_codes, weights=cd.weights * cd.prank, minlength=k) / w_by_stratum
    )
    return {"strata": cd.strata_levels, "share": share, "s_prank": s_prank}


def _raw(cd: CleanData) -> dict[str, np.ndarray]:
    raw = {
        "prank": cd.prank,
        "strata": cd.strata_levels[cd.strata_codes],
        "weights": cd.weights,
    }
    if cd.group_codes is not None:
        raw["group"] = cd.group_levels[cd.group_codes]
    return raw


def srank(*args, outcome=None, strata=None, weights=None, group=None) -> SrankResult:
    """Rank strata by the average percentile rank of their members.

    Call with arrays — ``srank(outcome, strata, weights=w)`` — or with a
    DataFrame / mapping of columns first:
    ``srank(df, outcome="income", strata="big_class", weights="weight")``.

    Parameters
    ----------
    outcome:
        Numeric array of outcomes (or a column name in data mode).
    strata:
        Array of the same length indicating strata membership. pandas
        Categorical keeps its category order.
    weights:
        Optional numeric array of sampling weights.
    group:
        Optional grouping factor, carried through to ``raw``.

    Returns
    -------
    SrankResult
        ``raw`` (complete cases with percentile ranks) and ``summary``
        (per-stratum share and average percentile rank).
    """
    outcome, strata, weights, group, _ = _resolve_inputs(
        "srank", args, outcome, strata, weights, group, None
    )
    cd = clean(outcome, strata, weights=weights, group=group)
    return SrankResult(raw=_raw(cd), summary=_summarize(cd))


def strat(
    *args,
    outcome=None,
    strata=None,
    weights=None,
    ordered: bool = False,
    group=None,
    group_name: str | None = None,
) -> StratResult:
    """Compute the stratification index proposed in Zhou (2012).

    Call with arrays — ``strat(outcome, strata, weights=w)`` — or with a
    DataFrame / mapping of columns first:
    ``strat(df, outcome="income", strata="big_class", weights="weight",
    group="education")``. In data mode ``group_name`` defaults to the group
    column name.

    Parameters
    ----------
    outcome:
        Numeric array of outcomes (or a column name in data mode).
    strata:
        Array of the same length indicating strata membership. pandas
        Categorical keeps its category order.
    weights:
        Optional numeric array of sampling weights.
    ordered:
        If True, strata are taken as pre-ordered ascendingly (by their level
        order); otherwise they are ordered by average percentile rank.
    group:
        Optional grouping factor. If supplied (with more than one level), the
        result includes a between-/within-group decomposition of the overall
        stratification.
    group_name:
        Label used for the group in printed output (R derives it from the
        expression passed as ``group``; Python uses the column name in data
        mode, else "group", unless given explicitly).

    Returns
    -------
    StratResult
        Overall index with approximate standard error (Goodman & Kruskal
        1963), per-stratum information, and the group decomposition when a
        group is supplied.

    References
    ----------
    Zhou, Xiang. 2012. "A Nonparametric Index of Stratification."
    Sociological Methodology, 42(1): 365-389.
    """
    outcome, strata, weights, group, group_name = _resolve_inputs(
        "strat", args, outcome, strata, weights, group, group_name
    )
    if group_name is None:
        group_name = "group"
    if not isinstance(ordered, bool):
        raise ValueError("ordered has to be a valid logical scalar")

    cd = clean(outcome, strata, weights=weights, group=group)
    strata_info = _summarize(cd)

    row_s_prank = strata_info["s_prank"][cd.strata_codes]
    if ordered:
        sort_by = cd.strata_codes.astype(float)
    else:
        sort_by = row_s_prank
    order = np.argsort(sort_by, kind="stable")
    y = cd.prank[order]
    r = sort_by[order]
    w = cd.weights[order]

    decomposition = None
    within_group = None
    if cd.group_codes is None or len(cd.group_levels) == 1:
        deno, nume = pair_sums(y, r, w)
        with np.errstate(invalid="ignore"):
            index = nume / deno
    else:
        c = cd.group_codes[order]
        sums = pair_sums_by(y, r, w, c, len(cd.group_levels))
        deno = sums["deno_within"] + sums["deno_between"]
        with np.errstate(invalid="ignore", divide="ignore"):
            index = (sums["nume_within"] + sums["nume_between"]) / deno
            decomposition = {
                "within": {
                    "weight": sums["deno_within"] / deno,
                    "strat": sums["nume_within"] / sums["deno_within"],
                },
                "between": {
                    "weight": sums["deno_between"] / deno,
                    "strat": sums["nume_between"] / sums["deno_between"],
                },
            }
            within_group = {
                group_name: cd.group_levels,
                "weight": sums["deno_by"] / sums["deno_within"],
                "strat": sums["nume_by"] / sums["deno_by"],
            }

    # approximate standard error (Goodman & Kruskal 1963)
    with np.errstate(invalid="ignore", divide="ignore"):
        arg = deno / (1.0 - index**2) / cd.n
        std_error = 1.0 / np.sqrt(arg)

    return StratResult(
        strat=float(index),
        std_error=float(std_error),
        strata_info=strata_info,
        decomposition=decomposition,
        within_group=within_group,
        group_name=group_name,
    )
