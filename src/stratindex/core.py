"""Public API: :func:`srank` and :func:`strat`.

Port of ``R/srank.R`` and ``R/strat.R`` from the R package ``strat``
(Xiang Zhou, https://cran.r-project.org/package=strat).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ._kernel import pair_sums, pair_sums_by
from ._utils import CleanData, clean, wtd_rank
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
    supplied = {"outcome": outcome, "strata": strata, "weights": weights}
    for name, value in zip(("outcome", "strata", "weights"), args, strict=False):
        if supplied[name] is not None:
            raise TypeError(f"{fn_name}() got multiple values for argument '{name}'")
        supplied[name] = value
    if supplied["outcome"] is None or supplied["strata"] is None:
        raise TypeError(f"{fn_name}() requires outcome and strata")
    return supplied["outcome"], supplied["strata"], supplied["weights"], group, group_name


def _stratum_mean_rank(strata_codes, weights, prank) -> tuple[np.ndarray, np.ndarray]:
    """Per-stratum weighted mean percentile rank and total weight."""
    w_by_stratum = np.bincount(strata_codes, weights=weights)
    s_prank = np.bincount(strata_codes, weights=weights * prank) / w_by_stratum
    return s_prank, w_by_stratum


def _sorted_kernel_input(prank, strata_codes, weights, ordered: bool, s_prank):
    """Rows sorted by stratum position — the layout the pairwise kernels expect."""
    sort_by = strata_codes.astype(float) if ordered else s_prank[strata_codes]
    order = np.argsort(sort_by, kind="stable")
    return prank[order], sort_by[order], weights[order], order


def _index_only(outcome, strata_codes, weights, ordered: bool) -> float:
    """Overall index for already-encoded complete cases (bootstrap replicate)."""
    n = len(outcome)
    w = weights / weights.sum() * n
    prank = wtd_rank(outcome, w) / n
    s_prank, _ = _stratum_mean_rank(strata_codes, w, prank)
    y, r, ws, _ = _sorted_kernel_input(prank, strata_codes, w, ordered, s_prank)
    deno, nume = pair_sums(y, r, ws)
    with np.errstate(invalid="ignore"):
        return float(np.divide(nume, deno))


def _bootstrap_se(cd: CleanData, ordered: bool, n_boot: int, random_state) -> float:
    rng = np.random.default_rng(random_state)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, cd.n, cd.n)
        reps[b] = _index_only(cd.outcome[idx], cd.strata_codes[idx], cd.weights[idx], ordered)
    reps = reps[np.isfinite(reps)]  # degenerate resamples (no comparable pairs)
    if len(reps) < 2:
        return float("nan")
    return float(reps.std(ddof=1))


def _summarize(cd: CleanData) -> dict[str, np.ndarray]:
    """Per-stratum population share and average percentile rank."""
    s_prank, w_by_stratum = _stratum_mean_rank(cd.strata_codes, cd.weights, cd.prank)
    return {
        "strata": cd.strata_levels,
        "share": w_by_stratum / cd.weights.sum(),
        "s_prank": s_prank,
    }


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
    se_method: str = "approx",
    n_boot: int = 200,
    random_state=None,
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
    se_method:
        ``"approx"`` (default) — the Goodman & Kruskal (1963) approximation,
        as in the R package; ``"bootstrap"`` — standard deviation of the
        index over ``n_boot`` resamples of the complete cases (percentile
        ranks and stratum order are recomputed in every replicate).
    n_boot:
        Number of bootstrap replicates (``se_method="bootstrap"`` only).
    random_state:
        Seed or ``numpy.random.Generator`` for the bootstrap.

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
    if se_method not in ("approx", "bootstrap"):
        raise ValueError('se_method has to be "approx" or "bootstrap"')
    if se_method == "bootstrap" and (not isinstance(n_boot, int) or n_boot < 2):
        raise ValueError("n_boot has to be an integer >= 2")

    cd = clean(outcome, strata, weights=weights, group=group)
    strata_info = _summarize(cd)
    y, r, w, order = _sorted_kernel_input(
        cd.prank, cd.strata_codes, cd.weights, ordered, strata_info["s_prank"]
    )

    decomposition = None
    within_group = None
    with np.errstate(invalid="ignore", divide="ignore"):
        if cd.group_codes is None or len(cd.group_levels) == 1:
            deno, nume = pair_sums(y, r, w)
            index = np.divide(nume, deno)
        else:
            c = cd.group_codes[order]
            sums = pair_sums_by(y, r, w, c, len(cd.group_levels))
            deno = sums["deno_within"] + sums["deno_between"]
            index = np.divide(sums["nume_within"] + sums["nume_between"], deno)
            decomposition = {
                "within": {
                    "weight": np.divide(sums["deno_within"], deno),
                    "strat": np.divide(sums["nume_within"], sums["deno_within"]),
                },
                "between": {
                    "weight": np.divide(sums["deno_between"], deno),
                    "strat": np.divide(sums["nume_between"], sums["deno_between"]),
                },
            }
            within_group = {
                group_name: cd.group_levels,
                "weight": sums["deno_by"] / sums["deno_within"],
                "strat": sums["nume_by"] / sums["deno_by"],
            }

    if se_method == "bootstrap":
        std_error = _bootstrap_se(cd, ordered, n_boot, random_state)
    else:
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
