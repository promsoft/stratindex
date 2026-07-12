"""pandas Categorical inputs: category order is respected (R factor semantics)."""

import numpy as np
import pandas as pd
import pytest

from stratindex import srank, strat

OUTCOME = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
STRATA = ["low", "high", "low", "mid", "high", "mid"]
ORDER = ["low", "mid", "high"]


def test_category_order_in_tables():
    cat = pd.Categorical(STRATA, categories=ORDER)
    res = srank(OUTCOME, cat)
    assert list(res.summary["strata"]) == ORDER  # not alphabetical


def test_series_of_category_dtype():
    ser = pd.Series(STRATA, dtype=pd.CategoricalDtype(categories=ORDER))
    res = srank(OUTCOME, ser)
    assert list(res.summary["strata"]) == ORDER


def test_ordered_true_uses_category_order():
    # reversing the category order flips the sign of the pre-ordered index
    outcome = [1.0, 2.0, 3.0, 4.0]
    strata = ["a", "a", "b", "b"]
    reversed_cat = pd.Categorical(strata, categories=["b", "a"])
    assert strat(outcome, np.asarray(strata), ordered=True).strat == pytest.approx(1.0)
    assert strat(outcome, reversed_cat, ordered=True).strat == pytest.approx(-1.0)


def test_alphabetical_categories_match_plain_strings():
    cat = pd.Categorical(STRATA, categories=sorted(set(STRATA)))
    for ordered in (False, True):
        a = strat(OUTCOME, cat, ordered=ordered)
        b = strat(OUTCOME, np.asarray(STRATA), ordered=ordered)
        assert a.strat == pytest.approx(b.strat)
        assert list(a.strata_info["strata"]) == list(b.strata_info["strata"])


def test_unused_categories_are_dropped_keeping_order():
    cat = pd.Categorical(STRATA, categories=["low", "extinct", "mid", "high"])
    res = srank(OUTCOME, cat)
    assert list(res.summary["strata"]) == ORDER


def test_missing_categorical_strata_rows_are_dropped():
    cat = pd.Categorical(["low", None, "low", "mid", "high", "mid"], categories=ORDER)
    res = srank(OUTCOME, cat)
    assert len(res.raw["prank"]) == 5
    assert list(res.summary["strata"]) == ORDER


def test_categorical_group_order_and_na():
    grp = pd.Categorical(["b", "a", "b", "a", "b", "a"], categories=["b", "a"])
    s = strat(OUTCOME, STRATA, group=grp)
    assert list(s.within_group["group"]) == ["b", "a"]

    grp_na = pd.Categorical(["b", None, "b", "a", "b", "a"], categories=["b", "a"])
    with pytest.raises(ValueError, match="group contains missing values"):
        strat(OUTCOME, STRATA, group=grp_na)


def test_unordered_index_is_order_independent():
    # without ordered=True, strata are ranked by mean percentile rank,
    # so the index must not depend on the level order at all
    cat = pd.Categorical(STRATA, categories=ORDER)
    a = strat(OUTCOME, cat)
    b = strat(OUTCOME, np.asarray(STRATA))
    assert a.strat == pytest.approx(b.strat)
