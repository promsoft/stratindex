"""Data-mode calls: strat(df, outcome="col", ...) with DataFrame or mapping."""

import numpy as np
import pandas as pd
import pytest

from stratindex import load_cpsmarch2015, srank, strat


@pytest.fixture(scope="module")
def cps():
    return load_cpsmarch2015()


def _subset(cps, n=800):
    return {k: v[:n] for k, v in cps.items()}


def test_mapping_input_matches_array_call(cps):
    d = _subset(cps)
    via_data = strat(d, outcome="income", strata="big_class", weights="weight")
    via_arrays = strat(d["income"], d["big_class"], weights=d["weight"])
    assert via_data.strat == via_arrays.strat
    assert via_data.std_error == via_arrays.std_error


def test_dataframe_input_matches_array_call(cps):
    df = pd.DataFrame(_subset(cps))
    via_data = strat(df, outcome="income", strata="big_class", weights="weight")
    via_arrays = strat(cps["income"][:800], cps["big_class"][:800], weights=cps["weight"][:800])
    assert via_data.strat == via_arrays.strat


def test_group_name_defaults_to_column_name(cps):
    d = _subset(cps)
    s = strat(d, outcome="income", strata="big_class", weights="weight", group="education")
    assert s.group_name == "education"
    assert "decomposition by education:" in str(s)
    # explicit group_name still wins
    s2 = strat(
        d,
        outcome="income",
        strata="big_class",
        weights="weight",
        group="education",
        group_name="edu",
    )
    assert s2.group_name == "edu"


def test_categorical_column_keeps_order(cps):
    df = pd.DataFrame(_subset(cps))
    order = ["unskilled manual", "skilled manual", "lower nonmanual", "upper nonmanual"]
    df["big_class"] = pd.Categorical(df["big_class"], categories=order)
    res = srank(df, outcome="income", strata="big_class", weights="weight")
    assert list(res.summary["strata"]) == order


def test_mixed_mode_array_value_in_data_call(cps):
    d = _subset(cps)
    w = np.ones(len(d["income"]))
    s = strat(d, outcome="income", strata="big_class", weights=w)
    assert s.strat == strat(d["income"], d["big_class"]).strat


def test_srank_data_mode(cps):
    d = _subset(cps)
    a = srank(d, outcome="income", strata="big_class", weights="weight")
    b = srank(d["income"], d["big_class"], weights=d["weight"])
    assert list(a.summary["share"]) == list(b.summary["share"])


def test_positional_weights_still_work(cps):
    d = _subset(cps)
    a = strat(d["income"], d["big_class"], d["weight"])
    b = strat(d["income"], d["big_class"], weights=d["weight"])
    assert a.strat == b.strat


def test_data_mode_errors(cps):
    d = _subset(cps)
    with pytest.raises(ValueError, match="outcome column 'nope' not found"):
        strat(d, outcome="nope", strata="big_class")
    with pytest.raises(TypeError, match="requires outcome= and strata="):
        strat(d, outcome="income")
    with pytest.raises(TypeError, match="columns as keywords"):
        strat(d, d["income"], outcome="income", strata="big_class")
    with pytest.raises(TypeError, match="multiple values for argument 'outcome'"):
        strat(d["income"], d["big_class"], outcome=d["income"])
    with pytest.raises(TypeError, match="requires outcome and strata"):
        strat(d["income"])
    with pytest.raises(TypeError, match="at most 3 positional"):
        strat(d["income"], d["big_class"], d["weight"], True)
