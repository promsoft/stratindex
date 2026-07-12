import math

import numpy as np
import pytest
from conftest import naive_strat

from stratindex import strat


def test_perfectly_stratified_is_one():
    s = strat([1.0, 2.0, 3.0, 4.0], ["a", "a", "b", "b"])
    assert s.strat == pytest.approx(1.0)
    assert s.std_error == pytest.approx(0.0)


def test_perfectly_antistratified_is_minus_one():
    # strata ordered by mean rank, so labels are relabeled; force order
    s = strat([4.0, 3.0, 2.0, 1.0], ["a", "a", "b", "b"], ordered=True)
    assert s.strat == pytest.approx(-1.0)


def test_hand_computed_example():
    # outcome [1, 2, 3, 4], strata [a, b, a, b], unit weights
    # pranks = [0.25, 0.5, 0.75, 1.0]; s_prank: a = 0.5, b = 0.75
    # sorted by stratum mean rank: a-rows before b-rows
    # valid pairs (a, b): (1,2)+, (1,4)+, (3,2)-, (3,4)+ each weight 1
    # strat = (3 - 1) / 4 = 0.5; se = sqrt((1 - 0.25) * 4 / 4)
    s = strat([1.0, 2.0, 3.0, 4.0], ["a", "b", "a", "b"])
    assert s.strat == pytest.approx(0.5)
    assert s.std_error == pytest.approx(math.sqrt(0.75))


def test_matches_naive_pipeline_on_random_data(rng):
    for ordered in (False, True):
        for use_weights in (False, True):
            for _ in range(5):
                n = int(rng.integers(5, 100))
                outcome = rng.normal(size=n).round(1)
                strata = rng.choice(list("abcd"), size=n)
                weights = rng.uniform(0.1, 4.0, size=n) if use_weights else None
                expected_index, expected_se = naive_strat(
                    outcome, strata, weights=weights, ordered=ordered
                )
                got = strat(outcome, strata, weights=weights, ordered=ordered)
                assert got.strat == pytest.approx(expected_index, rel=1e-12)
                assert got.std_error == pytest.approx(expected_se, rel=1e-12)


def test_group_decomposition_matches_naive(rng):
    for _ in range(5):
        n = int(rng.integers(10, 100))
        outcome = rng.normal(size=n).round(1)
        strata = rng.choice(list("abc"), size=n)
        group = rng.choice(["g1", "g2", "g3"], size=n)
        weights = rng.uniform(0.1, 4.0, size=n)
        expected_index, expected_se = naive_strat(outcome, strata, weights=weights, group=group)
        got = strat(outcome, strata, weights=weights, group=group)
        assert got.strat == pytest.approx(expected_index, rel=1e-12)
        assert got.std_error == pytest.approx(expected_se, rel=1e-12)


def test_decomposition_identities(rng):
    n = 150
    outcome = rng.normal(size=n)
    strata = rng.choice(list("abcd"), size=n)
    group = rng.choice(["g1", "g2"], size=n)
    s = strat(outcome, strata, group=group)
    d = s.decomposition
    assert d["within"]["weight"] + d["between"]["weight"] == pytest.approx(1.0)
    # overall index is the weight-average of the components
    assert s.strat == pytest.approx(
        d["within"]["weight"] * d["within"]["strat"]
        + d["between"]["weight"] * d["between"]["strat"]
    )
    assert np.sum(s.within_group["weight"]) == pytest.approx(1.0)
    assert list(s.within_group["group"]) == ["g1", "g2"]


def test_single_level_group_gives_no_decomposition():
    s = strat([1.0, 2.0, 3.0, 4.0], ["a", "b", "a", "b"], group=["g", "g", "g", "g"])
    assert s.decomposition is None
    assert s.within_group is None


def test_missing_values_are_dropped():
    base = strat([1.0, 2.0, 3.0, 4.0], ["a", "b", "a", "b"])
    padded = strat(
        [1.0, 2.0, 3.0, 4.0, np.nan, 5.0],
        ["a", "b", "a", "b", "a", None],
        weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    assert padded.strat == pytest.approx(base.strat)
    assert padded.std_error == pytest.approx(base.std_error)


def test_single_stratum_gives_nan():
    s = strat([1.0, 2.0, 3.0], ["a", "a", "a"])
    assert math.isnan(s.strat)


def test_input_validation():
    with pytest.raises(ValueError, match="numeric"):
        strat(["x", "y"], ["a", "b"])
    with pytest.raises(ValueError, match="equal length"):
        strat([1.0, 2.0], ["a"])
    with pytest.raises(ValueError, match="equal length"):
        strat([1.0, 2.0], ["a", "b"], weights=[1.0])
    with pytest.raises(ValueError, match="group contains missing values"):
        strat([1.0, 2.0], ["a", "b"], group=["g", None])
    with pytest.raises(ValueError, match="logical"):
        strat([1.0, 2.0], ["a", "b"], ordered="yes")
    with pytest.raises(ValueError, match="no complete cases"):
        strat([np.nan, np.nan], ["a", "b"])


def test_str_output_mentions_group_name():
    s = strat(
        [1.0, 2.0, 3.0, 4.0],
        ["a", "b", "a", "b"],
        group=["g1", "g2", "g1", "g2"],
        group_name="education",
    )
    text = str(s)
    assert "overall stratification:" in text
    assert "decomposition by education:" in text
    assert "within education" in text
