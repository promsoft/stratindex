import numpy as np
import pytest

from stratindex import srank


def test_hand_computed_summary():
    # outcome [1, 2, 3, 4], strata [a, b, a, b], unit weights
    # pranks = [0.25, 0.5, 0.75, 1.0]
    res = srank([1.0, 2.0, 3.0, 4.0], ["a", "b", "a", "b"])
    assert list(res.summary["strata"]) == ["a", "b"]
    assert res.summary["share"] == pytest.approx([0.5, 0.5])
    assert res.summary["s_prank"] == pytest.approx([0.5, 0.75])
    assert res.raw["prank"] == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_shares_sum_to_one_and_pranks_in_unit_interval(rng):
    n = 300
    res = srank(
        rng.normal(size=n).round(1),
        rng.choice(list("abcde"), size=n),
        weights=rng.uniform(0.1, 5.0, size=n),
    )
    assert np.sum(res.summary["share"]) == pytest.approx(1.0)
    # weighted midranks can slightly exceed n (as in Hmisc::wtd.rank):
    # the top rank is n + 0.5 * (1 - w_top), so prank < 1 + 0.5 / n
    bound = 1.0 + 0.5 / n
    assert (res.summary["s_prank"] > 0).all() and (res.summary["s_prank"] < bound).all()
    assert (res.raw["prank"] > 0).all() and (res.raw["prank"] < bound).all()


def test_group_is_carried_through_to_raw():
    res = srank([1.0, 2.0], ["a", "b"], group=["g1", "g2"])
    assert list(res.raw["group"]) == ["g1", "g2"]


def test_str_output_lists_strata():
    text = str(srank([1.0, 2.0, 3.0, 4.0], ["a", "b", "a", "b"]))
    assert "strata" in text and "share" in text and "s_prank" in text
