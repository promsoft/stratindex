import numpy as np
import pytest

from stratindex import load_cpsmarch2015, strat


def test_shape_and_dtypes(cps):
    assert set(cps) == {"income", "big_class", "micro_class", "education", "weight"}
    assert all(len(v) == 14358 for v in cps.values())
    assert cps["income"].dtype == float
    assert cps["weight"].dtype == float
    assert cps["micro_class"].dtype == int
    assert set(cps["big_class"]) == {
        "upper nonmanual",
        "lower nonmanual",
        "skilled manual",
        "unskilled manual",
    }
    assert set(cps["education"]) == {
        "less than HS",
        "HS or equivalent",
        "some college",
        "BA or above",
    }


def test_documented_example_regression(cps):
    """Regression snapshot of the example from the R docs.

    Values are produced by this package (validated against a naive reference
    implementation); cross-checked against the original R package in CI.
    """
    s = strat(
        cps["income"],
        cps["big_class"],
        weights=cps["weight"],
        group=cps["education"],
        group_name="education",
    )
    assert s.strat == pytest.approx(0.4127520133086936, rel=1e-9)
    assert s.std_error == pytest.approx(0.012960848514818133, rel=1e-9)
    assert s.decomposition["within"]["weight"] == pytest.approx(0.24348617732450617, rel=1e-9)
    assert s.decomposition["within"]["strat"] == pytest.approx(0.2683911994954791, rel=1e-9)
    assert s.decomposition["between"]["weight"] == pytest.approx(0.7565138226754938, rel=1e-9)
    assert s.decomposition["between"]["strat"] == pytest.approx(0.45921496171394943, rel=1e-9)
    np.testing.assert_allclose(
        s.within_group["strat"],
        [0.33331113, 0.26907302, 0.20722393, 0.19463748],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        s.strata_info["share"],
        [0.11895254, 0.16046984, 0.28615986, 0.43441776],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        s.strata_info["s_prank"],
        [0.47588702, 0.42982283, 0.33847693, 0.63900413],
        rtol=1e-6,
    )


def test_as_pandas(cps):
    pd = pytest.importorskip("pandas")
    df = load_cpsmarch2015(as_pandas=True)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (14358, 5)
    np.testing.assert_array_equal(df["income"].to_numpy(), cps["income"])
