import math

import numpy as np
import pytest

from stratindex import srank, strat


@pytest.fixture(scope="module")
def sample(rng):
    n = 300
    strata = rng.integers(0, 4, n)
    outcome = np.exp(rng.normal(10 + 0.3 * strata, 0.8)).round(0)
    weights = rng.uniform(0.5, 2.0, n)
    return outcome, strata, weights


def test_bootstrap_se_is_reproducible_and_sane(sample):
    outcome, strata, weights = sample
    approx = strat(outcome, strata, weights=weights)
    boot1 = strat(
        outcome, strata, weights=weights, se_method="bootstrap", n_boot=100, random_state=7
    )
    boot2 = strat(
        outcome, strata, weights=weights, se_method="bootstrap", n_boot=100, random_state=7
    )
    assert boot1.std_error == boot2.std_error  # reproducible with a seed
    assert boot1.strat == approx.strat  # the index itself is untouched
    assert 0 < boot1.std_error < 1
    # same order of magnitude as the Goodman-Kruskal approximation
    assert 0.2 < boot1.std_error / approx.std_error < 5


def test_bootstrap_accepts_generator(sample):
    outcome, strata, weights = sample
    gen = np.random.default_rng(42)
    s = strat(outcome, strata, weights=weights, se_method="bootstrap", n_boot=50, random_state=gen)
    assert math.isfinite(s.std_error)


def test_bootstrap_with_ordered_and_group(sample):
    outcome, strata, weights = sample
    group = np.where(np.arange(len(outcome)) % 2 == 0, "g1", "g2")
    s = strat(
        outcome,
        strata,
        weights=weights,
        ordered=True,
        group=group,
        se_method="bootstrap",
        n_boot=50,
        random_state=1,
    )
    assert math.isfinite(s.std_error)
    assert s.decomposition is not None  # decomposition still reported


def test_se_method_validation(sample):
    outcome, strata, _ = sample
    with pytest.raises(ValueError, match="se_method"):
        strat(outcome, strata, se_method="jackknife")
    with pytest.raises(ValueError, match="n_boot"):
        strat(outcome, strata, se_method="bootstrap", n_boot=1)


def test_repr_html_strat(sample):
    outcome, strata, weights = sample
    group = np.where(np.arange(len(outcome)) % 2 == 0, "g1", "g2")
    s = strat(outcome, strata, weights=weights, group=group, group_name="half")
    html_out = s._repr_html_()
    assert html_out.count("<table>") == 3  # overall, decomposition, strata
    assert "overall stratification" in html_out
    assert "decomposition by half" in html_out
    assert "within half" in html_out

    no_group = strat(outcome, strata, weights=weights)
    assert no_group._repr_html_().count("<table>") == 2  # overall, strata


def test_repr_html_srank_escapes_labels():
    res = srank([1.0, 2.0, 3.0, 4.0], ["a<b", "x&y", "a<b", "x&y"])
    html_out = res._repr_html_()
    assert "<table>" in html_out
    assert "a&lt;b" in html_out and "x&amp;y" in html_out
