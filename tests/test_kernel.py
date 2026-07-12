import numpy as np
import pytest
from conftest import naive_pair_sums, naive_pair_sums_by

from stratindex._kernel import (
    pair_sums,
    pair_sums_blocked,
    pair_sums_by,
    pair_sums_by_blocked,
)

# the fast kernel accumulates in a different order than the naive loop, and
# its between-group part is a difference of totals, so allow a small atol
TOL = dict(rtol=1e-9, atol=1e-8)


def _random_sorted_input(rng, n, n_ranks, n_groups):
    y = rng.choice(np.linspace(0.1, 1.0, max(2, n // 3)), size=n)
    r = np.sort(rng.choice(np.linspace(0.0, 1.0, n_ranks), size=n))
    w = rng.uniform(0.1, 3.0, size=n)
    c = rng.integers(0, n_groups, size=n)
    return y, r, w, c


def test_pair_sums_blocked_matches_naive_across_block_sizes(rng):
    for _ in range(10):
        n = int(rng.integers(2, 120))
        y, r, w, _ = _random_sorted_input(rng, n, n_ranks=7, n_groups=1)
        expected = naive_pair_sums(y, r, w)
        for block in (1, 3, 64, 4096):
            got = pair_sums_blocked(y, r, w, block=block)
            np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_pair_sums_by_blocked_matches_naive_across_block_sizes(rng):
    for _ in range(10):
        n = int(rng.integers(2, 120))
        m = int(rng.integers(1, 5))
        y, r, w, c = _random_sorted_input(rng, n, n_ranks=7, n_groups=m)
        expected = naive_pair_sums_by(y, r, w, c, m)
        for block in (1, 3, 64, 4096):
            got = pair_sums_by_blocked(y, r, w, c, m, block=block)
            for key in expected:
                np.testing.assert_allclose(got[key], expected[key], rtol=1e-12, err_msg=key)


def test_fast_pair_sums_matches_naive(rng):
    for _ in range(25):
        n = int(rng.integers(1, 200))
        y, r, w, _ = _random_sorted_input(rng, n, n_ranks=int(rng.integers(1, 12)), n_groups=1)
        np.testing.assert_allclose(pair_sums(y, r, w), naive_pair_sums(y, r, w), **TOL)


def test_fast_pair_sums_by_matches_naive(rng):
    for _ in range(25):
        n = int(rng.integers(1, 200))
        m = int(rng.integers(1, 6))
        y, r, w, c = _random_sorted_input(rng, n, n_ranks=int(rng.integers(1, 12)), n_groups=m)
        expected = naive_pair_sums_by(y, r, w, c, m)
        got = pair_sums_by(y, r, w, c, m)
        for key in expected:
            np.testing.assert_allclose(got[key], expected[key], err_msg=key, **TOL)


def test_fast_matches_blocked_on_larger_input(rng):
    n = 3000
    y, r, w, c = _random_sorted_input(rng, n, n_ranks=25, n_groups=4)
    np.testing.assert_allclose(pair_sums(y, r, w), pair_sums_blocked(y, r, w), rtol=1e-10)
    fast = pair_sums_by(y, r, w, c, 4)
    blocked = pair_sums_by_blocked(y, r, w, c, 4)
    for key in blocked:
        np.testing.assert_allclose(fast[key], blocked[key], rtol=1e-10, atol=1e-8, err_msg=key)


def test_group_sums_add_up_to_plain_sums(rng):
    n = 200
    y, r, w, c = _random_sorted_input(rng, n, n_ranks=11, n_groups=3)
    deno, nume = pair_sums(y, r, w)
    sums = pair_sums_by(y, r, w, c, 3)
    np.testing.assert_allclose(sums["deno_within"] + sums["deno_between"], deno, rtol=1e-12)
    np.testing.assert_allclose(sums["nume_within"] + sums["nume_between"], nume, rtol=1e-12)


@pytest.mark.parametrize("fn", [pair_sums, pair_sums_blocked])
def test_all_pairs_skipped_when_single_rank(fn):
    y = np.array([0.2, 0.5, 0.9])
    r = np.zeros(3)
    w = np.ones(3)
    assert fn(y, r, w) == (0.0, 0.0)


@pytest.mark.parametrize("fn", [pair_sums, pair_sums_blocked])
def test_all_pairs_skipped_when_single_outcome_value(fn):
    y = np.full(4, 0.5)
    r = np.array([0.0, 1.0, 2.0, 3.0])
    w = np.ones(4)
    assert fn(y, r, w) == (0.0, 0.0)


def test_empty_group_yields_zero_sums():
    y = np.array([0.2, 0.5, 0.9])
    r = np.array([0.0, 1.0, 2.0])
    w = np.ones(3)
    c = np.zeros(3, dtype=int)
    sums = pair_sums_by(y, r, w, c, 2)  # group 1 is empty
    assert sums["deno_by"][1] == 0.0 and sums["nume_by"][1] == 0.0
