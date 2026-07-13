import numpy as np
from conftest import (
    blocked_pair_sums,
    blocked_pair_sums_by,
    naive_pair_sums,
    naive_pair_sums_by,
)

from stratindex._kernel import (
    _signed_pair_sum_bit,
    _signed_pair_sum_vec,
    pair_sums,
    pair_sums_by,
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


def test_pair_sums_matches_naive(rng):
    for _ in range(25):
        n = int(rng.integers(1, 200))
        y, r, w, _ = _random_sorted_input(rng, n, n_ranks=int(rng.integers(1, 12)), n_groups=1)
        np.testing.assert_allclose(pair_sums(y, r, w), naive_pair_sums(y, r, w), **TOL)


def test_pair_sums_by_matches_naive(rng):
    for _ in range(25):
        n = int(rng.integers(1, 200))
        m = int(rng.integers(1, 6))
        y, r, w, c = _random_sorted_input(rng, n, n_ranks=int(rng.integers(1, 12)), n_groups=m)
        expected = naive_pair_sums_by(y, r, w, c, m)
        got = pair_sums_by(y, r, w, c, m)
        for key in expected:
            np.testing.assert_allclose(got[key], expected[key], err_msg=key, **TOL)


def test_matches_blocked_reference_on_larger_input(rng):
    n = 3000
    y, r, w, c = _random_sorted_input(rng, n, n_ranks=25, n_groups=4)
    np.testing.assert_allclose(pair_sums(y, r, w), blocked_pair_sums(y, r, w), rtol=1e-10)
    fast = pair_sums_by(y, r, w, c, 4)
    blocked = blocked_pair_sums_by(y, r, w, c, 4)
    for key in blocked:
        np.testing.assert_allclose(fast[key], blocked[key], rtol=1e-10, atol=1e-8, err_msg=key)


def test_fenwick_and_blockwise_numerators_agree(rng):
    # _signed_pair_sum picks between the two by a cost model; both must
    # compute the same sum for any block structure
    for _ in range(15):
        n = int(rng.integers(1, 300))
        n_y = int(rng.integers(1, 30))
        y_codes = rng.integers(0, n_y, size=n)
        w = rng.uniform(0.1, 3.0, size=n)
        n_blocks = int(rng.integers(1, n + 1))
        cuts = np.sort(rng.choice(np.arange(1, n), size=min(n_blocks - 1, n - 1), replace=False))
        bounds = [0, *cuts.tolist(), n]
        vec = _signed_pair_sum_vec(y_codes, w, bounds, n_y)
        bit = _signed_pair_sum_bit(y_codes, w, bounds, n_y)
        np.testing.assert_allclose(vec, bit, rtol=1e-10, atol=1e-10)


def test_group_sums_add_up_to_plain_sums(rng):
    n = 200
    y, r, w, c = _random_sorted_input(rng, n, n_ranks=11, n_groups=3)
    deno, nume = pair_sums(y, r, w)
    sums = pair_sums_by(y, r, w, c, 3)
    np.testing.assert_allclose(sums["deno_within"] + sums["deno_between"], deno, rtol=1e-12)
    np.testing.assert_allclose(sums["nume_within"] + sums["nume_between"], nume, rtol=1e-12)


def test_all_pairs_skipped_when_single_rank():
    y = np.array([0.2, 0.5, 0.9])
    r = np.zeros(3)
    w = np.ones(3)
    assert pair_sums(y, r, w) == (0.0, 0.0)


def test_all_pairs_skipped_when_single_outcome_value():
    y = np.full(4, 0.5)
    r = np.array([0.0, 1.0, 2.0, 3.0])
    w = np.ones(4)
    assert pair_sums(y, r, w) == (0.0, 0.0)


def test_empty_group_yields_zero_sums():
    y = np.array([0.2, 0.5, 0.9])
    r = np.array([0.0, 1.0, 2.0])
    w = np.ones(3)
    c = np.zeros(3, dtype=int)
    sums = pair_sums_by(y, r, w, c, 2)  # group 1 is empty
    assert sums["deno_by"][1] == 0.0 and sums["nume_by"][1] == 0.0
