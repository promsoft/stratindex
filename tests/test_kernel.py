import numpy as np
from conftest import naive_pair_sums, naive_pair_sums_by

from stratindex._kernel import pair_sums, pair_sums_by


def _random_sorted_input(rng, n, n_ranks, n_groups):
    y = rng.choice(np.linspace(0.1, 1.0, max(2, n // 3)), size=n)
    r = np.sort(rng.choice(np.linspace(0.0, 1.0, n_ranks), size=n))
    w = rng.uniform(0.1, 3.0, size=n)
    c = rng.integers(0, n_groups, size=n)
    return y, r, w, c


def test_pair_sums_matches_naive_across_block_sizes(rng):
    for _ in range(10):
        n = int(rng.integers(2, 120))
        y, r, w, _ = _random_sorted_input(rng, n, n_ranks=7, n_groups=1)
        expected = naive_pair_sums(y, r, w)
        for block in (1, 3, 64, 4096):
            got = pair_sums(y, r, w, block=block)
            np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_pair_sums_by_matches_naive_across_block_sizes(rng):
    for _ in range(10):
        n = int(rng.integers(2, 120))
        m = int(rng.integers(1, 5))
        y, r, w, c = _random_sorted_input(rng, n, n_ranks=7, n_groups=m)
        expected = naive_pair_sums_by(y, r, w, c, m)
        for block in (1, 3, 64, 4096):
            got = pair_sums_by(y, r, w, c, m, block=block)
            for key in expected:
                np.testing.assert_allclose(got[key], expected[key], rtol=1e-12, err_msg=key)


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
