import numpy as np
import pytest
from conftest import naive_wtd_rank

from stratindex._utils import wtd_rank


def test_unit_weights_are_midranks():
    # rank() midranks: ties share the average of their positions
    assert wtd_rank([1, 2, 2, 3]).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert wtd_rank([30, 10, 20]).tolist() == [3.0, 1.0, 2.0]


def test_weighted_hand_example():
    # w normalized to sum 4: [0.5, 1, 0.5, 2]
    # freqs: 10 -> 0.5, 20 -> 1.5, 30 -> 2; cumsum: 0.5, 2, 4
    # r_k = cumsum - 0.5 * (freq - 1): 0.75, 1.75, 3.5
    got = wtd_rank([10, 20, 20, 30], [1, 2, 1, 4])
    assert got == pytest.approx([0.75, 1.75, 1.75, 3.5])


def test_matches_naive_on_random_data(rng):
    for _ in range(20):
        n = int(rng.integers(2, 200))
        x = rng.normal(size=n).round(1)  # rounding creates ties
        w = rng.uniform(0.1, 5.0, size=n)
        np.testing.assert_allclose(wtd_rank(x, w), naive_wtd_rank(x, w), rtol=1e-12)


def test_equal_outcomes_get_exactly_equal_ranks(rng):
    x = rng.choice([1.0, 2.0, 3.0], size=100)
    w = rng.uniform(0.1, 5.0, size=100)
    r = wtd_rank(x, w)
    for v in (1.0, 2.0, 3.0):
        vals = r[x == v]
        assert (vals == vals[0]).all()
