import pytest

from src.eval.intervals import wilson_interval


def test_matches_published_value() -> None:
    # Newcombe (1998), Table I: 81/263 -> (0.2553, 0.3662) for the Wilson method.
    low, high = wilson_interval(81, 263)
    assert low == pytest.approx(0.2553, abs=5e-4)
    assert high == pytest.approx(0.3662, abs=5e-4)


def test_edges_stay_inside_unit_interval_and_are_not_degenerate() -> None:
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-12) and 0.0 < high < 1.0
    low, high = wilson_interval(10, 10)
    assert 0.0 < low < 1.0 and high == pytest.approx(1.0, abs=1e-12)


def test_zero_trials_is_refused() -> None:
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_successes_out_of_range_is_refused() -> None:
    with pytest.raises(ValueError):
        wilson_interval(5, 4)
