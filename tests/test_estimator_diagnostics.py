"""Day 21, Objective 2 — innovation whiteness diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from src.estimator.diagnostics import is_white, lag1_autocorrelation, white_noise_bound

SEED = 20260731


def test_lag1_autocorrelation_of_white_noise_is_near_zero() -> None:
    rng = np.random.default_rng(SEED)
    sequences = [list(rng.normal(size=500))]
    correlation, n_pairs = lag1_autocorrelation(sequences)
    assert n_pairs == 499
    assert abs(correlation) < 0.15  # loose: a single finite draw, not exactly 0


def test_lag1_autocorrelation_of_strongly_correlated_sequence_is_high() -> None:
    rng = np.random.default_rng(SEED)
    n = 500
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.normal(0, 0.1)
    correlation, n_pairs = lag1_autocorrelation([list(x)])
    assert n_pairs == n - 1
    assert correlation > 0.6


def test_boundaries_between_sequences_are_not_paired() -> None:
    """Two sequences, each internally constant-alternating (so a pair
    formed ACROSS the boundary would show strong correlation the
    within-sequence pairs do not) -- the result must reflect only the
    within-sequence pairs."""
    seq_a = [1.0, -1.0, 1.0, -1.0, 1.0]  # perfect anti-correlation within
    seq_b = [1.0, -1.0, 1.0, -1.0, 1.0]
    correlation, n_pairs = lag1_autocorrelation([seq_a, seq_b])
    # 4 pairs per sequence, 8 total -- NOT 9, which is what a naive
    # concatenation (ignoring the boundary) would produce.
    assert n_pairs == 8
    assert correlation == pytest.approx(-1.0, abs=1e-9)


def test_lag1_autocorrelation_too_short_returns_nan() -> None:
    correlation, n_pairs = lag1_autocorrelation([[1.0]])
    assert np.isnan(correlation)
    assert n_pairs == 0


def test_lag1_autocorrelation_constant_sequence_returns_nan() -> None:
    correlation, n_pairs = lag1_autocorrelation([[5.0, 5.0, 5.0, 5.0]])
    assert np.isnan(correlation)
    assert n_pairs == 3


def test_white_noise_bound_shrinks_with_more_pairs() -> None:
    assert white_noise_bound(100) > white_noise_bound(10000)


def test_white_noise_bound_matches_known_z_score() -> None:
    # 95% two-sided z ~ 1.959964
    bound = white_noise_bound(10000)
    assert bound == pytest.approx(1.959964 / 100, abs=1e-4)


def test_white_noise_bound_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        white_noise_bound(0)
    with pytest.raises(ValueError):
        white_noise_bound(10, confidence=1.5)


def test_is_white_true_for_genuine_white_noise() -> None:
    rng = np.random.default_rng(SEED)
    sequences = [list(rng.normal(size=200)) for _ in range(5)]
    white, correlation, bound, n_pairs = is_white(sequences)
    assert white is True
    assert n_pairs == 5 * 199
    assert abs(correlation) <= bound


def test_is_white_false_for_correlated_sequence() -> None:
    rng = np.random.default_rng(SEED)
    n = 200
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.8 * x[i - 1] + rng.normal(0, 0.1)
    white, correlation, bound, n_pairs = is_white([list(x)])
    assert white is False
    assert correlation > bound


def test_is_white_insufficient_evidence_is_not_white() -> None:
    white, correlation, bound, n_pairs = is_white([[1.0, 2.0, 3.0]])
    assert white is False
    assert n_pairs == 2
    assert np.isnan(bound)
