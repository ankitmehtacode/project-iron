"""Day 26, Objective 4 -- scripts/eval_joint_estimator.py's own logic.

Not a re-test of the joint filter (that's tests/test_estimator_joint.py);
this covers the evaluation script's own machinery: RMSE, coverage, margin
sign convention, and the directional check adapted to a joint-vs-
independent comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import eval_joint_estimator as ej  # noqa: E402


def test_rmse_of_empty_list_is_nan() -> None:
    assert np.isnan(ej._rmse([]))


def test_rmse_known_values() -> None:
    assert ej._rmse([9.0, 16.0]) == pytest.approx(np.sqrt(12.5))


def test_margin_positive_when_joint_beats_independent() -> None:
    # independent (baseline) RMSE 0.5, joint (candidate) RMSE 0.3 -> joint wins
    assert ej._margin(candidate_rmse=0.3, baseline_rmse=0.5) == pytest.approx(0.2)


def test_margin_negative_when_joint_loses_to_independent() -> None:
    assert ej._margin(candidate_rmse=0.6, baseline_rmse=0.5) == pytest.approx(-0.1)


def test_margin_nan_propagates() -> None:
    assert np.isnan(ej._margin(candidate_rmse=float("nan"), baseline_rmse=0.5))


def test_directional_check_pass_when_coverage_improves() -> None:
    result = ej._directional_check(
        "carrier", baseline_coverage=0.80, candidate_coverage=0.94
    )
    assert result is not None
    assert result.verdict == "PASS"


def test_directional_check_fail_overconfident_when_coverage_drops_below_nominal() -> (
    None
):
    result = ej._directional_check(
        "carrier", baseline_coverage=0.95, candidate_coverage=0.70
    )
    assert result is not None
    assert result.verdict == "FAIL_OVERCONFIDENT"


def test_directional_check_pass_with_cost_when_coverage_moves_above_nominal() -> None:
    result = ej._directional_check(
        "asset", baseline_coverage=0.95, candidate_coverage=0.995
    )
    assert result is not None
    assert result.verdict == "PASS_WITH_COST"


def test_directional_check_small_wiggle_within_tolerance_is_pass() -> None:
    result = ej._directional_check(
        "carrier", baseline_coverage=0.95, candidate_coverage=0.94
    )
    assert result is not None
    assert result.verdict == "PASS"


def test_directional_check_none_when_either_coverage_is_nan() -> None:
    assert ej._directional_check("carrier", float("nan"), 0.9) is None
    assert ej._directional_check("carrier", 0.9, float("nan")) is None


def test_carry_offset_is_declared_nonzero() -> None:
    """A zero offset would make the carried-asset scenario degenerate to
    the carrier's own track -- the offset must actually displace the
    synthetic asset for this evaluation to test anything."""
    assert np.linalg.norm(ej.CARRY_OFFSET_M) > 0.0
