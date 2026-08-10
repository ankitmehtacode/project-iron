"""Day 23, Objective 3 -- scripts/velocity_floor_frame_rate_sweep.py.

Regression anchors for the day's measured finding (not a re-test of the
filter itself, see tests/test_estimator_filter.py): the velocity-
covariance floor does not bind at this project's 12fps, and does not
bind anywhere in the swept range even at its closest approach.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import velocity_floor_frame_rate_sweep as sweep  # noqa: E402

from src.estimator.motion_model import (  # noqa: E402
    pedestrian_velocity_covariance_floor_mps2,
)


def test_natural_velocity_variance_is_positive_finite_at_12fps() -> None:
    natural = sweep._natural_velocity_variance_mps2(12.0)
    assert natural > 0.0
    assert natural < float("inf")


def test_floor_does_not_bind_at_12fps() -> None:
    """The Day-22/ADR-0010 finding, reproduced: floor stays well below
    natural convergence at this project's own frame rate."""
    natural = sweep._natural_velocity_variance_mps2(12.0)
    floor = pedestrian_velocity_covariance_floor_mps2(1.0 / 12.0)
    assert floor < natural
    assert floor / natural < 0.3


def test_floor_does_not_bind_at_its_closest_measured_approach() -> None:
    """fps=2 is where the swept ratio comes closest to 1.0 (Day 23) --
    still comfortably under it. If this ever flips, the floor's
    retirement finding needs revisiting, not just this test."""
    natural = sweep._natural_velocity_variance_mps2(2.0)
    floor = pedestrian_velocity_covariance_floor_mps2(1.0 / 2.0)
    assert floor < natural


def test_natural_velocity_variance_is_non_monotonic_in_fps() -> None:
    """Day 23's least-obvious finding: natural convergence is NOT
    monotonic in frame rate. It is high at very low fps, reaches a
    minimum in the low hundreds, and rises again at very high fps
    (differencing noisier-relative-to-motion samples). Guards against
    the intuitive-but-wrong assumption that "higher fps -> tighter
    convergence -> the floor binds sooner" -- measured, it is the
    opposite in the high-fps direction."""
    low = sweep._natural_velocity_variance_mps2(12.0)
    mid = sweep._natural_velocity_variance_mps2(120.0)
    high = sweep._natural_velocity_variance_mps2(1000.0)
    assert mid < low
    assert high > mid
