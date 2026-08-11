"""Day 23, Objective 3 -- scripts/velocity_floor_frame_rate_sweep.py.
Superseded Day 24, Objective 2: the floor formula this script sweeps was
per-timestep (scaled by dt_s), which is why Day 23 measured it as never
binding anywhere from 1-1000fps -- the formula shrank with dt_s faster
than natural convergence did. Day 24 re-derived the floor as an ABSOLUTE
bound (anchored to PEDESTRIAN_STOP_DURATION_S, not dt_s; see
motion_model.py's "Day 24 correction" docstring). Re-measured with the
corrected formula: the floor now binds at every fps from 2 to 1000, and
does not bind only at fps=1 (dt_s=1.0s), the single point where the two
derivations coincide by construction. These are regression anchors for
the corrected finding.
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


def test_floor_binds_at_12fps() -> None:
    """Day 24 correction: the absolute floor (2.25 (m/s)^2) now sits well
    above natural convergence (~0.065 (m/s)^2) at this project's own
    frame rate -- the opposite of Day 22-23's per-timestep-floor finding,
    which was an artifact of that formula's dt_s scaling, not a fact
    about this filter's physics."""
    natural = sweep._natural_velocity_variance_mps2(12.0)
    floor = pedestrian_velocity_covariance_floor_mps2(1.0 / 12.0)
    assert floor > natural
    assert floor / natural > 30.0


def test_floor_does_not_bind_only_at_1fps() -> None:
    """fps=1 (dt_s=1.0s=PEDESTRIAN_STOP_DURATION_S) is the single point
    where the corrected absolute formula and the old per-timestep formula
    coincide by construction -- natural convergence there (~3.86 (m/s)^2)
    is the one measured case that still exceeds the floor (2.25 (m/s)^2).
    Every other tested rate now binds (see test_floor_binds_at_12fps and
    the module docstring)."""
    natural = sweep._natural_velocity_variance_mps2(1.0)
    floor = pedestrian_velocity_covariance_floor_mps2(1.0)
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
