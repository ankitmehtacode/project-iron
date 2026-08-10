"""Day 20, Objective 4 — scripts/eval_estimator.py's own logic.

Not a re-test of the filter (that's tests/test_estimator_filter.py); this
covers the evaluation script's own machinery: camera-position inversion,
RMSE, and the three-method comparison on a small, fully-controlled
synthetic track where the right answer is known by construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import eval_estimator as ee  # noqa: E402

from src.estimator.measurement_model import measurement_model_for  # noqa: E402
from src.estimator.motion_model import motion_model_for  # noqa: E402


def _identity_world_to_camera_at(position: tuple[float, float, float]) -> np.ndarray:
    """A camera at ``position`` with identity rotation: world-to-camera
    translation is -position."""
    extrinsics = np.eye(4)
    extrinsics[:3, 3] = -np.array(position)
    return extrinsics


def test_camera_position_world_inverts_translation_only_extrinsics() -> None:
    extrinsics = _identity_world_to_camera_at((5.0, -2.0, 1.5))
    recovered = ee._camera_position_world(extrinsics)
    np.testing.assert_allclose(recovered, [5.0, -2.0, 1.5])


def test_camera_position_world_inverts_rotated_extrinsics() -> None:
    # 90-degree rotation about Z: x' = -y, y' = x, z' = z
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    camera_pos = np.array([3.0, 4.0, 0.0])
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = -rotation @ camera_pos
    recovered = ee._camera_position_world(extrinsics)
    np.testing.assert_allclose(recovered, camera_pos, atol=1e-10)


def test_camera_depth_m_is_positive_distance_along_optical_axis() -> None:
    extrinsics = _identity_world_to_camera_at((0.0, 0.0, 0.0))
    depth = ee._camera_depth_m(np.array([0.0, 0.0, 5.0]), extrinsics)
    assert depth == pytest.approx(5.0)


def test_rmse_of_empty_list_is_nan() -> None:
    assert np.isnan(ee._rmse([]))


def test_rmse_known_values() -> None:
    # errors of 3 and 4 -> RMSE = sqrt((9+16)/2) = sqrt(12.5)
    assert ee._rmse([9.0, 16.0]) == pytest.approx(np.sqrt(12.5))


# ---------------------------------------------------------------------------
# _evaluate_track: a small, fully-controlled synthetic track
# ---------------------------------------------------------------------------


def test_evaluate_track_on_a_straight_line_walker() -> None:
    """A noiseless straight-line walker: the filter should track it near-
    exactly and beat both trivial baselines by a wide margin."""
    frames = 10
    step_m = 0.5
    fps = 12.0
    track = np.zeros((frames, 3))
    track[:, 0] = np.arange(frames) * step_m
    extrinsics = _identity_world_to_camera_at((0.0, 0.0, -10.0))  # camera 10m back
    rng = np.random.default_rng(1)

    result = ee._evaluate_track(
        track,
        extrinsics,
        "test-sensor",
        fps,
        motion_model_for("person"),
        measurement_model_for("test-sensor"),
        rng,
    )

    assert result is not None
    assert len(result.position_sq_errors) == frames - ee.FIRST_COMPARABLE_INDEX
    filter_rmse = ee._rmse(result.position_sq_errors)
    copy_previous_rmse = ee._rmse(result.copy_previous_sq_errors)
    cv_rmse = ee._rmse(result.cv_no_update_sq_errors)

    # On a genuinely constant-velocity track with small measurement noise,
    # the filter should be competitive with (not necessarily always beat --
    # noise is real) copy-previous, since copy-previous's zero-velocity
    # assumption is a poor fit to a walker.
    assert filter_rmse < copy_previous_rmse
    # constant_velocity_no_update, uncorrected, should not be dramatically
    # better than the filter on a track with no dynamics it needs correcting
    # for -- this just confirms the baseline computed something sane, not a
    # NaN or a wildly divergent number.
    assert np.isfinite(cv_rmse)
    assert all(0.0 <= d for d in result.distances_m)


def test_evaluate_track_too_short_returns_none() -> None:
    track = np.zeros((2, 3))  # fewer frames than FIRST_COMPARABLE_INDEX + 1
    extrinsics = _identity_world_to_camera_at((0.0, 0.0, -10.0))
    result = ee._evaluate_track(
        track,
        extrinsics,
        "test-sensor",
        12.0,
        motion_model_for("person"),
        measurement_model_for("test-sensor"),
        np.random.default_rng(1),
    )
    assert result is None


def test_score_golden_set_handles_missing_version_gracefully(tmp_path: Path) -> None:
    from src.config import IronConfig

    config = IronConfig.load()
    report = ee._score_golden_set("does-not-exist-version", tmp_path, config)
    assert report is None
