"""Floor-points calibration file: synthetic homographies round-trip (Day 45)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.data.smartspaces import twin_rev_from_sha256
from src.pipeline.floor import write_floor_points

finite = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)


def _camera_homography(yaw: float, tilt: float, height: float) -> np.ndarray:
    """floor (x, y, 1) -> pixel, for a pinhole looking down at the floor."""
    k = np.array([[900.0, 0, 960], [0, 900.0, 540], [0, 0, 1]])
    forward = np.array(
        [np.cos(yaw) * np.cos(tilt), np.sin(yaw) * np.cos(tilt), -np.sin(tilt)]
    )
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    r = np.stack([right, down, forward])
    centre = np.array([0.0, 0.0, height])
    e = np.concatenate([r, (-r @ centre)[:, None]], axis=1)
    return k @ e[:, [0, 1, 3]]


@settings(max_examples=60, deadline=None)
@given(
    yaw=st.floats(min_value=-3.0, max_value=3.0),
    tilt=st.floats(min_value=0.35, max_value=1.2),
    height=st.floats(min_value=2.0, max_value=8.0),
    jitter=st.lists(st.tuples(finite, finite), min_size=4, max_size=8),
)
def test_a_synthetic_homography_round_trips(
    tmp_path_factory: object, yaw: float, tilt: float, height: float, jitter: list
) -> None:
    floor_to_image = _camera_homography(yaw, tilt, height)
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    side = np.array([-np.sin(yaw), np.cos(yaw)])
    # Calibration marks spread on the floor in front of the camera.
    base = np.array([[3, -2], [3, 2], [7, -3], [7, 3], [5, 0], [4, -1], [6, 1], [8, 0]])
    pts = base[: len(jitter)] + np.array(jitter)
    floor = pts[:, :1] * forward + pts[:, 1:] * side
    h = np.concatenate([floor, np.ones((len(floor), 1))], axis=1) @ floor_to_image.T
    image = h[:, :2] / h[:, 2:]
    out_dir = tmp_path_factory.mktemp("cal")  # type: ignore[attr-defined]
    path = Path(str(out_dir)) / "cam.yaml"
    projector = write_floor_points(
        path, "cam", image, floor, (0.0, 0.0, height), "test"
    )
    back, ok = projector.project(image)
    assert ok.all()
    np.testing.assert_allclose(back, floor, atol=1e-6)
    # Points that were not clicked map through the same homography.
    probe_floor = 5.5 * forward + 0.7 * side
    hp = np.r_[probe_floor, 1.0] @ floor_to_image.T
    probe, ok = projector.project((hp[:2] / hp[2])[None])
    assert ok.all()
    np.testing.assert_allclose(probe[0], probe_floor, atol=1e-6)
    assert projector.twin_rev == twin_rev_from_sha256(
        hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_a_reclick_is_a_new_twin_rev(tmp_path: Path) -> None:
    floor = np.array([[3, -2], [3, 2], [7, -3], [7, 3.0]])
    floor_to_image = _camera_homography(0.0, 0.6, 3.0)
    h = np.concatenate([floor, np.ones((4, 1))], axis=1) @ floor_to_image.T
    image = h[:, :2] / h[:, 2:]
    a = write_floor_points(tmp_path / "a.yaml", "cam", image, floor, (0, 0, 3.0))
    b = write_floor_points(tmp_path / "b.yaml", "cam", image + 0.5, floor, (0, 0, 3.0))
    assert a.twin_rev != b.twin_rev
