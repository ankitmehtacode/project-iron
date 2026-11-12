"""Foot/head anchoring (src/pipeline/anchor.py), Day 43, on a synthetic
pinhole camera whose answers are known in closed form."""

from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.anchor import (
    HEIGHT_PRIORS,
    Camera,
    anchor_boxes,
    covered95,
)
from src.pipeline.floor import FloorProjector

K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
CENTRE = np.array([0.0, 0.0, 5.0])


def _camera() -> Camera:
    f = np.array([1.0, 0.0, -0.5])
    f /= np.linalg.norm(f)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(f, x)
    return Camera(centre=CENTRE, rotation=np.stack([x, y, f]), intrinsic=K)


def _pixel(cam: Camera, p: np.ndarray) -> np.ndarray:
    pc = cam.rotation @ (p - cam.centre)
    uv = K @ pc
    return uv[:2] / uv[2]


def _box(
    cam: Camera, xy: tuple[float, float], height: float, feet_hidden_m: float = 0.0
) -> np.ndarray:
    """A 40-px-wide box from the visible bottom to the head top."""
    bottom = _pixel(cam, np.array([xy[0], xy[1], feet_hidden_m]))
    top = _pixel(cam, np.array([xy[0], xy[1], height]))
    return np.array([bottom[0] - 20, top[1], bottom[0] + 20, bottom[1]])


def _projector(cam: Camera) -> FloorProjector:
    extrinsic = np.concatenate(
        [cam.rotation, (-cam.rotation @ cam.centre)[:, None]], axis=1
    )
    return FloorProjector(
        camera_id="synthetic",
        image_to_floor=np.linalg.inv(K @ extrinsic[:, [0, 1, 3]]),
        camera_xyz_m=(0.0, 0.0, 5.0),
        twin_rev=0,
        source="smartspaces:K@E",
        front_sign=1.0,
    )


def test_foot_mode_is_the_old_projection() -> None:
    cam = _camera()
    boxes = np.stack(
        [
            _box(
                cam,
                (
                    12.0,
                    y,
                ),
                1.7,
            )
            for y in (-3.0, 0.0, 2.5)
        ]
    )
    a = anchor_boxes(cam, boxes, ["Person"] * 3, mode="foot")
    xy, ok = _projector(cam).project_boxes(boxes)
    assert ok.all()
    np.testing.assert_allclose(a.xy, xy, atol=1e-9)


def test_visible_feet_give_the_true_height_and_the_foot_anchor() -> None:
    cam = _camera()
    a = anchor_boxes(cam, _box(cam, (12.0, 1.0), 1.7)[None], ["Person"])
    assert a.anchor.tolist() == ["foot"]
    # Exact up to the box's top-centre pixel not being the head's own
    # projection (the synthetic box takes its x from the feet): ~1 mm here.
    assert a.implied_height_m[0] == pytest.approx(1.7, abs=0.01)
    np.testing.assert_allclose(a.xy[0], [12.0, 1.0], atol=1e-6)


def test_hidden_feet_fall_back_to_the_head_anchor() -> None:
    cam = _camera()
    box = _box(cam, (12.0, 1.0), 1.7, feet_hidden_m=0.9)[None]
    foot = anchor_boxes(cam, box, ["Person"], mode="foot")
    head = anchor_boxes(cam, box, ["Person"], mode="fallback")
    assert foot.anchor.tolist() == ["foot_flagged"]
    assert head.anchor.tolist() == ["head"]
    foot_err = np.linalg.norm(foot.xy[0] - [12.0, 1.0])
    head_err = np.linalg.norm(head.xy[0] - [12.0, 1.0])
    # The head error comes only from prior 1.716 vs true 1.7 along the ray.
    assert head_err < 0.2 < foot_err
    assert covered95(head.xy - [12.0, 1.0], head.cov).all()


def test_a_cut_off_head_keeps_the_flagged_foot() -> None:
    cam = _camera()
    box = _box(cam, (12.0, 1.0), 1.7, feet_hidden_m=0.9)
    box[1] = 0.0
    a = anchor_boxes(cam, box[None], ["Person"], mode="fallback")
    assert a.anchor.tolist() == ["foot_flagged"]


def test_a_class_without_a_prior_is_unchecked() -> None:
    cam = _camera()
    a = anchor_boxes(cam, _box(cam, (12.0, 1.0), 3.0)[None], ["Forklift"])
    assert a.anchor.tolist() == ["foot_unchecked"]
    assert "Forklift" not in HEIGHT_PRIORS


def test_a_ray_above_the_horizon_is_none() -> None:
    cam = _camera()
    sky = np.array([[940.0, 0.0, 980.0, 10.0]])  # bottom edge above horizon
    a = anchor_boxes(cam, sky, ["Forklift"])
    assert a.anchor.tolist() == ["none"] and np.isnan(a.xy).all()


def test_mode_is_validated() -> None:
    with pytest.raises(ValueError, match="mode"):
        anchor_boxes(_camera(), np.zeros((0, 4)), [], mode="head")


def test_estimate_tracks_records_the_anchor_without_moving_foot_points() -> None:
    from src.contracts.frames import FrameGeometry
    from src.pipeline.run import estimate_tracks
    from src.pipeline.tracker import Assignment

    cam = _camera()
    projector = _projector(cam)
    boxes = [_box(cam, (12.0 + 0.04 * f, 1.0), 1.7) for f in range(10)]
    boxes[5] = _box(cam, (12.2, 1.0), 1.7, feet_hidden_m=0.9)
    items = [
        Assignment(1, f, 0, tuple(float(v) for v in b), 0.9)  # type: ignore[arg-type]
        for f, b in enumerate(boxes)
    ]
    geometry = FrameGeometry(width=1920, height=1080)
    old, _ = estimate_tracks(items, projector, 30.0, geometry, "t")
    new, _ = estimate_tracks(
        items, projector, 30.0, geometry, "t", anchor_camera=cam, label="person"
    )
    assert [p.anchor for p in old] == ["foot_unchecked"] * 10
    assert [p.anchor for p in new] == ["foot"] * 5 + ["foot_flagged"] + ["foot"] * 4
    np.testing.assert_allclose(
        [p.raw_xy for p in new], [p.raw_xy for p in old], atol=1e-9
    )
    fallback, _ = estimate_tracks(
        items,
        projector,
        30.0,
        geometry,
        "t",
        anchor_camera=cam,
        anchor_mode="fallback",
        label="person",
    )
    assert fallback[5].anchor == "head"
    assert np.linalg.norm(np.subtract(fallback[5].raw_xy, (12.2, 1.0))) < 0.2
