"""Motion and observability are two questions and need two answers.

Day 9 replaced a metres threshold with "the rendered silhouette changed". That
was right about *observability* and wrong as a definition of *motion*: an agent
walking outside the frustum has an unchanging empty mask, so it read as "did
not move" and the coverage gap became inexpressible.

The two quantities:

* **Does the agent move?** World-space displacement. A property of the scene,
  the same from every camera. The *threshold* is derived from what a given
  sensor can resolve, but the quantity is not camera-dependent.
* **Can this camera see it move?** Silhouette on the sensor. A property of the
  view, and different for every camera looking at the same agent.

Collapsing them means a blind spot and an empty room produce the same label,
which is exactly the finding v3's coverage-gap clips exist to preserve.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import IronConfig
from src.data.scorecard import (
    FIRST_AGENT_INSTANCE_ID,
    Observability,
    observability_partition,
)


@pytest.fixture(scope="module")
def gate_config():  # type: ignore[no-untyped-def]
    return IronConfig.load().cascade.motion_gate_config()


def _clip(
    path: Path,
    *,
    frames: int,
    world_step_m: float,
    on_sensor: bool,
    depth_m: float = 6.0,
    size: tuple[int, int] = (720, 1280),
    area_px: int = 40_000,
) -> Path:
    """A clip with independently controlled world motion and visibility.

    The whole point is that these two knobs are separate. ``world_step_m``
    moves the agent in the scene; ``on_sensor`` decides whether any of it
    reaches this camera.
    """
    height, width = size
    instances = np.zeros((frames, height, width), dtype=np.int32)
    xyz = np.zeros((frames, 1, 3), dtype=np.float32)
    uv = np.zeros((frames, 1, 2), dtype=np.float32)

    for frame in range(frames):
        xyz[frame, 0, 0] = frame * world_step_m
        xyz[frame, 0, 2] = depth_m
        if on_sensor:
            start = frame * 200
            flat = instances[frame].reshape(-1)
            flat[start : start + area_px] = FIRST_AGENT_INSTANCE_ID
            uv[frame, 0, 0] = frame * 30.0

    np.savez_compressed(
        path,
        rgb=np.zeros((frames, height, width, 3), dtype=np.uint8),
        depth_m=np.full((frames, height, width), depth_m, dtype=np.float32),
        instances=instances,
        agent_xyz=xyz,
        track_uv=uv,
        track_occluded=np.zeros((frames, 1), dtype=bool),
        intrinsics=np.array([900.0, 900.0, width / 2, height / 2]),
        extrinsics=np.eye(4),
    )
    return path


def test_offfrustum_mover_is_moving_and_unobservable(
    tmp_path: Path, gate_config
) -> None:
    """The category error, pinned.

    An agent striding through the scene where this camera cannot see it must
    be **moving** and **not observable**. Labelling it "no motion" erases the
    difference between a blind spot and an empty room, and the coverage-gap
    clips exist precisely to keep that difference measurable.
    """
    clip = _clip(tmp_path / "blind.npz", frames=14, world_step_m=0.30, on_sensor=False)
    labels = observability_partition(clip, gate_config).after_warmup()

    assert np.all(labels == Observability.NOT_OBSERVABLE), (
        "world motion this camera cannot see must be NOT_OBSERVABLE, never "
        f"NO_MOTION; got {sorted(set(labels.tolist()))}"
    )


def test_a_genuinely_still_agent_is_no_motion(tmp_path: Path, gate_config) -> None:
    """The other side of the same coin: still is still, on or off sensor."""
    clip = _clip(tmp_path / "still.npz", frames=14, world_step_m=0.0, on_sensor=False)
    labels = observability_partition(clip, gate_config).after_warmup()
    assert np.all(labels == Observability.NO_MOTION)


def test_visible_mover_is_scored_not_excluded(tmp_path: Path, gate_config) -> None:
    """A large, fast, visible mover is the recall denominator."""
    clip = _clip(tmp_path / "seen.npz", frames=14, world_step_m=0.30, on_sensor=True)
    labels = observability_partition(clip, gate_config).after_warmup()
    assert np.all(labels == Observability.ABOVE_ENVELOPE)


def test_motion_threshold_is_derived_from_what_the_sensor_resolves(
    tmp_path: Path, gate_config
) -> None:
    """The threshold scales with distance; the quantity stays world motion.

    A fixed threshold in metres treats a 1 cm step at 2 m and at 20 m as the
    same event. They are not: one moves several pixels and the other moves a
    fraction of one. The bound must come from the sensor — a displacement
    below one pixel at the observing camera is not motion anything could have
    seen — while the thing being thresholded stays a world quantity, so it
    means the same from every camera.
    """
    # A step that subtends well under a pixel at 20 m but over a pixel at 2 m.
    # fx = 900, so one pixel at 20 m is 20/900 = 22 mm; at 2 m it is 2.2 mm.
    step = 0.006

    near = _clip(
        tmp_path / "near.npz",
        frames=14,
        world_step_m=step,
        on_sensor=False,
        depth_m=2.0,
    )
    far = _clip(
        tmp_path / "far.npz",
        frames=14,
        world_step_m=step,
        on_sensor=False,
        depth_m=20.0,
    )

    near_labels = observability_partition(near, gate_config).after_warmup()
    far_labels = observability_partition(far, gate_config).after_warmup()

    assert np.all(near_labels == Observability.NOT_OBSERVABLE), (
        "6 mm at 2 m subtends about 2.7 px and is real motion this camera "
        "simply cannot see from where it stands"
    )
    assert np.all(far_labels == Observability.NO_MOTION), (
        "the same 6 mm at 20 m subtends about a quarter of a pixel. No sensor "
        "here could resolve it, so calling it motion asks the gate to detect "
        "something no camera in this scene could have rendered"
    )
