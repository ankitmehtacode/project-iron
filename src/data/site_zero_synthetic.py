"""A synthetic stand-in for the site-zero-solo capture (Day 46).

The real capture (``data/raw/site_zero_solo/``: clips A, B, C, ``marks.csv``,
segment notes) does not exist yet. This module writes a directory with the
same layout and a KNOWN answer, so the code that will score the real capture
can be run end to end now:

- a floor seen by one pinhole camera whose pose is fixed here, so the
  image-to-floor homography is known exactly;
- ``marks.csv``: six ``calib`` marks and six ``test`` marks in floor metres;
- ``clicks.json``: the pixel of each calib mark, computed from the camera,
  standing in for the human's clicks on the calibration page;
- ``A.mp4``: the empty room; ``B.mp4``: a figure that walks in, stands on
  each test mark, and walks out; ``C.mp4``: two entries into and exits from
  the room zone, then an empty stretch;
- ``notes.yaml``: the segment notes (when the figure stands on which mark,
  which stretches are empty, how many entries and exits happened);
- ``zones.yaml``: the room zone, fixed here before any run;
- ``truth.json``: the figure's image box per clip and frame. It feeds the
  oracle detector only (the pretrained detector is not run on rendered
  frames: an appearance-learned capability on analytic primitives measures
  nothing, see iron-eval-discipline's synthetic-evaluability partition).

The figure is a flat upright rectangle (0.5 m x 1.7 m) in a plane of constant
floor ``y``, so its bottom edge projects to a horizontal image line and the
bottom-centre of its image box lies on the floor point it stands on, up to the
perspective lean of its sides. No real person and no real frame is involved.
The sensor noise (Gaussian, sigma 1.5 grey levels, seeded) exists so the
empty room is not bit-identical frame to frame; it is not a measured noise
model, so any gate number on these clips is a mechanism check, not a product
number.
"""

from __future__ import annotations

import csv
import functools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

FloatArray = npt.NDArray[np.float64]

WIDTH, HEIGHT, FPS = 640, 360, 10.0
CAMERA_ID = "site_zero_cam"
CAMERA_XYZ = (0.0, -1.0, 2.6)
PITCH_DEG = 30.0
FOCAL_PX = 500.0
FIGURE_HALF_WIDTH_M = 0.25
FIGURE_HEIGHT_M = 1.7
WALK_MPS = 1.2
NOISE_SIGMA = 1.5
FIGURE_BGR = (30, 30, 40)
"""Dark against the 110/150 floor: the gate differences grey levels."""

CALIB_MARKS = {
    "C1": (-2.0, 3.0),
    "C2": (2.0, 3.0),
    "C3": (-2.5, 8.0),
    "C4": (2.5, 8.0),
    "C5": (0.0, 5.5),
    "C6": (-1.0, 9.5),
}
TEST_MARKS = {
    "T1": (-1.0, 4.0),
    "T2": (1.0, 4.0),
    "T3": (0.0, 5.0),
    "T4": (-1.5, 6.5),
    "T5": (1.5, 6.5),
    "T6": (0.0, 7.5),
}
ROOM_ZONE = ((-1.5, 2.0), (3.5, 2.0), (3.5, 9.0), (-1.5, 9.0))
"""The room: everything right of the doorway line x = -1.5 m."""


def camera_matrices() -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(K, R, C)``: intrinsics, world-to-camera rotation, camera centre.

    Camera frame: x right, y down, z forward; it looks along +y, pitched
    down by :data:`PITCH_DEG`."""
    s, c = np.sin(np.radians(PITCH_DEG)), np.cos(np.radians(PITCH_DEG))
    z_axis = np.array([0.0, c, -s])
    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.stack([x_axis, y_axis, z_axis])
    k = np.array(
        [[FOCAL_PX, 0.0, WIDTH / 2], [0.0, FOCAL_PX, HEIGHT / 2], [0.0, 0.0, 1.0]]
    )
    return k, rotation, np.array(CAMERA_XYZ)


def project(points_xyz: FloatArray) -> FloatArray:
    """World ``[N, 3]`` to pixel ``[N, 2]`` (all points must be in front)."""
    k, rotation, centre = camera_matrices()
    cam = (np.asarray(points_xyz, np.float64).reshape(-1, 3) - centre) @ rotation.T
    if np.any(cam[:, 2] <= 0):
        raise ValueError("a point is behind the camera")
    uv = cam @ k.T
    out: FloatArray = uv[:, :2] / uv[:, 2:3]
    return out


def floor_to_image() -> FloatArray:
    """The true 3x3 floor-``(x, y, 1)``-to-pixel homography."""
    k, rotation, centre = camera_matrices()
    t = -rotation @ centre
    out: FloatArray = k @ np.column_stack([rotation[:, 0], rotation[:, 1], t])
    return out


@functools.lru_cache(maxsize=1)
def _background() -> npt.NDArray[np.uint8]:
    """A 0.5 m checker floor below the horizon, a flat wall above it."""
    image_to_floor = np.linalg.inv(floor_to_image())
    u, v = np.meshgrid(np.arange(WIDTH) + 0.5, np.arange(HEIGHT) + 0.5)
    pix = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], axis=1)
    mapped = pix @ image_to_floor.T
    w = mapped[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x, y = mapped[:, 0] / w, mapped[:, 1] / w
    floor = (w > 0) & (y > CAMERA_XYZ[1])
    tile = (np.floor(x / 0.5) + np.floor(y / 0.5)).astype(np.int64) % 2
    grey = np.where(floor, np.where(tile == 0, 150, 110), 190).astype(np.uint8)
    img = np.repeat(grey.reshape(HEIGHT, WIDTH, 1), 3, axis=2)
    return np.ascontiguousarray(img)


def figure_corners(x: float, y: float) -> FloatArray:
    """The figure's four image corners when it stands at floor ``(x, y)``."""
    h, a = FIGURE_HEIGHT_M, FIGURE_HALF_WIDTH_M
    return project(
        np.array([[x - a, y, 0.0], [x + a, y, 0.0], [x + a, y, h], [x - a, y, h]])
    )


@dataclass
class Clip:
    name: str
    positions: list[tuple[float, float] | None]
    """Per frame: where the figure stands, or ``None`` (not in the scene)."""
    segments: list[dict[str, Any]] = field(default_factory=list)
    expected_events: dict[str, dict[str, int]] = field(default_factory=dict)


def _walk(a: tuple[float, float], b: tuple[float, float]) -> list[tuple[float, float]]:
    """Samples from ``a`` toward ``b`` at walking speed, ``b`` excluded."""
    dist = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    n = max(1, int(np.ceil(dist / (WALK_MPS / FPS))))
    return [
        (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(n)
    ]


def _clip_a() -> Clip:
    n = int(10 * FPS)
    return Clip("A", [None] * n, [{"kind": "empty", "start_s": 0.0, "end_s": n / FPS}])


def _clip_b(stand_s: float) -> Clip:
    off = (-5.0, 4.0)
    pos: list[tuple[float, float] | None] = [None] * int(1 * FPS)
    segments = []
    here = off
    for name, mark in TEST_MARKS.items():
        pos.extend(_walk(here, mark))
        start = len(pos) / FPS
        pos.extend([mark] * int(stand_s * FPS))
        segments.append(
            {"kind": "on_mark", "mark": name, "start_s": start, "end_s": len(pos) / FPS}
        )
        here = mark
    pos.extend(_walk(here, (-5.0, 7.5)))
    pos.extend([None] * int(1 * FPS))
    return Clip("B", pos, segments)


def _clip_c() -> Clip:
    pos: list[tuple[float, float] | None] = [None] * int(1 * FPS)
    segments: list[dict[str, Any]] = []
    for y in (5.0, 6.0):
        outside, inside = (-5.0, y), (1.0, y)
        pos.extend(_walk(outside, inside))
        pos.extend([inside] * int(1 * FPS))
        pos.extend(_walk(inside, outside))
        start = len(pos) / FPS
        pos.extend([None] * int(3 * FPS))
        segments.append({"kind": "empty", "start_s": start, "end_s": len(pos) / FPS})
    start = len(pos) / FPS
    pos.extend([None] * int(6 * FPS))
    segments.append({"kind": "empty", "start_s": start, "end_s": len(pos) / FPS})
    return Clip("C", pos, segments, {"room": {"entered": 2, "exited": 2}})


def _visible_box(x: float, y: float) -> list[float] | None:
    corners = figure_corners(x, y)
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    x0, x1 = max(x0, 0.0), min(x1, float(WIDTH))
    y0, y1 = max(y0, 0.0), min(y1, float(HEIGHT))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return [float(x0), float(y0), float(x1), float(y1)]


def _write_video(path: Path, clip: Clip, rng: np.random.Generator) -> list[Any]:
    import cv2

    background = _background()
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open a video writer for {path}")
    boxes: list[Any] = []
    try:
        for p in clip.positions:
            frame = background.astype(np.float64)
            box = None
            if p is not None:
                corners = figure_corners(*p).round().astype(np.int32)
                canvas = frame.astype(np.uint8)
                cv2.fillConvexPoly(canvas, corners, FIGURE_BGR)
                frame = canvas.astype(np.float64)
                box = _visible_box(*p)
            frame += rng.normal(0.0, NOISE_SIGMA, frame.shape)
            writer.write(np.clip(frame.round(), 0, 255).astype(np.uint8))
            boxes.append(box)
    finally:
        writer.release()
    return boxes


def write_stand_in(root: Path, stand_s: float = 4.0, seed: int = 46) -> dict[str, Any]:
    """Write the stand-in capture under ``root``; return its known answers."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with open(root / "marks.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["name", "x_m", "y_m", "role"])
        for name, (x, y) in CALIB_MARKS.items():
            out.writerow([name, x, y, "calib"])
        for name, (x, y) in TEST_MARKS.items():
            out.writerow([name, x, y, "test"])
    clicks = {
        name: [float(u), float(v)]
        for name, (u, v) in zip(
            CALIB_MARKS,
            project(np.array([[x, y, 0.0] for x, y in CALIB_MARKS.values()])),
        )
    }
    (root / "clicks.json").write_text(
        json.dumps({"points": clicks, "camera_xyz_m": list(CAMERA_XYZ)}, indent=2)
    )
    (root / "zones.yaml").write_text(
        yaml.safe_dump(
            {
                "site_id": "site-zero-solo-synthetic",
                "zones": [{"name": "room", "polygon": [list(p) for p in ROOM_ZONE]}],
                "rules": {"min_presence_s": 0.5, "dwell_s": 5.0, "max_gap_s": 2.0},
            },
            sort_keys=False,
        )
    )
    truth: dict[str, Any] = {}
    clips = {}
    for clip in (_clip_a(), _clip_b(stand_s), _clip_c()):
        file = f"{clip.name}.mp4"
        boxes = _write_video(root / file, clip, rng)
        truth[clip.name] = {str(i): [b] for i, b in enumerate(boxes) if b is not None}
        clips[clip.name] = {
            "file": file,
            "segments": clip.segments,
            **(
                {"expected_events": clip.expected_events}
                if clip.expected_events
                else {}
            ),
        }
    (root / "truth.json").write_text(json.dumps(truth))
    notes = {"camera_id": CAMERA_ID, "clips": clips}
    (root / "notes.yaml").write_text(yaml.safe_dump(notes, sort_keys=False))
    return {
        "camera_id": CAMERA_ID,
        "test_marks": TEST_MARKS,
        "expected_events": {"C": {"room": {"entered": 2, "exited": 2}}},
        "empty_person_detections": 0,
    }
