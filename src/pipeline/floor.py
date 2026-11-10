"""Image pixels to floor-plane world coordinates.

Both calibration forms reduce to one 3x3 homography from image pixels to
floor ``(x, y)`` in metres, world frame ``X_Y_Z_UP`` with the floor at
``z = 0``:

- SmartSpaces ``calibration.json``: ``P = K @ E`` maps ``[x, y, z, 1]`` to
  pixels; on ``z = 0`` that is ``K @ E[:, [0, 1, 3]]``, inverted. The scene's
  person floor points sit at z = 0.007 m median (Warehouse_022, Day 43), so
  the plane is the floor.
- A four-point floor file for a camera without calibration: four (or more)
  image points and the floor points they lie on, solved by DLT
  (``cv2.findHomography`` with no RANSAC). It must also state the camera's
  floor-frame position, which the estimator's range-dependent noise needs and
  a floor homography alone cannot give without intrinsics.

``twin_rev`` is the calibration file's content hash
(:func:`src.data.smartspaces.twin_rev_from_sha256`): a recalibration is a new
revision because the bytes changed, not because someone bumped a counter.

A box is projected at its bottom-centre. When the feet are occluded the
visible box stops above the floor and the point lands too far from the
camera; that error is measured, not corrected (p90 0.30-1.74 m per camera on
ground-truth boxes, Warehouse_022).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml

from src.data.smartspaces import load_calibration, twin_rev_from_sha256

FloatArray = npt.NDArray[np.float64]


class CalibrationError(ValueError):
    """The calibration file cannot produce a floor projection for this camera."""


@dataclass(frozen=True)
class FloorProjector:
    """Pixels to floor metres for one camera, at one calibration revision."""

    camera_id: str
    image_to_floor: FloatArray
    """3x3, homogeneous pixel ``[u, v, 1]`` to homogeneous floor ``[x, y, w]``."""
    camera_xyz_m: tuple[float, float, float]
    twin_rev: int
    source: str
    """``smartspaces:K@E`` or ``floor_points``: which form produced it."""
    front_sign: float
    """Sign the homogeneous ``w`` takes for floor points in front of the
    camera (+1 for K@E; read off the measured points for a floor file)."""

    def project(self, uv: FloatArray) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
        """``[N, 2]`` pixels to ``([N, 2]`` floor metres, ``[N]`` valid``)``.

        Invalid means the pixel lies on or above the horizon (the ray never
        meets the floor in front of the camera); its row is NaN, never a
        clipped or guessed position."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        homogeneous = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)
        mapped = homogeneous @ self.image_to_floor.T
        w = mapped[:, 2]
        valid = np.isfinite(w) & (np.abs(w) > 1e-12) & (w * self.front_sign > 0)
        xy = np.full((uv.shape[0], 2), np.nan)
        xy[valid] = mapped[valid, :2] / w[valid, None]
        return xy, valid

    def project_boxes(
        self, boxes_xyxy: FloatArray
    ) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
        """Floor position of each box's bottom-centre."""
        boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
        feet = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]], axis=1)
        return self.project(feet)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def from_smartspaces(path: Path, camera_id: str) -> FloorProjector:
    calibration = load_calibration(path)
    if camera_id not in calibration.cameras:
        raise CalibrationError(
            f"{camera_id!r} is not in {path.name}; cameras: "
            f"{sorted(calibration.cameras)}"
        )
    camera = calibration.cameras[camera_id]
    floor_to_image = camera.intrinsic @ camera.extrinsic[:, [0, 1, 3]]
    rotation, translation = camera.extrinsic[:, :3], camera.extrinsic[:, 3]
    position = -rotation.T @ translation
    # Under P = K @ E, a point in front of the camera has positive depth, and
    # the floor-to-image map's third row IS that depth for z = 0. Its inverse
    # therefore returns w with the sign of 1/depth: positive in front.
    return FloorProjector(
        camera_id=camera_id,
        image_to_floor=np.linalg.inv(floor_to_image),
        camera_xyz_m=(float(position[0]), float(position[1]), float(position[2])),
        twin_rev=calibration.twin_rev,
        source="smartspaces:K@E",
        front_sign=1.0,
    )


def from_floor_points(path: Path, camera_id: str) -> FloorProjector:
    """A file with ``camera_id``, ``image_points`` ([u, v] each),
    ``floor_points`` ([x, y] metres each, same order, at least four) and
    ``camera_xyz_m``."""
    import cv2

    raw = yaml.safe_load(path.read_text())
    expected = {"camera_id", "image_points", "floor_points", "camera_xyz_m"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise CalibrationError(
            f"{path.name} must have exactly the keys {sorted(expected)}"
        )
    if raw["camera_id"] != camera_id:
        raise CalibrationError(
            f"{path.name} is for {raw['camera_id']!r}, not {camera_id!r}"
        )
    image = np.asarray(raw["image_points"], dtype=np.float64)
    floor = np.asarray(raw["floor_points"], dtype=np.float64)
    if image.shape != floor.shape or image.ndim != 2 or image.shape[1] != 2:
        raise CalibrationError("image_points and floor_points must both be [N, 2]")
    if image.shape[0] < 4:
        raise CalibrationError("a floor homography needs at least four points")
    homography, _ = cv2.findHomography(image, floor, method=0)
    if homography is None:
        raise CalibrationError(f"{path.name}: points are degenerate (collinear?)")
    xyz = [float(v) for v in raw["camera_xyz_m"]]
    if len(xyz) != 3:
        raise CalibrationError("camera_xyz_m must be [x, y, z]")
    position = (xyz[0], xyz[1], xyz[2])
    # The measured image points are in front of the camera by construction;
    # their w sign is the in-front sign.
    w = np.concatenate([image, np.ones((image.shape[0], 1))], axis=1) @ homography[2]
    if not (np.all(w > 0) or np.all(w < 0)):
        raise CalibrationError(
            f"{path.name}: the image points straddle the horizon line"
        )
    return FloorProjector(
        camera_id=camera_id,
        image_to_floor=homography,
        camera_xyz_m=position,
        twin_rev=twin_rev_from_sha256(_sha256(path)),
        source="floor_points",
        front_sign=float(np.sign(w[0])),
    )


def load_projector(path: Path, camera_id: str) -> FloorProjector:
    """Dispatch on the file's content, not its name."""
    text = path.read_text()
    try:
        is_smartspaces = "calibrationType" in json.loads(text)
    except json.JSONDecodeError:
        is_smartspaces = False
    if is_smartspaces:
        return from_smartspaces(path, camera_id)
    return from_floor_points(path, camera_id)
