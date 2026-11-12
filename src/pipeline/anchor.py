"""Where on the floor a box stands: foot anchor, implied-height check, and a
head-anchor fallback at a class-specific height prior (Day 43).

The slice projected every box at its bottom-centre (the foot anchor). When
the feet are hidden (shelving, another person, a forklift) the visible box
stops above the floor and the bottom-centre ray meets the floor far behind
the person: on ground-truth boxes the median error is 0.13-0.40 m but p90
reaches 8-75 m on low cameras (Warehouse_020/000, Day 43). This module adds
three things, all stated before any number was computed and none tuned:

1. **Implied height.** Put the person's feet where the foot ray meets the
   floor; the height at which the box-top ray passes vertically above that
   point is the height the foot anchor implies. A foot anchor whose implied
   height lies outside the class's plausible range is flagged.
2. **Head-anchor fallback.** For a flagged detection of a class with a
   height prior, intersect the box-top ray with the plane ``z = prior``
   instead. Not attempted when the box touches the top image edge (the
   head is cut off, so the top edge is not the head) or the class has no
   prior; the foot anchor is then kept and the detection stays flagged.
3. **Per-detection position covariance from ray geometry.** Pixel noise
   ``PIXEL_SIGMA_PX`` on the anchor pixel, pushed through the pixel-to-floor
   Jacobian (central differences), plus, for the head anchor, the prior's
   spread pushed along the ray: ``d(xy)/dh = d_xy / d_z``. Nothing models
   the foot anchor's own failure (occlusion), which is what the flag is
   for; coverage is measured, not assumed.

Priors and ranges (metres), with sources:

- Person range ``[1.352, 1.977]``: the extreme national mean adult heights
  for the 1996 birth cohort in NCD-RisC (2016), "A century of trends in
  adult human height", eLife 5:e13410 (women, Guatemala, 149.4 cm; men,
  Netherlands, 182.5 cm), each widened by two within-population standard
  deviations from Jelenkovic et al. (2016), as summarised by Our World in
  Data, "Human Height" (women 7.07 cm; men 7.76 cm from the stated 95% band
  163.2-193.6 cm).
- Person prior ``1.716 +/- 0.101``: an equal mixture of those two
  Jelenkovic distributions (women 164.7 cm, men 178.4 cm).
- AgilityDigit ``1.75``: Agility Robotics' stated Digit height (Digit 5 is
  1.81 m, Agility press release); FourierGR1T2 ``1.65``: Fourier GR-1 stated
  height (The Robot Report, production GR-1). Their ``0.05`` spread and
  their ranges (prior +/- 3 spread) are an ASSUMPTION, not cited.
- Every other class (Forklift, PalletTruck, NovaCarter, Transporter): no
  cited height, so no check and no fallback; the foot anchor is used and
  recorded as ``foot_unchecked``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.data.smartspaces import load_calibration

FloatArray = npt.NDArray[np.float64]

PIXEL_SIGMA_PX = 2.0
"""Stated, not measured: the anchor pixel's localisation error."""

TOP_EDGE_PX = 1.0
"""A box whose top is within this of the image top edge has a cut-off head."""


@dataclass(frozen=True)
class HeightPrior:
    mean_m: float
    sd_m: float
    range_m: tuple[float, float]
    source: str


HEIGHT_PRIORS: dict[str, HeightPrior] = {
    "Person": HeightPrior(
        1.716,
        0.101,
        (1.352, 1.977),
        "NCD-RisC 2016 eLife 5:e13410; Jelenkovic 2016 via Our World in Data",
    ),
    "person": HeightPrior(
        1.716,
        0.101,
        (1.352, 1.977),
        "NCD-RisC 2016 eLife 5:e13410; Jelenkovic 2016 via Our World in Data",
    ),
    "AgilityDigit": HeightPrior(
        1.75, 0.05, (1.60, 1.90), "Agility Robotics spec; spread ASSUMED"
    ),
    "FourierGR1T2": HeightPrior(
        1.65, 0.05, (1.50, 1.80), "Fourier GR-1 spec; spread ASSUMED"
    ),
}

ANCHORS = ("foot", "head", "foot_flagged", "foot_unchecked", "none")
"""``foot``: implied height in range. ``head``: fallback used.
``foot_flagged``: out of range, no usable fallback (cut-off head, or head ray
misses the prior plane). ``foot_unchecked``: the check did not run — the
class has no prior, or there was no full camera to run it with
(``src.pipeline.run.estimate_tracks`` without ``anchor_camera``, and runs
written before Day 43).
``none``: neither ray meets its plane in front of the camera."""


@dataclass(frozen=True)
class Camera:
    """A pinhole camera in the floor frame (z up, floor at 0)."""

    centre: FloatArray
    """``[3]`` world position."""
    rotation: FloatArray
    """``[3, 3]`` world to camera."""
    intrinsic: FloatArray

    def ray(self, uv: FloatArray) -> FloatArray:
        """``[N, 3]`` world directions (not normalised) through pixels."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        h = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)
        cam = h @ np.linalg.inv(self.intrinsic).T
        out: FloatArray = cam @ self.rotation  # R^T applied to each row
        return out

    def to_plane(
        self, uv: FloatArray, z: float | FloatArray
    ) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
        """Where each pixel's ray meets the horizontal plane ``z``: ``[N, 2]``
        and validity (in front of the camera)."""
        d = self.ray(uv)
        z = np.broadcast_to(np.asarray(z, dtype=np.float64), (d.shape[0],))
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (z - self.centre[2]) / d[:, 2]
        valid = np.isfinite(s) & (s > 0)
        xy = np.full((d.shape[0], 2), np.nan)
        xy[valid] = self.centre[:2] + s[valid, None] * d[valid, :2]
        return xy, valid

    def project(self, points: FloatArray) -> FloatArray:
        """``[N, 3]`` world points to ``[N, 2]`` pixels (NaN behind the camera)."""
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        cam = (p - self.centre) @ self.rotation.T
        uvw = cam @ self.intrinsic.T
        out = np.full((p.shape[0], 2), np.nan)
        front = uvw[:, 2] > 0
        out[front] = uvw[front, :2] / uvw[front, 2:3]
        return out

    def jacobian(self, uv: FloatArray, z: float | FloatArray) -> FloatArray:
        """``[N, 2, 2]`` d(floor xy)/d(pixel uv) on plane ``z``, central
        differences of half a pixel."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        out = np.full((uv.shape[0], 2, 2), np.nan)
        for k in range(2):
            step = np.zeros(2)
            step[k] = 0.5
            plus, _ = self.to_plane(uv + step, z)
            minus, _ = self.to_plane(uv - step, z)
            out[:, :, k] = plus - minus
        return out


def camera_from_smartspaces(path: Path, camera_id: str) -> Camera:
    cal = load_calibration(path).cameras[camera_id]
    rotation, translation = cal.extrinsic[:, :3], cal.extrinsic[:, 3]
    return Camera(
        centre=-rotation.T @ translation,
        rotation=rotation,
        intrinsic=np.asarray(cal.intrinsic, dtype=np.float64),
    )


@dataclass(frozen=True)
class Anchored:
    xy: FloatArray
    """``[N, 2]``; NaN where ``anchor == "none"``."""
    cov: FloatArray
    """``[N, 2, 2]`` position covariance (m^2)."""
    anchor: npt.NDArray[np.str_]
    implied_height_m: FloatArray
    """From the foot anchor; NaN when the foot ray misses the floor."""


def implied_height(
    camera: Camera, foot_xy: FloatArray, top_uv: FloatArray
) -> FloatArray:
    """Height (m) at which each box-top ray passes closest, horizontally, to
    the vertical line through its foot point."""
    d = camera.ray(top_uv)
    c = camera.centre
    dxy = d[:, :2]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.einsum("ij,ij->i", dxy, foot_xy - c[:2]) / np.einsum(
            "ij,ij->i", dxy, dxy
        )
    height: FloatArray = c[2] + s * d[:, 2]
    return height


def anchor_boxes(
    camera: Camera,
    boxes_xyxy: FloatArray,
    labels: npt.ArrayLike,
    mode: str = "fallback",
) -> Anchored:
    """``mode="foot"``: the slice's original bottom-centre projection, with
    the check and covariance recorded but no fallback. ``mode="fallback"``:
    head anchor where the foot is flagged and a fallback is usable."""
    if mode not in ("foot", "fallback"):
        raise ValueError(f"mode {mode!r}: expected 'foot' or 'fallback'")
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    labels = np.asarray(labels, dtype=str).reshape(-1)
    n = boxes.shape[0]
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    foot_uv = np.stack([cx, boxes[:, 3]], axis=1)
    top_uv = np.stack([cx, boxes[:, 1]], axis=1)
    foot_xy, foot_ok = camera.to_plane(foot_uv, 0.0)
    height = np.full(n, np.nan)
    if foot_ok.any():
        height[foot_ok] = implied_height(camera, foot_xy[foot_ok], top_uv[foot_ok])
    jf = camera.jacobian(foot_uv, 0.0)
    cov = np.einsum("nij,nkj->nik", jf, jf) * PIXEL_SIGMA_PX**2
    xy = foot_xy.copy()
    anchor = np.where(foot_ok, "foot", "none").astype("<U14")
    for cls in np.unique(labels):
        sel = labels == cls
        prior = HEIGHT_PRIORS.get(str(cls))
        if prior is None:
            anchor[sel & foot_ok] = "foot_unchecked"
            continue
        lo, hi = prior.range_m
        out = sel & (~foot_ok | ~((height >= lo) & (height <= hi)))
        anchor[out & foot_ok] = "foot_flagged"
        if mode == "foot" or not out.any():
            continue
        idx = np.flatnonzero(out & (boxes[:, 1] > TOP_EDGE_PX))
        if idx.size == 0:
            continue
        head_xy, head_ok = camera.to_plane(top_uv[idx], prior.mean_m)
        idx, head_xy = idx[head_ok], head_xy[head_ok]
        if idx.size == 0:
            continue
        jh = camera.jacobian(top_uv[idx], prior.mean_m)
        d = camera.ray(top_uv[idx])
        g = d[:, :2] / d[:, 2:3]
        xy[idx] = head_xy
        cov[idx] = (
            np.einsum("nij,nkj->nik", jh, jh) * PIXEL_SIGMA_PX**2
            + np.einsum("ni,nj->nij", g, g) * prior.sd_m**2
        )
        anchor[idx] = "head"
    xy[anchor == "none"] = np.nan
    return Anchored(xy=xy, cov=cov, anchor=anchor, implied_height_m=height)


CHI2_2DOF_95 = 5.991464547107979
"""95% quantile of chi-square with 2 degrees of freedom."""


def covered95(error_xy: FloatArray, cov: FloatArray) -> npt.NDArray[np.bool_]:
    """Whether each 2-D error lies inside its stated 95% ellipse."""
    e = np.asarray(error_xy, dtype=np.float64).reshape(-1, 2)
    inv = np.linalg.inv(cov)
    m2: FloatArray = np.einsum("ni,nij,nj->n", e, inv, e)
    inside: npt.NDArray[np.bool_] = m2 <= CHI2_2DOF_95
    return inside


PERSON_MAX_HEIGHT_M = 1.825 + 3 * 0.0776
"""2.058 m: the tallest national mean adult height (men, Netherlands, 1996
cohort, 182.5 cm; NCD-RisC 2016, eLife 5:e13410) plus three
within-population standard deviations (7.76 cm, Jelenkovic 2016 via Our
World in Data). A population bound, NOT an absolute one: the tallest
recorded human was 2.72 m (Robert Wadlow, Guinness World Records), which is
:data:`HUMAN_RECORD_HEIGHT_M`. A person taller than this violates the
range interval's lower end, and is reported as such."""

HUMAN_RECORD_HEIGHT_M = 2.72
"""Robert Wadlow, Guinness World Records "tallest man ever"."""


@dataclass(frozen=True)
class RangeInterval:
    """Bounds on a detection's horizontal range from the camera (m).

    ``r_foot``: the box bottom-centre ray meets the floor. An UPPER bound if
    the box bottom is at or above the feet (occlusion and an image-edge cut
    only raise it). ``r_head``: the box top-centre ray meets the plane
    ``z = max_height_m``. A LOWER bound if the box top is at or below the top
    of the head, the head is no higher than ``max_height_m``, and the camera
    is above that plane. Assumes an upright person on a flat floor at z = 0.
    NaN where a ray does not meet its plane in front of the camera."""

    r_foot: FloatArray
    r_head: FloatArray
    foot_xy: FloatArray
    head_xy: FloatArray
    max_height_m: float

    @property
    def width_m(self) -> FloatArray:
        out: FloatArray = self.r_foot - self.r_head
        return out


def range_interval(
    camera: Camera, boxes_xyxy: FloatArray, max_height_m: float = PERSON_MAX_HEIGHT_M
) -> RangeInterval:
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    foot_xy, foot_ok = camera.to_plane(np.stack([cx, boxes[:, 3]], axis=1), 0.0)
    head_xy, head_ok = camera.to_plane(
        np.stack([cx, boxes[:, 1]], axis=1), max_height_m
    )
    if camera.centre[2] <= max_height_m:
        head_ok[:] = False
        head_xy[:] = np.nan
    c = camera.centre[:2]
    r_foot = np.where(foot_ok, np.linalg.norm(foot_xy - c, axis=1), np.nan)
    r_head = np.where(head_ok, np.linalg.norm(head_xy - c, axis=1), np.nan)
    return RangeInterval(r_foot, r_head, foot_xy, head_xy, max_height_m)
