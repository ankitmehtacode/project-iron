from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence

import numpy as np

from src.geometry.ocr.text_detector import OCRResult as OCRDetection


# Module-level logger
logger = logging.getLogger(__name__)


# Data structures
@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters."""

    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(
                f"Focal lengths must be positive; got fx={self.fx}, fy={self.fy}."
            )


@dataclass(frozen=True, slots=True)
class TextAnnotation3D:
    """A recognised text string attached to a 3-D world point."""

    text: str
    position_3d: tuple[float, float, float]
    confidence: Optional[float] = None
    pixel_uv: tuple[int, int] = (0, 0)

    # Allow dict-style unpacking for backward-compatibility with callers
    # that expect plain dicts.
    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "position_3d": self.position_3d,
            "confidence": self.confidence,
            "pixel_uv": self.pixel_uv,
        }


# Core helpers
def _validate_depth_map(depth_map: np.ndarray) -> None:
    """Raise informative errors for obviously wrong depth maps."""
    if not isinstance(depth_map, np.ndarray):
        raise TypeError(f"depth_map must be a NumPy array; got {type(depth_map)}.")
    if depth_map.ndim != 2:
        raise ValueError(f"depth_map must be 2-D (H, W); got shape {depth_map.shape}.")
    if depth_map.size == 0:
        raise ValueError("depth_map must not be empty.")


def _validate_intrinsics(intrinsics: CameraIntrinsics) -> None:
    """Lightweight guard – the dataclass __post_init__ already checks fx/fy."""
    if not isinstance(intrinsics, CameraIntrinsics):
        raise TypeError(
            f"intrinsics must be a CameraIntrinsics instance; got {type(intrinsics)}."
        )


def _backproject(
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Vectorised pinhole back-projection."""
    x_cam = (u - intrinsics.cx) * z / intrinsics.fx
    y_cam = (v - intrinsics.cy) * z / intrinsics.fy
    return np.stack([x_cam, y_cam, z], axis=-1)  # (N, 3)


# Primary public function
def attach_text_to_3d(
    detections: Sequence[OCRDetection],
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    min_confidence: float = 0.0,
    depth_scale: float = 1.0,
) -> list[TextAnnotation3D]:
    """Project OCR detections onto 3-D coordinates using a depth map.
    The function is fully vectorised: pixel lookup and back-projection
    are done with NumPy array operations rather than Python loops.
    """
    _validate_depth_map(depth_map)
    _validate_intrinsics(intrinsics)

    if not detections:
        return []

    H, W = depth_map.shape

    # Filter by confidence (keep detections with confidence=None)
    kept: list[OCRDetection] = [
        d for d in detections if d.confidence is None or d.confidence >= min_confidence
    ]

    if not kept:
        logger.debug(
            "All %d detections filtered out by min_confidence=%.2f.",
            len(detections),
            min_confidence,
        )
        return []

    # Compute bounding-box centres (vectorised)
    bboxes = np.array([d.bbox for d in kept], dtype=np.float64)
    cx_px = bboxes[:, 0] + bboxes[:, 2] / 2.0  # u (horizontal)
    cy_px = bboxes[:, 1] + bboxes[:, 3] / 2.0  # v (vertical)

    #  Convert to integer indices and clip to valid bounds
    u_idx = np.clip(np.round(cx_px).astype(np.intp), 0, W - 1)
    v_idx = np.clip(np.round(cy_px).astype(np.intp), 0, H - 1)

    # Batch depth lookup
    raw_depths = depth_map[v_idx, u_idx].astype(np.float64) * depth_scale

    # Mask out invalid depth values
    valid_mask: np.ndarray = (raw_depths > 0) & np.isfinite(raw_depths)
    n_invalid = int(np.count_nonzero(~valid_mask))
    if n_invalid:
        logger.debug(
            "%d / %d detections skipped due to invalid depth (zero or NaN).",
            n_invalid,
            len(kept),
        )

    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size == 0:
        return []

    # Back-project valid points (vectorised)
    pts_3d = _backproject(
        cx_px[valid_idx],
        cy_px[valid_idx],
        raw_depths[valid_idx],
        intrinsics,
    )  # shape (M, 3)

    # Assemble output list
    results: list[TextAnnotation3D] = []
    for local_i, global_i in enumerate(valid_idx):
        det = kept[global_i]
        X, Y, Z = pts_3d[local_i]
        results.append(
            TextAnnotation3D(
                text=det.text,
                position_3d=(float(X), float(Y), float(Z)),
                confidence=det.confidence,
                pixel_uv=(int(u_idx[global_i]), int(v_idx[global_i])),
            )
        )

    return results


# level convenience wrapper for single frame processing
def process_frame_with_metadata(
    image: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    detector_fn: Any | None = None,
    min_confidence: float = 0.0,
    depth_scale: float = 1.0,
) -> list[TextAnnotation3D]:
    if detector_fn is None:
        from src.geometry.ocr.text_detector import (
            detect_text as detector_fn,
        )  # lazy import

    detections: list[OCRDetection] = detector_fn(image)
    return attach_text_to_3d(
        detections,
        depth_map,
        intrinsics,
        min_confidence=min_confidence,
        depth_scale=depth_scale,
    )


# Batch processing
def batch_process(
    frames: Sequence[tuple[np.ndarray, np.ndarray]],
    intrinsics: CameraIntrinsics,
    *,
    detector_fn: Any | None = None,
    min_confidence: float = 0.0,
    depth_scale: float = 1.0,
    yield_frame_index: bool = False,
) -> Iterator[list[TextAnnotation3D] | tuple[int, list[TextAnnotation3D]]]:
    for frame_idx, (image, depth_map) in enumerate(frames):
        annotations = process_frame_with_metadata(
            image,
            depth_map,
            intrinsics,
            detector_fn=detector_fn,
            min_confidence=min_confidence,
            depth_scale=depth_scale,
        )
        if yield_frame_index:
            yield frame_idx, annotations
        else:
            yield annotations
