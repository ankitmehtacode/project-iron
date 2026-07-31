"""Pinhole geometry operations that refuse to run on ill-defined inputs.

This module is deliberately *not* wired into
``src/geometry/projector_vectorized.py`` yet. That migration changes which
values reach the Parquet writer and must be made as a separate, measured
change with before/after numbers. Today the contract layer exists alongside the
legacy path so that new code can be written against it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.contracts.fields import DepthField
from src.contracts.frames import Intrinsics

FloatArray = npt.NDArray[np.float64]


def unproject(depth: DepthField, xy: FloatArray, K: Intrinsics) -> FloatArray:
    """Lift 2D pixel coordinates to 3D camera-space points using a depth field.

    Standard pinhole model, with the sampled depth as ``Z``::

        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy
        Z = depth at (x, y)

    Sampling is **nearest-neighbour**, not bilinear. Interpolating depth across
    an object boundary averages the foreground and background distances and
    produces a point floating in empty space between them — a "flying pixel".
    Nearest-neighbour picks one real surface. Bilinear interpolation is correct
    for smoothly varying signals like patch features, and wrong for depth,
    which is discontinuous exactly where objects are.

    Points whose depth is unusable are returned as ``NaN`` rather than as a
    plausible-looking coordinate. That covers three cases: the pixel is masked
    invalid, its depth is non-finite, or its depth is non-positive (a point at
    or behind the camera). Returning zeros instead would place them at the
    optical centre, where they would be indistinguishable from real
    observations.

    Args:
        depth: Metric depth field. Must be in metres.
        xy: ``[N, 2]`` pixel coordinates ``(x, y)`` in ``depth``'s frame.
        K: Intrinsics valid for ``depth``'s frame.

    Returns:
        ``[N, 3]`` float64 camera-space points ``(X, Y, Z)``, with ``NaN``
        rows where the depth was unusable.

    Raises:
        UnitsError: if ``depth`` is not in metres. Relative disparity has
            unknown scale and shift; unprojecting it yields a point cloud whose
            geometry is arbitrary but whose shape looks convincing.
        GeometryMismatch: if ``K.valid_for`` differs from ``depth.geometry``.
            Focal lengths are in pixels and only mean anything alongside the
            raster they were measured on.
        ValueError: if ``xy`` is not ``[N, 2]``.
    """
    depth.require_metric()
    K.require_geometry(depth.geometry)

    points = np.asarray(xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"unproject expects [N, 2] coordinates, got {points.shape}")

    height, width = depth.geometry.shape
    x = points[:, 0]
    y = points[:, 1]

    # Out-of-bounds coordinates are real — trackers extrapolate past the frame
    # edge. Clamp for the array lookup so it cannot wrap around to the opposite
    # edge, but remember which ones were out so they can be marked invalid
    # rather than silently answered with the nearest border pixel's depth.
    in_bounds = (x >= 0.0) & (x < float(width)) & (y >= 0.0) & (y < float(height))
    col = np.clip(np.floor(x), 0, width - 1).astype(np.intp)
    row = np.clip(np.floor(y), 0, height - 1).astype(np.intp)

    z = depth.data[row, col]
    usable = in_bounds & depth.valid_mask[row, col] & np.isfinite(z) & (z > 0.0)

    out = np.full((points.shape[0], 3), np.nan, dtype=np.float64)
    if not np.any(usable):
        return out

    z_ok = z[usable]
    out[usable, 0] = (x[usable] - K.cx) * z_ok / K.fx
    out[usable, 1] = (y[usable] - K.cy) * z_ok / K.fy
    out[usable, 2] = z_ok
    return out
