"""Depth rasters that carry their own units and frame geometry.

The bug this kills
------------------
Depth-Anything-V2 outputs *relative inverse depth* — disparity, on an arbitrary
per-frame scale with no metric meaning. The repository's Parquet schema
documents that same column as "Depth (meters)". Both statements cannot be true.
Nothing in the code checks, so the mislabel propagates into 3D points, into the
index, and into any answer computed from them, without a single exception ever
being raised.

:class:`DepthField` makes the units a required constructor argument. There is
no way to build one without saying what the numbers mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt

from src.contracts.errors import UnitsError
from src.contracts.frames import FrameGeometry

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

Units = Literal["meters", "disparity_rel", "log_depth"]
"""Physical interpretation of a depth raster.

- ``meters``: metric distance along the camera's optical axis. The only value
  for which unprojection to 3D is meaningful.
- ``disparity_rel``: relative inverse depth on an arbitrary per-frame scale.
  What Depth-Anything-V2 actually emits. Monotonic in depth but with unknown
  scale *and* unknown shift, so it cannot be converted to metres without an
  anchor.
- ``log_depth``: natural log of metric depth.
"""

METRIC_UNITS: Units = "meters"

ALL_UNITS: tuple[Units, ...] = get_args(Units)


@dataclass(frozen=True, eq=False)
class DepthField:
    """A depth raster plus everything needed to interpret it.

    Equality is disabled (``eq=False``): comparing two of these with ``==``
    would compare numpy arrays elementwise and return an array, which is
    ambiguous in a boolean context and would silently break any ``in`` or
    ``==`` check a caller wrote. Compare the fields you actually mean.

    Attributes:
        data: ``[H, W]`` float64 depth values.
        units: What those values mean. Required, never inferred.
        geometry: The pixel frame ``data`` is measured in.
        valid_mask: ``[H, W]`` bool, True where ``data`` is trustworthy. Depth
            models emit values everywhere, including for sky and for pixels
            outside the sensor's usable range; the mask is how a consumer
            distinguishes "far" from "no information".
    """

    data: FloatArray
    units: Units
    geometry: FrameGeometry
    valid_mask: BoolArray

    def __post_init__(self) -> None:
        if self.units not in ALL_UNITS:
            raise UnitsError(
                f"Unknown depth units {self.units!r}; expected one of {ALL_UNITS}"
            )
        if self.data.ndim != 2:
            raise ValueError(
                f"DepthField.data must be [H, W], got shape {self.data.shape}"
            )
        if self.data.shape != self.geometry.shape:
            raise ValueError(
                f"DepthField.data shape {self.data.shape} does not match "
                f"geometry {self.geometry} (expected {self.geometry.shape})"
            )
        if self.valid_mask.shape != self.data.shape:
            raise ValueError(
                f"DepthField.valid_mask shape {self.valid_mask.shape} does not "
                f"match data shape {self.data.shape}"
            )
        if self.valid_mask.dtype != np.bool_:
            raise ValueError(
                f"DepthField.valid_mask must be boolean, got {self.valid_mask.dtype}"
            )

    @property
    def is_metric(self) -> bool:
        """True when these values are in metres."""
        return self.units == METRIC_UNITS

    def require_metric(self) -> None:
        """Raise unless this field is metric.

        Raises:
            UnitsError: with an explanation of why no conversion is attempted.
        """
        if not self.is_metric:
            raise UnitsError(
                f"Expected depth in {METRIC_UNITS!r} but got {self.units!r}. "
                "No automatic conversion is provided: relative disparity has "
                "unknown scale and unknown shift, so converting it without an "
                "anchor would fabricate a metric result. Anchor the depth "
                "first, then label it 'meters'."
            )
