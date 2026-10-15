"""The detect capability's measured envelope, and R from it (Day 42, B6).

Data-model rule: R IS THE ENVELOPE. A detection's measurement noise comes
from measured curves of localisation error against object height, per
resolution, not from a constant. Every curve here is
``SYNTHETIC_DERIVED``: measured on rendered SmartSpaces frames, and not valid
for a real camera until validated on real footage.

- :class:`DetectEnvelope` holds, per resolution, three measured
  :class:`~src.model.envelope.EnvelopeCurve` s over object height in pixels:
  recall, centre-error sigma and size-error sigma. It is an
  :class:`~src.detect.contract.UncertaintySource`: a box's uncertainty is a
  ``gaussian_px`` read off the centre-error curve at the box's height.
- :func:`position_envelope_for` turns the pixel curve into the
  range-indexed metres curve :class:`~src.estimator.measurement_model.
  MeasurementModel` already takes: an object of physical height ``H`` at
  range ``z`` is ``fx * H / z`` px tall, and a pixel error ``s`` there is
  ``s * z / fx`` metres. The estimator's model is unchanged; its R now comes
  from this envelope.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.model.envelope import Envelope, EnvelopeCurve
from src.model.uncertainty import Uncertainty

SYNTHETIC_DERIVED = (
    "synthetic-derived: not valid for a real camera until validated on real " "footage"
)


class DetectEnvelopeError(ValueError):
    """Raised when a detect envelope is malformed or asked outside its use."""


@dataclass(frozen=True)
class ResolutionCurves:
    """Measured curves at one input resolution, x = object height in pixels
    AT THAT RESOLUTION."""

    resolution: str
    recall: EnvelopeCurve
    centre_sigma_px: EnvelopeCurve
    size_sigma_px: EnvelopeCurve
    matched_pairs: int

    def __post_init__(self) -> None:
        for curve in (self.recall, self.centre_sigma_px, self.size_sigma_px):
            if curve.independent_variable != "object_height_px":
                raise DetectEnvelopeError(
                    f"detect curves are indexed by object_height_px, got "
                    f"{curve.independent_variable!r}"
                )


@dataclass(frozen=True)
class DetectEnvelope:
    """One detector's measured envelope for one class, per resolution."""

    producer_model_sha: str
    producer_preprocess_sha: str
    object_class: str
    curves: dict[str, ResolutionCurves]
    fit_on: str
    manifest_sha: str
    label: str = SYNTHETIC_DERIVED

    def __post_init__(self) -> None:
        if self.label != SYNTHETIC_DERIVED:
            raise DetectEnvelopeError(
                "a detect envelope measured on SmartSpaces carries the "
                "synthetic-derived label; a real-footage envelope is a new type "
                "of record, not a relabelled one"
            )
        if not self.curves:
            raise DetectEnvelopeError("an envelope needs at least one resolution")

    def at(self, resolution: str) -> ResolutionCurves:
        try:
            return self.curves[resolution]
        except KeyError:
            raise DetectEnvelopeError(
                f"no curves measured at {resolution!r}; measured: "
                f"{sorted(self.curves)}"
            ) from None

    def source(self, resolution: str) -> "EnvelopeUncertainty":
        return EnvelopeUncertainty(self.at(resolution))


@dataclass(frozen=True)
class EnvelopeUncertainty:
    """Box uncertainty from the measured centre-error curve (B3's hook)."""

    curves: ResolutionCurves

    def uncertainty(self, x0: float, y0: float, x1: float, y1: float) -> Uncertainty:
        sigma = self.curves.centre_sigma_px.value_at(y1 - y0)
        return Uncertainty.gaussian_px(sigma, sigma)


def position_envelope_for(
    envelope: DetectEnvelope,
    resolution: str,
    camera_id: str,
    fx_px: float,
    object_height_m: float,
    twin_rev: int,
) -> Envelope:
    """The range-indexed position envelope :class:`MeasurementModel` takes.

    Each measured height point ``h`` becomes a range ``z = fx * H / h`` with
    sigma ``s(h) * z / fx`` metres. Points are re-sorted by range. The curve
    clamps outside the measured heights, and MeasurementModel inflates sigma
    outside its range: neither extrapolates.
    """
    if fx_px <= 0 or object_height_m <= 0:
        raise DetectEnvelopeError("fx and object height must be positive")
    curves = envelope.at(resolution)
    points = []
    for height_px, sigma_px in curves.centre_sigma_px.points:
        if height_px <= 0:
            continue
        z = fx_px * object_height_m / height_px
        points.append((z, sigma_px * z / fx_px))
    points.sort()
    return Envelope(
        capability=f"detect:{envelope.object_class}@{resolution}",
        camera_id=camera_id,
        twin_rev=twin_rev,
        curve=EnvelopeCurve(independent_variable="range_m", points=tuple(points)),
        sample_count=curves.matched_pairs,
        manifest_sha=envelope.manifest_sha,
    )


def from_artifact(payload: dict[str, object]) -> DetectEnvelope:
    """Build the envelope from ``artifacts/smartspaces/day42_detect_envelope
    .json`` (``scripts/fit_detect_envelope.py``). Only resolutions whose three
    curves were all measured are included; none borrows another's curve."""
    resolutions = payload["resolutions"]
    if not isinstance(resolutions, dict):
        raise DetectEnvelopeError("artifact has no resolutions block")
    curves: dict[str, ResolutionCurves] = {}
    for res, block in resolutions.items():
        raw = block["envelope_curves"]
        if not all(raw.values()):
            continue
        pairs = sum(b["matched_pairs"] for b in block["fit_val"]["bins"])

        def curve(points: list[list[float]]) -> EnvelopeCurve:
            return EnvelopeCurve(
                independent_variable="object_height_px",
                points=tuple((float(x), float(y)) for x, y in points),
            )

        curves[res] = ResolutionCurves(
            resolution=res,
            recall=curve(raw["recall_iou50"]),
            centre_sigma_px=curve(raw["centre_sigma_px"]),
            size_sigma_px=curve(raw["size_sigma_px"]),
            matched_pairs=int(pairs),
        )
    return DetectEnvelope(
        producer_model_sha=str(payload["producer_model_sha"]),
        producer_preprocess_sha="see the run_shas in the artifact",
        object_class=str(payload["class"]),
        curves=curves,
        fit_on=str(payload["fit_on"]),
        manifest_sha=str(payload["manifest_sha"]),
    )
