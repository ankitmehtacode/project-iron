"""Observation: one typed sensor reading, with mandatory uncertainty.

Every fact this system ever reports traces back to a chain of
observations (see :mod:`src.model.evidence`). An observation whose error
is unstated is indistinguishable, downstream, from ground truth — which is
exactly the fabrication risk the whole product architecture exists to
prevent. That is why :attr:`Observation.uncertainty` is a required
constructor argument with no default: see ``test_model_primitives.py`` for
the proof that omitting it, or passing ``None``, both raise.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.model.frame_of_reference import FrameOfReference
from src.model.measurement import Measurement
from src.model.uncertainty import Uncertainty
from src.model.ulid import ULID

EnvelopeStatus = str
"""One of the literal values in ``ENVELOPE_STATUSES``.

Kept as a plain ``str`` alias (validated at construction) rather than a
``Literal`` so :mod:`src.model.envelope`, which defines the authoritative
set, is the single place the vocabulary lives; a ``Literal`` here would
duplicate it and the two could drift.
"""

ENVELOPE_STATUSES: tuple[str, ...] = (
    "within_envelope",
    "degraded",
    "outside_envelope",
    "unknown",
)


class ObservationError(ValueError):
    """Raised when an :class:`Observation` violates its contract."""


@dataclass(frozen=True)
class Observation:
    """One typed reading from one sensor at one instant.

    Attributes:
        observation_id: Sortable, timestamp-embedding identity.
        sensor_id: Which sensor produced this — a camera, a badge reader,
            a door contact, an HRIS connector. Opaque, unique per
            deployment.
        ts_ns: When the underlying event occurred, nanoseconds since the
            epoch — not when it was processed.
        frame_ref: Pointer to the raw evidence backing this reading (a
            video frame reference for a camera, a log-line id for a badge
            reader). Sensor-specific meaning; always resolvable to
            something a person can inspect.
        measurement: The typed payload. See :mod:`src.model.measurement`.
        uncertainty: Required. See the module docstring.
        frame_of_reference: Geometry, canonicalizing transform, and twin
            revision this observation's coordinates (if any) are expressed
            in.
        producer_shas: Hashes of whatever produced this reading — a model
            export for a camera detection, a driver/firmware version for a
            hardware sensor. Empty tuple only for a raw, unprocessed
            hardware event (a badge swipe has no model in its path).
        envelope_status: This sensor's measured capability state at
            ``ts_ns`` — see :mod:`src.model.envelope`. ``"unknown"`` is a
            legitimate value (envelope not yet measured for this
            capability); it is not a default to reach for.
    """

    observation_id: ULID
    sensor_id: str
    ts_ns: int
    frame_ref: str
    measurement: Measurement
    uncertainty: Uncertainty
    frame_of_reference: FrameOfReference
    producer_shas: tuple[str, ...]
    envelope_status: EnvelopeStatus

    def __post_init__(self) -> None:
        if self.uncertainty is None:
            raise ObservationError(
                "Observation.uncertainty is required. An observation without "
                "a stated error is indistinguishable from ground truth "
                "downstream, which is the fabrication risk this type exists "
                "to prevent."
            )
        if not isinstance(self.uncertainty, Uncertainty):
            raise ObservationError(
                f"Observation.uncertainty must be an Uncertainty instance, "
                f"got {type(self.uncertainty).__name__}"
            )
        if not self.sensor_id:
            raise ObservationError("Observation.sensor_id must not be empty")
        if not self.frame_ref:
            raise ObservationError("Observation.frame_ref must not be empty")
        if self.envelope_status not in ENVELOPE_STATUSES:
            raise ObservationError(
                f"unknown envelope_status {self.envelope_status!r}; expected "
                f"one of {ENVELOPE_STATUSES}"
            )
