"""Measurement models: R IS the capability envelope (§15).

A ``MeasurementModel`` answers two questions: what does the state predict a
sensor would read (:meth:`h`), and how much should the filter trust one
actual reading (:meth:`R`). The second question already has an answer
elsewhere in this codebase — :class:`~src.model.envelope.Envelope` measures
exactly "how capable is this sensor, as a function of one independent
variable" (Day 13; Day 7 found the motion gate's own wake threshold varying
4.9x with subject speed, which is what :class:`EnvelopeCurve` exists to
represent instead of a scalar). Measurement noise is a function of distance,
resolution, and condition — the same shape of question — so :meth:`R` wires
the curve in directly rather than inventing a second noise model that could
drift out of step with what the envelope actually measures.

Outside the envelope: high R, never a discard
------------------------------------------------
:class:`~src.model.envelope.EnvelopeCurve.value_at` clamps outside its
measured range rather than extrapolating (Day 13's own design: a capability
measured between 0.2 and 4.9 m/s says nothing about 20 m/s). Taking that
clamped value at face value here would silently claim the *same* confidence
for a reading five metres past the last calibrated point as for one
measured there — an unearned certainty in exactly the region the curve
admits it knows the least about. :data:`OUTSIDE_ENVELOPE_INFLATION` scales
the clamped sigma up for any distance outside ``[points[0].x, points[-1].x]``,
so an out-of-envelope reading still enters the filter (never silently
dropped — a dropped reading is invisible to every downstream consistency
check) but enters *distrusted*, proportional to how far outside calibrated
territory it is trying to speak with calibrated confidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.model.envelope import Envelope, EnvelopeCurve

from src.estimator.motion_model import STATE_DIM

FloatArray = npt.NDArray[np.float64]

OUTSIDE_ENVELOPE_INFLATION = 5.0
"""Multiplier on sigma for a reading outside the envelope's measured range.

Not fitted to any dataset — there is no real camera yet (Day 19: blocked on
hardware procurement) to measure a true out-of-envelope error rate from.
Chosen to be large enough that an out-of-envelope reading visibly loses a
Kalman-gain contest against an in-envelope one at the same nominal distance,
without being infinite (infinite R is equivalent to a discard, which is
exactly the failure mode this module exists to avoid).
"""

_ILLUSTRATIVE_MANIFEST_SHA = "illustrative-unmeasured-2026-08-11"
"""Marks every envelope this module can build on its own as NOT a real
measurement. A real deployment loads a measured ``Envelope`` (the same way
``scripts/measure_envelope.py`` produces one for the motion gate) and passes
it to :func:`measurement_model_for` explicitly; nothing in this module
should ever be mistaken for that."""


class MeasurementModelError(ValueError):
    """Raised when a measurement model or its inputs are malformed."""


def _position_selection_matrix() -> FloatArray:
    """H: extracts [x, y, z] from the 6-dim state [x, y, z, vx, vy, vz]."""
    h = np.zeros((3, STATE_DIM), dtype=np.float64)
    h[0, 0] = h[1, 1] = h[2, 2] = 1.0
    return h


@dataclass(frozen=True)
class MeasurementModel:
    """One sensor/capability pair's observation model.

    Attributes:
        envelope: The Day-13 :class:`~src.model.envelope.Envelope` this
            model's R is derived from — ``envelope.camera_id`` is the
            sensor, ``envelope.capability`` is the capability, and
            ``envelope.curve`` must be a range-vs-sigma curve (metres in,
            metres out): ``curve.independent_variable`` is validated to
            contain "range" or "distance" so a caller cannot accidentally
            wire in a speed-indexed curve (e.g. the motion gate's) and get
            silently-wrong noise.
        outside_envelope_inflation: See module docstring.
    """

    envelope: Envelope
    outside_envelope_inflation: float = OUTSIDE_ENVELOPE_INFLATION

    def __post_init__(self) -> None:
        indep = self.envelope.curve.independent_variable
        if "range" not in indep and "distance" not in indep:
            raise MeasurementModelError(
                f"MeasurementModel needs a distance-indexed envelope curve "
                f"(independent_variable containing 'range' or 'distance'), "
                f"got {indep!r} — this curve measures capability against a "
                "different variable and would produce meaningless R here"
            )
        if not (self.outside_envelope_inflation >= 1.0) or not np.isfinite(
            self.outside_envelope_inflation
        ):
            raise MeasurementModelError(
                f"outside_envelope_inflation must be finite and >= 1.0 (an "
                f"inflation below 1.0 would trust an uncalibrated reading "
                f"MORE than a calibrated one), got {self.outside_envelope_inflation}"
            )

    @property
    def sensor(self) -> str:
        return self.envelope.camera_id

    @property
    def capability(self) -> str:
        return self.envelope.capability

    @property
    def sha(self) -> str:
        payload = {
            "sensor": self.sensor,
            "capability": self.capability,
            "twin_rev": self.envelope.twin_rev,
            "curve_points": list(self.envelope.curve.points),
            "outside_envelope_inflation": self.outside_envelope_inflation,
        }
        canonical = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def H(self) -> FloatArray:
        return _position_selection_matrix()

    def h(self, state: FloatArray) -> FloatArray:
        """Predicted measurement: the position sub-vector of ``state``."""
        result: FloatArray = self.H() @ state
        return result

    def is_within_envelope(self, distance_m: float) -> bool:
        xs = [x for x, _ in self.envelope.curve.points]
        return xs[0] <= distance_m <= xs[-1]

    def sigma_m(self, distance_m: float) -> float:
        """Position measurement sigma (metres) at ``distance_m``.

        Inflated outside the calibrated range — see module docstring.
        """
        if distance_m < 0 or not np.isfinite(distance_m):
            raise MeasurementModelError(
                f"distance_m must be finite and non-negative, got {distance_m}"
            )
        sigma = self.envelope.curve.value_at(distance_m)
        if not self.is_within_envelope(distance_m):
            sigma *= self.outside_envelope_inflation
        return sigma

    def R(self, distance_m: float) -> FloatArray:
        """Isotropic 3D measurement-noise covariance at ``distance_m``."""
        sigma = self.sigma_m(distance_m)
        result: FloatArray = (sigma**2) * np.eye(3, dtype=np.float64)
        return result


def illustrative_position_envelope(
    sensor: str, capability: str = "state_estimation", twin_rev: int = 0
) -> Envelope:
    """An UNMEASURED, illustrative range-vs-sigma envelope.

    For tests and synthetic evaluation only — see
    :data:`_ILLUSTRATIVE_MANIFEST_SHA`. Points chosen to be monotonically
    increasing and to span v3-indoor's own depth range (a few to several
    tens of metres, per ``docs`` on the synthetic indoor generator), not
    fitted to any real camera's measured resolution falloff, because no
    real camera has been measured yet (Day 19: blocked on hardware
    procurement — see ``docs/reference_hardware.md``).
    """
    curve = EnvelopeCurve(
        independent_variable="range_m",
        points=(
            (1.0, 0.03),
            (3.0, 0.06),
            (8.0, 0.15),
            (15.0, 0.35),
            (25.0, 0.70),
        ),
    )
    return Envelope(
        capability=capability,
        camera_id=sensor,
        twin_rev=twin_rev,
        curve=curve,
        sample_count=len(curve.points) * 2,
        manifest_sha=_ILLUSTRATIVE_MANIFEST_SHA,
    )


def measurement_model_for(
    sensor: str, capability: str = "state_estimation", envelope: Envelope | None = None
) -> MeasurementModel:
    """The swappable-component factory, mirroring
    :func:`~src.estimator.motion_model.motion_model_for`.

    Args:
        sensor: Camera/sensor id.
        capability: Which capability's envelope to use.
        envelope: A measured :class:`Envelope`. When omitted, falls back to
            :func:`illustrative_position_envelope` — never silently; the
            returned model's ``sha`` and the envelope's own
            ``manifest_sha`` both carry the "illustrative-unmeasured" marker
            so nothing downstream can mistake it for a real measurement.
    """
    if envelope is None:
        envelope = illustrative_position_envelope(sensor, capability)
    return MeasurementModel(envelope=envelope)
