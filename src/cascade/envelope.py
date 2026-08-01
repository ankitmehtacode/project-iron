"""The motion gate's measured capability envelope.

``MotionGateConfig.envelope_threshold_px`` is exact arithmetic about
*foreground* area: the gate wakes when the background model marks at least
``min_foreground_fraction`` of its raster. That is correct and it is not what a
scorecard needs, because a scorecard knows a mover's **silhouette** area, not
the foreground area the background model will assign it.

Measurement (``scripts/measure_envelope.py``) shows the two are not
proportional and that no single conversion exists:

    displacement (native px/frame)    silhouette area that wakes the gate
                      2, 3            never — absorbed into the background
                      4               250.0 gate px   (2.17x the derived value)
                      6               132.0           (1.15x)
                      8              110.0            (0.95x)
                     12, 20          108.0            (0.94x)
                     30               92.0            (0.80x)

A slow mover is learned as background regardless of size, so the envelope is a
function of speed. Using the derived 115.2 as a silhouette threshold would
call a 200 px mover at 4 px/frame a *gate defect* when the gate physically
cannot wake on it, and would call a 100 px mover at 20 px/frame a physical
limit when the gate does in fact wake on it.

This module holds the measured relation and the provenance needed to know
which measurement a scorecard was produced under. Changing the envelope
changes what counts as a defect, so two scorecards built on different
envelopes are not comparable and say so.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ENVELOPE_PATH = Path("configs/envelope/gate_320x180.envelope.json")


class EnvelopeError(RuntimeError):
    """Raised when an envelope is missing, malformed, or incomparable."""


@dataclass(frozen=True)
class MeasuredEnvelope:
    """Silhouette area needed to wake the gate, as a function of mover speed."""

    gate_width: int
    gate_height: int
    min_foreground_fraction: float
    derived_foreground_threshold_px: float
    measured_stack: str
    sha: str
    """Content hash of the measurement artifact. Cited by every scorecard."""

    speeds_gate_px: tuple[float, ...]
    """Displacements, gate px/frame, ascending."""

    thresholds_px: tuple[float | None, ...]
    """Silhouette area waking the gate at each speed.

    ``None`` means the sweep never crossed 50% **within the area range it
    tested** — which is not the same as "cannot wake". The first version of
    this model conflated the two: it swept only to 320 gate px, recorded
    ``None`` for the two slowest speeds, and the scorecard read that as
    physically unreachable. The Inspector then showed a 761 gate px agent
    labelled "unreachable at this speed" while the real gate was awake on that
    very frame. Re-measured to 1660 px, those speeds wake at 567 and 325.
    """

    swept_max_px: float | None = None
    """Largest silhouette area the sweep tested, if the artifact records it.

    Consumers use it to distinguish "unreachable" from "never measured". An
    envelope without it cannot make that distinction and says so rather than
    guessing."""

    @classmethod
    def load(cls, path: Path) -> "MeasuredEnvelope":
        if not path.exists():
            raise EnvelopeError(
                f"no measured envelope at {path}. Run "
                "scripts/measure_envelope.py --displacements ... --model-out "
                f"{path}. The scorecard will not guess a threshold: doing so "
                "would decide which misses are gate defects by assumption."
            )
        raw = path.read_bytes()
        data = json.loads(raw)
        samples = sorted(
            data["samples"], key=lambda s: s["displacement_gate_px_per_frame"]
        )
        if not samples:
            raise EnvelopeError(f"{path} contains no measured samples")
        return cls(
            gate_width=data["gate_width"],
            gate_height=data["gate_height"],
            min_foreground_fraction=data["min_foreground_fraction"],
            derived_foreground_threshold_px=data["derived_foreground_threshold_px"],
            measured_stack=data["measured_stack"],
            sha=hashlib.sha256(raw).hexdigest(),
            speeds_gate_px=tuple(
                float(s["displacement_gate_px_per_frame"]) for s in samples
            ),
            thresholds_px=tuple(
                None
                if s["wake_threshold_silhouette_gate_px"] is None
                else float(s["wake_threshold_silhouette_gate_px"])
                for s in samples
            ),
            swept_max_px=(data.get("swept_area_gate_px") or {}).get("stop"),
        )

    def uncrossed_speeds(self) -> tuple[float, ...]:
        """Speeds whose sweep never crossed 50% within the tested area range.

        These are the entries a caller must not treat as unreachable. If the
        artifact records how far the sweep went, they mean "needs more than
        ``swept_max_px``"; if it does not, they mean nothing trustworthy at all.
        """
        return tuple(
            speed
            for speed, threshold in zip(self.speeds_gate_px, self.thresholds_px)
            if threshold is None
        )

    def wake_threshold_px(self, speed_gate_px: float) -> float:
        """Silhouette area, in gate px, a mover at this speed needs to wake it.

        Returns ``inf`` only where the measurement genuinely has nothing to
        say: a speed whose sweep never crossed within the tested range, on an
        artifact that does not record how far it swept. Where the range *is*
        recorded, an uncrossed speed returns just past it — the honest reading
        is "more than we tested", and treating that as unreachable is the bug
        the Inspector caught.

        Between measured speeds the value is linearly interpolated. Outside the
        measured range it is clamped to the nearest measurement rather than
        extrapolated, because extrapolating a curve that has an asymptote at
        one end and a wall at the other invents behaviour that was not
        observed.
        """
        speeds, thresholds = self.speeds_gate_px, self.thresholds_px

        # An exact sample is a measurement, so it is returned as measured. It
        # must be checked before bracketing: a sample whose slower neighbour
        # never woke would otherwise be discarded as unresolvable, throwing
        # away the very value that was observed at that speed.
        for sample_speed, sample_threshold in zip(speeds, thresholds):
            if speed_gate_px == sample_speed:
                return (
                    self._uncrossed() if sample_threshold is None else sample_threshold
                )

        if speed_gate_px <= speeds[0]:
            first = thresholds[0]
            return self._uncrossed() if first is None else first
        if speed_gate_px >= speeds[-1]:
            last = thresholds[-1]
            return self._uncrossed() if last is None else last

        for index in range(len(speeds) - 1):
            low, high = speeds[index], speeds[index + 1]
            if not low <= speed_gate_px <= high:
                continue
            low_t, high_t = thresholds[index], thresholds[index + 1]
            # A None neighbour is a speed the sweep did not resolve, so there is
            # nothing to interpolate towards. The unresolved side wins: it is
            # the conservative reading and keeps a real defect from being
            # excused by a fabricated midpoint.
            if low_t is None or high_t is None:
                return self._uncrossed()
            if high == low:
                return high_t
            span = (speed_gate_px - low) / (high - low)
            return low_t + span * (high_t - low_t)

        raise EnvelopeError(f"speed {speed_gate_px} fell outside the sample grid")

    def _uncrossed(self) -> float:
        """Threshold to report for a speed the sweep never resolved.

        If the artifact records how wide it swept, the honest answer is "more
        than that" — a mover larger than the sweep may well wake the gate, and
        v3's ``speed_slow`` clips proved it does. Only when the range is
        unrecorded is there truly nothing to say, and then this is infinite.
        """
        if self.swept_max_px is None:
            return float("inf")
        return float(self.swept_max_px)

    def provenance(self) -> dict[str, object]:
        """What a scorecard records so its verdicts can be reproduced."""
        return {
            "envelope_sha": self.sha,
            "envelope_gate_raster": f"{self.gate_width}x{self.gate_height}",
            "envelope_min_foreground_fraction": self.min_foreground_fraction,
            "envelope_measured_stack": self.measured_stack,
        }

    def require_comparable(self, other: "MeasuredEnvelope") -> None:
        """Raise unless two scorecards measured against the same envelope.

        The envelope decides which misses count as gate defects. A recall
        computed under one envelope and compared against a recall computed
        under another is not a comparison, and the difference would read as a
        change in the gate.
        """
        if self.sha != other.sha:
            raise EnvelopeError(
                "these scorecards were produced under different capability "
                f"envelopes ({self.sha[:12]} vs {other.sha[:12]}). The envelope "
                "decides which misses are gate defects, so the numbers are not "
                "comparable. Re-score both under one envelope."
            )
