"""The capability envelope decides which misses are gate defects.

115.2 gate pixels used to draw that line, as an unargued constant. One day-6
miss sat at 104.0 against it — close enough that the number was doing real
work — and it turned out to be excusing a genuine gate defect.

These tests hold the three things that make the boundary defensible: the
derivation is explicit and follows the gate's own arithmetic, it is validated
against measurement rather than assumed, and a scorecard records which envelope
produced its verdicts so two of them cannot be silently compared across
different definitions of "defect".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cascade.envelope import (
    DEFAULT_ENVELOPE_PATH,
    EnvelopeError,
    MeasuredEnvelope,
)
from src.cascade.motion import MotionGateConfig
from src.data.scorecard import Scorecard, ScorecardError

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def envelope() -> MeasuredEnvelope:
    return MeasuredEnvelope.load(REPO_ROOT / DEFAULT_ENVELOPE_PATH)


def test_derivation_follows_the_gates_own_arithmetic() -> None:
    """The threshold is computed, not written down.

    The gate wakes when ``foreground_px / gate_px >= min_foreground_fraction``,
    so the smallest wakeable foreground region is the product of the two. If
    this ever becomes a literal again, it stops tracking the gate.
    """
    config = MotionGateConfig()
    assert config.envelope_threshold_px() == pytest.approx(
        config.min_foreground_fraction * config.gate_width * config.gate_height
    )
    assert config.envelope_threshold_px() == pytest.approx(115.2)


def test_threshold_moves_with_gate_resolution() -> None:
    """Halving each gate dimension quarters the raster and the threshold."""
    base = MotionGateConfig()
    smaller = MotionGateConfig(gate_width=160, gate_height=90)
    assert smaller.envelope_threshold_px() == pytest.approx(
        base.envelope_threshold_px() / 4
    )


def test_threshold_moves_with_foreground_fraction() -> None:
    base = MotionGateConfig()
    strict = MotionGateConfig(min_foreground_fraction=base.min_foreground_fraction * 2)
    assert strict.envelope_threshold_px() == pytest.approx(
        base.envelope_threshold_px() * 2
    )


def test_gate_raster_is_the_smaller_of_source_and_gate() -> None:
    """A source already below the gate size is not upscaled.

    Day 2's parity claim was vacuous because a 320x176 clip never touched the
    downscale path. The threshold has to follow the raster the gate actually
    processes, or it would be computed against a resolution that never existed.
    """
    config = MotionGateConfig()
    assert config.gate_pixels((176, 320)) == 176 * 320
    assert config.gate_pixels((720, 1280)) == 320 * 180


def test_measured_envelope_contradicts_the_derivation_at_low_speed(
    envelope: MeasuredEnvelope,
) -> None:
    """The derivation is about foreground area and is not a silhouette rule.

    This is the finding that made the envelope config-driven: a slow mover is
    absorbed into the background model at any size, so no silhouette threshold
    derived from the gate's arithmetic can predict its behaviour.
    """
    slow = envelope.speeds_gate_px[0]
    assert envelope.wake_threshold_px(slow) == float("inf"), (
        "at the slowest measured speed the gate never woke at any size; the "
        "derived 115.2 would have called those misses gate defects"
    )

    fast = envelope.speeds_gate_px[-1]
    derived = MotionGateConfig().envelope_threshold_px()
    assert envelope.wake_threshold_px(fast) < derived, (
        "at high speed the gate wakes on movers smaller than the derived "
        "threshold, because a displaced object marks both the pixels it "
        "arrived at and the ones it left"
    )


def test_envelope_is_monotone_in_speed(envelope: MeasuredEnvelope) -> None:
    """A faster mover never needs to be larger to be seen."""
    finite = [
        (s, envelope.wake_threshold_px(s))
        for s in envelope.speeds_gate_px
        if envelope.wake_threshold_px(s) != float("inf")
    ]
    thresholds = [t for _, t in finite]
    assert thresholds == sorted(
        thresholds, reverse=True
    ), f"threshold should fall as speed rises, got {finite}"


def test_interpolation_stays_inside_the_measured_range(
    envelope: MeasuredEnvelope,
) -> None:
    """Between samples the value is interpolated; outside it is clamped.

    Extrapolating a curve with an asymptote at one end and a wall at the other
    would invent behaviour that was never observed.
    """
    low, high = envelope.speeds_gate_px[0], envelope.speeds_gate_px[-1]
    assert envelope.wake_threshold_px(high * 100) == envelope.wake_threshold_px(high)
    assert envelope.wake_threshold_px(low / 100) == envelope.wake_threshold_px(low)


def test_unresolvable_neighbour_wins_the_interpolation(
    envelope: MeasuredEnvelope,
) -> None:
    """Interpolating across "never woke" must not manufacture a threshold.

    Averaging a finite threshold with an unmeasured one would produce a number
    the gate was never observed to achieve, and it would excuse real defects.
    """
    speeds = envelope.speeds_gate_px
    thresholds = [envelope.wake_threshold_px(s) for s in speeds]
    for index in range(len(speeds) - 1):
        if thresholds[index] == float("inf") and thresholds[index + 1] != float("inf"):
            midpoint = (speeds[index] + speeds[index + 1]) / 2
            assert envelope.wake_threshold_px(midpoint) == float("inf")
            return
    pytest.skip("this measurement has no unresolvable-to-resolvable boundary")


def test_missing_envelope_refuses_rather_than_guessing(tmp_path: Path) -> None:
    with pytest.raises(EnvelopeError, match="no measured envelope"):
        MeasuredEnvelope.load(tmp_path / "absent.json")


def test_scorecard_records_which_envelope_produced_it(
    envelope: MeasuredEnvelope,
) -> None:
    provenance = envelope.provenance()
    assert provenance["envelope_sha"]
    assert provenance["envelope_measured_stack"], (
        "the stack must be recorded; OpenCV 4.8 and 5.0 have different MOG2 "
        "implementations, so the same code measures a different envelope"
    )


def _card(set_sha: str, envelope_sha: str) -> Scorecard:
    return Scorecard(
        golden_set_version="v2-indoor",
        golden_set_sha=set_sha,
        domain="indoor",
        clips_scored=1,
        envelope={"envelope_sha": envelope_sha},
    )


def test_comparing_across_envelopes_raises() -> None:
    """The core guard: a different envelope is a different question.

    Recall under one envelope against recall under another looks exactly like a
    gate regression and is not one.
    """
    with pytest.raises(ScorecardError, match="different capability"):
        _card("set-a", "env-1").require_comparable(_card("set-a", "env-2"))


def test_comparing_across_golden_sets_raises() -> None:
    with pytest.raises(ScorecardError, match="different golden sets"):
        _card("set-a", "env-1").require_comparable(_card("set-b", "env-1"))


def test_same_set_and_envelope_compares_cleanly() -> None:
    _card("set-a", "env-1").require_comparable(_card("set-a", "env-1"))


def test_committed_envelope_matches_the_configured_gate(
    envelope: MeasuredEnvelope,
) -> None:
    """A measurement taken on a different gate does not describe this one."""
    config = MotionGateConfig()
    assert (envelope.gate_width, envelope.gate_height) == (
        config.gate_width,
        config.gate_height,
    )
    assert envelope.min_foreground_fraction == config.min_foreground_fraction


def test_envelope_artifact_is_valid_json_with_samples() -> None:
    data = json.loads((REPO_ROOT / DEFAULT_ENVELOPE_PATH).read_text())
    assert data["samples"], "an envelope with no samples cannot classify anything"
    assert any(
        s["wake_threshold_silhouette_gate_px"] is None for s in data["samples"]
    ), "the sweep must include a speed at which the gate never wakes"
    assert any(
        s["wake_threshold_silhouette_gate_px"] is not None for s in data["samples"]
    ), "and one at which it does, or the curve has no crossing to interpolate"
