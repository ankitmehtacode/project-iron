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

    A slow mover needs a far larger silhouette than the gate's arithmetic
    suggests — several times the derived value — so no scalar threshold
    predicts the gate's behaviour.

    This assertion used to say the gate *never* wakes at the slowest speeds.
    That was an artifact of a sweep that stopped at 320 gate px, not a property
    of the gate. The Inspector showed a 761 px agent labelled "unreachable at
    this speed" while the real gate was awake on that frame; re-measured to
    1660 px, the slowest speed wakes at 535.
    """
    slow = envelope.speeds_gate_px[0]
    fast = envelope.speeds_gate_px[-1]
    derived = MotionGateConfig().envelope_threshold_px()

    slow_threshold = envelope.wake_threshold_px(slow)
    assert slow_threshold > derived * 2, (
        "the slowest measured speed must need several times the derived "
        f"threshold, got {slow_threshold} against a derived {derived}"
    )
    assert envelope.wake_threshold_px(fast) < derived, (
        "at high speed the gate wakes on movers smaller than the derived "
        "threshold, because a displaced object marks both the pixels it "
        "arrived at and the ones it left"
    )
    assert slow_threshold / envelope.wake_threshold_px(fast) > 3, (
        "the whole point: the threshold varies several-fold across speed, so "
        "no single number can stand in for it"
    )


def test_an_uncrossed_speed_is_not_reported_as_unreachable() -> None:
    """ "Did not cross within the range swept" is not "cannot wake".

    Conflating the two is exactly how the first envelope came to claim a slow
    agent was unresolvable while the gate was demonstrably waking on it. When
    the artifact records how far it swept, an uncrossed speed must report that
    bound, not infinity.
    """
    bounded = MeasuredEnvelope(
        gate_width=320,
        gate_height=180,
        min_foreground_fraction=0.002,
        derived_foreground_threshold_px=115.2,
        measured_stack="test",
        sha="test",
        speeds_gate_px=(0.5, 5.0),
        thresholds_px=(None, 110.0),
        swept_max_px=1660.0,
    )
    assert bounded.wake_threshold_px(0.5) == 1660.0
    assert bounded.uncrossed_speeds() == (0.5,)

    unbounded = MeasuredEnvelope(
        gate_width=320,
        gate_height=180,
        min_foreground_fraction=0.002,
        derived_foreground_threshold_px=115.2,
        measured_stack="test",
        sha="test",
        speeds_gate_px=(0.5, 5.0),
        thresholds_px=(None, 110.0),
        swept_max_px=None,
    )
    assert unbounded.wake_threshold_px(0.5) == float(
        "inf"
    ), "with no recorded sweep range there is genuinely nothing to say"


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
    assert all(
        s["wake_threshold_silhouette_gate_px"] is not None for s in data["samples"]
    ), (
        "every sampled speed must have resolved a crossing. An uncrossed speed "
        "means the sweep was too narrow, and reading it as 'unreachable' is "
        "the defect that mislabelled v3's slow clips"
    )


def test_envelope_artifact_records_the_range_it_swept() -> None:
    """Without this, 'did not cross' cannot be told from 'cannot wake'."""
    data = json.loads((REPO_ROOT / DEFAULT_ENVELOPE_PATH).read_text())
    swept = data.get("swept_area_gate_px")
    assert swept and swept.get("stop"), (
        "the artifact must record how far it swept; the first envelope did not "
        "and its uncrossed speeds were misread as physical limits"
    )
    thresholds = [s["wake_threshold_silhouette_gate_px"] for s in data["samples"]]
    assert max(t for t in thresholds if t is not None) < swept["stop"], (
        "every crossing must sit inside the swept range, or the sweep stopped "
        "too early to have found it"
    )
