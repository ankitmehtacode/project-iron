"""Day 14, Objective 3 — the gate.* metric reframe.

Day 12's baseline rule showed motion_gate.precision/f1/false_positives all
indistinguishable from always-wake on v3-indoor: always-wake has recall
1.0 by definition, so precision/recall was never the frame that measures
what a gate is FOR — a gate's product is compute saved, not detections
made. These tests pin the four replacement metrics (gate.wake_fraction,
gate.compute_saved, gate.recall_retained, gate.miss_cost), their
baselines, and the structural pairing rule, without needing to render a
full synthetic clip (see test_synthetic_indoor.py for the end-to-end
version against real generated data).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.scorecard import (
    PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME,
    Metric,
    ScorecardError,
    Undefined,
    _gate_rates_from_totals,
    _metric_with_baselines,
    _validate_gate_metric_pairing,
)


# ---------------------------------------------------------------------------
# The pairing rule: wake_fraction never ships without recall_retained.
# ---------------------------------------------------------------------------


def test_wake_fraction_without_recall_retained_raises() -> None:
    wake_fraction_only = [
        Metric(
            name="gate.wake_fraction",
            value=0.5,
            unit="fraction",
            higher_is_better=False,
        )
    ]
    with pytest.raises(ScorecardError, match="gate.wake_fraction"):
        _validate_gate_metric_pairing(wake_fraction_only)


def test_wake_fraction_with_recall_retained_but_no_moving_fraction_raises() -> None:
    """Day 15: wake_fraction's pairing requirement grew a second member,
    dataset.moving_frame_fraction — recall_retained alone is no longer
    enough. A wake fraction read without the scene's own motion density
    reads as a gate defect when it may just be how much of the scene moves.
    """
    missing_moving_fraction = [
        Metric(
            name="gate.wake_fraction",
            value=0.5,
            unit="fraction",
            higher_is_better=False,
        ),
        Metric(
            name="gate.recall_retained",
            value=0.9,
            unit="fraction",
            higher_is_better=True,
        ),
    ]
    with pytest.raises(ScorecardError, match="gate.wake_fraction"):
        _validate_gate_metric_pairing(missing_moving_fraction)


def test_wake_fraction_with_recall_retained_and_moving_fraction_does_not_raise() -> None:
    paired = [
        Metric(
            name="gate.wake_fraction",
            value=0.5,
            unit="fraction",
            higher_is_better=False,
        ),
        Metric(
            name="gate.recall_retained",
            value=0.9,
            unit="fraction",
            higher_is_better=True,
        ),
        Metric(
            name="dataset.moving_frame_fraction",
            value=0.95,
            unit="fraction",
            higher_is_better=False,
        ),
    ]
    _validate_gate_metric_pairing(paired)  # must not raise


def test_recall_retained_alone_is_fine_without_wake_fraction() -> None:
    # The rule is directional: recall_retained does not require
    # wake_fraction (there is no product claim to check without the
    # savings number present in the first place).
    only_recall = [
        Metric(
            name="gate.recall_retained",
            value=0.9,
            unit="fraction",
            higher_is_better=True,
        )
    ]
    _validate_gate_metric_pairing(only_recall)  # must not raise


def test_unrelated_metrics_are_unaffected_by_the_pairing_rule() -> None:
    _validate_gate_metric_pairing(
        [
            Metric(
                name="coverage.frames_scored",
                value=100.0,
                unit="frames",
                higher_is_better=True,
            )
        ]
    )


# ---------------------------------------------------------------------------
# Baselines bracket the space: always-wake and never-wake.
# ---------------------------------------------------------------------------


def test_wake_fraction_baselines_bracket_the_space() -> None:
    m = _metric_with_baselines("gate.wake_fraction", 0.3, "fraction", False)
    names = {b.name: b.value for b in m.baselines}
    assert names["always_wake"] == pytest.approx(1.0)
    assert names["never_wake"] == pytest.approx(0.0)


def test_wake_fraction_always_wake_is_flag_worthy_unlike_recall() -> None:
    # A real gate at wake_fraction 1.0 (== always-wake) IS a defect —
    # unlike recall, where matching always-wake's 1.0 is not.
    m = _metric_with_baselines("gate.wake_fraction", 1.0, "fraction", False)
    assert m.flagged is True
    assert m.margin == pytest.approx(0.0)


def test_recall_retained_baselines_are_both_boundaries() -> None:
    m = _metric_with_baselines("gate.recall_retained", 0.9, "fraction", True)
    assert m.flagged is False
    assert not np.isfinite(m.margin)  # no flag-worthy baseline, same as Day 12's recall


def test_compute_saved_baselines_use_stated_placeholder_ceiling() -> None:
    m = _metric_with_baselines(
        "gate.compute_saved",
        5.0,
        "ms/frame (estimate)",
        True,
        max_compute_saved_ms=PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME,
    )
    names = {b.name: b.value for b in m.baselines}
    assert names["always_wake"] == pytest.approx(0.0)
    assert names["never_wake"] == pytest.approx(
        PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME
    )


def test_miss_cost_baselines_never_wake_is_beatable() -> None:
    m = _metric_with_baselines("gate.miss_cost", 3.0, "frames", False, moving_frames=50)
    names = {b.name: b.value for b in m.baselines}
    assert names["always_wake"] == pytest.approx(0.0)
    assert names["never_wake"] == pytest.approx(50.0)
    # 3 missed out of 50 possible beats never-wake's 50 by 47.
    assert m.margin == pytest.approx(47.0)
    assert m.flagged is False


def test_miss_cost_flags_a_gate_no_better_than_never_wake() -> None:
    m = _metric_with_baselines(
        "gate.miss_cost", 50.0, "frames", False, moving_frames=50
    )
    assert m.flagged is True
    assert m.margin == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The compute_saved formula: an estimate under a stated, unmeasured model.
# ---------------------------------------------------------------------------


def test_compute_saved_formula_is_suppressed_fraction_times_placeholder() -> None:
    presented = 100
    wakes_total = 40
    frames_suppressed = presented - wakes_total
    suppressed_fraction = frames_suppressed / presented
    expected = suppressed_fraction * PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME

    assert expected == pytest.approx(0.6 * PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME)


def test_placeholder_cost_constant_is_documented_as_unmeasured() -> None:
    # Regression guard against someone quietly turning this into a real
    # measurement without updating the label on every metric that reads it.
    import inspect

    from src.data import scorecard as scorecard_module

    source = inspect.getsource(scorecard_module)
    idx = source.index("PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME = 20.0")
    docstring_slice = source[idx : idx + 400]
    assert "NOT MEASURED" in docstring_slice


# ---------------------------------------------------------------------------
# Objective 3 (Day 15) — _condition_bucket, the occupied/empty/night/
# degenerate mapping over the golden-set Condition taxonomy.
# ---------------------------------------------------------------------------


def test_condition_bucket_empty_takes_priority() -> None:
    from src.data.golden import Condition
    from src.data.scorecard import _condition_bucket

    assert _condition_bucket((Condition.EMPTY, Condition.DAYLIGHT)) == "empty"


def test_condition_bucket_lights_transient_and_glare_are_degenerate() -> None:
    from src.data.golden import Condition
    from src.data.scorecard import _condition_bucket

    assert (
        _condition_bucket((Condition.LIGHTS_TRANSIENT, Condition.EVENING_ARTIFICIAL))
        == "degenerate"
    )
    assert _condition_bucket((Condition.GLARE, Condition.DAYLIGHT)) == "degenerate"


def test_condition_bucket_evening_artificial_alone_is_night() -> None:
    """v3-indoor has no clip like this (every EVENING_ARTIFICIAL clip is
    also LIGHTS_TRANSIENT there), but the bucket itself must exist and
    classify correctly for a future clip that carries steady artificial
    light without a transient lighting event.
    """
    from src.data.golden import Condition
    from src.data.scorecard import _condition_bucket

    assert _condition_bucket((Condition.EVENING_ARTIFICIAL,)) == "night"


def test_condition_bucket_defaults_to_occupied() -> None:
    from src.data.golden import Condition
    from src.data.scorecard import _condition_bucket

    assert _condition_bucket((Condition.DAYLIGHT, Condition.SINGLE_PERSON)) == "occupied"
    assert _condition_bucket(()) == "occupied"


def test_condition_bucket_partitions_v3_indoor_exactly() -> None:
    """Pins the real v3-indoor manifest's bucket counts so a manifest edit
    or a mapping-priority change is caught here, not discovered by someone
    reading a report with a table that no longer adds to 30.
    """
    from pathlib import Path

    from src.data.golden import load_golden_set
    from src.data.scorecard import _CONDITION_BUCKETS, _condition_bucket

    golden_root = Path(__file__).resolve().parents[1] / "configs" / "golden"
    golden = load_golden_set(golden_root, "v3-indoor")
    counts = {b: 0 for b in _CONDITION_BUCKETS}
    for clip in golden.clips:
        counts[_condition_bucket(clip.conditions)] += 1

    assert counts == {"occupied": 25, "empty": 1, "night": 0, "degenerate": 4}
    assert sum(counts.values()) == len(golden.clips) == 30


def test_dataset_moving_frame_fraction_has_a_registered_baseline() -> None:
    metric = _metric_with_baselines(
        "dataset.moving_frame_fraction", 0.97, "fraction", False
    )
    assert metric.baselines  # BaselineMissing would have raised otherwise


# ---------------------------------------------------------------------------
# Day 18, Objective 2 — Undefined, not NaN, for a zero-denominator metric.
#
# gate.recall_retained on a genuinely quiet clip is 0 true positives over 0
# moving frames: arithmetically NaN, and NaN used to satisfy "a value is
# present" (including the Day-14 co-emission check, which only looks at
# metric *names*) while carrying no information about why. The same
# zero-denominator problem existed, less visibly, in gate.wake_fraction,
# gate.compute_saved, dataset.moving_frame_fraction and
# coverage.observable_fraction whenever a run scores zero presented frames
# (a fully content-sha-refused golden set; a real clip shorter than the
# gate's warmup window) — closed the same way, for the same reason.
# ---------------------------------------------------------------------------


def test_metric_with_baselines_raises_on_a_bare_nan() -> None:
    """NaN cannot satisfy the co-emission requirement (Day 18).

    The raise fires before a Metric is constructed at all, so a bare NaN
    can never reach a scorecard — the same discipline BaselineMissing
    already applies to an unregistered metric name.
    """
    with pytest.raises(ScorecardError, match="gate.wake_fraction"):
        _metric_with_baselines("gate.wake_fraction", float("nan"), "fraction", False)


def test_metric_with_baselines_raise_names_the_metric_and_a_likely_cause() -> None:
    with pytest.raises(ScorecardError, match="zero-frame denominator"):
        _metric_with_baselines(
            "gate.recall_retained", float("nan"), "fraction", True
        )


def test_metric_with_baselines_accepts_undefined() -> None:
    """The escape hatch NaN does not get: an explicit, reasoned sentinel."""
    metric = _metric_with_baselines(
        "gate.recall_retained",
        Undefined(reason="no moving frames in denominator"),
        "fraction",
        True,
    )
    assert isinstance(metric.value, Undefined)
    assert metric.value.reason == "no moving frames in denominator"
    # No number, so no baseline comparison is meaningful either.
    assert not np.isfinite(metric.margin)
    assert metric.flagged is False
    # Baselines are still attached — Undefined is a value problem, not a
    # reason to skip the "every metric declares a trivial strategy" rule.
    assert metric.baselines


def test_undefined_metric_renders_as_undefined_never_as_a_number() -> None:
    from src.data.scorecard import Scorecard

    metric = _metric_with_baselines(
        "gate.recall_retained",
        Undefined(reason="no moving frames in denominator"),
        "fraction",
        True,
    )
    card = Scorecard(
        golden_set_version="vtest",
        golden_set_sha="deadbeef",
        domain="indoor",
        clips_scored=1,
        metrics=[metric],
    )

    rendered = card.render()
    assert "undefined (no moving frames in denominator)" in rendered
    assert "nan" not in rendered.lower()

    payload = card.as_dict()
    value = payload["metrics"][0]["value"]
    assert value == {"undefined": True, "reason": "no moving frames in denominator"}


def test_gate_rates_from_totals_recall_retained_is_undefined_with_zero_moving_frames() -> (
    None
):
    """Applied per condition bucket, not only in the aggregate (Day 18):

    a bucket that itself contains zero moving frames must say so even when
    the set as a whole has plenty — a quiet bucket sitting inside an
    otherwise-active set is exactly as undefined as a quiet aggregate.
    """
    totals = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 5,
        "scored_frames": 5,
        "below_envelope_frames": 0,
        "envelope_limited_misses": 0,
        "unobservable_frames": 0,
        "wakes_outside_envelope": 0,
        "frames_with_world_motion": 0,
    }
    rates = _gate_rates_from_totals(totals)

    assert isinstance(rates["recall_retained"], Undefined)
    assert rates["recall_retained"].reason == "no moving frames in denominator"
    # presented > 0 here (5 scored frames), so wake_fraction is a real
    # number — only recall_retained is undefined, because only recall's
    # own denominator (moving frames) is zero.
    assert rates["wake_fraction"] == pytest.approx(0.0)


def test_gate_rates_from_totals_recall_retained_is_a_real_number_when_moving_frames_exist() -> (
    None
):
    totals = {
        "tp": 3,
        "fp": 0,
        "fn": 1,
        "tn": 2,
        "scored_frames": 6,
        "below_envelope_frames": 0,
        "envelope_limited_misses": 0,
        "unobservable_frames": 0,
        "wakes_outside_envelope": 0,
        "frames_with_world_motion": 4,
    }
    rates = _gate_rates_from_totals(totals)

    assert rates["recall_retained"] == pytest.approx(0.75)
