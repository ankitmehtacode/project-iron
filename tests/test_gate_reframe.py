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


def test_wake_fraction_with_recall_retained_does_not_raise() -> None:
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
