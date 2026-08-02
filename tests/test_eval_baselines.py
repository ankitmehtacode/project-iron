"""The trivial-baseline requirement: enforcement, math, serialisation.

Three properties this file pins:

1. **Enforcement is at emit time.** A metric name with no registered baseline
   raises ``BaselineMissing`` when a caller tries to build a ``Metric`` for
   it — not at import time, not at report time.
2. **The margin math matches the sense of the metric.** Higher-is-better
   picks the highest baseline; lower-is-better picks the lowest; a degenerate
   predictor at or below the strongest baseline produces margin <= 0.
3. **The flag survives serialisation.** A ``Metric`` marked ``flagged=True``
   comes back through ``as_dict`` with ``flagged: true``, so scorecards
   round-tripping through JSON do not lose the warning.

The registry is a module-level singleton, so tests that would clear it
between cases either use disjoint metric names or use the ``_Registry``
class directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.scorecard import Metric, _metric_with_baselines, _strongest_baseline
from src.eval.baselines import (
    Baseline,
    BaselineMissing,
    _Registry,
    compute_baselines,
    margin,
    register_baseline,
    require_baseline,
)


def _b(name: str, value: float, *, flag_worthy: bool = True) -> Baseline:
    return Baseline(name, value, f"{name} baseline", flag_worthy=flag_worthy)


def test_unregistered_metric_raises_baseline_missing() -> None:
    with pytest.raises(BaselineMissing, match="no baseline is registered"):
        compute_baselines("this_metric_does_not_exist")
    with pytest.raises(BaselineMissing):
        require_baseline("this_metric_does_not_exist")


def test_metric_construction_raises_when_no_baseline_registered() -> None:
    with pytest.raises(BaselineMissing):
        _metric_with_baselines(
            "no_such_registered_metric", 0.5, "fraction", True
        )


def test_higher_is_better_margin_uses_strongest_baseline() -> None:
    baselines = [_b("weak", 0.2), _b("strong", 0.8)]
    assert margin(0.9, baselines, higher_is_better=True) == pytest.approx(0.1)
    assert margin(0.7, baselines, higher_is_better=True) == pytest.approx(-0.1)


def test_lower_is_better_margin_uses_lowest_baseline() -> None:
    baselines = [_b("weak", 100.0), _b("strong", 5.0)]
    assert margin(3.0, baselines, higher_is_better=False) == pytest.approx(2.0)
    assert margin(5.0, baselines, higher_is_better=False) == pytest.approx(0.0)
    assert margin(10.0, baselines, higher_is_better=False) == pytest.approx(-5.0)


def test_non_flag_worthy_baselines_do_not_drive_margin() -> None:
    # A boundary baseline (recall=1.0) plus a flag-worthy baseline (0.5)
    # should compute margin against the flag-worthy one only. Otherwise
    # every real recall would flag against the 1.0 boundary.
    baselines = [_b("boundary", 1.0, flag_worthy=False), _b("real", 0.5)]
    assert margin(0.7, baselines, higher_is_better=True) == pytest.approx(0.2)


def test_margin_is_nan_when_no_baselines_provided() -> None:
    assert not np.isfinite(margin(0.5, [], higher_is_better=True))


def test_degenerate_metric_is_flagged_higher_is_better() -> None:
    # motion_gate.precision's always-wake baseline is moving_fraction
    # (achievable to beat). A "degenerate" gate that just wakes on
    # everything matches the moving_fraction exactly, margin 0.0, flagged.
    m = _metric_with_baselines(
        "motion_gate.precision", 0.2, "fraction", True, moving_fraction=0.2
    )
    assert m.flagged is True
    assert m.margin == pytest.approx(0.0)


def test_boundary_baseline_does_not_flag() -> None:
    # motion_gate.recall's baselines are both boundaries (always-wake=1.0
    # and never-wake=0.0). Neither is flag_worthy — a real gate at recall
    # 1.0 might be doing something meaningful (precision + F1 will catch
    # it if not), and a real gate at recall 0.9 shouldn't be flagged just
    # because always-wake beats it by construction.
    m = _metric_with_baselines("motion_gate.recall", 0.9, "fraction", True)
    assert m.flagged is False
    assert not np.isfinite(m.margin)  # NaN: no flag-worthy baseline exists


def test_healthy_metric_is_not_flagged() -> None:
    # false_positives baseline (always_wake) with 100 non-moving frames is
    # 100. A gate producing 3 FPs beats that by 97; margin > 0, no flag.
    m = _metric_with_baselines(
        "motion_gate.false_positives",
        3.0,
        "frames",
        False,
        non_moving_frames=100,
    )
    assert m.flagged is False
    assert m.margin == pytest.approx(97.0)


def test_flag_survives_serialisation_roundtrip() -> None:
    # Precision matched exactly by always-wake → flag fires; the flag
    # (and the baselines that produced it) must round-trip through JSON.
    m = _metric_with_baselines(
        "motion_gate.precision", 0.3, "fraction", True, moving_fraction=0.3
    )
    payload = m.as_dict()
    assert payload["flagged"] is True
    assert isinstance(payload["baselines"], list)
    assert any(b["name"] == "always_wake" for b in payload["baselines"])
    assert payload["margin"] == pytest.approx(0.0)


def test_scorecard_render_shows_baseline_and_margin() -> None:
    from src.data.scorecard import Scorecard

    card = Scorecard(
        golden_set_version="test",
        golden_set_sha="0" * 64,
        domain="indoor",
        clips_scored=0,
    )
    card.metrics = [
        _metric_with_baselines(
            "motion_gate.precision", 0.75, "fraction", True, moving_fraction=0.2
        )
    ]
    text = card.render()
    assert "always_wake" in text
    assert "0.2000" in text  # baseline value
    # Precision 0.75 beats a 0.2 baseline by 0.55 → margin visible.
    assert "0.5500" in text or "+0.5500" in text


def test_registry_class_is_isolable() -> None:
    # A fresh _Registry instance shares nothing with the module singleton —
    # important for future tests that need to prove a metric raises without
    # perturbing every other test in the process.
    reg = _Registry()
    assert not reg.has("motion_gate.recall")
    with pytest.raises(BaselineMissing):
        reg.compute("motion_gate.recall")
