"""Scorecards under different observability proxies are not comparable
(Day 41, Objective 5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.scorecard import (
    OBSERVABILITY_BOX_AREA,
    OBSERVABILITY_INSTANCE_MASK,
    Observability,
    Partition,
    Scorecard,
    ScorecardError,
    gate_counts,
)


def _card(proxy: str) -> Scorecard:
    return Scorecard(
        golden_set_version="vtest",
        golden_set_sha="a" * 64,
        domain="indoor",
        clips_scored=1,
        observability_proxy=proxy,
        envelope={"envelope_sha": "e" * 64},
    )


def test_box_proxy_and_mask_scorecards_refuse_comparison() -> None:
    with pytest.raises(ScorecardError, match="partitioned observability"):
        _card(OBSERVABILITY_BOX_AREA).require_comparable(
            _card(OBSERVABILITY_INSTANCE_MASK)
        )
    with pytest.raises(ScorecardError, match="partitioned observability"):
        _card(OBSERVABILITY_INSTANCE_MASK).require_comparable(
            _card(OBSERVABILITY_BOX_AREA)
        )


def test_same_proxy_same_set_same_envelope_is_comparable() -> None:
    _card(OBSERVABILITY_BOX_AREA).require_comparable(_card(OBSERVABILITY_BOX_AREA))


def test_the_proxy_is_required_and_serialised() -> None:
    with pytest.raises(TypeError):
        Scorecard(  # type: ignore[call-arg]
            golden_set_version="vtest",
            golden_set_sha="a" * 64,
            domain="indoor",
            clips_scored=1,
        )
    assert (
        _card(OBSERVABILITY_BOX_AREA).as_dict()["observability_proxy"]
        == OBSERVABILITY_BOX_AREA
    )
    assert "observability: box_area" in _card(OBSERVABILITY_BOX_AREA).render()


def test_gate_counts_scores_a_partition_built_outside_the_npz_path() -> None:
    labels = np.array(
        [
            Observability.NO_MOTION,  # warmup, not scored
            Observability.ABOVE_ENVELOPE,  # woke: tp
            Observability.ABOVE_ENVELOPE,  # slept: fn
            Observability.NO_MOTION,  # woke: fp
            Observability.BELOW_ENVELOPE,  # slept: envelope-limited miss
        ],
        dtype=np.int8,
    )
    partition = Partition(
        labels=labels,
        warm=1,
        envelope_gate_px=115.2,
        world_moved_any=np.array([False, True, True, False, True]),
    )
    counts, _ = gate_counts(partition, np.array([True, True, False, True, False]))
    assert (counts["tp"], counts["fn"], counts["fp"], counts["tn"]) == (1, 1, 1, 0)
    assert counts["envelope_limited_misses"] == 1
    assert counts["frames_with_world_motion"] == 3
    with pytest.raises(ScorecardError):
        gate_counts(partition, np.array([True]))
