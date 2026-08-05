"""Objective 3 — Coverage and Absence.

STRUCTURAL, absolute: there is no code path that returns "nothing
happened" from an empty query result. Every test here either proves an
Absence with its Coverage basis, or gets CannotEstablish — never a bare
negative.
"""

from __future__ import annotations

import pytest

from src.model.coverage import (
    Absence,
    CannotEstablish,
    Coverage,
    CoverageError,
    Gap,
    Interval,
    prove_absence,
)

HOUR_NS = 3_600_000_000_000
DAY0 = 1_785_000_000 * 1_000_000_000


def _query() -> Interval:
    return Interval(DAY0, DAY0 + HOUR_NS)


def _live_coverage(subject_id: str = "cam-lobby-01", **overrides: object) -> Coverage:
    kwargs: dict[str, object] = dict(
        subject_id=subject_id,
        subject_kind="camera",
        interval=_query(),
        status="live",
        envelope_ref="motion_gate@cam-lobby-01@twin_rev=1",
        gaps=(),
        manifest_sha="sha-cov-1",
    )
    kwargs.update(overrides)
    return Coverage(**kwargs)  # type: ignore[arg-type]


def _never_moved(subject_id: str, interval: Interval) -> bool:
    return False


def _always_moved(subject_id: str, interval: Interval) -> bool:
    return True


# ---------------------------------------------------------------------------
# Interval algebra
# ---------------------------------------------------------------------------


def test_interval_rejects_non_positive_span() -> None:
    with pytest.raises(CoverageError):
        Interval(100, 100)
    with pytest.raises(CoverageError):
        Interval(100, 50)


def test_interval_subtract_splits_around_a_gap() -> None:
    whole = Interval(0, 100)
    gap = Interval(30, 60)
    pieces = whole.subtract(gap)
    assert pieces == (Interval(0, 30), Interval(60, 100))


def test_interval_subtract_no_overlap_returns_self() -> None:
    whole = Interval(0, 100)
    other = Interval(200, 300)
    assert whole.subtract(other) == (whole,)


# ---------------------------------------------------------------------------
# Coverage construction
# ---------------------------------------------------------------------------


def test_coverage_requires_envelope_ref() -> None:
    with pytest.raises(CoverageError):
        _live_coverage(envelope_ref="")


def test_gap_must_fall_within_coverage_interval() -> None:
    outside_gap = Gap(
        camera_id="cam-lobby-01",
        interval=Interval(DAY0 - 10, DAY0 - 5),
        reason="disconnect",
    )
    with pytest.raises(CoverageError):
        _live_coverage(gaps=(outside_gap,))


def test_coverage_rejects_unknown_status() -> None:
    with pytest.raises(CoverageError):
        _live_coverage(status="mostly_fine")


# ---------------------------------------------------------------------------
# STRUCTURAL: empty coverage log can never prove absence
# ---------------------------------------------------------------------------


def test_empty_coverage_log_cannot_establish_absence() -> None:
    result = prove_absence("cam-lobby-01", _query(), _never_moved, coverage_log=())
    assert isinstance(result, CannotEstablish)
    assert not isinstance(result, Absence)
    assert result.reason == "no_coverage"
    assert result.uncovered_subintervals == (_query(),)


def test_coverage_for_a_different_subject_does_not_count() -> None:
    other_camera_coverage = _live_coverage(subject_id="cam-dock-02")
    result = prove_absence(
        "cam-lobby-01", _query(), _never_moved, coverage_log=[other_camera_coverage]
    )
    assert isinstance(result, CannotEstablish)
    assert result.reason == "no_coverage"


def test_prove_absence_return_type_is_always_the_union() -> None:
    """No third value, no None, ever — for every coverage shape tried."""
    scenarios = [
        (),
        [_live_coverage()],
        [_live_coverage(status="degraded")],
        [_live_coverage(status="offline")],
    ]
    for coverage_log in scenarios:
        result = prove_absence("cam-lobby-01", _query(), _never_moved, coverage_log)
        assert isinstance(result, (Absence, CannotEstablish))


# ---------------------------------------------------------------------------
# The proof path
# ---------------------------------------------------------------------------


def test_unbroken_live_coverage_with_no_occurrence_proves_absence() -> None:
    result = prove_absence(
        "cam-lobby-01", _query(), _never_moved, coverage_log=[_live_coverage()]
    )
    assert isinstance(result, Absence)
    assert result.subject_id == "cam-lobby-01"
    assert result.coverage_basis
    assert all(c.status == "live" for c in result.coverage_basis)


def test_predicate_holding_is_not_a_proven_absence() -> None:
    result = prove_absence(
        "cam-lobby-01", _query(), _always_moved, coverage_log=[_live_coverage()]
    )
    assert isinstance(result, CannotEstablish)
    assert result.reason == "predicate_held"


# ---------------------------------------------------------------------------
# Degraded / partial coverage — the falsification-suite scenario
# ---------------------------------------------------------------------------


def test_degraded_status_blocks_absence_even_with_full_time_coverage() -> None:
    degraded = _live_coverage(status="degraded")
    result = prove_absence(
        "cam-lobby-01", _query(), _never_moved, coverage_log=[degraded]
    )
    assert isinstance(result, CannotEstablish)
    assert result.reason == "coverage_insufficient"
    assert any("degraded" in v for v in result.envelope_violations)


def test_occluded_and_offline_also_block_absence() -> None:
    for status in ("occluded", "offline"):
        cov = _live_coverage(status=status)
        result = prove_absence(
            "cam-lobby-01", _query(), _never_moved, coverage_log=[cov]
        )
        assert isinstance(result, CannotEstablish)


def test_partial_time_coverage_reports_the_uncovered_remainder() -> None:
    query = _query()
    half = Interval(query.start_ns, query.start_ns + HOUR_NS // 2)
    partial = _live_coverage(interval=half)
    result = prove_absence("cam-lobby-01", query, _never_moved, coverage_log=[partial])
    assert isinstance(result, CannotEstablish)
    assert result.reason == "no_coverage"
    assert result.uncovered_subintervals == (Interval(half.end_ns, query.end_ns),)


def test_gap_inside_live_coverage_blocks_absence_and_is_reported() -> None:
    query = _query()
    gap = Gap(
        camera_id="cam-lobby-01",
        interval=Interval(query.start_ns + 100, query.start_ns + 200),
        reason="dropped_frame",
    )
    covered_with_gap = _live_coverage(gaps=(gap,))
    result = prove_absence(
        "cam-lobby-01", query, _never_moved, coverage_log=[covered_with_gap]
    )
    assert isinstance(result, CannotEstablish)
    assert result.gaps == (gap,)


def test_multiple_coverage_records_stitch_to_prove_absence() -> None:
    query = _query()
    mid = query.start_ns + HOUR_NS // 2
    first_half = _live_coverage(interval=Interval(query.start_ns, mid))
    second_half = _live_coverage(interval=Interval(mid, query.end_ns))
    result = prove_absence(
        "cam-lobby-01", query, _never_moved, coverage_log=[first_half, second_half]
    )
    assert isinstance(result, Absence)
    assert len(result.coverage_basis) == 2


def test_absence_requires_nonempty_coverage_basis() -> None:
    with pytest.raises(CoverageError):
        Absence(
            subject_id="cam-lobby-01",
            interval=_query(),
            predicate="never_moved",
            coverage_basis=(),
        )


def test_cannot_establish_requires_reason() -> None:
    with pytest.raises(CoverageError):
        CannotEstablish(
            reason="", uncovered_subintervals=(), envelope_violations=(), gaps=()
        )
