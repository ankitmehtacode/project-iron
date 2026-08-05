"""Coverage and Absence — the primitive that makes this an evidence engine.

A search product returns "no results" for an empty query and lets the
reader assume that means nothing happened. That assumption is false every
time the camera was offline, occluded, or degraded during the window in
question, and nothing in a plain event log distinguishes "we looked and
saw nothing" from "we never looked". :class:`Coverage` records is what was
actually watched (per camera or zone, per interval, with a status and its
gaps); :func:`prove_absence` is the only sanctioned way to turn that
record into a negative claim, and it can only ever return a proven
:class:`Absence` or a structured :class:`CannotEstablish` — never a bare
"nothing happened".

STRUCTURAL, absolute
---------------------
There is no code path in this module that returns "nothing happened" from
an empty query result. :func:`prove_absence`'s return type is the union
``Absence | CannotEstablish`` and nothing else; an empty coverage log
(no :class:`Coverage` record at all for the requested camera/zone) yields
:class:`CannotEstablish`, never a proven absence — see
``test_model_coverage.py::test_empty_coverage_log_cannot_establish_absence``.
Silence is not absence, and the type system says so: there is no
constructor, no default, and no code path anywhere in this module that
produces a "nothing happened" value outside the union.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence, get_args

CoverageStatus = Literal["live", "degraded", "offline", "occluded"]
COVERAGE_STATUSES: tuple[CoverageStatus, ...] = get_args(CoverageStatus)

GapReason = Literal["dropped_frame", "backpressure", "disconnect"]
GAP_REASONS: tuple[GapReason, ...] = get_args(GapReason)


class CoverageError(ValueError):
    """Raised when a Coverage or Gap record violates its contract."""


@dataclass(frozen=True)
class Interval:
    """A half-open time interval ``[start_ns, end_ns)``."""

    start_ns: int
    end_ns: int

    def __post_init__(self) -> None:
        if self.end_ns <= self.start_ns:
            raise CoverageError(
                f"Interval must be positive: start_ns={self.start_ns} "
                f"end_ns={self.end_ns}"
            )

    def overlaps(self, other: "Interval") -> bool:
        return self.start_ns < other.end_ns and other.start_ns < self.end_ns

    def intersection(self, other: "Interval") -> "Interval | None":
        if not self.overlaps(other):
            return None
        return Interval(
            max(self.start_ns, other.start_ns), min(self.end_ns, other.end_ns)
        )

    def subtract(self, other: "Interval") -> tuple["Interval", ...]:
        """This interval minus ``other``, as zero, one, or two remaining pieces."""
        overlap = self.intersection(other)
        if overlap is None:
            return (self,)
        pieces = []
        if overlap.start_ns > self.start_ns:
            pieces.append(Interval(self.start_ns, overlap.start_ns))
        if overlap.end_ns < self.end_ns:
            pieces.append(Interval(overlap.end_ns, self.end_ns))
        return tuple(pieces)


@dataclass(frozen=True)
class Gap:
    """One interval during which coverage was interrupted.

    Written for every dropped frame, backpressure event, or disconnect —
    the caller is expected to emit one of these at the moment it happens,
    not reconstruct them later from absence of data.
    """

    camera_id: str
    interval: Interval
    reason: GapReason

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise CoverageError("Gap.camera_id must not be empty")
        if self.reason not in GAP_REASONS:
            raise CoverageError(
                f"unknown gap reason {self.reason!r}; expected one of {GAP_REASONS}"
            )


@dataclass(frozen=True)
class Coverage:
    """What was actually watched: a camera or zone, an interval, and a status.

    Attributes:
        subject_id: The camera_id or zone_id this record covers.
        subject_kind: Whether ``subject_id`` names a camera or a zone.
        interval: The interval this record attests to.
        status: The capability state in effect for the whole interval.
            A status that changes mid-interval is two Coverage records,
            not one with a status that is sometimes true.
        envelope_ref: Which :class:`~src.model.envelope.Envelope`
            (capability, at this status) this record's guarantees are
            measured against. Required: a coverage claim with no envelope
            reference cannot say what "live" was capable of detecting.
        gaps: Interruptions within ``interval``. Must each fall inside
            ``interval`` — a gap outside the interval it is attached to
            is a bookkeeping error, not data.
        manifest_sha: The run that produced this record.
    """

    subject_id: str
    subject_kind: Literal["camera", "zone"]
    interval: Interval
    status: CoverageStatus
    envelope_ref: str
    gaps: tuple[Gap, ...]
    manifest_sha: str

    def __post_init__(self) -> None:
        if not self.subject_id:
            raise CoverageError("Coverage.subject_id must not be empty")
        if self.status not in COVERAGE_STATUSES:
            raise CoverageError(
                f"unknown coverage status {self.status!r}; expected one of "
                f"{COVERAGE_STATUSES}"
            )
        if not self.envelope_ref:
            raise CoverageError(
                "Coverage.envelope_ref is required: a coverage claim with no "
                "envelope reference cannot say what its status was capable "
                "of detecting"
            )
        if not self.manifest_sha:
            raise CoverageError("Coverage.manifest_sha must not be empty")
        for gap in self.gaps:
            if (
                gap.interval.start_ns < self.interval.start_ns
                or gap.interval.end_ns > self.interval.end_ns
            ):
                raise CoverageError(
                    f"Gap {gap.interval} falls outside Coverage interval "
                    f"{self.interval}"
                )

    def uncovered_subintervals(self, query: Interval) -> tuple[Interval, ...]:
        """The parts of ``query`` this record does NOT attest to.

        Includes both the parts outside ``self.interval`` entirely and the
        gaps within it — both are places a predicate cannot be evaluated
        from this record alone.
        """
        overlap = self.interval.intersection(query)
        if overlap is None:
            return (query,)
        remaining = [overlap]
        for gap in self.gaps:
            next_remaining: list[Interval] = []
            for piece in remaining:
                next_remaining.extend(piece.subtract(gap.interval))
            remaining = next_remaining
        outside: list[Interval] = list(query.subtract(self.interval))
        return tuple(outside + remaining)


# ---------------------------------------------------------------------------
# Absence — derived, never stored.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Absence:
    """A proven negative claim: ``predicate`` did not hold in ``interval``.

    Never constructed directly by a query layer from an empty result —
    only by :func:`prove_absence`, which requires unbroken, envelope-valid
    Coverage across the whole interval before it will build one.
    """

    subject_id: str
    interval: Interval
    predicate: str
    coverage_basis: tuple[Coverage, ...]

    def __post_init__(self) -> None:
        if not self.predicate:
            raise CoverageError("Absence.predicate must not be empty")
        if not self.coverage_basis:
            raise CoverageError(
                "Absence.coverage_basis must not be empty: a proven absence "
                "with no coverage evidence backing it is exactly the "
                "silent-as-absence bug this type exists to prevent"
            )


@dataclass(frozen=True)
class CannotEstablish:
    """Absence could not be proven or refuted — the honest non-answer.

    Attributes:
        reason: Short machine-readable cause (e.g. ``"no_coverage"``,
            ``"partial_coverage"``, ``"envelope_violation"``).
        uncovered_subintervals: Parts of the query interval with no
            Coverage record, or that fall inside a Gap.
        envelope_violations: Subject/interval pairs where Coverage existed
            but its status was not sufficient to evaluate the predicate
            (e.g. ``degraded`` or ``occluded``).
        gaps: The specific Gap records responsible, when applicable.
    """

    reason: str
    uncovered_subintervals: tuple[Interval, ...]
    envelope_violations: tuple[str, ...]
    gaps: tuple[Gap, ...]

    def __post_init__(self) -> None:
        if not self.reason:
            raise CoverageError("CannotEstablish.reason must not be empty")


AbsenceResult = Absence | CannotEstablish
"""The only two values :func:`prove_absence` may return.

There is deliberately no third option and no ``None`` — a caller cannot
receive a value that is neither a proof nor a structured refusal.
"""

Predicate = Callable[[str, Interval], bool]
"""A predicate over (subject_id, interval): did the claim hold throughout?

Evaluating this is out of scope for Objective 3 — it is supplied by the
caller (typically backed by the event log) and :func:`prove_absence`'s job
is solely to establish whether the Coverage backing it is strong enough to
trust the answer, not to compute the answer itself.
"""

_ESTABLISHING_STATUSES: frozenset[CoverageStatus] = frozenset({"live"})
"""Only 'live' coverage can back a proven absence.

'degraded' and 'occluded' coverage watched *something*, but not reliably
enough to certify that a negative result reflects the world rather than
the sensor's own blind spot. 'offline' obviously cannot. Widening this set
is a product decision with its own measurement, not a Day-13 default.
"""


def prove_absence(
    subject_id: str,
    interval: Interval,
    predicate: Predicate,
    coverage_log: Sequence[Coverage],
) -> AbsenceResult:
    """Attempt to prove ``predicate`` never held for ``subject_id`` in ``interval``.

    Requires unbroken ``live`` Coverage (see ``_ESTABLISHING_STATUSES``)
    spanning the entire interval, sourced only from records naming
    ``subject_id``, before it will evaluate ``predicate`` at all.

    Returns:
        :class:`Absence` if coverage is unbroken and ``predicate`` never
            held; :class:`CannotEstablish` in every other case, including
            when there is no Coverage at all for ``subject_id`` — an empty
            ``coverage_log`` can never yield a proven absence.
    """
    relevant = [c for c in coverage_log if c.subject_id == subject_id]

    if not relevant:
        return CannotEstablish(
            reason="no_coverage",
            uncovered_subintervals=(interval,),
            envelope_violations=(),
            gaps=(),
        )

    uncovered: list[Interval] = [interval]
    for cov in relevant:
        next_uncovered: list[Interval] = []
        for piece in uncovered:
            overlap = piece.intersection(cov.interval)
            if overlap is None:
                next_uncovered.append(piece)
                continue
            next_uncovered.extend(piece.subtract(cov.interval))
        uncovered = next_uncovered

    violating_statuses = [
        cov
        for cov in relevant
        if cov.status not in _ESTABLISHING_STATUSES and cov.interval.overlaps(interval)
    ]
    all_gaps: list[Gap] = []
    for cov in relevant:
        if not cov.interval.overlaps(interval):
            continue
        for gap in cov.gaps:
            if gap.interval.overlaps(interval):
                all_gaps.append(gap)

    if uncovered or violating_statuses or all_gaps:
        return CannotEstablish(
            reason=(
                "no_coverage"
                if uncovered and not violating_statuses and not all_gaps
                else "coverage_insufficient"
            ),
            uncovered_subintervals=tuple(uncovered),
            envelope_violations=tuple(
                f"{cov.subject_id}:{cov.status}:{cov.interval.start_ns}-{cov.interval.end_ns}"
                for cov in violating_statuses
            ),
            gaps=tuple(all_gaps),
        )

    if predicate(subject_id, interval):
        return CannotEstablish(
            reason="predicate_held",
            uncovered_subintervals=(),
            envelope_violations=(),
            gaps=(),
        )

    return Absence(
        subject_id=subject_id,
        interval=interval,
        predicate=predicate.__name__
        if hasattr(predicate, "__name__")
        else repr(predicate),
        coverage_basis=tuple(relevant),
    )
