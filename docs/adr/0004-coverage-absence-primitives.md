# ADR 0004 — Coverage and Absence as primitives

- **Status:** Accepted
- **Date:** 2026-08-06
- **Decides for:** every negative claim this system will ever make ("no
  one entered the loading dock between 02:00 and 04:00")
- **Supersedes:** nothing — no prior negative-query surface existed

## Context

A query product that returns "no results" for an empty match lets the
reader assume that means nothing happened. On this system that assumption
is false every time the camera in question was offline, occluded, or
degraded during the window asked about, and a plain event log cannot
distinguish "we looked and saw nothing" from "we never looked" — both
produce zero matching rows.

This is not a hypothetical failure mode for this product. Day 6/7 already
found `coverage.observable_fraction` at 0.4365 on the golden set — the
system's cameras spend more than half their frames off-sensor for the
agents in that fixture. A negative answer computed against a log with
that much silent gap is not evidence, it is an artifact of where the
cameras happened to be pointed, and shipping a product that cannot tell
the difference is a liability the moment it is asked "did anyone access
the safe last night" and answers "no" from a camera that was down for
three of those hours.

## Decision

`Coverage` (`src/model/coverage.py`) records what was actually watched:
subject (camera or zone), interval, status (`live | degraded | offline |
occluded`), the envelope it was measured against, and any `Gap` within
it, written at the moment a frame drops, backpressure hits, or the sensor
disconnects — never reconstructed after the fact from absence of data.

`prove_absence(subject, interval, predicate, coverage_log)` is the only
sanctioned path from a query to a negative claim, and its return type is
the union `Absence | CannotEstablish` — nothing else, no `None`, no bare
`False`. It requires unbroken `live` coverage spanning the entire queried
interval, sourced only from records naming the queried subject, before it
will even evaluate the predicate. Anything short of that — no coverage
record at all, coverage for the wrong subject, a `degraded`/`occluded`/
`offline` status anywhere in the interval, or a `Gap` inside an otherwise
`live` record — falls through to `CannotEstablish` with a structured
reason, the specific uncovered subintervals, the envelope violations, and
the gaps responsible.

An empty `coverage_log` is the base case this ADR is written to close:
`prove_absence` with no `Coverage` records at all for the subject returns
`CannotEstablish(reason="no_coverage", ...)`, never a proven `Absence`.
There is no code path in `src/model/coverage.py` that returns a bare
"nothing happened" — every branch through `prove_absence` terminates in
one of the two union members.

## Consequences

Every negative claim this system makes from Day 13 onward costs a
`Coverage` record to back it — there is no shortcut that infers coverage
from the presence or absence of events, because that is exactly the
conflation this ADR exists to prevent. Pipeline stages that did not
previously need to track their own uptime, drop rate, and degradation
state now do, or their zone's negative queries will refuse rather than
answer. That is the intended trade: a refusal that says why is strictly
better than a wrong "no", and this system's whole value proposition is
being trustworthy enough to answer "did this happen" for a security
review.

## Open questions

- `_ESTABLISHING_STATUSES` currently admits only `live` coverage as
  sufficient to prove absence. Whether `degraded` coverage should ever
  count (with a correspondingly weaker claim) is a product decision that
  needs its own measurement — deferred, not decided by default.
- `Coverage` is per-subject; no aggregation across multiple cameras
  covering overlapping zones is implemented. A zone watched redundantly
  by two degraded cameras might jointly prove absence even though neither
  alone can — out of scope for Day 13.
