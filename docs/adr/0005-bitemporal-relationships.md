# ADR 0005 — Bitemporal relationships

- **Status:** Accepted
- **Date:** 2026-08-06
- **Decides for:** every typed link between two entities (identity
  resolution, containment, assignment) from Day 13 onward
- **Supersedes:** nothing — no prior relationship type existed

## Context

A badge swipe at 14:02 can resolve which enrolled identity an anonymous
track has been since 13:58. That sentence has two timestamps doing two
different jobs: 13:58 is when the fact became true in the world, 14:02 is
when the system learned it. A relationship type with a single timestamp
field can represent only one of them, and every retroactive-resolution
case this product needs to handle — identity resolution, track merges,
recalibration — is exactly the case where those two instants differ. Pick
13:58 and every downstream consumer sees the resolution as instantaneous,
losing the fact that scorecards and alerts computed between 13:58 and
14:02 were correct answers to the information available at the time. Pick
14:02 and the system asserts the identity became true at the moment it
was learned, which is false and undermines any later question of the form
"was this known at the time".

## Decision

`Relationship` (`src/model/relationship.py`) carries both axes as four
independently required constructor arguments, none defaulted:
`valid_from_ns` / `valid_to_ns` (when the fact was true — the valid-time
axis) and `asserted_at_ns` / `asserted_by` (when and how the system
learned it — the transaction-time axis). `valid_to_ns` is typed
`int | None` to allow an open-ended relationship, but even `None` must be
supplied explicitly; there is no default that lets a caller omit the
valid-time axis by accident. Supplying only one axis — the single-
timestamp relationship this ADR is written to prevent — is a `TypeError`
at the call site, not a validation failure discovered later.
`Relationship.is_retroactive` is a derived property (`asserted_at_ns >
valid_from_ns`) rather than a stored flag, so it cannot drift out of sync
with the two timestamps it is computed from.

Corrections compose with this: a `Correction` (`kind="identity_resolution"`)
records that a `Relationship` changed, and its required `invalidates`
list drives `ArtifactRegistry.apply_correction`'s transitive walk over
every derived artifact whose `input_closure` depended on the pre-
correction state — see the canonical test in
`tests/test_model_relationship.py`, which is the badge-swipe scenario
this ADR opens with, executed end to end.

## Consequences

Every relationship this system ever records costs two timestamps instead
of one, and every consumer that reads a relationship must decide which
axis it cares about — "what did we believe as of 13:59" is a different
query than "what do we believe now", and conflating them silently is
exactly what a single-timestamp design would have made easy to do by
accident. That cost buys the ability to answer both questions correctly,
which for a product whose output may end up in a security review is not
optional.

## Open questions

- No query surface yet answers "what was asserted as of transaction-time
  T" directly — `Relationship` records support it (filter on
  `asserted_at_ns <= T`), but no convenience function exists. Deferred to
  whichever Day first needs a time-travel query.
- `basis` (`asserted | observed | inferred`) is a flat enum on the
  relationship, independent of the four-class event split in ADR 0003.
  Whether a `Relationship` should instead be produced by (and typed
  against) one of the four event classes is an open design question, not
  decided here.
