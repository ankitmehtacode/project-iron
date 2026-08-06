# ADR 0003 — Event-class hierarchy over a boolean flag

- **Status:** Accepted
- **Date:** 2026-08-06
- **Decides for:** the shape of every fact this system records, from Day
  13 onward
- **Supersedes:** the `observed: bool` field on schema v1's `Event`
  (`src/events/schema.py`, unchanged and still readable; migrated by
  `scripts/migrate_events_v1_v2.py`)

## Context

Schema v1 has one epistemic bit: `observed`. `True` means seen; `False`
means "inferred", which in practice covered at least three different
situations that behave differently downstream:

1. A trajectory extrapolated across a dropped-frame gap — a claim about
   the past, derived from evidence either side of the blind spot.
2. A prediction that a subject will exit through a known door in the next
   ten seconds — a claim about the future, not yet true.
3. A dwell-time anomaly worth a human's attention but not confirmed by a
   second signal — a lead, not a claim.

All three set `observed=False`. All three therefore look identical to
every consumer that branches on the flag, and at least two of them must
never be treated the same way: alerting on (2) pages someone for a
forecast, and admitting (3) as evidence treats a guess as a fact. A
boolean cannot express "this claim is real and this shape of not-yet or
not-quite" — the shape has to live in comments and downstream `if` logic
that reruns the same judgment call at every consumption site, with no
guarantee two sites agree.

## Decision

Four frozen dataclasses, not a flag: `ObservedEvent`, `InferredEvent`,
`PredictedEvent`, `HypothesisEvent` (`src/model/events.py`). Shared fields
(subject, verb, object, zone, confidence, importance, evidence, state
refs, supersedes) live on a common base; each subclass adds only what
distinguishes it (`InferredEvent.basis`, `PredictedEvent.predicted_by`,
`HypothesisEvent.rationale`).

The two behavioural rules that motivated the split are enforced by
`functools.singledispatch`, not by an `if` on `event_class`:
`assemble_evidence` has no registered handler for `PredictedEvent`, and
`raise_alert` has none for `HypothesisEvent`. Both raise on the first
event of the excluded type they see — enforced by the absence of a
registration, which a future edit cannot accidentally invert the way it
can invert a conditional.

A prediction that comes true is never mutated into an observation:
`confirm_prediction()` returns a brand-new `ObservedEvent` with
`supersedes` pointing at the original `PredictedEvent`, which stays
frozen and independently retrievable.

The verb vocabulary is untouched. `Verb` and its GEOMETRIC/POSE/
INTERACTION groups stay in `src/events/schema.py`, versioned there; this
ADR is about how a claim's epistemic status is represented, not what
verbs exist.

## Consequences

Every consumer that used to branch on `observed` now branches on type —
`isinstance` or `match` — which mypy can check exhaustively. The four
classes have a mapping cost: v1 records only ever needed `observed=True`
or `observed=False`, and the migration script maps both onto exactly one
v2 class each (`ObservedEvent`, `InferredEvent`); v1 never represented
predictions or hypotheses at all, so nothing is lost, but nothing in v1
data will ever produce a `PredictedEvent` or `HypothesisEvent` either —
those only originate from v2-native code going forward.

## Open questions

- Should `InferredEvent.basis` and `PredictedEvent.predicted_by` be
  closed vocabularies (like `Verb`) rather than free text? Deferred until
  there are enough real basis strings in production to see whether they
  cluster.
- `HypothesisEvent` has no expiry field. A hypothesis that is never
  confirmed or refuted stays in the log forever; whether that needs a
  TTL or a "still open" query surface is a Day-14+ question.
