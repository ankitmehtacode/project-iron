# ADR 0006 — Episode with roled participants, not a separate Interaction primitive

- **Status:** Accepted
- **Date:** 2026-08-06
- **Decides for:** how segmented spans of activity (an episode, an
  interaction) are represented
- **Supersedes:** an earlier draft of IRON_DATA_MODEL that proposed
  `Interaction` as its own type

## Context

The architecture-review drafts leading into Day 13 initially split
"episode" (a segmented span with participants) and "interaction" (an
episode where two or more people did something to each other) into two
types. The reasoning at the time was that an interaction feels like a
qualitatively different thing worth its own name and its own schema.

Working through what `Interaction` would actually need turned up nothing
it required that `Episode` did not already have: the same time interval,
the same evidence references, the same `boundary_confidence` (an
interaction's boundaries are exactly as much a segmentation hypothesis as
any other episode's), and the same manifest provenance. The only thing
that distinguished "interaction" from "episode" was a property of the
participant list — at least two participants in a role other than
`object`. That is a predicate over `Episode.participants`, not a
different shape of data.

## Decision

There is one type, `Episode` (`src/model/episode.py`), with participants
carrying roles (`primary | counterpart | object | bystander | container`).
`Episode.is_interaction` is a derived `@property`
(`len([p for p in participants if p.role != "object"]) >= 2`), not a
class. No `Interaction` type exists in the module —
`tests/test_model_episode.py::test_is_interaction_derived_not_a_separate_type`
asserts this directly (`hasattr(episode_module, "Interaction")` is
`False`) as a regression guard against the type being reintroduced later
under a different justification.

This is a specific instance of a general rule this project applies
elsewhere: **prefer a derived predicate over a parallel type when the
only difference is a property of the data, not a difference in what
fields are needed to describe it.** The same reasoning is why
`PredictedEvent`-comes-true does not get its own "ConfirmedPrediction"
type (ADR 0003) — it becomes a plain `ObservedEvent` with `supersedes`
set — and why `Relationship.is_retroactive` (ADR 0005) is a property, not
a stored enum value.

The boundary is not absolute: `ObservedEvent` / `InferredEvent` /
`PredictedEvent` / `HypothesisEvent` in ADR 0003 *are* four separate
types, deliberately, because they differ in what a consumer is permitted
to do with them (evidence admission, alert eligibility) — a difference
enforceable only at the type level via `singledispatch`. `Interaction`
had no such enforcement need: nothing about being an interaction changes
what operations are valid on the record, only what a query might filter
for.

## Consequences

A consumer that wants "all interactions in the last hour" filters
`Episode` records by `.is_interaction` rather than querying a distinct
table or type. This is one predicate cheaper to maintain than a second
schema, at the cost of that predicate having to be recomputed (or
indexed) rather than stored — an acceptable trade at this data volume,
revisited only if interaction-filtering becomes a hot path that needs its
own index.

## Open questions

- If a future participant role needs interaction-like semantics that
  `Episode` genuinely cannot express (e.g. a role that changes what
  fields are required), that is the signal to split — this ADR is not a
  permanent ban on new types, only a record of why this specific split
  was rejected for the reason it was proposed.
