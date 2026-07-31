---
name: iron-events
description: Event schema, scripting layer, and narrative-grounding law for project-iron's video monitoring product. MUST be consulted when writing or modifying anything that creates, stores, queries, summarizes, or displays events — the event compiler, verb detectors, importance scoring, NL search, LLM narration, alerts, timeline UI, or investigator answers. Triggers on: "event", "scripting", "narrative", "who did what", "alert", "search the log", "summarize footage", "add a verb", "importance", "LLM query". The rule that keeps this product admissible as evidence: the log is truth, language is rendering.
---

# Iron Events — The Log Is Truth, Language Is Rendering

This product's output may end up in an internal investigation or a security audit. One
hallucinated sentence presented as fact ends the product category for us. The architecture
below makes that structurally impossible.

## The Grounding Law

1. **Facts originate ONLY in typed event records** `(event_id, site_id, ts_ns, subject, verb,
   object, zone, confidence, observed, importance, clip_ref, manifest_sha, schema_version)`.
2. **LLMs render and query; they never originate.** Narration formats event records. NL search
   compiles to structured queries over the log; answers come from query results. If the LLM
   layer is down, the product still works — that is the test of correct layering.
3. **Every narrative sentence resolves to ≥1 event_id** (automated test enforces this — a
   narrative sentence with no backing event fails CI). Every event carries a `clip_ref` where
   the envelope allows; an event is evidence because you can watch it.

## Verb Law

- **Closed vocabulary, versioned.** v1 = GEOMETRIC {entered, exited, dwelled, approached,
  followed, loitered, ran, fell} + POSE {sat, stood, bent_down, reached, lying} + INTERACTION
  {picked_up, put_down, carried, handed_over, opened, closed}. Adding a verb = schema version
  bump + detection recipe + labeled eval data + published precision/recall floor BEFORE it may
  fire in production. ~30 verbs done reliably beats open vocabulary at 60%.
- INTERACTION verbs require an `object`; the compiler rejects incoherent tuples.
- Zone transitions get hysteresis; pose verbs get temporal smoothing. Event chatter at
  boundaries is a bug, not noise.
- Interaction evidence is a continuity chain (object stationary → hand-region intersect →
  object co-moves with track), stored with the event.

## The `observed` Flag Is Sacred

`observed=false` marks inferred facts (blind-spot trajectory predictions, retroactive identity
resolutions). Downstream MUST render inferred distinctly (dashed vs solid, "likely" language)
and MUST NOT let inferred events satisfy queries that ask what was *seen* unless the query
opts in. A system that renders predictions identically to observations fabricates evidence.

## Capability Envelopes Gate Events

Every camera carries an auto-computed envelope (re-ID valid to X m, interaction events to Y m,
per resolution). Events outside envelope are suppressed or confidence-flagged — never guessed.
`line_of_sight(subject, asset)` is the honest verb for "viewed"; `read` is unprovable and
therefore not a verb.

## Importance & Identity in Events

Importance = policy weight (customer zones/assets/hours config) × statistical surprise (site
priors) × identity factor (unknown person, low-confidence ID). Three lanes: alert / notable /
routine. Identity is always `(EntityRef, confidence)`; UI language is calibrated ("likely
Ankit, 0.93") — asserting certainty the posterior doesn't have is a defect. Corrections
(revocation, retroactive resolution) append correction records and propagate; the log itself
is append-only.
