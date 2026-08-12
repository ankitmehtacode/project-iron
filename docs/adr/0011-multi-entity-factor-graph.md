# ADR 0011 — Multi-entity factor graph: component-based joint inference

- **Status:** Proposed (Objective 2 measurement complete; Objectives 3-4
  conditional on it, recorded below as they land)
- **Date:** 2026-08-13 (Day 26)
- **Decides for:** whether and how the single-entity filter (Day 20, ADR
  0010) extends to joint estimation across coupled entities, and what
  "coupled" means operationally
- **Supersedes:** nothing directly — fills the gap `src/model/episode.py`
  has left explicitly open since Day 13 ("the multi-entity factor graph...
  still out of scope")
- **Related:** [[iron-data-model-day13]]; ADR 0010 (the single-entity
  filter this extends); `src/estimator/motion_model.py`'s
  `carrier_entity_id` interface (built Day 20, unused until this ADR);
  the closed-form-vs-instrumented rule (`.claude/skills/iron-eval-discipline/SKILL.md`,
  added Day 26 Objective 1) — this ADR is that rule's first pre-emptive
  application, not just its worked example.

## Context

Multi-entity joint estimation — solving several coupled entities' states
together rather than independently — has been deferred since Day 13 as
explicitly out of scope, and `asset_carried`'s `carrier_entity_id` field
(Day 20) exists as an unused interface hook for exactly this work. Going
into Day 26, the design was recalled as resting on an analytical claim,
paraphrased as living in "Data model v0.3 §3": that entity coupling is
sparse — components stay small (informally, "typically 1-6 entities") —
because entities only couple through active relationships (proximity,
carried-object, shared-zone).

**Grep-verified before any measurement code was written: no such
document, section, or sentence exists anywhere in this repository**
(`docs/`, `src/model/`, `FOUNDATION_REPORT.md` all checked). This is a
stronger finding than "the claim was unmeasured" — the claim was
unwritten. `src/model/relationship.py`'s `Relationship.predicate` is a
free string with only illustrative examples (`"same_identity_as"`,
`"contains"`, `"assigned_to"`); no closed vocabulary for proximity,
carried-object, or shared-zone coupling exists in the schema, and no
golden set tags any of the three as ground truth. This ADR treats the
sparsity claim as the implicit assumption underlying `episode.py`'s and
`motion_model.py`'s deferred-work language, not as a cited design
decision — and measures it before code is built on top of it, per the
closed-form-vs-instrumented rule this same day's Objective 1 added to
`iron-eval-discipline`.

## Objective 2 — component-size distribution, measured

`scripts/measure_component_sparsity.py`. Of the three named relationship
types:

- **Proximity within a declared metric threshold** — MEASURABLE directly
  from GT `agent_xyz` (every golden set carries exact synthetic position
  per agent, per frame).
- **Carried-object coupling** — NOT MEASURABLE on any golden set this
  project owns. No carried-asset entity exists in any clip's GT or
  manifest; `BAG_CARRIED` is a clip-level `Condition` tag (scene
  metadata), not a per-frame entity relationship.
- **Shared-zone occupancy** — NOT MEASURABLE. No golden set tags
  per-agent zone/room membership; the synthetic generator records one
  fixed `room_size` per scene, not a subdivided zone graph.

So the entire measurable signal is proximity-based connectivity, on the
one golden set with multi-agent clips at all: v3-indoor (agent counts
1/2/3/6 across 30 clips: 1×1, 22×2, 5×3, 2×6). v4.1-gate (0-1 agents/clip)
and v5-cessation (always exactly 1 agent/clip) are component-size 1
everywhere by construction — not because sparsity held, but because there
is nothing in their GT to couple.

### Component-size distribution and threshold sensitivity, v3-indoor

Per-frame MAX component size (connected components under a proximity
threshold, pooled across all 30 clips, n=1200 frames):

| threshold (m) | p50 | p95 | max | frac. frames merged |
| ---: | ---: | ---: | ---: | ---: |
| 0.30 | 1 | 1 | 2 | 1.3% |
| 0.75 | 1 | 1 | 2 | 4.7% |
| 1.00 | 1 | 2 | 2 | 8.0% |
| 1.05 | 1 | 2 | 3 | 21.3% |
| **1.10** | 1 | 2 | **6** | 28.0% |
| 1.20 | 1 | 2 | 6 | 35.7% |
| 1.50 (declared default) | 2 | 3 | 6 | 55.3% |
| 2.00 | 2 | 6 | 6 | 81.4% |
| 3.00+ | 2 | 6 | 6 | 91-97% (plateau) |

**The knee: 1.05m → 1.10m.** The pooled max component size jumps from 3
to 6 in one sweep step — a single scene (`crowded_6agents`, the only
6-agent clip pair) fully merging from several small pairs into one
6-entity block. Per-clip breakdown pinpoints it precisely:
`crowded_6agents__cam_a`/`__cam_b` reach a FIRST pairwise merge at just
0.30m (the smallest threshold tested — two of the six agents pass within
touching distance at some frame, an ordinary crossing) but do not reach
FULL merge (all six in one component) until **1.10m**. Every other
clip (2-3 agents) merges at similarly low thresholds (0.75-1.75m) but
never exceeds its own agent count, because there is nothing larger to
merge with.

**Component lifetime, declared default threshold (1.5m):** merges are not
transient blips once they form — p50 = 21 frames, p95 = 30.2, max = 31
(clips run 40 frames total, so a merge typically persists most or all of
a scene once it starts). This is the opposite of "cheap": a persistent
merge holds a joint-solve block open for the scene's full duration, not a
few frames of incidental crossing.

### Decision gate: PASS, narrowly — proceed to Objective 3

At the declared default threshold (1.5m, proxemics' "close social"
boundary — the natural physical proxy for "close enough to plausibly
interact or hand off an object"), pooled p95 = 3 and the observed maximum
is 6 — at the gate's own stated bound (`p95 component size ≤ 6`), not
comfortably below it. **This is a narrow pass, not a comfortable one, and
the "6" ceiling is itself an artifact of this dataset's largest authored
scene having exactly six agents — not evidence that six is where real
crowding would stop.** Proceeding to Objective 3 on this basis, with that
caveat carried forward explicitly rather than smoothed into "sparsity
holds."

### Honest limitation, stated plainly

These are 30 authored synthetic clips with 1-6 agents, only two of which
reach 6. **The crowded-lobby case that would actually break sparsity —
dozens of people, not six — cannot be measured on data this project
owns.** What this measurement DOES show: even this project's own small
"crowded" scene fully merges at a proximity threshold (1.10m) well inside
ordinary personal/social space, which is itself informative — sparsity is
not structurally guaranteed by "people happen to be far apart" in an
indoor scene; it depends on keeping the coupling threshold tight (e.g.
≤1.0m, appropriate for literal carried-object coupling, where sparsity
held cleanly: max component size 2 at every threshold below ~1.0m) or on
scenes staying genuinely uncrowded. What remains unmeasured and would
settle it: whether real deployment scenes (an office floor, a lobby, a
meeting room at capacity) show p95 component sizes climbing well past 6
at a physically-realistic coupling threshold — this needs either a larger
authored synthetic scene (20-50+ agents) or real capture data, neither of
which exists in this project's golden sets today (real camera capture is
still blocked on procurement — see [[iron-blocked-on-humans]]).

## Objective 3 — design, conditional on the gate above: landed

`src/estimator/joint.py`. Components as the unit of inference: a
`Component` is one carrier plus zero or more entities it carries,
declared statically by the caller — Objective 2's proximity graph was a
MEASUREMENT to sanity-check before building anything, not the grouping
key the filter itself uses; the actual coupling factor implemented today
is `carrier_entity_id` specifically (proximity alone creates no shared
kinematic factor between two unrelated nearby entities).

### Rigid coupling plus slip — the motivating case

A carried entity's own kinematics are not tracked; its state is a 3D
offset from its carrier (`carried_absolute_position = carrier_position +
offset`), nearly constant between updates ("rigid") with a small
process-noise density representing shift-while-held ("slip") —
exactly the design `motion_model.py`'s `asset_carried` docstring named
since Day 20 and left unimplemented pending this day. State layout: `[6
carrier dims (pos, vel)] + [3 dims per carried entity (offset)]` — 6+3k
for k carried entities. F: carrier's own motion-model F, identity on
every offset block. Q: carrier's own motion-model Q, `slip_sigma² * dt *
I3` on every offset block (declared 0.05 m/√s, same order as IMM's
`static`-mode sway noise). H: position-only, picking `carrier_pos` for a
carrier observation or `carrier_pos + offset_i` for a carried-entity
observation (linear, no EKF needed). Joseph-form update, generalized
dimension, applied sequentially for however many entities are observed
at one timestep.

**Tested directly, not just claimed:** bootstrap a carrier+carried
component, feed ONLY carrier observations for several further steps (no
carried-entity observation at all), and confirm the carried entity's
implied absolute position moves with the carrier
(`test_carried_entity_posterior_moves_with_carrier_without_further_observation`).
This is the entire product value of coupling — the carrier's motion
informs the carried entity's estimate between the object's own, possibly
sparse, detections — and it falls out of the predict step automatically
once the state layout above is correct; no special-cased "propagate the
carried entity" logic was needed.

### Size-1 delegates to Day 20/25's own filter, verbatim

A component with no carried entities is not joint at all:
`run_joint_filter` calls
`src.estimator.filter.run_single_entity_filter` directly for that case —
not a reimplementation that happens to agree, the actual function call —
so it reproduces Day 20/25's single-entity behaviour (mean, cov, and
config B's adopted velocity floor) bit-for-bit BY CONSTRUCTION, tested
directly against a shared observation sequence
(`test_size_one_component_reproduces_single_entity_filter_exactly`).

### NEES dof, tested at sizes 1/2/3

`consistency.compute_nees` already infers `dof = error.shape[0]` — no
joint-specific NEES function was needed, only correctly-sized joint
error/cov arrays. Tested directly (`test_nees_dof_matches_component_size`,
parametrized): size-1 reports dof=6 (identical to Day 25), size-2
(1 carried entity) reports dof=9, size-3 (2 carried entities) reports
dof=12 — confirming "getting this right" was about the state
construction, not the metric.

### STRUCTURAL, re-tested against the single-entity precedent

- **Prior firewall (§17):** `run_joint_filter`/`resolve_joint_state`
  signatures inspected directly (`inspect.signature`), no
  prior-shaped parameter — same test pattern as
  `run_single_entity_filter`'s own.
- **Consistency residuals required:** `JointStateEstimate` independently
  re-checks the non-empty-residuals-with-an-nis-entry rule at
  construction (not inherited from `StateEstimate` — a genuinely separate
  type, since the joint state's dimension is variable and
  `StateEstimate.mean` is hard-validated to exactly `STATE_DIM=6`).
- **graph_rev reproducibility:** re-solving a joint component at an
  earlier revision after MORE factors (a second, unrelated component) are
  appended to the same graph reproduces bit-identically — tested directly,
  mirroring `test_estimator_filter.py`'s own reproducibility test.

### Not implemented today — a real scope limit, not a discovered gap

Bootstrap requires every declared component member to be observed at the
component's FIRST timestamp; a carried entity discovered mid-track
(dynamic membership) is out of scope. Two skeletons name what would close
this, both `NotImplementedError` with a docstring, not a silent
placeholder: `resolve_data_association` (hypothesis management — which
carrier a carried entity belongs to, when that's genuinely ambiguous) and
`HybridDiscreteContinuousState` (a discrete "picked up"/"set down" mode
that would let component membership change without restarting the
filter). `resolve_joint_state` also raises `NotImplementedError` for
`horizon_kind="smoothed"`, matching `StateQuery`'s own documented
convention. None of these were needed for today's evaluation
(Objective 4), which builds and tears down one component's full
membership per synthetic track.

### One convention carried forward, not enforced by code

A `StateGraph` is used per-component, exactly as the single-entity
precedent uses one per track (`run_single_entity_filter`'s own docstring
already noted "nothing in this Day-20 scope exercises" sharing a graph
across unrelated entities). `resolve_joint_state` does not filter
factors by component identity — it relies on `graph_rev` pinning the
same way `resolve_state` always has. Mixing two components' factors into
one graph and querying at the LATEST revision could pick up the wrong
component's estimate; this was not fixed today because nothing in this
project's actual call sites does that (today's evaluation harness builds
a fresh graph per component, matching `scripts/eval_estimator.py`'s own
per-track pattern). Flagged here so a future multi-component orchestrator
does not assume this guarantee exists.
