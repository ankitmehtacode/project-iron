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

## Objective 4 — joint vs independent, evaluated: the motivating case is real, and it has a cost

`scripts/eval_joint_estimator.py`. No golden set carries a real carried-
object entity (Objective 2's finding), so a carried asset's GT is
synthesized: the carrier's own GT position plus a fixed, declared, RIGID
offset (0.25m to the side, 0.30m below the carrier's own reference point —
no synthetic slip added to GT itself, so this tests whether coupling
correctly exploits a genuinely rigid attachment). Trivial baseline (Day 12
rule): INDEPENDENT per-entity filtering — the carrier under config B (Day
25's adopted floor-enabled single-entity filter), the asset under the
`asset_carried` motion model that has existed, unused, since Day 20
(inflated CV process noise, no coupling).

### The motivating case passes cleanly

| set | asset RMSE: indep → joint | margin | asset coverage: indep → joint | verdict |
| --- | --- | ---: | --- | --- |
| v5-cessation (n=1021) | 0.2051m → 0.1626m | **+0.0425m** | 0.8570 → 0.8737 | PASS |
| v3-indoor (n=2736) | 0.1691m → 0.1426m | **+0.0265m** | 0.9635 → 0.9269 | PASS (small wiggle, within tolerance) |

Joint estimation beats independent filtering on the carried asset by a
real, consistent margin on both sets, and does not degrade the asset's
own calibration toward overconfidence on either. This is not a marginal
result — it is coupling doing exactly the thing it was built to do
(Objective 3's own tested claim: "if A moves, laptop 7's posterior must
move," now confirmed to also make the posterior MORE ACCURATE, not just
directionally correct).

### The carrier's own calibration is not neutral — it is measurably worse on v5-cessation specifically

| set | carrier RMSE: indep → joint | margin | carrier coverage: indep → joint | verdict |
| --- | --- | ---: | --- | --- |
| v5-cessation (n=1021) | 0.1888m → 0.1608m | **+0.0280m** | 0.9089 → 0.8874 | **FAIL_OVERCONFIDENT** (Δ −0.0215) |
| v3-indoor (n=2736) | 0.1552m → 0.1408m | **+0.0144m** | 0.9912 → 0.9675 | PASS (Δ +0.0238, calibration improves) |

**On both sets the carrier's RAW ACCURACY improves under joint estimation**
(more measurements per step lowers RMSE, as expected) — but the
CALIBRATION effect is set-dependent and, on v5-cessation, crosses the
directional criterion's own non-negotiable line: coverage moves from
already-near-nominal (0.9089) to measurably overconfident (0.8874), a
degradation the criterion correctly flags regardless of the accuracy
gain sitting right next to it. This is exactly the danger Objective 4's
own framing named in advance: "a joint solve that sharpens covariance
without justification is the exact danger the criterion now names."

**The effect is regime-concentrated, not uniform**, per-regime carrier
coverage on v5-cessation:

| regime | n | indep cov | joint cov | Δ toward nominal |
| --- | ---: | ---: | ---: | ---: |
| static | 260 | 0.8923 | 0.9154 | **+0.0067** (improves) |
| onset | 91 | 0.9121 | 0.8791 | −0.0209 |
| sustained | 282 | 0.9149 | 0.8865 | −0.0163 |
| cessation | 373 | 0.9276 | 0.8901 | **−0.0375** (worst) |

`static` is the only regime where coupling improves the carrier's
calibration; every regime with real carrier motion degrades it, and
`cessation` degrades hardest — nearly double the pooled effect, and in
the exact regime this entire multi-day investigation (Days 21-26) has
already found the estimator's calibration most fragile. The plausible
mechanism, stated as a hypothesis and not yet confirmed by a second
measurement (per this project's own discipline against acting on a
first read): during a regime where the carrier's true velocity is
changing, the coupled update lets the asset's own (independently noisy)
position observation contribute extra apparent confidence to the
carrier's state through the shared Kalman gain, exactly when the
carrier's motion is least predictable and the floor (Day 25) is doing
the most work to keep velocity uncertainty honest — an interaction this
session did not isolate further.

### No same-session fix, per this project's own discipline

The mechanism is hypothesized, not confirmed, and Days 21-25 have
repeatedly shown that tuning a parameter (here, candidates would be
`OFFSET_SLIP_SIGMA_MPS_SQRT_S` or excluding the carrier←asset direction
of the cross-covariance update) in the SAME session that found a
miscalibration reads as motivated regardless of whether the reasoning is
sound. This is recorded as Day 27's first item: confirm the mechanism
with a second measurement (e.g., sweep `OFFSET_SLIP_SIGMA_MPS_SQRT_S` and
check whether the carrier's cessation-regime overconfidence tracks it
monotonically, which would support the hypothesis above) BEFORE changing
anything.

### Day-10 validity gate and falsification test 5

Re-run, unaffected by today's work: `state_estimation` **PASSES** on both
v5-cessation and v3-indoor (unchanged — gating logic does not depend on
which filter scored the set). Falsification test 5 (behaviour-query
shape): re-run, still **PASSES/PARTIAL** exactly as Day 20 left it
(`tests/test_falsification.py`, 8/8 green) — joint estimation produces a
richer `StateEstimate`-adjacent output but still nothing that resolves to
an `ActivityMode` behaviour label, so the test's own PARTIAL verdict is
unchanged, not silently inherited.

### Status: capability validated, not yet a production recommendation

Joint estimation is a real, measured capability addition — the carried-
asset improvement is not marginal and holds on both sets. It is NOT
recommended for unconditional production use in its current form: the
carrier's own posterior takes a real, regime-concentrated calibration
cost on at least one golden set, and per this project's own no-trade
discipline, a capability that helps one entity while measurably degrading
another's calibration is a stated trade, not a clean win, until the
mechanism is understood. Status stays **Proposed** (not Accepted) pending
Day 27's confirmation measurement.

## Day 27, Objective 1 — the mechanism confirmed, and the naive fix trades the motivating case away

### The cross-covariance confounder does not apply

Checked before anything else, per instruction. Git history: the fix
(`JointStateEstimate.carried_position_cov_m2`) and the eval script that
first produced the carrier-overconfidence finding
(`scripts/eval_joint_estimator.py`) landed in the SAME commit
(`cd0f763`) — there is no earlier, buggy version of the eval script that
was ever run. More directly: the bug was isolated to the ASSET's
absolute-position covariance, `Cov(carrier_pos + offset)`, which
genuinely needs the cross term. The CARRIER's own marginal covariance
(`joint_estimate.cov_array()[:6, :6]`) is a direct sub-block of the joint
covariance matrix — a marginal of a joint Gaussian needs no summation,
so there was never a formula to get wrong there. **The confounder is
ruled out; the carrier finding stands on its own math.**

### The mechanism, instrumented directly (not reasoned about)

`scripts/eval_joint_estimator.py` now reports, per regime, the carrier's
covariance shrinkage from coupling (`1 - joint_pos_cov_trace /
independent_pos_cov_trace`) against the ACTUAL squared-error reduction
the coupling delivers — Day 25's rule applied to Day 26's own finding: a
closed-form prediction ("coupling should help") is a hypothesis; this is
the measurement.

**v5-cessation, constant slip model (Day 26's original):**

| regime | covariance shrinkage | actual error reduction | unjustified gain |
| --- | ---: | ---: | ---: |
| static | 40.2% | 47.6% | −7.3% (conservative) |
| onset | 32.0% | **−7.7%** (worse!) | **+39.7%** |
| sustained | 38.6% | 26.0% | +12.7% |
| cessation | 40.8% | 21.5% | **+19.3%** |

The mechanism is exactly as hypothesized: the constant slip model shrinks
the carrier's covariance by roughly the SAME 32-41% regardless of regime
— it does not know whether the carrier's motion is steady or changing.
The ACTUAL benefit the coupling delivers varies enormously by regime
(near-full justification in `static`, negative in `onset`). The
covariance shrinkage is a constant-rate side effect of adding a second
measurement stream per step; the error reduction is regime-dependent.
Where they diverge most (`onset`, `cessation`) is exactly where the
no-trade criterion flagged overconfidence.

### The derived fix: partially confirms the physics, and reveals a real trade

`OffsetSlipModel = "acceleration_scaled"` (`src/estimator/joint.py`):
effective slip sigma = `OFFSET_SLIP_SIGMA_MPS_SQRT_S * (|carrier
acceleration| / PERSON_SIGMA_A_MPS2)`, both already-declared constants,
carrier acceleration estimated causally from the two most recent carrier
velocity states already in the factor chain (no new state dimension, no
fitted parameter).

**v5-cessation, acceleration-scaled model:**

| regime | covariance shrinkage | actual error reduction | unjustified gain |
| --- | ---: | ---: | ---: |
| static | 13.4% | 18.2% | −4.8% |
| onset | 14.2% | **−8.1%** | **+22.3%** (still bad) |
| sustained | 11.1% | 11.5% | −0.4% |
| cessation | 12.4% | 11.4% | **+0.9%** (was +19.3%) |

**Overall carrier verdict on v5-cessation: FAIL_OVERCONFIDENT → PASS**
(coverage 0.9089→0.9109, Δ +0.0020). Cessation — the regime this entire
investigation exists to fix — has its unjustified gain reduced by more
than 20x. This is a real, physics-derived success: the fix works exactly
where it was targeted.

**But `onset` is NOT fixed** — the unjustified gain barely moves in
relative terms, because onset's ACTUAL error reduction stays negative
(joint estimation is genuinely worse there, not just less-confidently
better) — the acceleration-scaled model reduces the covariance's
overclaim but cannot make an intrinsically-poor coupling fit look good.

**And the asset's own benefit is destroyed — the trade the objective
warned about, found exactly as it predicted:**

| set | asset margin, constant model | asset margin, acceleration-scaled |
| --- | ---: | ---: |
| v5-cessation | **+0.0425m** | **−0.0051m** |
| v3-indoor | +0.0265m | **−0.0049m** |

On both golden sets the asset's RMSE margin flips from a clear win to
essentially zero or slightly negative. The mechanism: acceleration-scaled
slip noise collapses toward the Q regularization floor whenever the
carrier's estimated acceleration is near zero — which is MOST of a
walking track (pedestrian motion is mostly steady). A near-rigid offset
sounds correct (the true synthetic offset genuinely never changes), but
an almost-zero process noise also means the Kalman gain on subsequent
asset observations shrinks toward zero: the filter locks onto whatever
the (single, noisy) bootstrap measurement implied and stops meaningfully
averaging in later asset observations to refine it. The constant model's
uniformly-nonzero slip noise was, inadvertently, doing double duty — not
just modeling physical slip, but keeping the offset estimate ELASTIC
enough to keep refining from new measurements. The acceleration-scaled
model removes that side benefit along with the overclaimed confidence.

### Verdict: the derived model is a finding about the physics, not an adoptable fix

Per instruction: report this as a finding, not paper over it. The
acceleration-scaling hypothesis is CONFIRMED as the mechanism (cessation
overconfidence tracks estimated carrier acceleration almost exactly, and
scaling slip noise by it closes the gap by over 20x) — but the specific
derived form tested today optimizes carrier calibration at the direct
expense of the asset's own accuracy, which is the motivating case this
entire capability exists to deliver. **A change that protects the
carrier by destroying the coupling's benefit has traded the motivating
case away — not a fix, a different, worse tradeoff.** Neither model
(constant or acceleration-scaled, as derived today) is adopted as the
default. `offset_slip_model` stays available as an explicit opt-in
parameter (default `"constant"`, unchanged from Day 26) so the tradeoff
this section documents is not silently picked for a caller.

No further same-session tuning was attempted (Day 24's own rule: a
tuned parameter that succeeds is a fitted parameter wearing a physics
costume). The gap suggests the two effects (confidence calibration,
estimate elasticity) need to be decoupled — e.g. a slip model with an
acceleration-scaled component ADDED to a small constant floor, so the
offset never becomes too rigid to keep averaging, rather than one term
doing both jobs. This is recorded as Day 28's first item, to be derived
and tested with the same discipline, not assumed to work because the
reasoning sounds right.

## Day 27, Objective 2 — a hard cap with a specified degradation path

Day 26's own gate was a narrow pass, not a comfortable one, and gave no
reason to expect sparsity holds past 6 entities. Rather than relax the
proximity threshold (which would hide the problem, not solve it), the
solver now has a specified, provenance-recording behaviour for when a
component would exceed what was actually measured.

### The cap: 6, and why

`ComponentCapConfig` (`src/estimator/joint.py`), config-driven and
versioned (a `.sha` over its declared fields, same convention as
`ImmConfig.sha`), passed explicitly to `run_joint_filter` rather than a
module-global. `DEFAULT_COMPONENT_CAP_CONFIG`:
`max_component_size=6, degradation_action="independent_fallback"`.

**6 is not a round number** — it is the exact bound Day 26 Objective 2's
own decision gate used to authorize building any solver at all ("proceed
to Objective 3 only if p95 component size ≤ 6"). The solver was designed
against, and evaluated against
(`scripts/eval_joint_estimator.py`), data topping out at 6 entities.
Capping the solver's own operation at that SAME measured bound means it
never runs in a size regime nothing has validated it against.

**`independent_fallback`, not `split_weakest_coupling`:** the latter
would need a measured notion of relationship information strength to
rank factors by, and this project has only ever measured coupling
DENSITY (Day 26 Objective 2), never coupling INFORMATIVENESS — building
a ranking heuristic today would be exactly the unmeasured analytical
claim Day 25/26's rule warns against. Independent fallback regresses to
Day 20/25's single-entity filter: well-tested, well-understood, with a
known accuracy/calibration profile.

### STRUCTURAL: the overflow path cannot silently solve jointly

`DegradedComponentEstimate` is a distinct TYPE (not a flag on
`JointStateEstimate`) with two required fields, no default:
`degradation_action`, `cap_config_sha`. `resolve_joint_state`'s return
type is `StateEstimate | JointStateEstimate | DegradedComponentEstimate`
— a caller scoring a component's estimate must `isinstance`-branch on
this, so a capped result can never be silently treated as a genuine
joint solve. `DegradedComponentEstimate.__post_init__` also verifies
`entity_estimates` covers exactly `component.entity_ids` — no entity can
be silently dropped during degradation. Tested explicitly
(`tests/test_estimator_joint.py`): a size-4 component under a cap of 2
degrades and returns a `DegradedComponentEstimate` with every entity's
own independent estimate recoverable via `estimate_for`; the dataclass
fields are confirmed to have no default (`dataclasses.fields`); an
incomplete `entity_estimates` tuple raises at construction; a
desynchronized observation stream (one entity missing an observation at
a shared timestamp) raises rather than silently degrading partially;
degraded-path `graph_rev` reproducibility is re-tested against the same
pattern the joint and single-entity paths already use.

### Cost measured — and an important caveat about what this specific measurement shows

`scripts/measure_component_cap_cost.py`. No general person-to-person
proximity coupling is implemented (only carrier+carried), so there is no
genuinely-6-PERSON joint solve in this codebase to cap. To still measure
against real data, `v3-indoor`'s `crowded_6agents` scene (the one Day 26
found merges at 1.10m) supplies 6 real GT tracks, with one HONESTLY
LABELED "carrier" and the other five "carried" purely to exercise a
genuine 6-entity component using the topology that exists — not a claim
that five people are rigidly attached to a sixth.

| entity | uncapped RMSE | capped RMSE | cost (capped − uncapped) | uncapped cov | capped cov |
| --- | ---: | ---: | ---: | ---: | ---: |
| agent-0 (carrier) | 0.3822m | 0.1335m | **−0.2487m** | 0.1316 | 1.0000 |
| agent-1 | 0.2424m | 0.1637m | −0.0788m | 0.3684 | 0.9474 |
| agent-2 | 0.2363m | 0.1492m | −0.0871m | 0.3158 | 1.0000 |
| agent-3 | 0.2936m | 0.1182m | −0.1754m | 0.1316 | 0.9211 |
| agent-4 | 0.2242m | 0.1387m | −0.0855m | 0.2368 | 1.0000 |
| agent-5 | 0.3439m | 0.1546m | −0.1893m | 0.1316 | 1.0000 |

**The measured cost is negative on every entity — capping HELPS here,
substantially.** This is real and correctly measured, but it is NOT
evidence that joint estimation is generally worse, or that capping is
"free": it is a direct consequence of the honest-labeling caveat above.
These six people are genuinely independent walkers with no real rigid
relationship; forcing them into one rigid-coupling-plus-slip component
is a badly mismatched physical model, so the joint solve actively hurts,
and falling back to independent filtering is strictly better. **This
measures the cap's safety value in the opposite failure mode from the
one the objective asked about: protection against inappropriately
coupling unrelated entities, not the accuracy given up when a genuinely
well-matched component gets capped.** Day 26/27's own carrier+actual-
asset evaluation (a physically appropriate 2-entity component) already
answers the latter question directly: joint beat independent by
+0.0425m/+0.0265m RMSE on the two golden sets. If a genuinely
well-coupled 6-entity component existed and were capped, the plausible
cost would be foregoing an improvement of that order — not the ~0.15m
"improvement" this specific test shows, which is really a demonstration
of what happens when component MEMBERSHIP is wrong, a concern this
project has never had the data to settle (Day 26 Objective 2's own
finding: proximity density was measured, not whether nearby entities are
actually coupled).

**What this leaves genuinely open:** whether a real 6-entity component
that IS actually coupled (e.g. a family/group moving together, several
items carried by one person) would show a positive cost when capped.
Answering that needs either real multi-entity carried-object data or a
larger authored synthetic scene with a genuine group relationship — both
already on the Day 26 punch list, unresolved by today's work.
