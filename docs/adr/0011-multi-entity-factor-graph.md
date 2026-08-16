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

## Day 28, Objective 1 — the two-term model: derivation well-determined, acceptance criterion not met

### The derivation, and why the trap named in the prompt does not apply here

`OffsetSlipModel = "two_term"` (`src/estimator/joint.py`): `"constant"`
PLUS `"acceleration_scaled"`, combined as VARIANCES — the physically
correct combination for two independent noise sources, whose variances
add while their sigmas do not:

    offset_variance = OFFSET_SLIP_SIGMA_MPS_SQRT_S^2
                     + (OFFSET_SLIP_SIGMA_MPS_SQRT_S * |a_hat|/PERSON_SIGMA_A_MPS2)^2

The constant term is baseline grip compliance under steady carry (Day
26's own physical reading, unchanged). The acceleration term is slip
induced by a change in motion (Day 27's own physical reading,
unchanged). Both terms reuse the SAME already-declared constants —
`OFFSET_SLIP_SIGMA_MPS_SQRT_S` and `PERSON_SIGMA_A_MPS2` — and no third,
separately-tuned parameter was introduced to weight one term against the
other. The instantaneous ratio between what each term contributes is
therefore not free: it falls out of the acceleration ratio
`|a_hat|/PERSON_SIGMA_A_MPS2` already established Day 27 — exactly 1 at
the nominal acceleration bound, below 1 in steady motion (floor
dominates), above 1 in a sharp transient (acceleration term dominates).
No sweep was run to find this ratio; it was not adjustable in the first
place. **The derivation is not underdetermined** — the trap the Day 28
prompt named (a fitted ratio wearing a physics costume) does not apply
to this specific failure, which is a different one, below.

### Three-way comparison, v5-cessation (n=1021 carrier, n=1021 asset)

| model | carrier margin | carrier no-trade | asset margin | asset no-trade |
| --- | ---: | --- | ---: | --- |
| constant (Day 26) | +0.0280m | **FAIL_OVERCONFIDENT** (Δ −0.0215) | +0.0425m | PASS (Δ +0.0167) |
| acceleration_scaled (Day 27) | +0.0115m | PASS (Δ +0.0020) | −0.0051m | PASS (Δ +0.0137) |
| two_term (Day 28) | +0.0115m | PASS (Δ +0.0020) | **−0.0052m** | PASS (Δ +0.0137) |

Cessation-regime unjustified gain, the number this whole investigation
exists to fix:

| model | static | onset | sustained | cessation |
| --- | ---: | ---: | ---: | ---: |
| constant | −7.3% | +39.7% ⚠ | +12.7% ⚠ | +19.3% ⚠ |
| acceleration_scaled | −4.8% | +22.3% ⚠ | −0.4% | +0.9% |
| two_term | −4.8% | +22.1% ⚠ | −0.5% | +0.9% |

### Three-way comparison, v3-indoor (n=2736 carrier, n=2736 asset)

| model | carrier margin | carrier no-trade | asset margin | asset no-trade |
| --- | ---: | --- | ---: | --- |
| constant (Day 26) | +0.0144m | PASS (Δ +0.0238) | +0.0265m | PASS (Δ −0.0096) |
| acceleration_scaled (Day 27) | +0.0066m | PASS (Δ +0.0000) | −0.0049m | PASS (Δ +0.0134) |
| two_term (Day 28) | +0.0066m | PASS (Δ +0.0000) | **−0.0050m** | PASS (Δ +0.0134) |

### two_term is numerically almost indistinguishable from acceleration_scaled alone

Both golden sets, every regime, every reported statistic: two_term
lands within 0.0001–0.0002 of acceleration_scaled and nowhere near
constant's numbers. Most tellingly, the ASSET's `static`-regime margin —
the regime where the floor term should matter most, since a held object
is (by GT construction) not accelerating and the acceleration term
should collapse toward zero, leaving the floor to do its Day-26 job —
does not recover:

| model | asset margin, v5-cessation `static` | asset margin, v3-indoor `static` |
| --- | ---: | ---: |
| constant | **+0.0581m** | +0.0221m |
| acceleration_scaled | +0.0001m | −0.0037m |
| two_term | **−0.0000m** | **−0.0038m** |

If the floor term were doing the protective work the derivation
predicts, two_term's `static` margin should sit near constant's. It
sits at acceleration_scaled's instead, on both sets.

### Root cause, measured directly: the acceleration ESTIMATE, not the model, is the problem

Instrumented `|a_hat|` (the causal, one-step-lagged carrier acceleration
estimate that feeds the acceleration term) directly against
`PERSON_SIGMA_A_MPS2` frame-by-frame on a v5-cessation track:

| step | GT regime | \|a_hat\| (m/s²) | ratio to PERSON_SIGMA_A_MPS2 |
| ---: | --- | ---: | ---: |
| 6 | sustained | 7.32 | 4.9x |
| 10 | sustained | 8.81 | 5.9x |
| 20 | sustained | 9.75 | 6.5x |
| 28 | sustained | 2.65 | 1.8x |
| 49 | static | 8.82 | 5.9x |

Every value shown is from a regime GT labels as steady (no motion
change) — the exact regime the derivation depends on `|a_hat|` reading
LOW so the floor can dominate. Instead the causal acceleration estimate
sits 2–7x the reference constant throughout, never once dropping near
zero. The mechanism: `a_hat` is built from the DIFFERENCE of two
consecutive Kalman-filtered velocity ESTIMATES divided by `dt_s ≈
0.083s` (12 fps) — a division that amplifies whatever estimation noise
sits in each velocity state, and that amplified noise floor turns out
to be several times `PERSON_SIGMA_A_MPS2` regardless of the carrier's
TRUE acceleration. Given

    acceleration_variance = floor_variance * ratio^2

a ratio of 5–7x (typical, per the table above, not an outlier) puts the
acceleration term at 25–49x the floor's variance — the floor is
mathematically present in every step's sum but numerically negligible
in all but a handful of them, which is exactly why two_term tracks
acceleration_scaled instead of interpolating toward constant during
steady regimes.

**This is a measuring-apparatus finding, in the same family as Day 24's
and Day 25's — but the apparatus at fault is not a test or a metric,
it is the state estimator's own causal acceleration ESTIMATE, which
this project had not previously used as an INPUT to another model's
noise term and had therefore never characterized as a noise source in
its own right.** `PERSON_SIGMA_A_MPS2 = 1.5` m/s² is a bound on
plausible TRUE human acceleration; it was never validated as a bound on
the NOISE FLOOR of a finite-difference estimate of that acceleration at
12 fps from filtered velocity states. Those are different quantities,
and the two-term model's derivation silently assumed they were
interchangeable.

### Acceptance criterion, evaluated explicitly — NOT MET

Stated in advance: *"the carrier's cessation overconfidence must resolve
AND the asset's RMSE margin over independent filtering must survive.
Both, or the model is reported as another measured trade and not
adopted."*

- Carrier cessation overconfidence resolves: **YES** (unjustified gain
  +19.3% → +0.9%, matching acceleration_scaled almost exactly).
- Asset RMSE margin survives: **NO** (+0.0425m → −0.0052m on
  v5-cessation, +0.0265m → −0.0050m on v3-indoor — the same collapse
  Day 27 measured for acceleration_scaled alone, to within 0.0001m).

Both were required. One failed. **two_term is not adopted.** Per the
acceptance criterion's own wording, this is reported as another
measured trade, the same disposition as Day 27's acceleration_scaled.
`offset_slip_model` default stays `"constant"`, unchanged since Day 26;
`"two_term"` is available as an explicit opt-in alongside the other two,
carrying this section's caveat.

### What today adds to Day 27's open question

Day 27 closed by suggesting a floor term would "decouple" confidence
calibration from estimate elasticity. Today's measurement shows that
suggestion was half right: the floor term as derived is real, is
correctly combined, and would decouple the two effects PROVIDED its
input — the acceleration estimate — reads near zero during steady
motion. It does not, at this frame rate, with this estimator. The
decoupling Day 27 predicted requires either (a) an acceleration
estimate with a noise floor genuinely below `PERSON_SIGMA_A_MPS2`
during steady motion (e.g. smoothed over more than two consecutive
velocity states, at the cost of more lag), or (b) re-deriving
`PERSON_SIGMA_A_MPS2` itself — or a separate reference constant — from
the ESTIMATOR's own measured noise floor rather than from a physical
bound on true human acceleration. Neither was attempted today: doing so
under the "must resolve, not merely improve" pressure of this
objective's own acceptance criterion is exactly the condition under
which a fitted-looking number stops being distinguishable from a
derived one. Recorded for Day 29, not attempted same-session.

## Day 29 — the slip question closed: NO, at this frame rate and sensor quality

**Framing.** Day 28's root cause was not that the two-term model's physics
was wrong — it was that the physical bound (`PERSON_SIGMA_A_MPS2`, a bound
on plausible TRUE human acceleration) was being compared against a
quantity that had never been characterized as a NOISE SOURCE in its own
right: the causal, finite-difference acceleration estimate `|a_hat|`. The
general rule this closes with (also added to
`.claude/skills/iron-eval-discipline/SKILL.md`, Day 29): **before deriving
a model from a measured quantity, characterize that quantity's noise floor
against the scale the model depends on.** This is the fourth attempt at
slip (Day 26 constant, Day 27 acceleration-scaled, Day 28 two-term, Day 29
closing measurement) and Objective 2 existed to close the question, not
open a fifth.

### Objective 1 — the noise floor, measured over the full population

`scripts/measure_acceleration_noise_floor.py`. Day 28's finding (`|a_hat|`
2-7x `PERSON_SIGMA_A_MPS2` on a five-point spot check of one v5-cessation
track) is confirmed and sharpened over every scored frame of both golden
sets, at the one frame rate either set carries (12fps — no other frame
rate exists in this project's data; reported as a bounded null, not
extrapolated):

| set | steady-regime noise floor (std) | SNR vs `PERSON_SIGMA_A_MPS2` (1.5 m/s²) |
| --- | ---: | ---: |
| v5-cessation | 6.8607 m/s² | 0.2186 |
| v3-indoor | 7.3290 m/s² | 0.2047 |

Per-regime SNR, using each regime's OWN mean GT `\|accel\|` as the signal
(not a pooled median across onset/cessation/maneuver — an early version of
this measurement used a pooled median and it washed out to ~0.0, because
"cessation" per Day 23's own redefinition mixes genuine anticipatory
deceleration with post-stop recovery frames whose GT acceleration is
already back near zero; caught by inspecting the divergence between that
number and the per-regime means, not trusted on first read):

| regime | v5-cessation SNR | v3-indoor SNR |
| --- | ---: | ---: |
| onset | 0.0739 | 0.0000 |
| cessation | 0.3506 | (n=0 — v3-indoor carries no cessation frames) |
| maneuver | **1.9421** | (n=0) |

Only `maneuver` — sharp heading/speed jumps, `MANEUVER_SPEED_THRESHOLD_MPS_PER_FRAME`'s
own ~6 m/s² definition — clears SNR=1. `onset` and `cessation`, the
regimes this entire investigation exists to fix, do not, by a wide margin.

**Attribution, instrumented (three configs, identical observation draws):**

| config | steady-regime noise floor (std), v5-cessation / v3-indoor |
| --- | --- |
| baseline (real Q, real R) | 6.8607 / 7.3290 m/s² |
| `q_near_zero` (process noise ≈ 0) | 6.8439 / 7.3117 m/s² — **unchanged** |
| `r_near_zero` (measurement noise ≈ 0) | 0.7423 / 0.0211 m/s² — **9x/300x collapse** |

The noise floor is almost entirely MEASUREMENT noise propagated through
the filter, not process-noise inflation. This is what Objective 2's
candidate selection rests on.

**Closed-form cross-check, per the closed-form-vs-instrumented rule** (report
both, note divergence): the naive independent-sample formula
(`sqrt(2 * mean(Var(v)) / dt²)`) predicts 34.99 / 36.96 m/s² — roughly 5x
the instrumented number. The naive formula assumes consecutive filtered
velocity estimates are independent; they are not (a Kalman filter smooths
velocity across updates, so `v[t-1]` and `v[t-2]` are strongly positively
correlated, and their difference has far lower variance than independence
would predict). Reported as its own instance of this project's
closed-form-vs-instrumented finding family — the closed form was not
wrong about its own formula, it was wrong about what physical assumption
the formula silently required.

### Objective 2 — Candidate B tested, prediction confirmed, verdict NO

Given the R-dominated attribution, Candidate A (a Kalman-filtered,
model-based acceleration state) and Candidate B (an explicit windowed
average of the existing causal `a_hat` sequence) are, physically, the same
remedy: more temporal averaging to suppress measurement-noise-driven
error. A Kalman smoother's advantage over a boxcar average is a better
small-N constant and adaptive weighting, not a different asymptotic
noise-vs-window-length scaling once R dominates. **Only Candidate B was
tested** — it needs no state-model change (no new motion-model kind, no
touching `STATE_DIM=6` anywhere `src/estimator/joint.py`/`filter.py`
assume it), and its lag cost is directly, transparently measurable as a
window length. If B's lag cost disqualifies it against this project's own
already-declared `PEDESTRIAN_STOP_DURATION_S` bound, Candidate A would
need to beat the same underlying averaging physics to do meaningfully
better, which the ablation gives no reason to expect — so B's result
closes the question for both, consistent with today's own discipline
against a fifth attempt.

**Prediction, stated before running** (`scripts/measure_slip_remedy_smoothing.py`):
naive `1/sqrt(K)` noise scaling from Objective 1's pooled noise floor
predicts `SNR>=1` needs `K ≈ (7.1/1.5)² ≈ 22` frames (~1.83s at 12fps),
~1.83x `PEDESTRIAN_STOP_DURATION_S` (1.0s) — predicted, before running, to
be self-defeating for cessation: a slip model reacting ~1.8s after a stop
would still be averaging in pre-stop motion for most of the stop's own
~1s settling window.

**Measured** (causal boxcar average of the already-causal `a_hat`
sequence, Fibonacci-spaced window sweep):

| K (frames) | lag (s) | v5-cessation SNR | v3-indoor SNR |
| ---: | ---: | ---: | ---: |
| 1 | 0.083 | 0.2186 | 0.2047 |
| 3 | 0.250 | 0.4405 | 0.3918 |
| 8 | 0.667 | 0.4116 | 0.4702 |
| 13 | 1.083 | 0.6042 | 0.6448 |
| 21 | 1.750 | 0.8620 | 0.8468 |
| 34 | 2.833 | **1.4557** | **1.0544** |

SNR crosses 1.0 between K=21 and K=34 on both sets — roughly K≈25-30
frames, ~2.1-2.5s — close to, and if anything worse than, the
pre-registered prediction of ~1.83s. Even under the most charitable
reading (using cessation's own higher mean GT acceleration, 2.4057 m/s²
on v5-cessation, instead of the generic `PERSON_SIGMA_A_MPS2` reference),
crossing SNR=1 lands around K≈13-14 frames (~1.1-1.2s) — still at or just
past `PEDESTRIAN_STOP_DURATION_S`, not comfortably inside it. **The
prediction holds under every reading tried:** the lag needed to suppress
this estimator's noise floor to the scale the slip model needs is
comparable to or longer than the physical duration of the cessation event
itself.

### Verdict

**NO — acceleration-conditioned slip is not achievable at this frame rate
(12fps) and this sensor quality (this project's illustrative,
unmeasured-but-declared measurement envelope) with any remedy tested:
the causal acceleration estimate's noise floor is measurement-noise-
dominated, and the smoothing window needed to suppress it below the scale
the slip model depends on introduces a lag comparable to or exceeding the
~1s duration of the cessation event the model exists to react to.** This
retires a question that has consumed four days (Day 26-29). It is a good
outcome per this project's own stated framing: a negative answer with
evidence closes an open problem and lets the carrier/asset trade below be
made as a stated engineering decision with quantified cost, rather than
staying an unresolved question blocking `offset_slip_model`'s default.

`offset_slip_model` stays `"constant"` by default (unchanged since Day
26); `"acceleration_scaled"` and `"two_term"` remain available as explicit
opt-ins, each carrying its own day's measured trade, and no fourth
acceleration-conditioned variant will be derived against this same noise
floor — the floor itself, not any one derivation, is now the documented
reason.

### The carrier/asset trade, restated as a decision, not an open problem

With acceleration-conditioned slip closed, Day 26-27's original finding
stands as the production choice to make explicitly: joint estimation with
`"constant"` slip measurably helps the carried asset (RMSE margin
+0.0425m/+0.0265m, Day 26 Objective 4) at a measurable calibration cost to
the carrier specifically in the cessation regime (coverage 0.9089→0.8874,
`FAIL_OVERCONFIDENT`, Day 26). Two production postures are available,
and this project has not yet chosen between them:

- **Carrier calibration protects evidence integrity.** Ship joint
  estimation only where the carrier's own posterior calibration is not
  degraded (e.g. gated off in cessation-heavy scenes, or off by default
  with `offset_slip_model` opt-in), accepting that the carried-asset
  accuracy benefit is foregone in exactly the regime — cessation, a
  handoff or set-down moment — where a carried asset's position often
  matters most.
- **Asset accuracy protects the coupling's motivating case.** Ship joint
  estimation by default, accepting the carrier's measured cessation-regime
  overconfidence as a stated, bounded cost, on the grounds that the
  carried asset's own accuracy is coupling's entire reason to exist.

Both sides are now fully measured (Day 26 Objective 4, Day 27 Objective 1,
this section). **Which side to ship is a product decision, not an
engineering one, and should be made explicitly by whoever owns that
tradeoff — not left to whichever default this ADR happens to already
carry.** Recorded here as the punch-list item Day 26 originally deferred
("capability validated, not yet a production recommendation") now that the
one open technical question blocking that decision (could acceleration-
conditioning avoid the trade entirely) has a measured NO.

### Objective 3 — the seven stubbed predicates, filled

Day 28 typed seven constraints and left every predicate raising
`NotImplementedError`. All seven now run. Status, by name:

| constraint | kind | status |
| --- | --- | --- |
| `one_body_one_place` | hard | implemented Day 28 |
| `max_pedestrian_velocity` | hard | **implemented** — `speed <= PEDESTRIAN_MAX_SPEED_MPS` (12.5 m/s) |
| `mass_conservation` | hard | **implemented** — gap displacement reachable at that bound |
| `gravity_floor_transition` | hard | **implemented** — `a_z >= -g`, free-fall bound |
| `wall_impermeability` | twin_dependent | **`Unevaluable("no twin geometry")`** |
| `portal_required` | twin_dependent | **`Unevaluable("no twin geometry")`** |
| `stair_or_lift_for_floor_change` | twin_dependent | **`Unevaluable("no twin geometry")`** |
| `visibility_and_accessibility` | twin_dependent | **`Unevaluable("no twin geometry")`** |

**The third outcome, enforced by type.** `evaluate_constraint` no longer
returns `None`. Its outcome set is closed and three-valued per kind —
`Satisfied | Unevaluable | HardConstraintViolation` and
`Satisfied | Unevaluable | TwinRevisionHypothesis` — and the pruning path
still accepts only `HardConstraintViolation`. Removing `None` rather than
adding `Unevaluable` beside it is the substance of the fix: a caller
writing `if evaluate_constraint(...) is None` would have classified
`Unevaluable` as a violation, and `is not None` would have classified it
as one too; one of the two readings is wrong whichever way the caller
guesses, and neither looks wrong at the call site. With `None` gone there
is no `is None` idiom left to get backwards. `check_hard_constraints`
returns a `HardConstraintCheck` carrying the unevaluable set explicitly,
so "nothing objected" is no longer readable as "everything passed"; its
`match` is exhaustive, verified by removing a case and confirming `mypy`
errors, so a fourth outcome type cannot be silently skipped later.

**Two of Day 28's three "hard" constraints were described in
twin-dependent terms.** `gravity_floor_transition` was described as
requiring a "modeled transition (stairs, lift, ramp)" — word for word the
claim `stair_or_lift_for_floor_change` already makes as a *twin-dependent*
constraint; one physical claim filed under both kinds.
`mass_conservation` was described via "a modeled portal or occlusion
boundary" — also twin geometry. Both keep their names and both are
genuinely hard once reduced to their twin-free core (free-fall bound;
reachability bound), with the twin-dependent halves left where they
already were. The general tell: a hard constraint whose description names
the twin is mis-typed.

**A new constant was needed, and why that is a finding.** The objective
asked for thresholds cited from the pedestrian bounds the motion model
already declares. None could serve: **every pedestrian constant this
project declared before today is TYPICAL-scale, and a hard constraint
needs an IMPOSSIBILITY-scale bound.** `PERSON_SIGMA_A_MPS2` is a Gaussian
process-noise density — in the model it parameterizes, acceleration is
unbounded, so it bounds nothing — and `PERSON_SIGMA_A_MPS2 *
PEDESTRIAN_STOP_DURATION_S` = 1.5 m/s is, by its own docstring, a
comfortable walking pace. Used as a hard max-speed threshold it would
irreversibly prune anyone jogging. `PEDESTRIAN_MAX_SPEED_MPS = 12.5` is
declared instead, cited to human physiology (~12.4 m/s peak sprint), with
the reasoning recorded at the constant. A hard constraint's job is to be
never-wrong, not tight; it sits on an irreversible path, so its error
budget is one-sided, and discriminating power at the typical scale
belongs in the likelihood where being wrong is recoverable.

**GT violation count: 0**, across 9,507 constraint evaluations on both
golden sets (`scripts/measure_gt_constraint_violations.py`) — the result a
correctly-derived hard-constraint set requires. Four qualifications, all
measured rather than left as caveats, and each one a limit on how much
that zero establishes:

1. *The measurement does discriminate.* The rejected typical-scale
   threshold (1.5 m/s) would violate on **46.0% of v5-cessation GT frames
   (487/1059)** and 5.6% of v3-indoor. This corrects the reasoning that
   first justified `PEDESTRIAN_MAX_SPEED_MPS`, which argued the
   mis-derivation would *pass* the acceptance test because "the golden
   sets' walkers move at ~0.5 m/s" — a figure taken from the Day-21 report
   rather than measured against the sets as they stand. v5-cessation
   postdates that report: mean GT speed 1.24 m/s, peak 7.74 m/s. Asserting
   a property of the data from a stale document instead of measuring it is
   this day's own subject matter, committed while documenting it.
2. *`gravity_floor_transition` is untested by this data.* `agent_xyz`'s
   vertical component is a constant 0.86 m in every golden track (the
   generator places each agent at `height_m / 2.0` and never moves it
   vertically), so GT vertical acceleration is identically zero. Its zero
   is a bounded null, not a pass. `one_body_one_place` is likewise vacuous
   on v5-cessation (single-agent clips, 0 pairs evaluated); it is
   exercised on v3-indoor (2,680 pairs, closest approach 0.0185 m).
3. *The generator does produce unphysical motion, in a quantity no
   constraint bounds.* v5-cessation GT reaches **36.58 m/s² of horizontal
   acceleration — 3.7g, 24x `PERSON_SIGMA_A_MPS2` — with 3.4% of frames
   above 1g.** A human on foot cannot decelerate at 3.7g. Not fixed by
   adding a fifth constraint today: the bound that would catch it needs
   the same impossibility-vs-typical derivation above, and inventing it
   inside an acceptance measurement is how a threshold gets fitted to the
   data it is meant to judge. Day-30 item.
4. *v3-indoor GT is exactly constant-velocity* — peak GT acceleration
   0.00 m/s². Nothing in that set can test an acceleration-conditioned
   model in either direction, which is the concrete reason its per-regime
   SNR column above reads 0.0000 for `onset` and n=0 elsewhere.

**These do not weaken Objective 2's NO — they strengthen it.** The slip
verdict was measured against a GT whose cessation transients are *larger
than physically possible* (up to 24x the declared human acceleration
bound). A real pedestrian's cessation acceleration is smaller, so the true
signal is smaller, so the real-world SNR is *lower* than the 0.35 measured
for cessation. The question closes on a favorable-case measurement.

The first run of the GT measurement reported 16 `gravity_floor_transition`
violations at up to -36.58 m/s², 3.7x free fall. Every one was a defect in
the measuring script, which read `agent_xyz` index 2 as vertical, citing
`src/model/world.py`'s +z-up world frame — a real convention, just not the
one that array is in. Index 1 is vertical; z is camera-facing depth, and
an abrupt depth-axis speed change reads exactly like an impossible fall
when the axis is mislabelled. Verified against the generator
(`scripts/gen_synthetic_indoor.py:198`) rather than re-assumed. The
nonzero count was the instrument, not the subject — again — which is why
it was diagnosed rather than reported.

Status: **Accepted** for the joint-estimation capability and the
`offset_slip_model="constant"` default; the carrier/asset production
posture above is a separate, still-open decision, not part of this
acceptance. The constraint registry is **Accepted** for its four hard
constraints and the typed `Unevaluable` outcome; the four twin-dependent
constraints are **Blocked on twin geometry**, which is a real dependency
with a named interface (`TwinGeometry`), not an unimplemented predicate.

## Day 31 revision — every v5-cessation number re-run on v6-motion

**Nothing above is edited.** Day 30 established that v5-cessation's GT is
physically impossible (14 of its 22 stop events peak above 1g, median
1.54g, max 3.7g), which made every conclusion in this ADR provisional.
This section re-runs them on v6-motion and says which survive.

### 1. Joint vs independent — SURVIVES, on both sets

`scripts/eval_joint_estimator.py --version v6-motion`, 19 tracks, 1558
scored frames:

| quantity | independent | joint | margin |
| --- | ---: | ---: | ---: |
| carrier RMSE | 0.1258 m | 0.1005 m | **+0.0253 m** |
| carried-asset RMSE | 0.1323 m | 0.1028 m | **+0.0295 m** |

Positive on both entities, in the same direction and the same order of
magnitude as v5-cessation's (+0.0425 m on the carried asset). Coupling
helps, and it helps most where the ADR says it should. **Day 26's
headline is unaffected by the contamination.**

### 2. Carrier overconfidence — PARTIALLY SURVIVES, and the surviving half is not the half the slip models were built for

The signature is a covariance shrink larger than the error reduction it
is justified by. Per regime, `unjustified gain`:

| regime | v5-cessation | v6-motion |
| --- | ---: | ---: |
| static | −0.0735 | −0.0565 |
| onset | **+0.3974** | **+0.1281** |
| sustained | **+0.1269** | **+0.1367** |
| cessation | **+0.1925** | **−0.0646** |
| maneuver | **+0.5330** | (empty) |

`onset` and `sustained` keep the signature — smaller in `onset`, flat in
`sustained`. **`cessation` loses it entirely**, and so does the headline:
the carrier's own directional no-trade check moves from
**FAIL_OVERCONFIDENT** (delta −0.0215) on v5-cessation to **PASS**
(delta +0.0160) on v6-motion.

This matters because the cessation half is the one that motivated Days
26-28's three slip models. The carrier was measured as overconfident
exactly where the generator was producing 3.7g stops, and on physically
reachable motion it is not overconfident there at all. The `onset` and
`sustained` signatures are real, were never the stated motivation, and
are now the only live part of this finding.

**The state-carry caveat applies here and is load-bearing.** A per-regime
number is not an independent measurement: the joint filter carries
covariance across the regime boundary, so `onset`'s and `sustained`'s
signatures on v5-cessation were computed on frames whose filter state had
just been through an impossible transient. That they shrink (onset,
+0.3974 → +0.1281) rather than vanish is consistent with a real effect
that v5 was inflating. It is not proof of one.

### 3. The slip verdict — SURVIVES, and the a fortiori argument is now measured rather than argued

Day 29 closed the acceleration-conditioned slip question with a measured
NO, and defended it a fortiori: v5-cessation's transients are larger than
physically possible, so a real pedestrian's signal is smaller, so the
real-world SNR is *lower* than the one the NO was measured at. That was an
argument. v6-motion makes it a measurement.

`scripts/measure_acceleration_noise_floor.py`, both sets:

| quantity | v5-cessation | v6-motion |
| --- | ---: | ---: |
| noise floor (steady-regime std of `a_hat`) | 6.8607 m/s² | 7.7275 m/s² |
| pooled SNR vs `PERSON_SIGMA_A_MPS2` | 0.2186 | **0.1941** |
| `onset` SNR | 0.0739 | 0.0807 |
| `cessation` SNR | **0.3506** | **0.0458** |
| `maneuver` SNR | 1.9421 | (empty) |
| cessation-regime mean GT \|a\| | 2.4057 m/s² | 0.3543 m/s² |

**The direction holds.** Cessation's SNR falls **7.7x**, from 0.3506 to
0.0458, because the regime's mean true acceleration falls 6.8x while the
noise floor does not (it rises 12.6%). v5-cessation was the favourable
case, exactly as Day 29 claimed without being able to show it. A slip
model that could not be derived at SNR 0.35 cannot be derived at 0.046.

The mechanism is unchanged and re-confirmed by ablation on the new set:
near-zero R collapses the v6 floor 7.7275 → 0.4788 (16x), near-zero Q
leaves it at 7.7131 (unchanged). Measurement-noise dominated on both
sets, so every candidate remedy is still some form of temporal averaging,
and the window needed is still longer than the event.

**`offset_slip_model` stays `"constant"`. The question stays closed** —
now on a measurement of the physical case rather than an inference about
it.

One number does move in the unfavourable direction and is reported rather
than buried: the noise floor itself rose 6.8607 → 7.7275 m/s² (+12.6%).
That makes the NO stronger, not weaker — a higher floor against a smaller
signal is worse on both sides of the ratio.

### 4. The component-size cap and the sparsity measurement — NOT re-run

`scripts/measure_component_sparsity.py` and
`measure_component_cap_cost.py` measure graph structure (pairwise
adjacency, component sizes) rather than filter behaviour. Their inputs
are agent POSITIONS, and v5-cessation's positions are ordinary — it is
its velocity discontinuities that are impossible, and a component
membership test does not read velocity. They are unaffected by
construction, not by measurement, and re-running them would confirm an
arithmetic identity rather than test a conclusion. Day 26/27's open
question (the cap's cost on a genuinely well-matched component) is
untouched: it still needs real coupled multi-entity data, which neither
v5 nor v6 contains.
