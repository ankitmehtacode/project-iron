---
name: iron-eval-discipline
description: Measurement-first engineering law for project-iron. MUST be consulted before ANY change claimed to improve accuracy, quality, speed, or memory — model swaps, preprocessing changes, threshold tuning, quantization, prompt changes, "optimizations", refactors of inference code. Also triggers on: "improve", "fix accuracy", "better results", "tune", "upgrade the model", "switch to", benchmark requests, and before merging any PR that touches src/. The rule is: no number, no merge.
---

# Iron Eval Discipline — No Number, No Merge

The failure mode this skill exists to prevent: a change that "obviously helps" ships, nothing
measures it, quality silently drops 30%, and nobody notices for a month because the pipeline
still runs. Everything below is designed so that cannot happen.

## The Three Laws

1. **No number, no merge.** Any PR claiming an improvement includes the eval scorecard delta
   (before/after on the frozen golden set, same manifest discipline). Any PR that regresses a
   metric states why in the description and gets explicit sign-off. "It should be better" is
   not a number.
2. **Red before green.** When a bug is found or suspected, FIRST write a failing test that
   encodes it (`@pytest.mark.known_bug`, xfail non-strict), THEN fix. The test is the proof the
   bug existed and the guarantee it stays dead. Fixing without the red test first is rejected.
3. **Baselines are frozen artifacts.** Golden sets and synthetic ground-truth sets are
   content-addressed and append-only. Editing a golden clip is forbidden; add a new versioned
   set. Comparing against a mutated baseline is not a comparison.

## The Eval Assets (know what exists before inventing)

- **Golden set**: real clips spanning the operational envelope — nominal, night, rain, blur,
  occlusion-heavy, texture-poor, camera shake, mid-clip entry, degenerate (all-black,
  blown-out, 1-frame, T not divisible by tubelet). Report metrics at 1080p AND 720p renditions.
- **Synthetic GT set** (Kubric): exact depth/tracks/occlusion/3D. Use it for any question of
  the form "what is the error in units" — never eyeball real footage for that.
- **Query benchmark**: NL/search queries with known answers → success@1/@5.

## Metric Protocol (use standard protocols; do not invent)

Depth: AbsRel, RMSE, δ<1.25, scale-invariant log error, static-point Z std (flicker).
Tracking: TAP-Vid `<δˣ_avg`, OA, AJ; plus IDF1 and ID-switches/min on occlusion subsets.
3D: endpoint error in mm **bucketed by distance** (1–5 m, 5–15 m, 15 m+) — one aggregate hides
far-field failure. Semantics: same-object retrieval mAP, temporal embedding stability,
patch-boundary discontinuity, FP32↔INT8 per-patch cosine reporting mean **and p1** (mean is
always ~0.99 and always meaningless). Latency: p50/p95/p99/max — pilots fail on p99, not mean.

## Parameter-Scaling Bugs — fix before you tune

**A bug whose severity scales with a parameter must be fixed before that parameter is
tuned.** Tune first and the sweep measures the bug's gradient, and the result gets read as a
property of the system. The best-looking setting is then whichever one the defect happens to
hurt least, and it ships with a number attached.

Worked example (ADR 0002). The V-JEPA2 frame→temporal-slot rule was a clamp,
`min(t, T_out - 1)`, where the encoder's layout calls for `t // tubelet`. They agree only at
the bottom of the range, so the damage scales almost linearly with clip length:

| `T` | misassigned frames | share |
| ---: | ---: | ---: |
| 4 | 1 of 4 | 25% |
| 8 | 5 of 8 | 62% |
| 16 | 13 of 16 | **81%** |
| 64 | 61 of 64 | **95%** |

The audit ran at `T=4` — the mildest configuration the bug has, and the reason it stayed
hidden. A clip-length sweep on the pre-fix mapper would have found 4 frames "best" and
concluded short clips beat long ones, which is a real, reproducible measurement of the clamp
rather than of clip length.

Same shape elsewhere: an envelope threshold validated at one mover speed (day 7), a parity
claim measured on a clip below the downscale trigger (day 2), a determinism check whose two
runs shared one process (day 6).

**Detection heuristic — run before any parameter sweep.** List the known-open defects touching
that code path and ask, for each: *does its severity depend on the parameter I am about to
sweep?* If yes, the sweep is blocked until it is closed. Cheap version: compute the defect's
effect at the low, middle, and high end of the sweep range. If those three numbers differ, you
are about to measure the defect.

Corollary: **the mildest configuration is the worst place to audit.** Reproduce a suspected
bug at the setting the product will actually use, not the one that is fastest to run.

## The Synthetic-Evaluability Partition — classify the capability before building the fixture

**Geometry-derived capabilities are synthetically evaluable. Appearance-learned ones are not.
Signal-absence capabilities are the hardest to evaluate synthetically at all, and point tracking
is gated by set composition rather than belonging to either row.**

| | examples | why | fixture |
|---|---|---|---|
| Geometry-derived | motion, occlusion topology, coverage, frustum visibility | ground truth is computable from the scene description; the model infers nothing the renderer does not already know | synthetic is **better** than real — exact GT, no annotation |
| Appearance-learned | depth, semantics, re-ID, detection | the model runs on shading gradients, texture statistics and object recognition | synthetic is **worse** than real — analytic primitives delete exactly those cues |
| Signal-absence | the motion gate: value is concentrated in frames where nothing happens | the model's job is recognising *absence* correctly, and a renderer's idea of "nothing happening" is perfect, noiseless stillness — not what absence looks like on a real sensor | synthetic is **worst** of the three — see below; even a *correct* renderer still cannot supply the texture of real quiet |

Worked example (appearance-learned). v3-indoor carries *exact* ground-truth depth in metres,
which made it look like an ideal depth fixture. Measured, DA-V2 scored **rank correlation
−0.5924** against that ground truth — it ordered the pixels backwards — while a scale-and-shift
alignment produced a respectable-looking **AbsRel 0.1538**. The same weights on real footage
produced a 4.44 dynamic range with the floor correctly nearer than the ceiling. A flat matte wall
at 15 m is not an easy depth target; it is an absent one.

The trap is that **exact GT does not make a set a fixture for that capability.** Ground truth
answers "what is true"; it says nothing about whether the input carries the signal the model
needs. Motion survives on primitives because motion *is* geometry. Depth does not, because
monocular depth is learned appearance.

### Point tracking is not fixed on either side — it is a property of the set, not the capability

Day 16 found point tracking refuses on v3-indoor (corner density 2.34e-04, below the 1e-03
floor — fast motion blurs the frame, starving the Shi-Tomasi corner detector) and **passes** on
v4-gate (1.11e-03 — static, unblurred scenes hand the same detector plenty to lock onto). Same
generator, same primitives, same capability, opposite verdicts. Point tracking's validity gate
measures local texture/corner availability, and that is a property of what a given set happened
to render — motion speed, blur, edge density — not a property of the point-tracking capability
itself the way "appearance-learned" is a property of depth. **A new set must declare its texture
density (or run the validity gate) before anyone assumes point tracking transfers from a sibling
set that happened to pass or fail it.** Treating a capability's validity verdict as portable
across sets in the same family is exactly the mistake this finding closes.

### Signal-absence capabilities: synthetic data's worst case, not just another hard one

The motion gate's entire product is compute saved on frames where nothing worth reporting
happens — its value is concentrated almost entirely in the *absence* of signal, the opposite of
every capability above. A renderer's version of "nothing happening" is perfect, bit-identical
stillness: no sensor noise, no HVAC-driven shadow drift, no autofocus hunt, no compression
artifact. Real absence is never that clean — those exact effects are what a background-
subtraction gate reacts to. A synthetic quiet scene is not a smaller-signal version of a real
quiet scene; it is missing the specific texture the capability is being asked to correctly
ignore.

Worked example: Day 16 authored v4-gate specifically to contain quiet scenes (empty rooms, a
motionless occupant, brief entries) and found the generator could not represent stillness at
all — its articulated agents' leg-gait animation was driven by elapsed clip time rather than by
whether the agent was moving, so a `speed_scale=0.0` "motionless" occupant rendered a silhouette
that changed on 38 of 40 frames. Day 17 fixed that specific defect (gait phase now tracks
distance travelled, not wall-clock time — see `Agent.progress_at`), and the renderer can now
produce a genuinely bit-identical-across-frames stationary agent. That closes one way this
generator could not represent absence; it does not close the general problem. **Open question,
recorded rather than answered: does this generator have a sensor-noise model?** As of Day 17, no
— an empty room still renders bit-identical frame to frame, not because that is the noise floor
of nothing happening, but because nothing is modelled at all. Until it does, every quiet-scene
measurement (any `dataset.moving_frame_fraction` near zero) is optimistic by an unknown margin,
and the gate cannot be honestly evaluated on synthetic data regardless of how carefully the
scenes are authored — the ceiling is not the scene design, it is the absence of a noise model.

In practice:

- Classify the capability *before* authoring a fixture, and pick the data source from the class —
  including whether it is signal-absence, not just geometry-vs-appearance.
- Every capability gets a **validity gate** that runs before its metric — see
  `src/data/validity.py`. A gate that fails records `unmeasurable_here` with evidence; it never
  emits a blank, a zero, or nothing at all.
- For point tracking specifically: run the validity gate (or otherwise declare texture density)
  on every new set. Do not assume a verdict carries over from a sibling set in the same family.
- Report **band populations alongside band scores, always.** A band holding 95% of pixels sets
  the aggregate by itself while the band you care about scores zero without moving the headline.
- A validity registry that only ever refuses is indistinguishable from a broken one. Implement
  at least one passing capability so the mechanism is falsifiable.
- For a signal-absence capability, a synthetic "quiet" measurement is a mechanism check, not a
  product number, until the generator has a stated noise model — say so on every scorecard it
  produces, the same way appearance-learned refusals are stated rather than implied.

## Bounded Nulls — a search result carries its bounds

**"Not found within the measurement window" and "does not exist" are distinct
results and must be distinct types.** A search that returns null carries the range it
searched. Any consumer that treats an unbounded null as a physical limit is a defect.

Worked example (day 7 → day 8). The capability-envelope sweep tested silhouette areas
40–320 gate px. At the two slowest speeds the wake rate never crossed 50% inside that window,
so the model recorded `null` — and the scorecard read `null` as *physically unreachable*, then
labelled real, resolvable motion as beyond the camera's capability. Re-measured to 1660 px, the
slow-speed crossing sat at 535. **There was no absorption floor; there was an edge of the
search.** 68 frames of correctly-detected motion had been excluded from the denominator.

The rule in practice:

- A sweep artifact records `searched_range` alongside every result. No range, no conclusion.
- An uncrossed search returns the bound (`"> 320 px"`), never a bare `None`.
- Interpolating *across* an uncrossed sample fabricates a value that was never observed —
  propagate the unresolved state instead.
- Before concluding "X cannot happen", check whether X was inside the window at all. The
  cheapest version: does the extreme of the searched range still fail? If the sweep never
  reached failure, it never established a limit.

Related trap, same family: **a metric that can be rescued by fitting is not evidence on its
own.** Depth alignment (day 9) fits scale and shift before scoring, and a near-constant
prediction on a scene that is 95% background fits that background and posts AbsRel 0.15 while
ordering pixels backwards. Pair any fitted metric with an alignment-invariant one — rank
correlation, here — and gate on that.

## Trivial Baselines — a number without one is not a result

**Every metric ships with the score of a trivial solution that ignores the capability being
measured.** Report `value (baseline, margin)` always; a non-positive margin means the metric is
measuring something other than what it claims — not that the system under test is bad, but that
the metric cannot currently tell the difference between a real system and one that knows
nothing.

Worked examples, both from this project, both caught after the number had already shipped:

- **Constant-prediction depth.** A near-constant disparity prediction landed on v3-indoor's
  dominant distance band and scored aggregate δ<1.25 = 0.9163 — while the band containing every
  agent, the one the product actually cares about, scored 0.0000 without moving the headline.
  The aggregate needed a constant-prediction baseline next to it to be interpretable at all.
- **Positional-constancy retrieval.** Same-object retrieval mAP came back at 18x chance on
  v3-indoor semantics. The diagnostic that explained it: 79% of GT tracks never left their
  16-px patch across the 4-frame encoder window, so "same object" had collapsed into "same
  patch index." A position-only baseline (rank by assumed co-location, ignore the encoder
  entirely) at that same query protocol would have shown the mAP sitting on top of it, not
  above it — and would have caught the defect before the number was reported instead of after.

**Minimum baseline per capability class:**

| capability | trivial baseline(s) |
|---|---|
| depth | constant prediction (report per distance band — see the worked example above) |
| retrieval / semantics | position-only (patch-index / co-location identity), and chance (1/n) |
| tracking | copy-previous-frame, static-point |
| binary wake/gate decisions | always-decide-yes, always-decide-no |

**Enforcement, not convention.** `src/eval/baselines.py` makes this structural: a metric name
with no registered baseline raises `BaselineMissing` at the moment a `Metric` is constructed,
before the number can reach a scorecard or a report. See `src/data/scorecard.py`'s
`_metric_with_baselines` for the only path that is allowed to build a `Metric` in this project.
Some baselines are theoretical boundaries no real system can beat (always-wake's recall of 1.0,
zero false negatives) — these still render as context but are excluded from the margin/flag
computation, or every system would flag against a bound nothing can cross.

## Closed-Form vs Instrumented — a prediction about a running system is a hypothesis

**If the real thing can be instrumented, instrument it. A closed-form comparison against a
running system is a hypothesis about that system, not a measurement of it, no matter how
carefully the closed form is derived.** The two are easy to conflate because a closed-form
value is exact — but exact about the formula, not about what the system actually did with it.

Worked example (ADR 0010, Days 22-25). The velocity-covariance floor's derivation was correct
from Day 22 onward. What was never correct was how "does it bind" got answered:
`scripts/velocity_floor_frame_rate_sweep.py` computed the floor's closed-form value and compared
it against a SEPARATE run of the filter with the floor disabled — a natural-convergence number,
not the floor-enabled filter's own output. Day 22 through Day 24 all read "closed-form floor >
natural convergence" as "the floor binds," and it does follow logically — but nobody had actually
run the floor-enabled filter and read `estimate.cov_array()` back to check. Day 25 did exactly
that (`scripts/velocity_floor_binding_audit.py`): ran config B for real on real tracks, and found
the floor's minimum reading equalled the closed-form value to floating-point precision on every
scored frame — the inference was correct, but two days (Day 22's "measurably inert" and Day 23's
"never binds at any frame rate 1-1000fps") had already been spent on a derivation bug whose
symptom was, itself, only ever checked against the same kind of closed-form stand-in.

This is a different failure shape than a bug that produces a wrong number about the system: it
is an analytical shortcut standing in for a measurement, and it can produce a wrong number about
a component that is working exactly as designed. The closed-form side of the comparison was never
false — the floor's formula was correct at every step once Day 24 fixed it. What was false was
treating "the formula says X" as equivalent to "the running system does X."

**In practice:**

- Before shipping a claim of the form "component A's output implies component B's behavior,"
  check whether B can be run directly and its actual output read back. If yes, that reading is
  the measurement; the closed-form inference is at best a cross-check, never a substitute.
- A closed-form check is legitimate as a FAST proxy during iteration, but the claim that ships in
  a report or gates a decision needs the instrumented number, not the proxy's inference from it.
- Treat "verified against a synthetic/analytical comparison" as a flag to ask "was the real
  system ever actually run and read?" — the same way a fitted metric (Bounded Nulls, above) is a
  flag to ask "was this rescued by the fit?"
- Any analytical claim that a design or a scaling argument rests on (e.g. "coupling is sparse, so
  components stay small") gets the same treatment before code is built on top of it: measure it
  against real or synthetic GT before treating it as a constraint the implementation may assume.

## Honesty Clauses

- Report the metric that looks bad. Omitting an unfavorable bucket is falsification.
- A test that can't run (missing weights) SKIPS with a reason — it never fake-passes.
- Per-verb / per-capability precision floors gate production: a detector below its floor is
  disabled by policy, not shipped with hope.
- If you cannot measure a claim yet, say so in the PR and file the eval task — do not soften
  the claim's wording as a substitute for measuring it.
