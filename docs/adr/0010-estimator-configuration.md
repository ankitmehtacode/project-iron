# ADR 0010 — Estimator configuration: config B (velocity covariance floor) now adopted

- **Status:** **Superseded 2026-08-16 (Day 30) — the adopted
  configuration REVERTS from B to A.** Config B's velocity uncertainty is
  a constant (pinned at the 1.5 m/s floor on 100.00% of scored frames on
  v6-motion, 99.12%/99.89% on v5-cessation/v3-indoor), so it never
  estimated the quantity its calibration result was credited to; and on
  physically reachable motion the cessation overconfidence it was adopted
  to fix does not exist (config A coverage 0.9920 on v6-motion, against
  0.5013 on v5-cessation). See "Day 30 revision (Objective 3)" at the
  end. Every cessation number in this ADR before that section was
  measured on v5-cessation, whose stop events are physically impossible
  (median peak GT deceleration 15.13 m/s² = 1.54g across its 22 stop
  events, max 36.58 m/s² = 3.7g; 63.6% of events above 1g, where a
  world-class sprinter peaks near 1g) — marked PROVISIONAL in "Day 30
  revision (Objective 1)", which also contains a claim about IMM that
  Objective 3 then corrects. **No prior text is edited or retracted.**
  Prior status line, unchanged: Accepted; revised 2026-08-11 (Day 23); revised again
  2026-08-11 (Day 24); revised again 2026-08-12 (Day 25) — the adopted
  configuration CHANGES, from A to B. **Revised again 2026-08-13 (Day
  26) — the A→B flip's provenance is settled with numbers: config B now
  produces materially different POSITION ESTIMATES from config A (RMSE
  deltas of up to 62% per regime), not merely a different pass/fail
  label under a redesigned criterion. See "Day 26 revision
  (Objective 1)".** See "Day 25 revision (Objective 1)" (confirms the
  floor binds on a real run — no remaining defect) and "Day 25 revision
  (Objective 3)" (the no-trade criterion made directional; under it,
  config B clears the bar on both golden sets with one stated, bounded,
  safe-direction cost, and is adopted). Day 24's summary, for continuity:
  Objective 1 traced Day 21's founding NEES-815 number to n=1 hand-traced
  track, not an aggregate — under-supported as originally reported,
  though the underlying finding is independently established at real n
  (373 frames, Day 23). Objective 2 found the velocity floor Day 22-23
  measured as permanently inert was itself mis-derived; corrected, it
  binds at every practical frame rate.
- **Date:** 2026-08-10
- **Decides for:** which of the four Day-22 estimator configurations
  (single model / single model + velocity floor / IMM / IMM + velocity
  floor) is adopted for production use, and on what evidence
- **Supersedes:** nothing directly — extends the diagnosis Day 21 opened
  and did not close
- **Related:** [[iron-data-model-day13]] (IMM built Day 21, not yet
  validated as an improvement); `FOUNDATION_REPORT.md` Day 21 (cessation
  diagnosis, IMM build, no-trade criterion NOT satisfied), Day 22 (this
  ADR's original evidence), Day 23 (the first revision below:
  v5-cessation, the frame-rate sweep, and the reconciled regime
  labeling), Day 24 (the second revision: the NEES-815 frame-support
  audit and the corrected, absolute velocity floor), and Day 25 (the
  third revision: the floor's binding confirmed against a real run, and
  the no-trade criterion redesigned directionally, changing the adopted
  configuration to B); [[check-the-measuring-apparatus]] (the pooled-NEES
  question this ADR answers, and the matrix-script gap Day 23 found while
  re-running the validity gates, are both instances of that pattern)

## Context

Day 21 diagnosed a specific estimator failure and built a specific fix for
it. The diagnosis: a single constant-velocity Kalman filter tracking a
walking person acquires a tight velocity covariance over a sustained walk
— repeated updates against an accurate position sensor pull steady-state
variance down — and that hard-won confidence becomes precisely wrong the
instant motion stops. One frame-by-frame trace showed NEES climbing to
~800 immediately after a stop (chi-square(6) 95% bound: 12.6) and decaying
back to nominal over roughly 10-14 frames as the covariance caught up to
the fact that velocity had actually gone to zero. The fix Day 21 built:
IMM, a three-mode mixture (`static`, `constant_velocity`, `maneuvering`)
that lets the filter's own posterior swap dynamics models frame by frame
instead of running one Q wide enough to cover both regimes badly.

IMM did not clear its own bar. Evaluated per regime against a no-trade
criterion (a transient regime must improve materially toward nominal
coverage, and no steady regime may degrade), Day 21 found the criterion
NOT SATISFIED on both golden sets: v3-indoor's sustained regime — a
regime that needed no fix — went from 99.4% to 65.9% pooled NEES coverage
under IMM, and v4.1-gate's static regime went from 81.7% to 43.4%. Day 21
closed with two open questions rather than tuning IMM to force a pass, per
the day's own instruction against iterating a model to hit a target
number: (1) was the metric itself — pooled NEES, which collapses IMM's
mixture posterior to one (mean, cov) before testing it against a
single-Gaussian chi-square bound — measuring the collapse rather than the
filter, and (2) does a much simpler, physically-derived fix address the
cessation mechanism directly, without IMM's added complexity. Day 22 was
scoped to answer both before any estimator decision.

## Decision

1. **The pooled-NEES metric was partially invalid, and does not rescue
   IMM.** NEES assumes a single Gaussian posterior; IMM's combined
   estimate is a Gaussian mixture. `src/estimator/consistency.py` now
   declares this explicitly — `compute_nees` raises `PosteriorFamilyError`
   on anything but a genuine single-Gaussian posterior — and two
   mixture-valid alternatives were implemented: `compute_mixture_nees`
   (probability-weighted per-mode NEES) and
   `empirical_coverage_by_sampling` (nonparametric HPD-region coverage via
   Monte Carlo, assuming no parametric shape for the mixture region at
   all). Re-evaluated under both, per regime: the exact magnitude moves —
   v3-indoor sustained's coverage under IMM is 79.6% by the sampling
   metric, not the 65.9% the collapsed number reported — but the
   direction does not. Sustained still degraded by a real ~20 points
   relative to the single model's 99.4%, not merely as an artifact of
   collapsing the mixture. v4.1-gate static's number is 43.4% under EVERY
   metric tested, collapsed or not, because IMM's mode probability there
   is concentrated enough on one mode that collapsing changes almost
   nothing. **Conclusion: the metric question was a real but partial red
   herring.** It changes how bad IMM's degradation looks; it does not
   change whether IMM has one.

2. **The physically-derived velocity covariance floor is sound and
   measurably inert on this data.** `pedestrian_velocity_covariance_floor_mps2`
   derives a floor on a person-kind mode's posterior velocity variance
   from `PERSON_SIGMA_A_MPS2` (1.5 m/s², already declared in
   `src/estimator/motion_model.py` since Day 20) — the same reasoning
   Q's own velocity-block entry already uses: a single predict step
   already asserts up to `(sigma_a * dt)^2` of velocity uncertainty enters
   regardless of the prior, so the posterior may never legitimately claim
   less. Measured on both golden sets, at every regime, config B (single
   model + floor) is **numerically identical to config A to displayed
   precision** — not approximately close, bit-for-bit identical RMSE and
   coverage. Verified directly against the raw per-frame velocity
   covariance on the actual `brief_entry` cessation track (not just
   aggregate RMSE, to rule out a wiring bug masked by rounding): at this
   project's 12 fps (`dt_s ≈ 0.083`), the floor evaluates to `~0.0156
   (m/s)^2`, while this filter's natural Kalman steady-state velocity
   variance — even after only 6 onset frames, well short of full
   convergence — is already `~0.065-0.09 (m/s)^2`, four to six times
   looser than the floor. **The floor never binds at this frame rate, on
   this data, so it changes nothing.** Config D (IMM + floor) is
   correspondingly identical to config C for the same reason. This is
   reported as the finding it is: a physically-derived constraint that
   does not help is real information about the model, not a bug to
   explain away — Day 22's own instruction, taken at its word.

3. **Four-way results, per regime, both golden sets** (position RMSE /
   coverage — mixture-valid sampling coverage for C and D, NEES-based
   coverage for A and B / frame count):

   | set | regime | A | B | C | D |
   | --- | --- | --- | --- | --- | --- |
   | v3-indoor | static (n=38) | 0.085m / 1.000 | = A | 0.044m / 1.000 | = C |
   | v3-indoor | onset (n=284) | 0.156m / 0.965 | = A | 0.147m / 0.961 | = C |
   | v3-indoor | sustained (n=2414) | 0.114m / 0.994 | = A | 0.205m / **0.796** | = C |
   | v3-indoor | cessation (n=0) | empty | empty | empty | empty |
   | v3-indoor | maneuver (n=0) | empty | empty | empty | empty |
   | v4.1-gate | static (n=175) | 0.237m / 0.817 | = A | 0.060m / **0.434** | = C |
   | v4.1-gate | onset (n=12) | 0.190m / 0.833 | = A | 0.194m / 0.917 | = C |
   | v4.1-gate | sustained (n=0) | empty | empty | empty | empty |
   | v4.1-gate | cessation (n=3, THIN) | 0.159m / 0.000 | = A | 0.143m / 0.333 | = C |
   | v4.1-gate | maneuver (n=0) | empty | empty | empty | empty |

   `maneuver` is empty on both sets by construction (every synthetic
   agent walks one straight line at one constant speed — Day 21). B=A and
   D=C everywhere, per Decision 2. IMM (C/D) roughly halves position RMSE
   in three of the four non-empty regimes it touches, while degrading
   coverage in the two regimes with real frame counts (v3-indoor
   sustained, v4.1-gate static) — accuracy and calibration moving in
   opposite directions, the same pattern Day 21 already found.

   **Superseded for cessation specifically, 2026-08-11:** v4.1-gate's
   `cessation (n=3, THIN)` row above used the pre-Day-23 regime label,
   which (see "Day 23 revision" below) covered only the anticipatory
   frame before a stop, not the recovery tail — this table is left
   exactly as measured on 2026-08-10 for the record, not edited in
   place. The corrected label, corrected v4.1-gate numbers (n=39), and
   the new v5-cessation set (n=373) are in the revision section.

4. **The no-trade criterion is UNSCOREABLE on both golden sets, for every
   candidate, independent of the numbers above.** The criterion (restated
   Day 22 around cessation specifically, since Day 21 pinned the failure
   there and not at onset or maneuver): cessation coverage must move
   materially toward nominal, and static/sustained must not degrade.
   Cessation's own frame count is 0 on v3-indoor and 3 on v4.1-gate, both
   below the 10-frame floor this project already uses to flag a
   conclusion as unsupported (`MIN_REGIME_FRAMES_FOR_A_CONCLUSION`).
   `scripts/eval_estimator.py`'s `_no_trade_verdict` now returns a third
   state — `unscoreable` — for exactly this case, rather than silently
   computing a delta from 3 frames and calling it a pass or a fail. This
   is reported plainly rather than glossed: even v4.1-gate cessation's
   NEES pass rate moving from 0.0 to 0.33 under IMM, or its RMSE
   improving from 0.159m to 0.143m, are numbers from 3 frames — real
   numbers, correctly computed, and not evidence of anything at the
   confidence this project requires before calling a regime fixed.

5. **Adopted configuration: A (Day 20's single model, unchanged).** No
   configuration meets the no-trade criterion — not because B, C, or D
   were measured and found wanting on cessation specifically, but because
   neither available golden set contains enough cessation frames to
   measure that question at all. This is recorded as this decision's
   terminal state for today, not a placeholder awaiting more code: **the
   blocker is data, not an unbuilt estimator.**

6. **Rejected, and why:**
   - **IMM (config C)** is not adopted for production use. Independent of
     the cessation question, it measurably degrades two regimes with real
     frame counts (v3-indoor sustained, v4.1-gate static) under every
     valid consistency metric tested, mixture-aware or not, with no
     offsetting evidence — because none exists in this data — that it
     fixes the regime it was built for.
   - **The velocity covariance floor (config B)** is not rejected on its
     physics; it is simply not a fix for anything observable today, since
     it never binds at this project's 12fps cadence. It is kept as tested,
     documented, opt-in machinery (`velocity_covariance_floor=True`), not
     defaulted on — a constraint that does not currently do anything
     should not ship as if it does.
   - **Config D** inherits both rejections: C's measured degradation and
     B's inertness.

## Consequences

- Config A carries forward into every layer that builds on today's
  single-entity core — multi-entity association, smoothing, hypothesis
  management — with its already-known characteristics unchanged: good
  calibration on v3-indoor (static/onset/sustained all ≥96%), poor
  calibration on v4.1-gate static (81.7%), and a diagnosed-but-unfixed
  cessation blowup. Anything built on top of a configuration wrong at
  cessation stays wrong in every later layer, per this day's own framing
  — that risk is unchanged by today's work, not resolved by it.
- Cessation is now a harder blocker than "no fix has been tried yet": it
  is "no fix can currently be CERTIFIED, because the golden sets that
  would certify one don't contain the regime in enough volume." This
  reframes the highest-leverage next step away from estimator engineering
  and toward data: more walk-then-stop synthetic clips would make the
  no-trade criterion scoreable without touching the filter at all.
- The floor's inertness is itself a finding worth carrying forward: if a
  floor-shaped fix is revisited, deriving it from a DIFFERENT physical
  quantity (an asserted minimum stopping-relevant variance, not one
  single predict step's own Q injection) is the specific thing to try
  next — today's derivation was correct as far as it went and simply
  turned out to be looser than what this filter's own Kalman convergence
  already achieves at 12fps.
- IMM's mode-probability calibration (Day 21 Objective 5, documented as
  uncalibrated) remains unvalidated; nothing here changes that status.

## Open questions

- Would a floor derived from the ACTUAL stopping deceleration profile
  (rather than one timestep of Q's own injection) bind where today's did
  not? Untried — the day's own rule against fitting a floor to the data
  applies equally to inventing a second derivation just to force a
  different answer; a next attempt needs its own independent physical
  basis, stated before it is measured, not chosen because it moves the
  number.
- Can the synthetic generator produce enough walk-then-stop clips to make
  cessation scoreable at all (target: ≥10 frames per set, ideally more)?
  This is arguably the single highest-leverage open item from today,
  since it unblocks re-evaluating every configuration in this ADR against
  the criterion they actually need to pass.
- Whether IMM's degradation on steady regimes (Day 21's "mixing overhead"
  hypothesis — a standing 2.5%-per-mode tax from the transition matrix
  even after mode probability converges) is the mechanism, independent of
  cessation — still unconfirmed by a second measurement, still open.

## Day 23 revision — v5-cessation results, the no-trade criterion scored

Day 22 closed with two open questions this section answers, plus a third
Day 23 asked that Day 22 did not: (1) can the generator supply enough
cessation volume — **yes**; (2) does IMM's degradation survive contact
with real cessation volume, or was it an artifact of 0-3 frames —
**survives, and gets worse**; (3) at what frame rate would the velocity
floor begin to bind — **none tested, including 12fps and every rate
above it**.

### The regime label was the first defect, fixed before any of this

Day 23 Objective 1 found that the pre-Day-23 `cessation` regime
(`src/estimator/regime.py`) covered only the *anticipatory* window before
a stop — the instant GT speed reached zero, every following frame was
immediately `static`, including the ~10-14 frame recovery tail Day 21
traced by hand (NEES 759.9→9.1, reproduced directly against the real
`brief_entry` track; Day 21's own reported 815→16 differs only by minor
codebase drift since). `classify_track` now runs a second pass that
reclassifies a `static` run's opening frames back to `cessation` for a
recovery window (`PEDESTRIAN_STOP_DURATION_S`, reused not fitted — 12
frames at 12fps). Measured effect: v4.1-gate's cessation count rose
3→39. **This made the phenomenon measurable. It did not supply
behavioral diversity** — 39 frames from ~2 underlying stop trajectories
is one labeling fix away from three-frame evidence, still thin. That is
what v5-cessation is for.

### v5-cessation: 373 cessation frames, from 19 diverse tracks

Day 23 Objective 2 built `synthetic-indoor-v5-cessation` — 19 clips: 8
radial and 8 lateral stop events spanning slow/medium/fast approach,
near/far distance, abrupt/gradual deceleration, and a long-hold variant,
plus 2 stop-then-restart clips and 1 double-stop clip. Measured directly
via `classify_track` over every scene's raw GT track: 1059 total frames,
cessation 373, static 260, sustained 282, onset 129, maneuver 15
(declared out of scope). The mint-time gate
(`enforce_regime_volume`/`RegimeVolumeError`) refuses to mint a set that
cannot clear ≥200 cessation frames and ≥30 for every other non-exempt
regime — this set cleared both by a wide margin. Registered lane S with
full per-asset clearance (`configs/datasets.yaml`); the Day-10 validity
gates confirm PASS on `motion_geometry`/`state_estimation`, REFUSE on
`depth`/`appearance_semantics`/`point_tracking` — geometry-yes,
appearance-no, exactly as this set's own purpose predicts. (Note: the
eval script's own per-track scoring drops each track's first 2 frames as
filter warm-up — `FIRST_COMPARABLE_INDEX` in `scripts/eval_estimator.py`
— so the *scored* onset count below is 91, not 129; cessation/static/
sustained are unaffected since none of v5-cessation's 19 tracks start
inside those regimes.)

### Four-way per-regime table, v5-cessation (scored via `scripts/eval_estimator.py --version v5-cessation`)

| regime | n | A/B RMSE | A/B coverage¹ | C/D RMSE | C/D coverage² | C/D NEES/NIS pass³ | margin(copy-prev), A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| static | 260 | 0.1381m | 0.9962 | 0.0832m | 0.0808 | 0.0615 | +0.1338m |
| onset (scored) | 91 | 0.1991m | 0.8132 | 0.2686m | 0.9011 | 0.7802 | +0.1413m |
| sustained | 282 | 0.1470m | 0.9574 | 0.2067m | 0.9504 | 0.9220 | +0.1789m |
| cessation | 373 | 0.2499m | 0.5013 | 0.1432m | 0.3727 | 0.3083 | +0.0368m |
| maneuver (exempt) | 15 | 0.1789m | 0.4000 | 0.2608m | 0.8000 | 0.7333 | +0.2855m |

¹A/B coverage is NEES pass rate (single-Gaussian posterior, valid for
these configs). ²C/D coverage is the mixture-valid sampling-HPD
coverage (`empirical_coverage_by_sampling`), not the collapsed-Gaussian
number — Day 22's `PosteriorFamilyError` discipline, applied. ³C/D
NEES/NIS pass is the collapsed-Gaussian diagnostic Day 22 built
specifically to flag as non-authoritative for a mixture posterior;
listed because Objective 3 asked for NEES/NIS pass rates explicitly, not
as the calibration verdict — use the coverage column for that. B=A and
D=C in every cell (floor still inert — see below). Every config beats
`constant_velocity_no_update` by very large margins everywhere (dead
reckoning accumulates unbounded drift); omitted from the table as
uninformative, consistent with Day 22.

Re-run on v4.1-gate under the corrected regime label (same command):
cessation n=39 (was 3), A RMSE 0.4643m / coverage 0.1282, C/D RMSE
0.0803m / coverage 0.0256 (sampling-HPD) — same direction as
v5-cessation, more extreme. v3-indoor is untouched by any of Day 23's
work and remains cessation n=0 — a genuine data gap specific to that
set (it has no stop events at all), not something v5-cessation was ever
meant to fix.

### No-trade verdict: now scoreable, and NOT_SATISFIED

| version | A→B | A→C | A→D |
| --- | --- | --- | --- |
| v3-indoor | UNSCOREABLE (n=0) | UNSCOREABLE (n=0) | UNSCOREABLE (n=0) |
| v4.1-gate | NOT_SATISFIED (cessation Δ +0.0000, n=39) | NOT_SATISFIED (cessation Δ −0.1026, static Δ −0.3604 REGRESSION) | NOT_SATISFIED (same as C) |
| v5-cessation | NOT_SATISFIED (cessation Δ +0.0000, n=373) | NOT_SATISFIED (cessation Δ −0.1287, static Δ −0.8231 REGRESSION) | NOT_SATISFIED (same as C) |

No candidate config satisfies the criterion on either set that has
cessation frames. IMM does not merely fail to *improve* cessation
coverage — it makes it worse (0.5013→0.3727 on v5-cessation) while
cratering static calibration harder than on any set measured to date
(0.9962→0.0808, an 91-point drop — worse than v3-indoor's or
v4.1-gate's own IMM static regressions). This is now measured on 373
autocorrelated-but-behaviorally-diverse cessation frames from 19
distinct stop events across two camera geometries, not 3 frames from one
track. **The direction of Day 21/22's finding survives at full strength;
the magnitude of config A's own cessation overconfidence moderates once
measured on a diverse set** — v4.1-gate's thin-sample cessation coverage
under config A was 0.1282; v5-cessation's properly-powered estimate is
0.5013. Both are decisively below the 0.95 target and both are real; the
first was also an artifact of n=2 underlying trajectories, which this
section is what resolves.

### Frame-rate dependence of the velocity floor: it never binds

`scripts/velocity_floor_frame_rate_sweep.py` runs config A (floor
disabled) over a controlled synthetic constant-velocity walk at a swept
range of frame rates and compares the filter's own converged posterior
velocity variance against the floor's closed-form value
(`(PERSON_SIGMA_A_MPS2 * dt_s)^2`) at the same `dt_s`. Measured, 1-1000
fps:

| fps | natural (m/s)² | floor (m/s)² | floor/natural |
| --- | --- | --- | --- |
| 1 | 3.864052 | 2.250000 | 0.5823 |
| 2 | 0.656961 | 0.562500 | **0.8562 (closest approach)** |
| 12 (this project) | 0.065043 | 0.015625 | 0.2402 |
| 90 | 0.010296 | 0.000278 | 0.0270 |
| 120 | 0.011324 | 0.000156 | 0.0138 |
| 1000 | 0.547952 | 0.000002 | 0.0000 |

The floor never binds anywhere in this range. The least-obvious finding:
natural convergence is **not monotonic** in frame rate — it is worst at
very low fps (large per-step process noise), improves to a minimum
around fps≈90-120, then gets worse again at very high fps (differencing
positions that are close together in time, against fixed measurement
noise, amplifies velocity-estimate noise — a classical
differentiation-of-a-noisy-signal effect). The floor shrinks
monotonically as `dt_s²` throughout, so it becomes *more* inert, not
less, as fps rises past 12 — the intuitive guess ("higher fps converges
tighter, so the floor binds sooner") is backwards. Its closest approach
to binding in the tested range is ~86% of natural, at 2fps — already
below any frame rate this product would run at, and the ratio still does
not cross 1.0 there.

**This closes Day 22's open question 2 (generator volume: yes) and
answers Day 23's frame-rate question directly: the floor as currently
derived is retired as a live finding.** It is not deleted — it remains
tested, documented, opt-in machinery per Decision 6, since a physically
sound constraint that does not currently bind is real information, not
a bug — but it is no longer an open research thread to revisit "at a
different frame rate," because no frame rate this product could plausibly
run at makes it relevant. Day 22's open question 1 (a floor derived from
the *actual stopping deceleration profile*, a different physical basis
entirely) is untouched by this finding and remains open if a floor-shaped
fix is ever revisited. Open question 3 (IMM's mixing-overhead hypothesis)
is also untouched — still unconfirmed — though v5-cessation's sharper
static regression under IMM is one more data point consistent with
*something* systematic in IMM's steady-regime behavior, not a
confirmation of the specific mechanism.

### Decision: unchanged. Config A remains adopted — now on stronger evidence

Day 22 adopted config A because no alternative could be shown to clear
the no-trade bar, and neither could the bar itself be evaluated at
cessation. Day 23 evaluated it, on 373 frames from 19 diverse stop
events, and no alternative clears it — IMM fails cessation and regresses
static harder than previously measured, and the floor is inert
everywhere a camera for this product could plausibly run. This is a
strictly stronger result than Day 22's, in the same direction: config A
was provisionally correct-by-elimination; it is now correct-by-measurement.

## Day 24 revision (Objective 1) — how much did Day 21's NEES-815 number actually support?

Day 24's first objective asked a narrower question than it looks:
Day 21's headline diagnosis — NEES climbing to ~815 and decaying over
10-14 frames after a stop — motivated everything from IMM (Day 21) through
the floor (Day 22) through v5-cessation (Day 23). How many frames actually
stood behind that number when it was first reported?

**Answer: exactly one hand-traced track.** Day 21 Objective 2's
`brief_entry` trace (reproduced verbatim from the Day 21 report):

```
frame  regime      NEES     bound   within
 0-3   onset       0.8-2.2  12.59   True        <- fine
 4     cessation   165.6    12.59   False       <- the stop itself
 5-14  static      815->16  12.59   False (all) <- decaying tail
15+    static      <10      12.59   True        <- settled
```

is a single-track, hand-traced narrative — 10 frames (5-14) of one
recovery tail from one stop event. It was never an aggregate statistic.
Day 21's own per-regime coverage table, printed in the SAME report
section, could not have shown this finding even in principle: under the
then-current regime label, those 10 decaying-NEES frames were themselves
classified `static` (the label bug Day 23 Objective 1 later fixed), so
they are folded into v4.1-gate's `static` row (n=175, coverage 0.8171) —
diluted by ~165 genuinely-converged static frames elsewhere in the set —
not visible in the `cessation` row (n=3, coverage 0.0000) printed
alongside it, which the trace also does not correspond to (that n=3 was
the anticipatory instant(s) before a stop, a different frame set
entirely). **Day 21's own aggregate table and its own headline number
were, without anyone noticing at the time, describing two different
slices of the data that happened to share a label.**

Regime definition, then vs now: **then** (Day 21, pre-Day-23-fix),
`classify_track` (`src/estimator/regime.py`) used priority
static > onset > cessation > maneuver > sustained and assigned
`cessation` only to the anticipatory window immediately before GT speed
reaches zero; every frame from the stop onward, including the entire
recovery tail, fell to `static`. **Now** (Day 23 Objective 1's fix,
unchanged today), a second labeling pass reclassifies a `static` run's
opening frames back to `cessation` if they fall inside a recovery window
of `PEDESTRIAN_STOP_DURATION_S` (12 frames at 12fps) following a
moving-to-static transition — which is what raised v4.1-gate's cessation
count 3→39, and is the label under which v5-cessation's 373 frames were
classified from the start.

**Verdict, one sentence: Day 21's NEES-815 diagnosis was under-supported
as originally reported (n=1 track, hand-traced, not even the same frames
as its own printed aggregate row) — the *existence* of the cessation
miscalibration is now established (373 frames, 19 stop events, config A
coverage 0.5013, decisively below the 0.95 target), but that took until
Day 23 Objective 2-3, two full days after the diagnosis was first acted
on.** This reframes Days 21-22 accurately: they established that IMM and
the floor were plausible responses to a real-looking signal and built
both in good faith, but neither day could have certified that the signal
generalized beyond one anecdote, and Day 22 said so explicitly (the
no-trade criterion's own "UNSCOREABLE" verdict was this same fact,
already surfacing structurally two days before it was stated in these
terms).

**Process note — this is the second instance of the same omission
pattern.** Day 20's per-distance-bucket margin existed in the underlying
data but wasn't surfaced in the report's own table until Day 21
Objective 1 went looking for it. Here, the "how many frames actually
support this" question had its answer available in Day 23's own
Objective 1 body (`FOUNDATION_REPORT.md` names the `brief_entry` track
directly) but was never assembled into an explicit statement of support
— Day 23's headline described the fix and the outcome, not the
evidentiary weight of the number that motivated the fix. Two instances is
a pattern, not a coincidence: a number can be technically present in a
report and still functionally missing if no sentence ever states what it
does or doesn't support. Going forward, any report section that opens a
multi-day investigation on the strength of one measured number should
state that number's own sample size in the same paragraph, not leave it
inferable from a trace printed for a different purpose.

## Day 24 revision (Objective 2) — the floor was mis-derived, not inert; corrected, it changes the field

**Headline: the "floor never binds at any frame rate" finding Day 23
reported as decisive was itself an artifact of a derivation bug. Fixed,
the floor binds at every practical frame rate this project could run at,
and config B (single model + floor) satisfies the no-trade criterion on
v4.1-gate outright and comes close on v5-cessation. This does not
automatically make B the adopted configuration — see the open question at
the end of this section — but it means Day 22-23's "the floor is real
physics that simply doesn't help" conclusion is retracted. The floor
helps; it was measured with a broken ruler.**

### The derivation bug: per-timestep, not absolute

`pedestrian_velocity_covariance_floor_mps2` (`src/estimator/motion_model.py`)
computed, since Day 22:

```
sigma_v_floor^2 = (PERSON_SIGMA_A_MPS2 * dt_s)^2        [units: (m/s^2 * s)^2 = (m/s)^2]
```

where `dt_s` was the CURRENT PREDICT STEP's own timestep — the same `dt_s`
used to compute `F`/`Q` for that one step. This is dimensionally a
velocity variance (that is why it passed review), but it answers "how
much velocity uncertainty does one sample interval's own process noise
inject" — a quantity that shrinks as the sample interval shrinks, i.e.
as frame rate rises. Day 23's frame-rate sweep measured exactly that
consequence: the floor shrank monotonically as `dt_s^2` while natural
Kalman convergence did not shrink nearly as fast, so the floor fell
further behind natural convergence at every fps above 1, never catching
up anywhere in a 1-1000fps sweep.

The floor exists to bound a different, PHYSICAL question: how much could
a person's velocity plausibly have changed since the filter last had
strong evidence pinning it down, given that a person can go from walking
to at-rest in about `PEDESTRIAN_STOP_DURATION_S` (~1s, already declared,
Day 22)? That physical fact does not depend on how often a camera happens
to sample the walk. A floor that answers it correctly must therefore be
ABSOLUTE — the same value regardless of frame rate — not a per-step
quantity. Corrected:

```
sigma_v_floor^2 = (PERSON_SIGMA_A_MPS2 * PEDESTRIAN_STOP_DURATION_S)^2
                 = (1.5 m/s^2 * 1.0 s)^2
                 = (1.5 m/s)^2
                 = 2.25 (m/s)^2                          [constant, every fps]
```

Units at every step: `PERSON_SIGMA_A_MPS2` is m/s², `PEDESTRIAN_STOP_DURATION_S`
is s; their product is m/s (a velocity, the same physical quantity `dt_s`
occupied in the old formula, just anchored to a fixed physical duration
instead of a variable sampling interval); squaring gives (m/s)², the same
units `cov`'s velocity-diagonal entries already carry. The two formulas
necessarily coincide at exactly one point — `dt_s == PEDESTRIAN_STOP_DURATION_S`,
i.e. 1 fps — which is why Day 23's sweep table already contained the
"correct" value (2.2500) in its fps=1 row without anyone noticing it was
the special case, not a data point on a curve that should have varied.

### Corrected floor vs converged σ_v on v5

At this project's 12fps (`dt_s ≈ 0.0833s`), re-measuring natural Kalman
convergence directly (`scripts/velocity_floor_frame_rate_sweep.py`,
unchanged script, corrected floor value): natural steady-state velocity
variance is **0.0650 (m/s)²**, versus the corrected floor's **2.2500
(m/s)²** — the floor is now ~34.6x LARGER than natural convergence, i.e.
it binds, hard, exactly where Day 21's diagnosis said a floor was needed.
Re-running the full frame-rate sweep confirms this holds everywhere
tested except the single fps=1 coincidence point:

| fps | natural (m/s)² | floor (m/s)² | binds? |
| ---: | ---: | ---: | --- |
| 1 | 3.8641 | 2.2500 | no |
| 2 | 0.6570 | 2.2500 | **YES** |
| 12 (this project) | 0.0650 | 2.2500 | **YES** |
| 90 | 0.0103 | 2.2500 | **YES** |
| 120 | 0.0113 | 2.2500 | **YES** |
| 1000 | 0.5480 | 2.2500 | **YES** |

The hypothesis stated at the top of today's objective — "a floor that
never binds anywhere from 1 to 1000 fps is more likely mis-derived than
physically irrelevant" — is confirmed. The floor now binds at every
tested rate from 2 to 1000 fps; the one exception (1 fps) is the
coincidence point above, not a counterexample to the fix.

### Four-way re-evaluation, corrected floor, all three golden sets

Re-ran `scripts/eval_estimator.py`'s A/B/C/D comparison (config B/D now
genuinely differ from A/C for the first time since the floor was built).

**v5-cessation** (n=373 cessation frames, 19 stop events):

| regime | n | A RMSE/cov | B RMSE/cov | C RMSE/cov | D RMSE/cov |
| --- | ---: | --- | --- | --- | --- |
| static | 260 | 0.1381m / 0.9962 | 0.1923m / 0.9923 | 0.0832m / 0.0808 | 0.0839m / 0.7154 |
| onset (scored) | 91 | 0.1991m / 0.8132 | 0.1988m / 0.9890 | 0.2686m / 0.9011 | 0.2701m / 0.9011 |
| sustained | 282 | 0.1470m / 0.9574 | 0.1887m / 0.9929 | 0.2067m / 0.9504 | 0.1962m / 0.9787 |
| cessation | 373 | 0.2499m / 0.5013 | **0.1990m / 0.9946** | 0.1432m / 0.3727 | 0.1450m / 0.8606 |
| maneuver (exempt) | 15 | 0.1789m / 0.4000 | 0.1697m / 1.0000 | 0.2608m / 0.8000 | 0.1625m / 1.0000 |

Config B's cessation coverage moves **0.5013 → 0.9946** — from
decisively-overconfident to essentially nominal — while static moves
0.9962→0.9923 (negligible) and sustained moves 0.9574→0.9929 (toward
MORE conservative, i.e. underconfident, not overconfident — see the
no-trade discussion below for why this still registers as a "regression"
under the criterion as coded).

No-trade verdicts, v5-cessation:

| A→ | cessation Δtoward-nominal | static Δ | sustained Δ | verdict |
| --- | --- | --- | --- | --- |
| B | **+0.4040 (IMPROVED)** | +0.0038 | −0.0355 REGRESSION | NOT_SATISFIED |
| C | −0.1287 (not improved) | −0.8231 REGRESSION | +0.0071 | NOT_SATISFIED |
| D | +0.3592 (IMPROVED) | −0.1885 REGRESSION | −0.0213 REGRESSION | NOT_SATISFIED |

**v4.1-gate** (n=39 cessation frames, ~2 underlying trajectories, Day 23's corrected label):

| regime | n | A RMSE/cov | B RMSE/cov | C RMSE/cov | D RMSE/cov |
| --- | ---: | --- | --- | --- | --- |
| static | 139 | 0.1023m / 0.9928 | 0.1437m / 1.0000 | 0.0567m / 0.5468 | 0.0568m / 0.5468 |
| onset | 12 | 0.1899m / 0.8333 | 0.1833m / 0.9167 | 0.1935m / 0.9167 | 0.1889m / 0.9167 |
| sustained | 0 | empty | empty | empty | empty |
| cessation | 39 | 0.4643m / 0.1282 | **0.1748m / 1.0000** | 0.0803m / 0.0256 | 0.0901m / 0.1282 |
| maneuver | 0 | empty | empty | empty | empty |

No-trade verdicts, v4.1-gate:

| A→ | cessation Δtoward-nominal | static Δ | verdict |
| --- | --- | --- | --- |
| B | **+0.7718 (IMPROVED)** | −0.0072 | **SATISFIED** |
| C | −0.1026 (not improved) | −0.3604 REGRESSION | NOT_SATISFIED |

**Config B (single model + corrected floor) clears the no-trade bar
outright on v4.1-gate.** It does not clear it on v5-cessation, purely
because of `sustained`'s −0.0355 delta — cessation itself improves
strongly on both sets (+0.7718 on v4.1-gate, +0.4040 on v5-cessation).

v3-indoor (unaffected finding — cessation stays UNSCOREABLE, n=0, a
genuine data gap untouched by this fix): static/onset/sustained coverage
under B moves 1.0000→1.0000, 0.9648→0.9965, 0.9938→0.9975 respectively —
each equal to or above config A, no regression by any reading.

### Why v5-cessation's "regression" is a softer failure than it looks

The no-trade criterion (`_no_trade_verdict`, `scripts/eval_estimator.py`)
scores `sustained`/`static` degradation as `|coverage − 0.95|` growing —
symmetric around nominal, so moving from underconfident-but-close
(A: 0.9574, |Δ|=0.0074) to more-underconfident (B: 0.9929, |Δ|=0.0429) on
v5-cessation's sustained regime counts against B exactly as harshly as
moving toward overconfidence would. But overconfidence and
underconfidence are not equally dangerous for this product: overconfident
coverage means the filter's stated uncertainty band is a LIE (Day 21's
whole diagnosis), which is the failure mode that produces false
confidence in wrong state; underconfident coverage means the filter
claims a wider band than it strictly needs, which is conservative, not
misleading. B's only "regression" on the only properly-powered set is
entirely in the safe direction. This is noted as an observation, not
acted on today — changing how the criterion is scored specifically
because it would flip B's verdict is exactly the "iterate a metric until
it passes" pattern the project's rules prohibit, however defensible the
argument sounds in isolation. It is recorded here as Day 25's first
open question, to be decided BEFORE looking at whether the decision
changes, not after.

### Decision: unchanged today — config A remains adopted, pending one open question

Config A remains adopted as of Day 24, unchanged — but on materially
weaker relative footing than Day 23 recorded, since Day 23's stated
reason for rejecting B ("floor never binds, so it changes nothing") is
now known to be false. The correct summary of today's evidence: **B is a
serious, no-longer-inert candidate that clears the no-trade bar on one
golden set and misses it on the other only via a safe-direction
deviation the criterion does not currently distinguish from a dangerous
one.** Re-deciding the adopted configuration on that basis today, in the
same session that discovered it, would repeat the exact pattern Day 21
was instructed to avoid (iterating toward a target rather than deciding
in advance what would count as evidence). Day 25's first item is
therefore: settle whether the no-trade criterion should weight
over-confidence and under-confidence deviations asymmetrically — decided
on its own methodological merits, stated in advance of re-scoring B
against it — and only then revisit config B's adoption.

### Rejected, and why (Day 24 addendum)

- **IMM (configs C/D)** — unaffected by today's fix, still rejected per
  Day 23: degrades static/sustained on both sets with real frame counts,
  with no offsetting cessation win large enough to justify it (C's own
  cessation coverage is still the worst of any config, 0.3727 on
  v5-cessation).
- **The velocity floor (config B)** — NOT rejected. Explicitly
  un-rejected by this section; carried forward as an open, high-priority
  candidate rather than retired machinery.

## Day 25 revision (Objective 1) — the floor vs converged σ_v contradiction was never real; both numbers now measured directly

Day 25 opened by treating Day 24's "binds hard, 34.6x" claim as unverified
rather than settled: that number came from comparing the floor's
closed-form value against `scripts/velocity_floor_frame_rate_sweep.py`'s
synthetic-walk convergence measurement — never from running the actual
floor-enabled filter (config B) on a real golden set and reading its
posterior covariance back. A floor that is correctly wired should never
be readable below itself once enabled (`apply_velocity_covariance_floor`
is a `max()` clamp on the posterior — `src/estimator/filter.py:193-199`),
but "should" is not "was measured to."

**Method.** New script, `scripts/velocity_floor_binding_audit.py`: runs
config A (floor disabled) and config B (floor enabled) for real on
v5-cessation's actual tracks, and reads `estimate.cov_array()`'s
velocity-diagonal entries back per scored frame — the same real per-frame
covariance `eval_estimator.py`'s coverage numbers are already computed
from, now also surfaced as a distribution (`FrameRecord.velocity_variance_diag_mps2`,
new field; `_sigma_v_distribution`, new helper — both in
`scripts/eval_estimator.py`, and both consumed by the regular
`--version`/`by_regime` report path, not just this audit script).

**Floor value** (unchanged from Day 24, re-derived here with units shown
at every step): `PERSON_SIGMA_A_MPS2 [1.5 m/s²] * PEDESTRIAN_STOP_DURATION_S
[1.0 s] = 1.5000 m/s` → floor variance `2.2500 (m/s)²`.

**Converged σ_v, v5-cessation, config A (floor DISABLED — natural convergence)**:

| regime | n | min (m/s) | p50 (m/s) | max (m/s) | p50/floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| static | 260 | 0.2294 | 0.2562 | 0.3110 | 0.1708 |
| onset | 91 | 0.2482 | 0.5954 | 1.8014 | 0.3969 |
| sustained | 282 | 0.2400 | 0.2728 | 0.5182 | 0.1819 |
| cessation | 373 | 0.2297 | 0.2548 | 0.3352 | 0.1699 |

**Converged σ_v, v5-cessation, config B (floor ENABLED)**:

| regime | n | min (m/s) | p50 (m/s) | max (m/s) |
| --- | ---: | ---: | ---: | ---: |
| static | 260 | 1.5000 | 1.5000 | 1.5000 |
| onset | 91 | 1.5000 | 1.5000 | 1.8014 |
| sustained | 282 | 1.5000 | 1.5000 | 1.5000 |
| cessation | 373 | 1.5000 | 1.5000 | 1.5000 |

Cross-checked on v4.1-gate (static/onset/cessation; `sustained` is empty
by construction on that set) — same shape: config A natural p50 0.17-0.38x
the floor in every populated regime, config B pinned at exactly 1.5000 in
every populated regime.

**Verdict: none of (a)/(b)/(c).** The contradiction the objective set out
to resolve does not survive contact with a real run:

- **Not (b).** Config B's minimum observed σ_v equals the floor to
  floating-point precision in every regime on both golden sets — the
  clamp fires on essentially every scored frame. `onset`'s max (1.8014)
  exceeding the floor is not a bypass; it is the clamp correctly doing
  nothing on the frames where natural uncertainty is *already* above the
  floor (onset is exactly where velocity is genuinely changing fast, so
  higher natural uncertainty there is expected, not a defect). There is
  no P0 finding here — the floor is implemented, wired into the
  covariance-update path at exactly the line the module docstring says it
  is, and reached by real production configs.
- **Not (a).** The floor (1.5 m/s) is not far below converged σ_v; it is
  far *above* natural convergence (~0.25-0.60 m/s, 0.17-0.40x the floor)
  in every regime — the opposite of (a)'s premise.
- **Not (c).** Natural convergence does not exceed 0.8 m/s in every
  regime — only `onset`'s max (1.8014) and a handful of high frames do;
  every regime's p50 sits at 0.25-0.60 m/s, comfortably inside the
  "confident" range (c) describes as absent.

**What actually happened:** Day 23's original "the floor never binds
anywhere from 1-1000fps" finding was entirely a consequence of the
dt_s-scaling bug Day 24 already found and fixed. Day 24's fix and its own
re-measurement (the 34.6x/binds-hard finding, and the four-way table
showing B's cessation coverage jumping 0.5013→0.9946) were both already
correct; today's contribution is confirming that conclusion against a
real run's actual posterior covariance rather than a closed-form
comparison, closing the one gap in how it had been checked. The 0.5-0.8
m/s order-of-magnitude estimate in the objective's own framing (a
"walking speed" argument) differs from the derivation's actual output
(1.5 m/s, a "stopping deceleration × duration" argument) by roughly 2x —
both are legitimate, independently-reasoned order-of-magnitude estimates
of the same physical quantity from different starting facts, not a new
discrepancy; Day 24 already used the acceleration-based derivation and
this is not revisited today per the objective's own instruction not to
adjust a constant to make the floor bind.

No constant was adjusted to reach this result — both numbers were
measured as-is, from the code as Day 24 left it.

## Day 25 revision (Objective 3) — the no-trade criterion made directional; config B now adopted

**This was revised because the criterion was symmetric and the
underlying risk is not — not because a configuration needed to pass.**
The redesign below was written and committed to (in `scripts/eval_estimator.py`,
with unit tests) BEFORE it was run against config B's own numbers, exactly
per Day 24's closing instruction: "settle whether the no-trade criterion
should weight over-confidence and under-confidence deviations
asymmetrically... and only then revisit config B's adoption." The
sequence below records that order, so this reads as a principled
redesign rather than a metric loosened until something passed.

### The asymmetry, stated as a rule before any config was re-scored

Overconfidence (empirical coverage below the 0.95 nominal — the filter's
stated uncertainty band understates its true error) and underconfidence
(coverage above 0.95 — the band is wider than it needs to be) are not
equally dangerous for an evidence system. Overconfidence is a lie the
filter tells about its own certainty, and every downstream decision
inherits it; the system becomes most certain exactly where it is most
wrong (this project's Day 20 headline finding, `brief_entry`'s
onset-transient overconfidence, was this exact failure mode). Underconfidence
is inefficient — wider bounds, more conservative alerts — never
misleading.

The criterion (`_no_trade_verdict`, `scripts/eval_estimator.py`) now
returns one of five explicit types, never a bare boolean or a status
string a caller could collapse to "it passed":

- `NoTradeUnscoreable` — cessation too thin to judge anything (unchanged
  from Day 22).
- `NoTradeNoImprovement` — cessation didn't move materially toward
  nominal; nothing to weigh a cost against.
- `NoTradeFailOverconfident` — **non-negotiable.** A steady regime
  (`static`/`sustained`) moved toward overconfidence (candidate coverage
  below 0.95) by more than `NO_TRADE_DEGRADATION_TOLERANCE` (0.02,
  unchanged). Independent of how much cessation improved.
- `NoTradePass` — cessation improved materially; no steady regime
  degraded toward overconfidence beyond tolerance.
- `NoTradePassWithCost(regime, magnitude, costs=...)` — cessation
  improved materially; at least one steady regime degraded, but strictly
  toward underconfidence. The magnitude is a required field, not a note —
  the trade is numeric in the record, not implied.

Direction is read off the CANDIDATE's own coverage-error sign against
0.95 (below = overconfident), not the sign of the change — a regime
already overconfident at baseline that stays overconfident is still
`overconfident`, independent of whether the gap narrowed.

### Four-way re-evaluation under the directional criterion

**v5-cessation** (n=373 cessation frames):

| A→ | cessation Δtoward-nominal | verdict (Day 24, symmetric) | verdict (Day 25, directional) | changed? |
| --- | --- | --- | --- | --- |
| B | +0.4040 (IMPROVED) | NOT_SATISFIED (sustained −0.0355 scored as a plain regression) | **PASS_WITH_COST** (sustained, magnitude 0.0355) | **YES** |
| C | −0.1287 (not improved) | NOT_SATISFIED | FAIL_OVERCONFIDENT (static 0.9962→0.0808, magnitude 0.8231) | no (still rejected, different reason recorded) |
| D | +0.3592 (IMPROVED) | NOT_SATISFIED | FAIL_OVERCONFIDENT (static 0.9962→0.7154, magnitude 0.1885) | no (still rejected) |

**v4.1-gate** (n=39 cessation frames):

| A→ | cessation Δtoward-nominal | verdict (Day 24) | verdict (Day 25) | changed? |
| --- | --- | --- | --- | --- |
| B | +0.7718 (IMPROVED) | SATISFIED | **PASS** (no cost anywhere — static 0.9928→1.0000 is itself an improvement, not a degradation) | no (already satisfied; now clean rather than merely satisfied) |
| C | −0.1026 (not improved) | NOT_SATISFIED | FAIL_OVERCONFIDENT (static 0.9928→0.5468, magnitude 0.3604) | no |

**v3-indoor**: UNSCOREABLE for every candidate on both the old and new
criterion (0 cessation frames by construction — unaffected finding,
confirms the redesign did not change behaviour where there is nothing to
score).

**IMM's expected outcome, checked as instructed.** The objective's own
prediction — "IMM still fails, because Day 23 found it regressing static
harder than any set to date and static regression means overconfidence
in the regime that should be easiest" — is confirmed on both sets. IMM
(C and D) does not pass under the directional criterion either;
`static`'s coverage collapse (0.9962→0.0808 on v5-cessation, 0.9928→0.5468
on v4.1-gate) is squarely in the overconfident direction (both landing
far below 0.95), so `NoTradeFailOverconfident` is the correct, and only
possible, outcome — there is no "examine whether IMM's degradation is
genuinely toward underconfidence" step to run, because it plainly is not:
coverage collapsing toward 0 is overconfidence by definition. The
directional criterion did not loosen anything for IMM; it gave the same
rejection a more specific, falsifiable reason.

**Config B is the only verdict that changes**, and it changes in the
direction the redesign's own stated rationale predicts: B's only
"regression" anywhere in either golden set is `sustained` moving from
slightly-underconfident (0.9574) to more-underconfident (0.9929) on
v5-cessation — coverage moving further above nominal, the safe direction
by the criterion's own definition. Under the old symmetric scoring this
counted against B exactly as hard as moving toward overconfidence would;
under the corrected, directional scoring it is recorded as a stated,
bounded, numeric cost (`magnitude=0.0355`) rather than a rejection.

### Decision: config B (single model + velocity covariance floor) is now adopted

Following this project's decision framework:

1. **Problem.** Config A is overconfident specifically in the cessation
   regime (v5-cessation coverage 0.5013, v4.1-gate 0.1282 — both far
   below the 0.95 nominal, the exact failure mode Day 21 diagnosed).
2. **Constraints.** A replacement must not become overconfident anywhere
   it wasn't already (the non-negotiable side of the criterion above);
   any other cost must be stated numerically, not hidden in an aggregate.
3. **Alternatives compared.** A (status quo, overconfident at cessation),
   B (single model + floor), C (IMM), D (IMM + floor).
4. **Tradeoffs.** B fixes cessation coverage on both golden sets
   (0.5013→0.9946 on v5-cessation, 0.1282→1.0000 on v4.1-gate) at the cost
   of `sustained` becoming more conservative by 0.0355 coverage points on
   v5-cessation only — a single, bounded, safe-direction cost, not a new
   failure mode. C/D fix cessation less completely (C's own cessation
   coverage, 0.3727, is the worst of any config) while introducing a
   genuine, dangerous overconfidence regression in `static` on both sets —
   rejected on the same grounds Day 21-23 already established.
5. **Recommendation: adopt config B.** It is the only candidate that
   clears the non-negotiable side of the criterion on both golden sets
   while materially fixing the problem the criterion exists to catch.
6. **Why it wins.** The one cost it carries is exactly the kind this
   criterion was redesigned to treat differently from a danger: a wider,
   more conservative bound in a regime that was already close to nominal,
   not a filter lying about its own certainty.
7. **Future maintenance cost.** `velocity_covariance_floor_enabled` is
   already a per-motion-model flag (Day 22) — adopting B is a
   configuration change (`motion_model_for("person",
   velocity_covariance_floor=True)` at whatever call site constructs the
   production filter; no such call site exists yet, since detection/
   tracking are still unbuilt — Day 20's own scope note), not new code.
   The floor itself (`pedestrian_velocity_covariance_floor_mps2`) is
   analytic and declared, not fitted, so it carries no retraining or
   re-tuning burden going forward. The one open item this decision
   creates: `sustained`'s 0.0355-point conservatism on v5-cessation
   specifically should be watched, not re-litigated, if a future golden
   set shows it growing rather than staying flat.

This reverses config A's Day 20-24 default. It is recorded as a decision
made on Day 25's own evidence, following redesign-then-score ordering
Day 24 set out in advance — not a same-session reaction to a metric that
happened to flip.

### Rejected, and why (Day 25 addendum)

- **IMM (configs C/D)** — unaffected by the criterion redesign; the
  directional criterion gives the same rejection Day 21-24 already
  established a more specific, falsifiable reason (`FAIL_OVERCONFIDENT`,
  not just "regressed").

## Day 26 revision (Objective 1) — provenance of the A→B flip

**The adopted configuration changed after the criterion that judges it
was redesigned. That sequence is the shape that reads as motivated in
hindsight, so this section answers the one question that settles whether
it was: did the ESTIMATOR change, or only the SCORING of it?**

### The question, answered plainly: (i)

Day 22 measured config B as bit-for-bit identical to config A — "every
RMSE, every coverage figure, to displayed precision." Day 25 found the
corrected floor binds on essentially every scored frame. A constraint
binding constantly changes the Kalman gain (a wider floored covariance
means new measurements are weighted more heavily in the update step),
which changes the POSTERIOR MEAN, not just its reported uncertainty — so
(i) and (ii) make different, checkable predictions: if (i), position RMSE
per regime must differ between A and B today; if (ii), it must not.

Re-measured directly (`scripts/eval_estimator.py --config A --config B`,
current codebase, both golden sets) — position RMSE per regime, config A
vs config B:

**v5-cessation:**

| regime | n | A RMSE (m) | B RMSE (m) | Δ (m) | Δ (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| static | 260 | 0.1381 | 0.1923 | +0.0542 | +39.2% |
| onset | 91 | 0.1991 | 0.1988 | −0.0003 | −0.2% |
| sustained | 282 | 0.1470 | 0.1887 | +0.0417 | +28.4% |
| cessation | 373 | 0.2499 | 0.1990 | **−0.0509** | **−20.4%** |
| maneuver | 15 | 0.1789 | 0.1697 | −0.0092 | −5.1% |

**v4.1-gate:**

| regime | n | A RMSE (m) | B RMSE (m) | Δ (m) | Δ (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| static | 139 | 0.1023 | 0.1437 | +0.0414 | +40.5% |
| onset | 12 | 0.1899 | 0.1833 | −0.0066 | −3.5% |
| cessation | 39 | 0.4643 | 0.1748 | **−0.2895** | **−62.4%** |

**(i) is true: the Day-22 equivalence predated the floor actually being
reached, and B now produces materially different POINT ESTIMATES from A —
not merely a different reported confidence around the same estimates.**
The mechanism is exactly the Kalman-gain argument above, and it cuts both
ways, which is itself evidence this is a real estimator effect and not an
artifact: `static`/`sustained` RMSE gets WORSE under B (the floor forces
more weight onto each new position measurement even when the filter's own
prior velocity estimate was already accurate, adding noise it didn't
need), while `cessation` RMSE improves dramatically (the same extra
weight is exactly what lets the filter track a real, sudden velocity
change instead of coasting on a stale, overconfident prior). A pure
criterion change, with A and B computing identical estimates, could not
produce a bidirectional accuracy effect like this — it can only relabel
an unchanged number as pass or fail. Day 22's bit-identical finding was
real and correctly measured — for the per-timestep floor formula that
existed then, which the clamp's own `max()` semantics meant almost never
fired. It stopped describing config B the moment Day 24 corrected that
formula; nothing between Day 24 and today re-checked whether the RMSE
table itself, not just the coverage table, had moved.

### This was decided as an estimator change, not laundered through a criterion change

Three facts, each checkable independently of the others and of today's
own re-measurement above, are what keep this decision from reading as
"the criterion was loosened until something passed":

1. **The asymmetry argument was stated before the re-scoring, not
   after.** Day 24's ADR revision closed with an explicit, forward-dated
   instruction: "settle whether the no-trade criterion should weight
   over-confidence and under-confidence deviations asymmetrically —
   decided on its own methodological merits, stated in advance of
   re-scoring B against it." Day 25's "Day 25 revision (Objective 3)"
   section states the rationale (overconfidence is a lie the filter tells
   about its own certainty; underconfidence is merely inefficient) and
   commits it to code and unit tests BEFORE the four-config
   re-evaluation is run in that same section — the file diff order
   (`_no_trade_verdict`'s rewrite, then its tests, then the re-evaluation
   call) is checkable directly in commit `3cbc556`.
2. **IMM still failed under the new criterion — the load-bearing evidence
   against motivated reasoning.** If the redesign had been tuned, however
   subtly, to let B through, the cheapest tell would be IMM ALSO
   slipping through on some regime where its degradation happened to
   read as underconfidence. It did not: IMM (C/D) hits
   `FAIL_OVERCONFIDENT` on both golden sets, `static` coverage collapsing
   to 0.0808 (v5-cessation) and 0.5468 (v4.1-gate), both decisively below
   the 0.95 nominal — the criterion still fails the exact thing it was
   built to fail. A criterion redesigned specifically to rescue B would
   have had no principled reason to keep rejecting IMM this hard.
3. **B's cost is numeric, not waved away.** `NoTradePassWithCost` on
   v5-cessation names the regime and the magnitude directly:
   `sustained`, coverage moving 0.9574→0.9929 (further above the 0.95
   nominal — underconfident, the safe direction), magnitude **0.0355**
   coverage points. On v4.1-gate there is no cost at all (`static` moves
   0.9928→1.0000, itself an improvement, not a degradation) — B is a
   clean `PASS` there. A bounded cost with a number attached, on exactly
   one regime of one set, is a decision; "the criterion changed and B
   happened to pass" would not have a number to point to at all.

Read together with today's RMSE deltas: config B is not "config A with a
different pass/fail label." It is a measurably different estimator —
worse raw position accuracy on two steady regimes, dramatically better on
the regime the whole investigation exists to fix — adopted because the
criterion that judges the tradeoff was redesigned first, on its own
merits, and then applied to a config that turned out, independently, to
have actually changed underneath it.

## Day 30 revision (Objective 1) — the cessation evidence is measured on impossible motion, and is marked PROVISIONAL

**Nothing above is edited. This section states which of it survives.**

### The measurement

`scripts/measure_gt_acceleration_distribution.py` reports GT acceleration
magnitude per regime, using the same `classify_track` partition every
per-regime number in this ADR was computed against. On v5-cessation
(1021 frames with a defined acceleration, 19 tracks, 22 stop events):

| regime | n | p50 | p95 | max | frames >1g |
| --- | ---: | ---: | ---: | ---: | ---: |
| static | 260 | 0.000 | 0.000 | 0.000 | 0 (0.00%) |
| onset | 91 | 0.000 | 1.593 | 20.259 | 2 (2.20%) |
| sustained | 282 | 0.000 | 0.000 | 5.976 | 0 (0.00%) |
| cessation | 373 | 0.000 | **15.281** | **36.581** | 26 (6.97%) |
| maneuver (exempt) | 15 | 9.566 | 24.255 | 27.854 | 7 (46.67%) |

All values m/s². 1g = 9.80665 m/s².

**The gating question, answered: yes, they concentrate in cessation.**
74.3% of every >1g frame in the set (26/35) is cessation-regime, and
cessation is the only populated regime whose p95 is itself above 1g.

The frame fraction understates it, because a cessation regime is mostly a
recovery tail sitting at exactly zero acceleration while the transient
itself is one or two frames. Counted by EVENT instead: **14 of the set's
22 moving-to-static transitions (63.6%) peak above 1g, with a median peak
of 15.13 m/s² — 1.54g.** The median stop event in the set built to
measure stopping is not a stop; it is a collision.

Split by the set's own authored deceleration profile
(`PathSegment.ease_out`), which is the parameter that was supposed to make
half of them gradual:

| profile | events | peak range (m/s²) | above 1g |
| --- | ---: | --- | ---: |
| abrupt (`ease_out=False`) | 12 | 11.74 – 36.58 | **12/12** |
| gradual (`ease_out=True`) | 10 | 3.89 – 27.62 | 2/10 |

Every abrupt stop is impossible, which is by construction — an
`ease_out=False` leg ends at constant velocity and the next leg is a
zero-length pause, so the velocity step is the full approach speed in one
frame. The gradual ones are impossible only when fast
(`radial_fast_far_gradual` 27.62, `lateral_fast_near_gradual` 22.20),
because the quadratic ease-out's derivative is `2(1 - tail_t)`: it *jumps
speed to 2x* at the tail's start and then has one `ease_fraction` (0.3 of
a 1.2s leg ≈ 0.36s) to shed all of it. "Gradual" was never a physiological
profile; it was a smoothing of the second half of a discontinuity.

v3-indoor, for contrast: max GT acceleration 0.0001 m/s² over 2736 frames
— exactly constant-velocity to float32 storage precision, zero stop
events, zero cessation frames. The two golden sets bracket reality
without containing it.

### What this does and does not invalidate

**Provisional (measured on motion no body can produce):**

- Day 21's founding NEES ~815 diagnosis, and Day 23's re-measurement of
  it at n=373. The filter's covariance was responding to a velocity
  discontinuity — `PathSegment(ease_out=False)` is an *instantaneous* stop
  by construction, and even `ease_out=True`'s quadratic ramp doubles speed
  instantaneously at the tail's start. A filter is overconfident against a
  teleport-to-zero by definition, and no calibration is achievable against
  one, so "config A is overconfident at cessation" was never a falsifiable
  claim on this data.
- Every `cessation` row of every four-way table above, on v5-cessation and
  on v4.1-gate alike (v4.1-gate's `brief_entry` stops are built by the same
  mechanism).
- The A→B adoption's *load-bearing* evidence: cessation coverage
  0.5013→0.9946 (v5-cessation) and 0.1282→1.0000 (v4.1-gate).

**Not provisional:**

- **IMM's rejection.** C/D hit `FAIL_OVERCONFIDENT` on `static`
  (0.9962→0.0808 on v5-cessation, 0.9928→0.5468 on v4.1-gate). `static`'s
  GT acceleration is identically zero in this set — 0 frames above 1g, max
  0.000 m/s² — so IMM's static regression is measured on motion that is
  not merely physical but trivially so. Unaffected.
- **The directional criterion itself** (Day 25 Objective 3). A scoring
  rule is not a measurement.
- **Day 26's finding that A and B produce different point estimates.**
  That is a Kalman-gain fact about the two configurations, visible in
  `static` and `sustained` RMSE (+39.2%, +28.4%) as much as in cessation.
- **Day 29's slip NO.** Explicitly survives *a fortiori*: larger true
  acceleration means higher SNR, so a model that failed on the easier
  signal fails harder on the real one. That protection is directional and
  does **not** extend to the cessation diagnosis, which is why this
  section exists.

### The adoption is NOT retracted here

Config B remains adopted as of this revision. Contaminated evidence is not
disproof, and retracting a decision on the strength of "the data was
wrong" without measuring the alternative would substitute one
unsupported conclusion for another. Two things settle it, both scoped to
Day 30 and reported in `FOUNDATION_REPORT.md` §Day-30:

1. **Re-measurement on physical motion** — v6-motion, generated under
   physiological acceleration bounds (Objective 2), with the four-way
   evaluation re-run against it (Objective 3).
2. **A test of config B on its own terms** — whether its velocity
   uncertainty is a floor that binds on every frame, in which case B
   satisfies a calibration criterion by refusing to become confident
   rather than by estimating better (Objective 3).

Whichever way those land, this ADR's cessation evidence is
provisional until they are in.

## Day 30 revision (Objective 3) — config B is not adopted. It never estimated velocity uncertainty, and the defect it was adopted to fix does not exist on physical motion

**Nothing above is edited. This section reverses the decision and says on
what evidence.**

### Finding 1 — config B's velocity uncertainty is a constant

`scripts/velocity_floor_pinning_audit.py`, and a new
`fraction_at_floor` in `scripts/eval_estimator.py`'s
`_sigma_v_distribution` so every report path carries it:

| set | scored frames | pinned at the floor |
| --- | ---: | ---: |
| v6-motion | 1558 | **1558 (100.00%)** |
| v5-cessation | 1021 | 1012 (99.12%) |
| v3-indoor | 2736 | 2733 (99.89%) |

Per regime, the only frames not pinned are in `onset` on two sets (90.1%
and 98.9% pinned) — the one regime where velocity genuinely changes fast
enough that natural uncertainty already exceeds the floor. Every other
regime on every set is at 100.0%.

Day 25 measured the same underlying quantity and reported
`min == p50 == max == 1.5000` per regime, reading it as confirmation
that the clamp fires. It is that. It is also the signature of something
Day 25 had no reason to look for, because it was asking a wiring
question: **config B does not estimate velocity uncertainty. It reports
1.5 m/s.**

Config A's own natural converged σ_v is 0.2548–0.5954 m/s (v5-cessation,
p50 per regime), so the floor sits **2.5–5.9x above** what the filter
actually converges to. And 1.5 m/s is not a bound — Day 29 established it
is `PERSON_SIGMA_A_MPS2 × PEDESTRIAN_STOP_DURATION_S`, a comfortable adult
walking pace, the same product that `PEDESTRIAN_MAX_SPEED_MPS`'s docstring
rejects as a maximum speed.

**A filter that is never confident cannot be caught being overconfident.**
Config B's cessation coverage moving 0.5013 → 0.9946 on v5-cessation is
therefore not evidence that it models cessation better. Any sufficiently
large constant would have produced it.

### Finding 2 — on physical motion there is no overconfidence to fix

Four-way re-run on v6-motion, under the **unchanged** directional
criterion (`scripts/eval_estimator.py --version v6-motion --config A
--config B --config C --config D`). v5-cessation was re-run in the same
invocation and reproduces Day 25/26 exactly, so the two sets differ only
in the data:

| regime | n | A RMSE / cov | B RMSE / cov | C RMSE / cov | D RMSE / cov |
| --- | ---: | --- | --- | --- | --- |
| static | 328 | 0.0839m / 0.9909 | 0.1260m / 0.9970 | 0.0546m / 0.9726 | 0.0535m / 0.9970 |
| onset | 110 | 0.1126m / 0.9727 | 0.1279m / 1.0000 | 0.0714m / 1.0000 | 0.0713m / 1.0000 |
| sustained | 746 | 0.0956m / 0.9879 | 0.1326m / 0.9987 | 0.1380m / 0.9464 | 0.1459m / 0.9584 |
| cessation | 374 | **0.0946m / 0.9920** | 0.1280m / 0.9973 | 0.0747m / 0.9572 | 0.0736m / 0.9920 |
| maneuver | 0 | empty | empty | empty | empty |

**Config A's cessation coverage on physical motion is 0.9920 —
essentially nominal.** On v5-cessation the same configuration, the same
code, the same criterion measured 0.5013. The cessation overconfidence
this ADR has been about since Day 21 is a response to a
teleport-to-zero, not to a person stopping.

No-trade verdicts on v6-motion:

| A→ | cessation Δtoward-nominal | verdict |
| --- | --- | --- |
| B | −0.0053 | **NO_IMPROVEMENT** |
| C | +0.0348 | NO_IMPROVEMENT |
| D | +0.0000 | NO_IMPROVEMENT |

And config B's cost on physical motion is no longer one bounded,
safe-direction deviation. It is worse in **every** regime: static
+50.2%, onset +13.6%, sustained +38.7%, cessation **+35.3%** position
RMSE — including the regime it was adopted for.

### Decision: config A is re-adopted. Config B's adoption is withdrawn

Following the same framework Day 25 used, applied to better data:

1. **Problem.** As stated on Day 25: config A is overconfident at
   cessation. **That premise is now measured to be false on physical
   motion** (coverage 0.9920).
2. **Constraints.** Unchanged. The directional criterion is not
   revisited here — it is applied exactly as Day 25 committed it, which
   is what makes this a re-scoring rather than a re-litigation.
3. **Alternatives compared.** A, B, C, D on v6-motion.
4. **Tradeoffs.** B costs 13–50% position RMSE across every regime and
   buys a calibration improvement of −0.0053 at cessation, i.e. none.
5. **Recommendation: config A.** No candidate satisfies the criterion on
   v6-motion, and the status quo is the only one that does not pay for
   the failure.
6. **Why it wins.** It is the only configuration that still estimates
   velocity uncertainty at all.
7. **Future maintenance cost.** Reverting is a configuration change, not
   a code change — `velocity_covariance_floor_enabled` stays as tested,
   documented, opt-in machinery, exactly as Day 22 left it. The floor
   itself is not deleted: it is a correctly-implemented constraint whose
   *derivation* uses a typical-scale constant where a bound is needed,
   and that is a live open question, not dead code.

### The criterion has a blind spot, and it is separate from this reversal

Config B fails on v6-motion, so the criterion happens to reject it. It
would not have caught the pinning: a constant-uncertainty filter
satisfies both halves of the directional criterion trivially — cessation
coverage improves, and no steady regime moves toward overconfidence,
because nothing moves at all. **The criterion never asks whether the
reported uncertainty is informative.** That is a defect in the criterion
independent of today's verdict, and it is carried to Day 31 rather than
patched here, for the same reason Day 24 declined to redesign a criterion
in the session that discovered it mattered.

### Correction — Day 30 Objective 1's own claim about IMM was wrong

The "Day 30 revision (Objective 1)" section above lists **IMM's
rejection** under *"Not provisional"*, arguing that `static`'s GT
acceleration is identically zero on v5-cessation so IMM's static
regression was measured on trivially physical motion. **That reasoning
is invalid, and the v6-motion re-run refutes it directly.**

| set | A static coverage | C static coverage |
| --- | ---: | ---: |
| v5-cessation | 0.9962 | **0.0808** |
| v6-motion | 0.9909 | **0.9726** |

IMM's catastrophic static-coverage collapse does not reproduce on
physical motion. The error in the argument: a per-frame GT regime label
describes the *world* at that frame, not the *filter's state*, and a
filter's covariance at frame `t` is a function of the entire preceding
trajectory. IMM's mode probabilities are explicitly history-dependent, so
an impossible transient contaminates the static frames that follow it,
however physical those frames' own GT is. Checking that a regime's GT is
clean is not sufficient to establish that a measurement taken during that
regime is clean.

This does **not** make IMM adoptable: on v6-motion it is
`NO_IMPROVEMENT` like everything else, because there is no cessation
defect left to improve, and it is worse than A on `sustained` RMSE
(0.1380 vs 0.0956) while better on the other three. What it does mean is
that **the recorded reason for rejecting IMM — a dangerous, measured
overconfidence regression in `static` — is not supported by physical
data**, and Days 21–25's rejection of IMM is now provisional on the same
grounds as everything else measured on v5-cessation.

## Day 31 revision — the criterion now catches config B at the moment it was adopted

**Nothing above is edited.** Day 30 reverted the adopted configuration to
A on two findings: config B's velocity uncertainty is a constant, and the
cessation overconfidence it was adopted to fix does not exist on physical
motion. Both were found by hand, months of report-days after the fact.
This section records that the criterion itself now finds the first one.

### The new dimension

`src/estimator/informativeness.py` adds an informativeness margin: the
estimator's expected log predictive density minus that of the best
constant-variance predictor for the same errors, sharing the estimator's
own means so the comparison isolates the second moment. Nats per frame.
The baseline is fitted by maximum likelihood on the data it is scored
against, which makes it stronger than any calibration-tuned constant and
therefore makes a margin against it a lower bound. A calibration figure
can no longer be emitted without one (`CalibrationAndInformativeness` has
no defaulted fields).

`NoTradeUninformative` is a new typed verdict, checked after the
non-negotiable overconfidence test and before any pass.

### All four configurations, both axes, v6-motion

Coverage is unchanged from Day 30's table; the margin column is new.

| regime | n | A cov / margin | B cov / margin | C cov / margin | D cov / margin |
| --- | ---: | --- | --- | --- | --- |
| static | 328 | 0.9909 / **+0.839** | 0.9970 / **−1.002** | 0.9726 / +1.440 | 0.9970 / +1.407 |
| onset | 110 | 0.9727 / **+3.320** | 1.0000 / **−1.108** | 1.0000 / **−5.990** | 1.0000 / **−5.618** |
| sustained | 746 | 0.9879 / **+0.930** | 0.9987 / **−0.944** | 0.9464 / +1.381 | 0.9584 / **−0.133** |
| cessation | 374 | 0.9920 / **+0.765** | 0.9973 / **−1.060** | 0.9572 / +1.857 | 0.9920 / +1.126 |

Margins in nats/frame; bold where the margin is at or below the declared
`UNINFORMATIVE_MARGIN_NATS` of 0.05, or where it is the point of the row.

**Config B is uninformative in every regime, as predicted — and not
merely near zero. It is negative.** Config B's per-frame uncertainty
makes the observed truth roughly `e` times *less* likely per frame than a
single fitted constant does. It is not a constant in disguise; it is a
worse constant, reporting a velocity uncertainty of 1.5 m/s against
errors that warrant far less.

**Config A is informative in every regime.** This is the falsifiability
half: if the metric returned "uninformative" for everything it would be
measuring nothing.

**IMM (C/D) is informative in three regimes and badly uninformative at
`onset`** — −5.99 and −5.62 nats. Its onset coverage of 1.0000 reads as
ideal under calibration alone and costs about six nats a frame, which is
a new finding about IMM that no previously-reported number showed. The
mixture is scored on its own density here, not a collapsed Gaussian, so
Day 22's `PosteriorFamilyError` caveat does not apply to it.

Two honest limits on the table. First, the baseline's own coverage is
reported alongside every margin and is near nominal in most cells
(0.89-0.97) — confirming it really is a cheapest-way-to-pass predictor —
but it is 0.6455 in config A's `onset` cell, where the fitted constant is
itself overconfident. A's +3.320 there is therefore partly the baseline
being poor rather than A being good, and the safe reading of that cell is
its sign, not its magnitude. Second, every cell inherits the
state-carry caveat below.

### The retroactive result: the criterion would have caught the adoption

Config B was adopted on Day 25 on v5-cessation, with an A→B verdict of
`PASS_WITH_COST` (cessation +0.4040, cost: sustained 0.0355 toward
underconfidence). Re-scored on that same data under the extended
criterion:

```
A -> B (v5-cessation):
  cessation delta-toward-nominal: +0.4040 (n=373, IMPROVED)
  ...but static informativeness COLLAPSED: +3.1066 -> -0.5538 nats
  NO-TRADE CRITERION: UNINFORMATIVE
```

**The verdict that adopted config B becomes `UNINFORMATIVE` on the
unchanged data.** The five days between adoption and reversal were spent
recovering a fact the criterion can now state on the day the
configuration is first scored.

On v6-motion every candidate is `NO_IMPROVEMENT` — cessation is not
broken there, so the new verdict never has to fire, and config A remains
adopted for the reasons Day 30 gave.

A third confirmation of Day 30's contamination finding falls out of this
axis independently: config A's own cessation margin is **−4.998 nats on
v5-cessation** and **+0.765 on v6-motion**. On impossible motion even the
adopted filter's uncertainty is worse than a constant.

### The state-carry caveat, applied to every per-regime number in this ADR

**Regime-partitioned metrics are not independent measurements when the
estimator carries state across the partition boundary.** A filter's
covariance at frame `t` is a function of the entire preceding trajectory;
a GT regime label describes the world at that frame, not the filter's
state. So a per-regime figure measures "the filter, during frames the
world labelled X" — not "the filter on X".

This is not hypothetical. Day 30's correction found IMM's `static`
coverage moving 0.0808 → 0.9726 between two sets whose `static` GT
acceleration is identically zero in both: nothing about the static frames
changed, only what preceded them. Every per-regime row above, on both
sets, in this ADR and in ADR 0011, carries that caveat. It is strongest
for regimes that follow a transient (`static` and `cessation`, both of
which sit downstream of a stop) and weakest for `sustained`, which is
mostly its own history.
