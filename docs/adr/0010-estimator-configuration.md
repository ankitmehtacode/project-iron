# ADR 0010 — Estimator configuration: config A remains in use

- **Status:** Accepted; revised 2026-08-11 (Day 23); **revised again
  2026-08-11 (Day 24)** — see "Day 24 revision (Objective 1)" and
  "Day 24 revision (Objective 2)" below. Objective 1: Day 21's founding
  NEES-815 number is traced to n=1 hand-traced track, not an aggregate —
  under-supported as originally reported, though the underlying finding
  is now independently established at real n (373 frames, Day 23).
  Objective 2: **the velocity floor Day 22-23 measured as permanently
  inert was itself mis-derived; corrected, it binds at every practical
  frame rate and materially changes the evidence, though the adopted
  configuration (A) is unchanged today pending one open methodological
  question (see Day 24 revision (Objective 2), final section).**
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
  labeling), and Day 24 (the second revision: the NEES-815 frame-support
  audit and the corrected, absolute velocity floor);
  [[check-the-measuring-apparatus]] (the pooled-NEES question this ADR
  answers, and the matrix-script gap Day 23 found while re-running the
  validity gates, are both instances of that pattern)

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
