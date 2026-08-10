# ADR 0010 — Estimator configuration: config A remains in use

- **Status:** Accepted
- **Date:** 2026-08-10
- **Decides for:** which of the four Day-22 estimator configurations
  (single model / single model + velocity floor / IMM / IMM + velocity
  floor) is adopted for production use, and on what evidence
- **Supersedes:** nothing directly — extends the diagnosis Day 21 opened
  and did not close
- **Related:** [[iron-data-model-day13]] (IMM built Day 21, not yet
  validated as an improvement); `FOUNDATION_REPORT.md` Day 21 (cessation
  diagnosis, IMM build, no-trade criterion NOT satisfied) and Day 22
  (this ADR's evidence); [[check-the-measuring-apparatus]] (the pooled-NEES
  question this ADR answers is another instance of that pattern)

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
