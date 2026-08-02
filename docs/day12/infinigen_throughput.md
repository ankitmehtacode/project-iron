# Infinigen throughput measurement, and the Day-11 correction

**Status:** measured (coarse-stage solving), extrapolated (full scene),
proxy-measured (raytracing). No full generation was run to produce this —
per the Day-12 objective, the instruction was to measure, extrapolate, and
recommend, not to run the full pipeline.

## The Day-11 overclaim, and what it should have said

Day 11 concluded: *"no renderer available to us can score
appearance-learned capabilities in a decision-quality session budget on
this hardware."* That conflates two different claims:

1. **What Day 11 actually measured**: generation did not complete within
   a ~21-minute session budget on this box.
2. **What Day 11's sentence claimed**: Infinigen cannot produce a passing
   sample, ever, on this hardware.

(1) is true and measured. (2) does not follow from (1) — a renderer that
needs an hour is not a renderer that cannot score appearance. The
`Infinigen-Indoors-depth-candidate` registry entry and this report are
corrected accordingly: the depth validity gate on this candidate is
**UNMEASURED**, not **REFUSED**. No sample has ever reached the gate, so
there is no gate evidence in either direction. The corrected status is
**blocked on generation throughput**, which is a compute-and-time
problem with a knowable cost — not a capability verdict.

## What was actually measured: per-stage coarse-solving time

Source: `docs/day11/infinigen_render_stall.log`, the same run Day 11
already produced and killed. No new generation was run to get this table
— it is a re-read of evidence already on disk.

| stage | room | wall-clock |
| --- | --- | ---: |
| `solve_rooms` (floorplan) | — | 6.4 s |
| `on_floor_and_wall_0` | bathroom | 120.3 s |
| `on_floor_and_wall_1` | bedroom | 134.8 s |
| `on_floor_and_wall_2` | dining-room | skipped (no objects) |
| `on_floor_and_wall_3` | kitchen | 228.1 s |
| `on_floor_and_wall_4` | living-room | 246.2 s |
| `on_floor_freestanding_0` | bathroom | skipped (no objects) |
| `on_floor_freestanding_1` | bedroom | 67.5 s |
| `on_floor_freestanding_2` | dining-room | 228.5 s |
| `on_floor_freestanding_3` | kitchen | killed at 237.1 s, iteration 9/100 (incomplete) |

**Mean completed-stage duration: 170.9 s (2.85 min/stage)**, across 6
stages that ran to completion (2 stages were instant no-ops — the solver
found nothing to place). Coarse-stage placement is per-room,
per-placement-category (wall-adjacent vs. freestanding, at minimum), and
this 5-room scene had at least 10 non-trivial stage slots, of which the
observed run reached 7 before the 21m35s kill — under complete coverage of
even the coarse task.

**Bottleneck identified: constraint-satisfaction search (simulated
annealing over object placement), not raytracing.** Every stage above is
CPU-bound Python/numpy optimization — placing furniture subject to spatial
and semantic constraints — and none of it touches the GPU. The log
confirms Metal GPU acceleration is available and would be used
(`Cycles will use use_device_type='METAL'`), but the run never reached the
`render` task where that matters. The 21m35s spent on Day 11 bought
partial coarse-stage furniture placement and zero rendered pixels.

## Extrapolation: one scene, and the sample the gates need

Coarse-only estimate for ONE scene, at the measured 170.9 s/stage rate: a
5-room scene with an estimated 10-14 non-trivial coarse stages implies
**29-40 minutes for coarse alone**, before `populate`, `fine_terrain`, or
`render` run at all. This is an extrapolation from a partial run, not a
completed measurement, and it should be read as a floor: variance across
stages was large (67.5 s to 246.2 s), and rooms not yet reached in this
scene could be slower or faster.

`populate`, `fine_terrain`, and `render` remain **entirely unmeasured** —
Day 11 never reached them, and this report does not manufacture numbers
for them from nothing. What can be said: a direct, bounded, isolated probe
of raytracing throughput (below) gives a per-frame lower bound for the
`render` sub-stage specifically, on hardware that would use Metal GPU
acceleration for it.

### Raytracing throughput, isolated — measured

`scripts/probe_cycles_throughput.py` renders a minimal bpy scene (single
cube, one sun light, Cycles engine, Metal GPU device) at 320x240 to measure
pure raytracing cost, separate from Infinigen's scene composition:

| samples | per-frame seconds (3 renders) | mean (steady-state) |
| ---: | --- | ---: |
| 4 | 3898.35, 0.666, 0.547 | **0.61 s** (first render excluded — see below) |
| 64 | 0.919, 0.893, 0.911 | **0.91 s** |

The first render at each sample count is a one-time cost, not a per-frame
one: 3898 seconds on the very first call is Cycles compiling its Metal
shaders and loading its denoising kernels for the first time in the
process — logged explicitly ("Loading denoising kernels (may take a few
minutes the first time)"). Every render after that first one, at both 4
and 64 samples, lands in **well under one second per frame**.

**This is the headline finding of the throughput probe: once warmed up,
Metal-GPU raytracing on this box is roughly 200-300x faster than the
CPU-bound coarse-stage constraint solving measured above** (0.5-0.9 s/frame
vs. 170.9 s/stage). Sample count barely matters at this scene's triangle
count (4 vs. 64 samples: 0.61s vs. 0.91s) — the fixed per-frame overhead
(BVH build, driver dispatch) dominates over per-sample raytracing cost at
this polygon budget.

**Caveat, stated plainly**: a bare cube is not a furnished room. Cycles
cost scales with polygon count, material complexity, and light-bounce
depth, and a real Infinigen interior has orders of magnitude more geometry
and shading complexity than one cube. This number is a **floor**, not a
prediction of real render-stage cost — but the gap between it and the
measured solving-stage cost (170.9 s/stage) is wide enough that even a
furnished scene's render pass is unlikely to be the bottleneck. The
constraint-solving stage remains the dominant cost by a wide margin on the
evidence gathered today.

Reproduce with:
`.venv-infinigen/bin/python scripts/probe_cycles_throughput.py --samples 4 64`

### The gates' statistical requirement

The depth validity gate (`src/data/validity.py::gate_depth`) needs
enough frames that rank correlation and dynamic-range statistics are not
single-digit — the same bar Day 10 and Day 11 applied to v2/v3-indoor,
which scored on 30 golden-set clips of ~40 frames each (1,200+ frames).
Matching that scale from Infinigen at the measured coarse-stage rate
alone (ignoring populate/fine_terrain/render, which is optimistic) implies
multiple scene-hours of CPU-bound solving before a single frame is even
queued for raytracing.

## Decision memo: overnight local vs. one day of rented GPU

| option | wall-clock | what it unblocks | cost |
| --- | --- | --- | --- |
| **Overnight local** (this Mac, unattended, ~10 hrs) | Coarse-stage rate implies roughly 15-20 scenes if coarse dominates at the measured 30-40 min/scene rate — before `populate`/`fine_terrain`/`render` are even counted. Realistic yield: **unknown, likely 3-8 complete scenes** once the unmeasured stages are included. | Possibly enough for the depth gate's per-image statistics (rank correlation, dynamic range) if 3-8 scenes yield enough frames; NOT enough for the appearance gate's per-region material-diversity statistic, which wants many distinct rooms/materials. | $0 marginal; ties up this machine for a session, competing with every other Day-12+ CPU-bound eval (this report itself hit that contention live — see the probe caveat above). |
| **One day of rented GPU** (e.g., a cloud instance with a discrete GPU Infinigen/Cycles can target — this Mac's Metal path does not generalize to a rented Linux box, which would need CUDA/OptiX) | Constraint solving is CPU-bound regardless of GPU, so a rented GPU box does not shorten the measured bottleneck (coarse-stage annealing) — it only speeds the render sub-stage this report could not measure at scale. Net effect on wall-clock is **unknown without also measuring CPU throughput on the rented instance**, which may simply have more/faster cores rather than a categorically different bottleneck. | Same matrix cells as local, faster if the rented instance's CPU is meaningfully faster than this Mac's; the GPU itself mainly buys back render-stage time, which was not this run's bottleneck. | Real dollar cost, unknown until an instance is chosen; do not commit to a specific one without first re-running this same coarse-stage timing probe on candidate hardware — extrapolating from THIS machine's CPU to another vendor's CPU is exactly the kind of unmeasured claim Day 11 got called out for. |

**Recommendation: do not choose yet.** The information that would make this
decision measured rather than guessed is: (a) the actual `populate` /
`fine_terrain` / `render` per-stage costs on a scene that reaches those
stages, and (b) a coarse-stage timing run on whatever GPU instance is
being considered, before renting a full day of it. Day 13's cheapest next
step is a single, patient overnight local run (killed and inspected each
morning rather than session-bounded) to fill gap (a); it costs nothing but
wall-clock this machine is otherwise idle for.

## The upside this unblocks that no dataset purchase can

Independent of the depth question: a **passing** Infinigen sample would be
licence-clean synthetic data (BSD-3 generator, no third-party assets, no
SMPL trap — see the per-asset clearance already recorded in
`configs/datasets.yaml`) usable for **re-ID training**, not just eval. No
public re-ID dataset in the registry is lane S without a conditional
per-asset caveat (RandPerson, UnrealPerson, ClonedPerson, PersonX all
carry "verify — goal is lane S"), and every real alternative is lane R
(eval-only) or blocked outright (DukeMTMC). If Infinigen's appearance gate
passes once throughput is solved, it is the only currently-visible path to
a synthetic re-ID *training* set this project could legally ship a model
trained on. That upside is independent of today's depth question and is
worth stating plainly: it is a reason to solve the throughput problem, not
just a reason to want the depth number.
