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

## Honesty Clauses

- Report the metric that looks bad. Omitting an unfavorable bucket is falsification.
- A test that can't run (missing weights) SKIPS with a reason — it never fake-passes.
- Per-verb / per-capability precision floors gate production: a detector below its floor is
  disabled by policy, not shipped with hope.
- If you cannot measure a claim yet, say so in the PR and file the eval task — do not soften
  the claim's wording as a substitute for measuring it.
