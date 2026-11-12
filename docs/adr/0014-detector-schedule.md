# ADR 0014 — Detector schedule for the slice (cascade decision)

- **Status:** Proposed
- **Date:** 2026-11-12 (Day 43)
- **Decides for:** which frames the person detector runs on in the Day-43
  slice (motion gate, keep-alive, fixed stride, or adaptive stride), by a
  decision rule fixed before the deciding numbers existed
- **Related:** `src/pipeline/frontier.py` (the candidates and the rule,
  `decide`); `scripts/cascade_frontier.py`; `scripts/day43_frontier_tables.py`;
  `artifacts/day43/frontier_w022_pretrained.json` (the deciding run),
  `frontier_{w022,020,000}_oracle.json`, `frontier_tables_*.md`,
  `stride15_audit_*.json`; runs `outputs/runs/day43-2cf9a30/rtdetr_v2_s_gate-{off,default}`
  (gate-default stamped `1507dc6`; only `scripts/` and `artifacts/` differ)
  and `outputs/runs/day43-7ed3392/oracle_gate-*`; ADR 0013 (the detector);
  `docs/blocker_ledger.yaml` entry `adr-0014-detector-schedule-decision`

> **Claude Code must not change the status line.** Acceptance is a human
> decision, recorded by the human editing `Status:` to `Accepted` (or
> `Rejected`).

## Context

The detector is the slice's only expensive stage: RT-DETRv2-S on this CPU
took p50 0.70–1.03 s and p99 1.01–1.50 s a frame (the run's own
`cascade.jsonl` accounting, Camera_0003 and Camera_0002; other jobs shared
the CPU, so these are not clean latency figures), against a 30 fps source. Every
frame cannot be afforded; the question is which frames to drop and what that
costs in events.

Seven schedules were compared by replaying each over per-frame detections
that already existed (the detection cache; exact because the detector is
stateless per frame). The replay reproduces the official evals exactly for
every schedule that also ran as a real pipeline run (oracle gate-off,
default and keep-alive; pretrained gate-off and default: event counts, both
trackers' ID switches, RMSE per bucket, invocation totals).

## Decision rule (fixed before the pretrained numbers existed)

On pretrained detections, Warehouse_022, 4 cameras x 60 s: choose the
candidate with the fewest detector invocations whose 95% event-recall
interval (percentile bootstrap over GT person tracks, 2000 rounds, seed 0)
overlaps gate-off's. If none does, fixed stride 6 is the default. No
threshold tuned to make a candidate pass.

## Evidence (pretrained RT-DETRv2-S, Warehouse_022, lane S synthetic)

| candidate | invocations (per cam-min) | recall [95%] | precision | paired vs gate-off: gained / lost, Δrecall [95%] |
|---|---|---|---|---|
| gate_off | 7200 (1800) | 65/78 [0.756, 0.911] | 65/87 | reference |
| gate_keepalive | 6022 (1506) | 59/78 [0.686, 0.829] | 59/81 | +0 / −6, −0.077 [−0.154, −0.013] |
| gate_default | 4172 (1043) | 42/78 [0.439, 0.650] | 42/53 | +0 / −23, −0.295 [−0.397, −0.194] |
| stride_6 | 1200 (300) | 60/78 [0.651, 0.873] | 60/72 | +2 / −7, −0.064 [−0.157, +0.012] |
| **adaptive** | **1156 (289)** | **59/78 [0.638, 0.874]** | **59/69** | **+1 / −7, −0.077 [−0.159, +0.000]** |
| motion_adaptive | 705 (176) | 35/78 [0.347, 0.556] | 35/45 | +1 / −31, −0.385 [−0.507, −0.273] |
| stride_15 | 480 (120) | 46/78 [0.471, 0.706] | 46/59 | +2 / −21, −0.244 [−0.381, −0.121] |

Passing the rule: adaptive, stride_6, gate_keepalive. **Chosen by the rule:
`adaptive`** (stride 15 frames while the shadow tracker holds no live track,
stride 6 while it holds one).

## Decision

Run the detector on the `adaptive` schedule.

## What the evidence does and does not support

- **Adaptive and stride 6 are the same schedule on this scene.** Someone is
  in view almost all the time, so adaptive runs at stride 6 almost always:
  1156 vs 1200 invocations (3.7% fewer) and one fewer matched event (59 vs
  60). The rule picks adaptive on that 3.7%; its cost is a stateful
  scheduler (a shadow tracker) where stride 6 needs none. On a scene with
  long empty stretches adaptive's saving would be larger; no such scene was
  measured.
- **Both lose events against gate-off.** Paired, adaptive loses 7 GT events
  gate-off matched and gains 1; the paired interval's upper end is +0.000,
  so a real loss is the likelier reading. The rule is about interval
  overlap, which is weaker than a paired test; that is the rule as fixed,
  recorded rather than re-argued here.
- **The motion gate as built is the worst trade on the frontier.** It
  spends 1043 invocations per camera-minute (3.6x adaptive) and loses 23
  events: on oracle boxes it lost 29 of 76, on all three scenes measured.
  Keep-alive recovers most of that at 1506 per camera-minute.
- **Interval overlap is weak where recall is low.** On the two other scenes
  (oracle only, 60 s each, Warehouse_020 and Warehouse_000) the same rule
  would pick `motion_adaptive`, because their recall intervals are wide
  (gate-off recall 29/45 and 36/54, limited by the floor projection, not the
  schedule; see the Day-43 report). Those scenes are not the deciding data
  and carry no pretrained run.
- **Stride 15 loses events rather than delaying them.** Against gate-off it
  misses 21 (pretrained) to 28 (oracle) GT events on Warehouse_022; of the
  oracle misses 16 are absent, 6 fire within the 2.0 s tolerance on a track
  attributed to another person, and for 6 the same person, zone and verb
  fire only 4.1–31.5 s away, which reads as a different visit more than as
  a late event. Events stride 15 does match are a median 0.28 s late.
- **The intervals are probably too narrow.** The bootstrap resamples GT
  person tracks per camera; four cameras watch the same 60 s of one scene,
  so one person can be up to four units. That makes passing the overlap
  rule harder, not easier, but the "95%" is nominal.
- **Tracker finding, not decided here:** on pretrained detections at every
  frame the two-stage Kalman tracker makes 453 ID switches against the
  IoU-only baseline's 93; at adaptive's cadence 83 against 40. The pipeline
  tracker is worse than its own baseline on real detections at 30 fps.
- Everything above is one synthetic scene (lane S), one CPU, four cameras,
  60 s each. Not evidence about real cameras.

## Consequences

- `src/pipeline/run.py`'s `GATE_MODES` does not yet include `adaptive`;
  the frontier replays it, the product run cannot select it. Wiring it is
  the next engineering step once this ADR is accepted.
- Maintenance cost: one more gate mode, and a scheduler whose cadence
  depends on tracker state — a run's invocation count is no longer a
  function of the video alone. The manifest must record the schedule and
  its strides.
- Revisit when: a scene with long empty stretches is measured (adaptive vs
  stride 6 separates), the tracker's ID-switch regression is fixed (the
  live-track signal adaptive reads comes from that tracker), or a real
  camera exists.

## Day 44 update (appended; the text above is left as written)

Two things changed under this decision: the product tracker is now the
IoU-only tracker (Kalman behind `--tracker kalman`), and the bootstrap is
clustered by GT person (scene-wide object id, summed across the four
cameras; 17 clusters) instead of per-camera tracks. The intervals above are
the per-camera ones; the ones below supersede them. Same cache, same
schedules, same rule (`artifacts/day44/frontier_w022_pretrained_{iou,kalman}.json`,
`frontier_tables_pretrained_{iou,kalman}.md`).

| candidate | invocations (per cam-min) | IoU-only: recall [95%] | IoU-only: paired vs gate-off | Kalman: recall [95%] | Kalman: paired vs gate-off |
|---|---|---|---|---|---|
| gate_off | 7200 (1800) | 58/78 [0.630, 0.836] | reference | 65/78 [0.774, 0.886] | reference |
| gate_keepalive | 6011 / 6022 | 53/78 [0.560, 0.800] | +0/−5, −0.064 [−0.129, +0.000] | 59/78 [0.697, 0.825] | +0/−6, −0.077 [−0.158, +0.000] |
| gate_default | 4172 (1043) | 36/78 [0.371, 0.567] | +1/−23, −0.282 [−0.359, −0.182] | 42/78 [0.465, 0.639] | +0/−23, −0.295 [−0.392, −0.182] |
| stride_6 | 1200 (300) | 55/78 [0.643, 0.773] | +7/−10, −0.038 [−0.110, +0.055] | 60/78 [0.628, 0.861] | +2/−7, −0.064 [−0.170, +0.000] |
| **adaptive** | **1154 / 1156 (288 / 289)** | **56/78 [0.640, 0.793]** | **+8/−10, −0.026 [−0.118, +0.078]** | **59/78 [0.610, 0.854]** | **+1/−7, −0.077 [−0.192, +0.000]** |
| motion_adaptive | 705 (176) | 32/78 [0.337, 0.500] | +4/−30, −0.333 [−0.434, −0.203] | 35/78 [0.356, 0.562] | +1/−31, −0.385 [−0.489, −0.263] |
| stride_15 | 480 (120) | 40/78 [0.389, 0.610] | +4/−22, −0.231 [−0.329, −0.148] | 46/78 [0.458, 0.733] | +2/−21, −0.244 [−0.366, −0.107] |

**The rule chooses `adaptive` under both trackers** (passing under both:
adaptive, stride_6, gate_keepalive). The decision above stands.

What changed in the evidence:

- **With the IoU-only tracker adaptive is indistinguishable from gate-off**
  (paired −0.026 [−0.118, +0.078]); with Kalman it was likelier a loss.
- **The tracker switch costs events at every frame.** Paired, IoU-only minus
  Kalman: gate-off −0.090 [−0.185, −0.010] (58 vs 65 of 78); at adaptive
  −0.038 [−0.146, +0.109], not distinguishable
  (`artifacts/day44/paired_iou_vs_kalman_pretrained.json`). It buys ID
  switches 93 vs 453 at every frame, 40 vs 83 at adaptive, and precision
  58/66 vs 65/87.
- The Kalman tracker's excess switches come from its low-score second stage
  meeting RT-DETR's duplicate boxes, not from the Kalman prediction
  (`artifacts/day44/kalman_switch_diagnosis.json`). A tracker that keeps the
  Kalman prediction without the second stage (92 switches) was measured in
  that ablation but not on events; it is not a candidate here.
- 17 clusters is a small sample: the intervals are wide, and the overlap
  rule passes candidates it cannot tell apart.
