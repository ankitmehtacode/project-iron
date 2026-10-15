# ADR 0013 — Detector fine-tuning on lane-S SmartSpaces

- **Status:** Proposed
- **Date:** 2026-10-15 (Day 42)
- **Decides for:** whether the detector stage's pretrained weights may be
  fine-tuned on PhysicalAI-SmartSpaces (lane S), under which split, selection
  and promotion rules, and what result counts as an improvement
- **Related:** `src/detect/` (contract and backends, Day 42 B3);
  `src/data/coco_export.py` and `scripts/smartspaces_to_coco.py` (split
  rules, B4); `scripts/eval_detection_zero_shot.py` and
  `artifacts/smartspaces/day42_detection_zero_shot.json` (the zero-shot
  baseline, B5); `configs/datasets.yaml` entries `PhysicalAI-SmartSpaces`,
  `RT-DETRv2-S-COCO`, `YOLOX-S-COCO`; ADR 0012 (the same lane-S question for
  re-ID adapters); `docs/blocker_ledger.yaml` entries
  `adr-0013-detector-fine-tuning-decision`,
  `detector-base-training-data-licence`,
  `smartspaces-nvidia-terms-of-service`, `smartspaces-smpl-provenance`

> **Claude Code must not change the status line.** Acceptance is a human
> decision, recorded by the human editing `Status:` to `Accepted` (or
> `Rejected`).

## Context

Day 42 built the detector stage and measured it once, zero-shot: a
COCO-trained RT-DETRv2-S (licence signed by the human, Apache-2.0) on
SmartSpaces frames, against an empty predictor and a motion-blob baseline,
at 1080p, 720p and 360p. Two gaps are structural, not tuning:

1. **Vocabulary.** COCO has no forklift, pallet truck, AMR or humanoid
   robot class (Day-42 H8). Six of SmartSpaces' seven classes cannot be
   detected by name at all, whatever the weights' quality.
2. **Domain.** Warehouse viewpoints (high, oblique, far) and rendered
   appearance differ from COCO photographs.

Fine-tuning on lane-S data could address both. It also carries every risk
this repository has recorded for SmartSpaces: unresolved NVIDIA terms, SMPL
provenance of the person renders, Cosmos Transfer re-renders that share
their source's trajectories, and a base model whose own training images
(COCO, Flickr-sourced) carry licences nobody here has read.

## Decision (proposed)

Detector fine-tuning on lane-S SmartSpaces is **allowed in principle**,
under all of the following:

1. **Re-renders are never training data.** Enforced in
   `src/data/coco_export.py`: a re-render cannot be placed in train, a
   re-render and its source are never split apart, and a re-render of a
   train scene is excluded from the export entirely.
2. **Selection uses SmartSpaces val only.** Hyperparameters, checkpoint
   choice, score thresholds and candidate choice are made on the
   SmartSpaces val split. CHIRLA is report-only: read once, after
   selection, never fed back (and today it holds no full frames at all —
   Day-42 H6).
3. **Warehouse classes need fine-tuning or an open-vocabulary route.**
   Neither is assumed to work; each is a measured experiment against the
   zero-shot baseline.
4. **Promotion stays blocked** while any of these is open: NVIDIA's terms
   clause (`smartspaces-nvidia-terms-of-service`), SMPL provenance
   (`smartspaces-smpl-provenance`), and the base model's training-data
   licence (`detector-base-training-data-licence`; enforced by
   `DatasetRegistry.require_shippable`, Day 42). A fine-tuned checkpoint's
   manifest records all three and refuses `promoted` while any is open.
5. **Fine-tuning starts only after the zero-shot baseline exists** — it does
   (`artifacts/smartspaces/day42_detection_zero_shot.json`) — and only under
   the stop rule below, fixed now, before any training run.

## Stop rule (written before the first training run)

A fine-tuned checkpoint counts as an improvement only if ALL hold, measured
by `scripts/eval_detection_zero_shot.py`'s scoring (pycocotools, height
buckets, transient frames excluded) on the SmartSpaces **val** split,
source stratum, 1080p:

1. Person AP@[.5:.95] exceeds the zero-shot value by **>= 0.05 absolute**,
   and a 95% bootstrap interval over val **images** (1,000 resamples) for
   that difference excludes 0.
2. Person recall@IoU0.5 does not fall by more than **0.02** in any height
   bucket holding >= 30 instances, at 1080p or at 720p.
3. For each warehouse class the run claims to add: class recall@IoU0.5
   **>= 0.50** over buckets holding >= 30 instances, where zero-shot has
   no class at all.
4. The val set is the one exported on Day 42 (export sha `66c683ffa66f`) or
   a later export with a recorded rule; it is never re-drawn after seeing a
   fine-tuned result.

Anything less is reported as no improvement. The rule may be changed only by
a new ADR revision written before the run it would govern.

## Consequences

- Day 43 can fine-tune without re-litigating the split, selection or
  promotion rules; it must still state its training budget in advance.
- A fine-tuned detector trained here can be evaluated and reported, but
  cannot ship until the three blockers in decision 4 are resolved by humans.
- The val split is thin (72 images, 6 cameras, 3 scenes; four classes under
  30 instances). Decision 3's classes may be unmeasurable on it; that is
  reported, not worked around by moving data between splits.

## Alternatives considered

- **Fine-tune on everything, including re-renders.** Rejected: re-renders
  are their source's trajectories, so any split that admits them leaks.
- **Use CHIRLA for selection.** Rejected: lane R, eval-only, with an
  unresolved deployment-scope restriction (ADR 0012's reasoning applies).
- **Open-vocabulary detector only, no fine-tuning.** Not rejected; it is
  one of the two routes in decision 3 and needs its own licence check
  under the model registry before any weights are fetched.
