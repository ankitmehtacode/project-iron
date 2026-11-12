"""Day 44, time-boxed: why the two-stage Kalman tracker makes 453 ID switches
where the IoU-only tracker makes 93 (RT-DETRv2-S, Warehouse_022, every
frame). Diagnosis only; nothing is fixed.

Two views, both replayed over the run's saved detections (no detector):

1. Ablation, 2 x 2: Kalman prediction on/off x low-score second stage on/off,
   every other TrackerConfig field at its default. ID switches counted by
   ``match_frames`` on the tracker's assignments (boxes before floor
   projection; the evaluator counts on projected track points, which drops
   unprojectable boxes, so totals can differ by a few).
2. Per-switch causes for the Kalman tracker. A switch is a GT person whose
   matched track changes from A to B at frame f. Flags (not exclusive):
   - ``fp_birth``: B's first box matched no GT person at IoU 0.5: born on a
     false positive (or a robot) and later took the person over.
   - ``duplicate``: at f, two or more detections overlap this person's GT box
     at IoU >= 0.5 (the detector boxed the person twice).
   - ``low_score``: B's box at f scored below ``high_score``: it reached B
     through the low-score second stage.
   - ``gating``: A was still live at f, and A's last raw box overlaps B's box
     at f at IoU >= ``stage1_min_iou``: the IoU-only association (last box as
     the prediction) could have kept the person on A; the Kalman prediction
     did not.
   - ``a_ended``: A's last assignment is ``max_missed`` or more detector runs
     before f: A had ended (a re-birth, not a hand-off).
   Primary cause: the first flag that holds in the order above, else
   ``other``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.evaluate import match_frames  # noqa: E402
from src.pipeline.run import Detections, track_camera  # noqa: E402
from src.pipeline.tracker import (  # noqa: E402
    KALMAN_TWO_STAGE,
    TrackerConfig,
    iou_matrix,
)
from src.pipeline.truth import load_truth, person_truth  # noqa: E402

ORDER = ("fp_birth", "duplicate", "low_score", "gating", "a_ended")


def switches(truth: Any, assignments: list[Any], n: int) -> tuple[Any, list[Any]]:
    counts, matches = match_frames(
        truth,
        [a.frame_index for a in assignments],
        [a.track_id for a in assignments],
        np.array([a.box_xyxy for a in assignments]).reshape(-1, 4),
        n,
    )
    return counts, matches


def diagnose(
    truth: Any, dets: Detections, label: str, cfg: TrackerConfig
) -> tuple[Counter[str], Counter[str], int]:
    assignments = track_camera(dets, label, cfg)
    counts, matches = switches(truth, assignments, dets.n_frames)
    by_track: dict[int, list[Any]] = {}
    for a in assignments:
        by_track.setdefault(a.track_id, []).append(a)
    for items in by_track.values():
        items.sort(key=lambda a: a.frame_index)
    run_index = {int(f): i for i, f in enumerate(dets.run_frames)}
    person_runs = dict((f, (b, s)) for f, b, s in dets.person_runs(label))
    gt_by_frame: dict[int, np.ndarray] = {}
    for f, box in zip(truth.frame, truth.box_xyxy):
        gt_by_frame.setdefault(int(f), []).append(box)  # type: ignore[arg-type]
    primary: Counter[str] = Counter()
    flags: Counter[str] = Counter()
    last: dict[int, int] = {}
    for f, gt_row, tid, _ in matches:
        gid = int(truth.object_id[gt_row])
        prev = last.get(gid)
        last[gid] = tid
        if prev is None or prev == tid:
            continue
        b_items = by_track[tid]
        b_now = next(a for a in b_items if a.frame_index == f)
        first = b_items[0]
        gts = np.array(gt_by_frame.get(first.frame_index, [])).reshape(-1, 4)
        fl = set()
        if gts.size == 0 or iou_matrix(np.array([first.box_xyxy]), gts).max() < 0.5:
            fl.add("fp_birth")
        boxes, _ = person_runs.get(f, (np.zeros((0, 4)), None))
        if (
            len(boxes)
            and (iou_matrix(truth.box_xyxy[gt_row][None], boxes) >= 0.5).sum() >= 2
        ):
            fl.add("duplicate")
        if b_now.score < cfg.high_score:
            fl.add("low_score")
        a_before = [a for a in by_track[prev] if a.frame_index < f]
        if a_before:
            a_last = a_before[-1]
            runs_since = run_index[f] - run_index[a_last.frame_index]
            a_live = runs_since < cfg.max_missed
            if a_live and (
                iou_matrix(np.array([a_last.box_xyxy]), np.array([b_now.box_xyxy]))[
                    0, 0
                ]
                >= cfg.stage1_min_iou
            ):
                fl.add("gating")
            if not a_live:
                fl.add("a_ended")
        for k in fl:
            flags[k] += 1
        primary[next((k for k in ORDER if k in fl), "other")] += 1
    return primary, flags, counts.id_switches


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--ground-truth", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    manifest = json.loads((args.run / "manifest.json").read_text())
    label = manifest["identity"]["person_label"]
    gt = load_truth(args.ground_truth)
    cams = list(manifest["cameras"])
    ablation = {}
    for kalman in (False, True):
        for second in (False, True):
            cfg = dataclasses.replace(
                KALMAN_TWO_STAGE, use_kalman=kalman, second_stage=second
            )
            total = 0
            for cam in cams:
                dets = Detections.load(args.run / cam / "detections.npz")
                truth = person_truth(gt, cam, dets.n_frames)
                c, _ = switches(truth, track_camera(dets, label, cfg), dets.n_frames)
                total += c.id_switches
            ablation[f"kalman={kalman},second_stage={second}"] = total
            print(f"kalman={kalman} second_stage={second}: {total} ID switches")
    primary: Counter[str] = Counter()
    flags: Counter[str] = Counter()
    total = 0
    per_camera = {}
    for cam in cams:
        dets = Detections.load(args.run / cam / "detections.npz")
        truth = person_truth(gt, cam, dets.n_frames)
        pr, fl, n = diagnose(truth, dets, label, KALMAN_TWO_STAGE)
        primary.update(pr)
        flags.update(fl)
        total += n
        per_camera[cam] = {"switches": n, "primary": dict(pr), "flags": dict(fl)}
    out = {
        "run": str(args.run),
        "detector": manifest["detector"],
        "counted_on": "tracker assignments (boxes), before floor projection",
        "ablation_id_switches": ablation,
        "kalman_two_stage": {
            "switches": total,
            "primary_cause": dict(primary.most_common()),
            "flags_not_exclusive": dict(flags.most_common()),
            "per_camera": per_camera,
        },
        "cause_order": list(ORDER) + ["other"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["kalman_two_stage"]["primary_cause"]))
    print(json.dumps(out["kalman_two_stage"]["flags_not_exclusive"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
