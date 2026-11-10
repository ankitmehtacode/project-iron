"""Score one run against SmartSpaces ground truth. Three numbers, each with
its baseline, per camera and pooled:

1. World-position RMSE by camera distance bucket, on frames where a track
   box matches a GT person box (IoU >= 0.5). Pipeline = estimator config A;
   baseline = the raw floor projection of the same boxes (no estimator).
2. ID switches (CLEAR-MOT counting on per-frame Hungarian matches at IoU
   >= 0.5, without CLEAR-MOT's keep-previous-match refinement). Pipeline =
   two-stage IoU/Kalman tracker; baseline = IoU-only tracker
   (:data:`~src.pipeline.tracker.IOU_ONLY`) re-run on the same cached
   detections.
3. Event precision/recall against events the same compiler derives from
   each GT person's floor trajectory over the frames that person is visible
   on that camera. A predicted event matches a GT event with the same verb
   and zone, within ``EVENT_TOLERANCE_S``, whose subject is the GT person
   that track matched most often; one-to-one, greedy by time gap. Baseline =
   events compiled from the raw projections (no estimator).

Every number here is lane S synthetic (NVIDIA SmartSpaces), calibration
given: not evidence about real cameras.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from src.events.schema import Verb
from src.pipeline.events import (
    CompiledEvent,
    EventRules,
    Trajectory,
    Zone,
    ZoneConfig,
    compile_trajectory,
)
from src.pipeline.run import (
    Detections,
    TrackPoint,
    frame_ts_ns,
    read_points,
    track_camera,
    trajectories,
)
from src.pipeline.tracker import IOU_ONLY, iou_matrix
from src.pipeline.truth import CameraTruth, load_truth, person_truth

FloatArray = npt.NDArray[np.float64]

MATCH_IOU = 0.5
DISTANCE_EDGES_M = (0.0, 10.0, 20.0, 30.0, float("inf"))
EVENT_TOLERANCE_S = 2.0
LABEL = (
    "lane S synthetic (NVIDIA PhysicalAI-SmartSpaces), calibration given; "
    "not evidence about real cameras or real crowding"
)


@dataclass
class MotCounts:
    gt: int = 0
    matches: int = 0
    misses: int = 0
    false_positives: int = 0
    id_switches: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def _hypotheses_by_frame(
    frames: Sequence[int], track_ids: Sequence[int], boxes: FloatArray
) -> dict[int, tuple[list[int], FloatArray]]:
    out: dict[int, tuple[list[int], list[FloatArray]]] = {}
    for f, t, b in zip(frames, track_ids, boxes):
        ids, bs = out.setdefault(int(f), ([], []))
        ids.append(int(t))
        bs.append(b)
    return {f: (ids, np.array(bs).reshape(-1, 4)) for f, (ids, bs) in out.items()}


def match_frames(
    truth: CameraTruth,
    hyp_frames: Sequence[int],
    hyp_tracks: Sequence[int],
    hyp_boxes: FloatArray,
    n_frames: int,
) -> tuple[MotCounts, list[tuple[int, int, int, int]]]:
    """Per-frame Hungarian matching. Returns the counts and the matches as
    ``(frame, gt_row, track_id, hyp_row)`` with ``hyp_row`` indexing the
    hypothesis arrays."""
    counts = MotCounts()
    hyps = _hypotheses_by_frame(hyp_frames, hyp_tracks, hyp_boxes)
    hyp_row: dict[tuple[int, int], int] = {
        (int(f), int(t)): i for i, (f, t) in enumerate(zip(hyp_frames, hyp_tracks))
    }
    last_track: dict[int, int] = {}
    matches: list[tuple[int, int, int, int]] = []
    gt_starts = np.searchsorted(truth.frame, np.arange(n_frames), side="left")
    gt_ends = np.searchsorted(truth.frame, np.arange(n_frames), side="right")
    for f in range(n_frames):
        g0, g1 = int(gt_starts[f]), int(gt_ends[f])
        ids, boxes = hyps.get(f, ([], np.zeros((0, 4))))
        counts.gt += g1 - g0
        if g1 == g0:
            counts.false_positives += len(ids)
            continue
        iou = iou_matrix(truth.box_xyxy[g0:g1], boxes)
        paired: list[tuple[int, int]] = []
        if iou.size:
            r, c = linear_sum_assignment(1.0 - iou)
            paired = [(int(i), int(j)) for i, j in zip(r, c) if iou[i, j] >= MATCH_IOU]
        counts.matches += len(paired)
        counts.misses += (g1 - g0) - len(paired)
        counts.false_positives += len(ids) - len(paired)
        for i, j in paired:
            gid = int(truth.object_id[g0 + i])
            tid = ids[j]
            if gid in last_track and last_track[gid] != tid:
                counts.id_switches += 1
            last_track[gid] = tid
            matches.append((f, g0 + i, tid, hyp_row[(f, tid)]))
    return counts, matches


def _bucket(distance: float) -> str:
    for lo, hi in zip(DISTANCE_EDGES_M, DISTANCE_EDGES_M[1:]):
        if lo <= distance < hi:
            return f"[{lo:g},{hi:g})"
    raise ValueError(distance)


def _rmse(errors: list[float]) -> float | None:
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else None


def _error_row(e: dict[str, list[float]]) -> dict[str, float | int | None]:
    """RMSE (the reported number) and the median beside it: RMSE is
    dominated by occluded-feet boxes whose bottom edge is not the floor."""
    row: dict[str, float | int | None] = {k: _rmse(v) for k, v in e.items()}
    for k, v in e.items():
        row[f"{k}_p50"] = float(np.median(v)) if v else None
    row["n"] = len(e["estimator"])
    return row


def position_errors(
    truth: CameraTruth,
    points: Sequence[TrackPoint],
    matches: Sequence[tuple[int, int, int, int]],
    camera_xy: FloatArray,
) -> dict[str, dict[str, list[float]]]:
    """``bucket -> {"estimator": [...], "raw_projection": [...]}`` errors (m)."""
    out: dict[str, dict[str, list[float]]] = {}
    for _, gt_row, _, hyp_row in matches:
        p = points[hyp_row]
        g = truth.floor_xy[gt_row]
        bucket = out.setdefault(
            _bucket(float(np.linalg.norm(g - camera_xy))),
            {"estimator": [], "raw_projection": []},
        )
        bucket["estimator"].append(float(np.linalg.norm(np.array(p.est_xy) - g)))
        bucket["raw_projection"].append(float(np.linalg.norm(np.array(p.raw_xy) - g)))
    return out


def gt_trajectories(truth: CameraTruth, camera_id: str, fps: float) -> list[Trajectory]:
    out = []
    for gid in np.unique(truth.object_id):
        rows = np.flatnonzero(truth.object_id == gid)
        out.append(
            Trajectory(
                subject_id=f"gt/{int(gid)}",
                ts_ns=np.array([frame_ts_ns(int(f), fps) for f in truth.frame[rows]]),
                xy=truth.floor_xy[rows],
                frame_refs=tuple(
                    f"{camera_id}/frame-{int(f)}" for f in truth.frame[rows]
                ),
                score=1.0,
            )
        )
    return out


def match_events(
    predicted: Sequence[CompiledEvent],
    truth: Sequence[CompiledEvent],
    subject_of: dict[str, str],
) -> dict[str, Any]:
    """Greedy one-to-one by time gap; see the module docstring."""
    tol = int(EVENT_TOLERANCE_S * 1e9)
    candidates = []
    for i, p in enumerate(predicted):
        for j, g in enumerate(truth):
            gap = abs(p.ts_ns - g.ts_ns)
            if (
                p.verb == g.verb
                and p.zone == g.zone
                and gap <= tol
                and subject_of.get(p.subject_id) == g.subject_id
            ):
                candidates.append((gap, i, j))
    used_p: set[int] = set()
    used_g: set[int] = set()
    for _, i, j in sorted(candidates):
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    by_verb = {}
    for verb in (Verb.ENTERED, Verb.EXITED, Verb.DWELLED):
        np_ = sum(1 for i, p in enumerate(predicted) if p.verb == verb)
        ng = sum(1 for j, g in enumerate(truth) if g.verb == verb)
        tp = sum(1 for i in used_p if predicted[i].verb == verb)
        by_verb[verb.value] = {"predicted": np_, "gt": ng, "matched": tp}
    return {
        "predicted": len(predicted),
        "gt": len(truth),
        "matched": len(used_p),
        "by_verb": by_verb,
    }


def _pr(block: dict[str, Any]) -> dict[str, float | None]:
    p, g, m = block["predicted"], block["gt"], block["matched"]
    return {
        "precision": m / p if p else None,
        "recall": m / g if g else None,
    }


def evaluate_run(run_dir: Path, gt_path: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    identity = manifest["identity"]
    zones = ZoneConfig(
        site_id=identity["zones"]["site_id"],
        zones=tuple(
            Zone(
                name=z["name"],
                polygon=tuple((float(x), float(y)) for x, y in z["polygon"]),
            )
            for z in identity["zones"]["zones"]
        ),
        rules=EventRules(**identity["zones"]["rules"]),
    )
    person_label = identity["person_label"]
    gt = load_truth(gt_path)
    all_points = read_points(run_dir / "tracks.parquet")
    result: dict[str, Any] = {
        "label": LABEL,
        "run": str(run_dir),
        "manifest_sha": manifest["manifest_sha"],
        "detector": manifest["detector"],
        "ground_truth_sha256": gt.source_sha256,
        "match_iou": MATCH_IOU,
        "event_tolerance_s": EVENT_TOLERANCE_S,
        "cameras": {},
    }
    pooled_err: dict[str, dict[str, list[float]]] = {}
    pooled_mot = {"pipeline": MotCounts(), "iou_only_baseline": MotCounts()}
    pooled_events = {"estimator": [0, 0, 0], "raw_projection": [0, 0, 0]}
    for camera_id, cam in manifest["cameras"].items():
        detections = Detections.load(run_dir / camera_id / "detections.npz")
        truth = person_truth(gt, camera_id, detections.n_frames)
        points = [p for p in all_points if p.camera_id == camera_id]
        counts, matches = match_frames(
            truth,
            [p.frame_index for p in points],
            [p.track_id for p in points],
            np.array([p.box_xyxy for p in points]).reshape(-1, 4),
            detections.n_frames,
        )
        baseline = track_camera(detections, person_label, IOU_ONLY)
        base_counts, _ = match_frames(
            truth,
            [a.frame_index for a in baseline],
            [a.track_id for a in baseline],
            np.array([a.box_xyxy for a in baseline]).reshape(-1, 4),
            detections.n_frames,
        )
        errors = position_errors(
            truth, points, matches, np.array(cam["camera_xyz_m"][:2])
        )
        majority: dict[str, str] = {}
        votes: dict[int, Counter[int]] = {}
        for _, gt_row, tid, _ in matches:
            votes.setdefault(tid, Counter())[int(truth.object_id[gt_row])] += 1
        for tid, c in votes.items():
            majority[f"{camera_id}/t{tid}"] = f"gt/{c.most_common(1)[0][0]}"
        gt_events = [
            e
            for t in gt_trajectories(truth, camera_id, detections.fps)
            for e in compile_trajectory(t, zones.zones, zones.rules)
        ]
        event_blocks = {}
        for name, use_estimate in (("estimator", True), ("raw_projection", False)):
            predicted = [
                e
                for t in trajectories(points, use_estimate=use_estimate)
                for e in compile_trajectory(t, zones.zones, zones.rules)
            ]
            block = match_events(predicted, gt_events, majority)
            block.update(_pr(block))
            event_blocks[name] = block
            pooled_events[name][0] += block["predicted"]
            pooled_events[name][1] += block["gt"]
            pooled_events[name][2] += block["matched"]
        for b, e in errors.items():
            slot = pooled_err.setdefault(b, {"estimator": [], "raw_projection": []})
            for k in slot:
                slot[k].extend(e[k])
        for key, mot in (("pipeline", counts), ("iou_only_baseline", base_counts)):
            for field_name, value in mot.as_dict().items():
                setattr(
                    pooled_mot[key],
                    field_name,
                    getattr(pooled_mot[key], field_name) + value,
                )
        result["cameras"][camera_id] = {
            "frames": detections.n_frames,
            "gt_person_ids": int(np.unique(truth.object_id).size),
            "mot": {
                "pipeline": counts.as_dict(),
                "iou_only_baseline": base_counts.as_dict(),
            },
            "rmse_m_by_distance": {b: _error_row(e) for b, e in sorted(errors.items())},
            "events": event_blocks,
        }
    result["pooled"] = {
        "rmse_m_by_distance": {
            b: _error_row(e)
            for b, e in sorted(
                pooled_err.items(), key=lambda kv: float(kv[0][1:].split(",")[0])
            )
        },
        "mot": {k: v.as_dict() for k, v in pooled_mot.items()},
        "events": {
            k: {
                "predicted": p,
                "gt": g,
                "matched": m,
                "precision": m / p if p else None,
                "recall": m / g if g else None,
            }
            for k, (p, g, m) in pooled_events.items()
        },
    }
    return result
