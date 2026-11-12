"""Recall-versus-invocations frontier for detector scheduling (Day 43).

Every candidate is a *schedule*: which frames the detector runs on. All of
them are replayed over one per-frame detection source (the detection cache,
or the oracle's ground-truth boxes), so no detector runs here; that is exact
for the same reason the gate-as-mask is (``src.pipeline.run``): the detector
is stateless per frame, so its output on a frame does not depend on which
other frames it ran on.

Candidates, fixed before any number was computed:

- ``gate_off``: every frame.
- ``gate_default``: frames the motion gate is awake on.
- ``gate_keepalive``: the gate, plus every frame on which a shadow tracker
  holds a live track. Mirrors ``detect_camera`` line for line.
- ``stride_6`` / ``stride_15``: frames ``f % k == 0`` (5 fps / 2 fps at 30).
- ``adaptive``: run when ``f - last_run >= 15`` while the shadow tracker has
  no live track, ``>= 6`` while it has one; frame 0 always runs.
- ``motion_adaptive``: ``adaptive``'s cadence, but only on frames the motion
  gate is awake (no motion, no run).

The shadow tracker is ``Tracker(TrackerConfig())`` stepped on the tracked
class of each run, as in ``detect_camera``. Nothing here is tuned: strides
are the ones named in the Day-43 instruction and every threshold is the
pipeline's stated default.

Metrics per candidate, per camera and pooled, from the evaluator's own
functions (``src.pipeline.evaluate``): event precision/recall (estimator),
ID switches for the pipeline tracker and the IoU-only baseline, position
RMSE by distance bucket, and invocations per camera-minute (a count). Event
recall/precision and ID switches carry a 95% percentile bootstrap interval
over *units*: one unit per GT person track (camera, object id), carrying its
GT events, the matched events and the predicted events attributed to it,
plus one unit per predicted track that matched no GT person (its events are
false positives that belong to no GT track). Per-class coverage: the share
of GT objects (any class, per camera) on which the schedule ran the detector
at least ``min_hits`` times while the object was visible, with a Wilson
interval; it depends only on the schedule and the GT, not on the detector's
output.

Lane S synthetic (NVIDIA SmartSpaces), calibration given: not evidence
about real cameras.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from src.contracts.frames import FrameGeometry
from src.eval.intervals import wilson_interval
from src.pipeline.evaluate import (
    _error_row,
    gt_trajectories,
    match_event_pairs,
    match_frames,
    position_errors,
)
from src.pipeline.events import ZoneConfig, compile_trajectory
from src.pipeline.floor import FloorProjector
from src.pipeline.run import Detections, estimate_tracks, track_camera, trajectories
from src.pipeline.tracker import IOU_ONLY, Tracker, TrackerConfig
from src.pipeline.truth import CameraTruth

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

CANDIDATES = (
    "gate_off",
    "gate_default",
    "gate_keepalive",
    "stride_6",
    "stride_15",
    "adaptive",
    "motion_adaptive",
)
ADAPTIVE_IDLE_STRIDE = 15
ADAPTIVE_LIVE_STRIDE = 6
BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class FrameDets:
    """Detector output on one frame, every class."""

    boxes: FloatArray
    scores: FloatArray
    labels: tuple[str, ...]


@dataclass(frozen=True)
class CameraSource:
    """Everything one camera contributes: per-frame detections, the gate's
    per-frame wake decision, the projector and the person ground truth."""

    camera_id: str
    fps: float
    width: int
    height: int
    frames: Sequence[FrameDets]
    gate_awake: BoolArray
    projector: FloorProjector
    truth: CameraTruth
    objects: dict[str, list[npt.NDArray[np.int64]]]
    """class -> one array of visible frame indices per GT object."""


def _person(d: FrameDets, label: str) -> tuple[FloatArray, FloatArray]:
    keep = np.array([lb == label for lb in d.labels], dtype=bool)
    return d.boxes[keep].reshape(-1, 4), d.scores[keep]


def schedule(
    name: str,
    frames: Sequence[FrameDets],
    gate_awake: BoolArray,
    person_label: str,
    tracker_config: TrackerConfig | None = None,
) -> npt.NDArray[np.int64]:
    """The frames candidate ``name`` runs the detector on."""
    n = len(frames)
    if gate_awake.shape != (n,):
        raise ValueError(f"gate mask has shape {gate_awake.shape}, need ({n},)")
    if name == "gate_off":
        return np.arange(n, dtype=np.int64)
    if name == "gate_default":
        return np.flatnonzero(gate_awake).astype(np.int64)
    if name in ("stride_6", "stride_15"):
        k = int(name.split("_")[1])
        return np.arange(0, n, k, dtype=np.int64)
    if name not in ("gate_keepalive", "adaptive", "motion_adaptive"):
        raise ValueError(f"unknown candidate {name!r}; known: {CANDIDATES}")
    shadow = Tracker(tracker_config)
    runs: list[int] = []
    last: int | None = None
    for f in range(n):
        if name == "gate_keepalive":
            awake = bool(gate_awake[f]) or shadow.live_tracks > 0
        else:
            stride = (
                ADAPTIVE_LIVE_STRIDE if shadow.live_tracks > 0 else ADAPTIVE_IDLE_STRIDE
            )
            awake = last is None or f - last >= stride
            if name == "motion_adaptive":
                awake = awake and bool(gate_awake[f])
        if awake:
            runs.append(f)
            last = f
            boxes, scores = _person(frames[f], person_label)
            shadow.step(f, boxes, scores)
    return np.array(runs, dtype=np.int64)


def detections_for(
    source: CameraSource, run_frames: npt.NDArray[np.int64], backend: str
) -> Detections:
    frames: list[npt.NDArray[np.int64]] = []
    boxes: list[FloatArray] = []
    scores: list[FloatArray] = []
    labels: list[str] = []
    for f in run_frames:
        d = source.frames[int(f)]
        frames.append(np.full(len(d.labels), int(f), dtype=np.int64))
        boxes.append(d.boxes.reshape(-1, 4))
        scores.append(d.scores)
        labels.extend(d.labels)
    return Detections(
        camera_id=source.camera_id,
        fps=source.fps,
        n_frames=len(source.frames),
        width=source.width,
        height=source.height,
        run_frames=run_frames,
        frame=np.concatenate(frames) if frames else np.zeros(0, np.int64),
        boxes=np.concatenate(boxes) if boxes else np.zeros((0, 4)),
        scores=np.concatenate(scores) if scores else np.zeros(0),
        labels=np.array(labels, dtype=str),
        producer=("frontier", "frontier", "frontier"),
        backend=backend,
    )


def _switches_by_gt(
    truth: CameraTruth, matches: Sequence[tuple[int, int, int, int]]
) -> Counter[int]:
    """ID switches per GT object, counted exactly as ``match_frames`` does."""
    last: dict[int, int] = {}
    out: Counter[int] = Counter()
    for _, gt_row, tid, _ in matches:
        gid = int(truth.object_id[gt_row])
        if gid in last and last[gid] != tid:
            out[gid] += 1
        last[gid] = tid
    return out


@dataclass
class Unit:
    """One bootstrap unit: a GT person track, or an unmatched predicted track."""

    gt_events: int = 0
    matched: int = 0
    predicted: int = 0
    switches: int = 0
    switches_baseline: int = 0


@dataclass
class CameraResult:
    invocations: int
    minutes: float
    events: dict[str, int]
    mot: dict[str, int]
    mot_baseline: dict[str, int]
    errors: dict[str, dict[str, list[float]]]
    units: list[Unit]
    class_coverage: dict[str, list[int]] = field(default_factory=dict)
    event_log: dict[str, list[list[Any]]] = field(default_factory=dict)
    """``gt``: ``[subject, verb, zone, ts_ns]`` per GT event; ``predicted``:
    ``[track, attributed GT subject or None, verb, zone, ts_ns]``; ``pairs``:
    ``[predicted index, gt index]`` as :func:`match_event_pairs` made them.
    Lets a reader pair candidates event by event and tell a lost event from
    one shifted past the tolerance."""


def evaluate_schedule(
    source: CameraSource,
    run_frames: npt.NDArray[np.int64],
    person_label: str,
    zones: ZoneConfig,
    backend: str,
    tracker_config: TrackerConfig | None = None,
) -> CameraResult:
    tracker_config = tracker_config or TrackerConfig()
    dets = detections_for(source, run_frames, backend)
    assignments = track_camera(dets, person_label, tracker_config)
    points, _ = estimate_tracks(
        assignments,
        source.projector,
        source.fps,
        FrameGeometry(source.width, source.height),
        "frontier",
    )
    n = len(source.frames)
    truth = source.truth
    counts, matches = match_frames(
        truth,
        [p.frame_index for p in points],
        [p.track_id for p in points],
        np.array([p.box_xyxy for p in points]).reshape(-1, 4),
        n,
    )
    baseline = track_camera(dets, person_label, IOU_ONLY)
    base_counts, base_matches = match_frames(
        truth,
        [a.frame_index for a in baseline],
        [a.track_id for a in baseline],
        np.array([a.box_xyxy for a in baseline]).reshape(-1, 4),
        n,
    )
    camera_xy = np.array(source.projector.camera_xyz_m[:2])
    errors = position_errors(truth, points, matches, camera_xy)

    votes: dict[int, Counter[int]] = {}
    for _, gt_row, tid, _ in matches:
        votes.setdefault(tid, Counter())[int(truth.object_id[gt_row])] += 1
    subject_of = {
        f"{source.camera_id}/t{tid}": f"gt/{c.most_common(1)[0][0]}"
        for tid, c in votes.items()
    }
    gt_events = [
        e
        for t in gt_trajectories(truth, source.camera_id, source.fps)
        for e in compile_trajectory(t, zones.zones, zones.rules)
    ]
    predicted = [
        e
        for t in trajectories(points, use_estimate=True)
        for e in compile_trajectory(t, zones.zones, zones.rules)
    ]
    pairs = match_event_pairs(predicted, gt_events, subject_of)

    units: dict[str, Unit] = {
        f"gt/{int(g)}": Unit() for g in np.unique(truth.object_id)
    }
    for g in gt_events:
        units[g.subject_id].gt_events += 1
    for _, j in pairs:
        units[gt_events[j].subject_id].matched += 1
    for p in predicted:
        owner = subject_of.get(p.subject_id)
        if owner is None:
            units.setdefault(f"pred/{p.subject_id}", Unit()).predicted += 1
        else:
            units[owner].predicted += 1
    for gid, k in _switches_by_gt(truth, matches).items():
        units[f"gt/{gid}"].switches += k
    for gid, k in _switches_by_gt(truth, base_matches).items():
        units[f"gt/{gid}"].switches_baseline += k
    if sum(u.switches for u in units.values()) != counts.id_switches:
        raise AssertionError("per-unit ID switches do not sum to match_frames")

    run_set = np.zeros(n, dtype=bool)
    run_set[run_frames] = True
    min_hits = tracker_config.min_hits
    class_coverage = {
        cls: [len(objs), sum(int(run_set[v].sum()) >= min_hits for v in objs)]
        for cls, objs in source.objects.items()
    }
    return CameraResult(
        invocations=int(run_frames.size),
        minutes=n / source.fps / 60.0,
        events={
            "predicted": len(predicted),
            "gt": len(gt_events),
            "matched": len(pairs),
        },
        mot=counts.as_dict(),
        mot_baseline=base_counts.as_dict(),
        errors=errors,
        units=list(units.values()),
        class_coverage=class_coverage,
        event_log={
            "gt": [[g.subject_id, g.verb.value, g.zone, g.ts_ns] for g in gt_events],
            "predicted": [
                [
                    p.subject_id,
                    subject_of.get(p.subject_id),
                    p.verb.value,
                    p.zone,
                    p.ts_ns,
                ]
                for p in predicted
            ],
            "pairs": [[i, j] for i, j in pairs],
        },
    )


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def bootstrap(
    units: Sequence[Unit],
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, list[float | None]]:
    """95% percentile intervals over units for recall, precision and switch
    totals. A resample with no GT event (or no prediction) has no recall (or
    precision) and is left out of that interval; how many were is reported."""
    rng = np.random.default_rng(seed)
    a = np.array(
        [
            [u.gt_events, u.matched, u.predicted, u.switches, u.switches_baseline]
            for u in units
        ],
        dtype=np.float64,
    ).reshape(-1, 5)
    if a.shape[0] == 0:
        return {}
    idx = rng.integers(0, a.shape[0], size=(rounds, a.shape[0]))
    s = a[idx].sum(axis=1)
    out: dict[str, list[float | None]] = {}
    for key, num, den in (("recall", 1, 0), ("precision", 1, 2)):
        ok = s[:, den] > 0
        vals = s[ok, num] / s[ok, den]
        out[key] = (
            [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
            if vals.size
            else [None, None]
        )
        out[f"{key}_undefined_resamples"] = [float(rounds - int(ok.sum()))]
    for key, col in (("id_switches", 3), ("id_switches_baseline", 4)):
        out[key] = [
            float(np.percentile(s[:, col], 2.5)),
            float(np.percentile(s[:, col], 97.5)),
        ]
    return out


def summarise(results: Sequence[CameraResult]) -> dict[str, Any]:
    """Pool camera results (sums of counts, concatenated errors and units)."""
    inv = sum(r.invocations for r in results)
    minutes = sum(r.minutes for r in results)
    ev = {k: sum(r.events[k] for r in results) for k in ("predicted", "gt", "matched")}
    mot = {k: sum(r.mot[k] for r in results) for k in results[0].mot}
    mot_b = {
        k: sum(r.mot_baseline[k] for r in results) for k in results[0].mot_baseline
    }
    errors: dict[str, dict[str, list[float]]] = {}
    for r in results:
        for b, e in r.errors.items():
            slot = errors.setdefault(b, {"estimator": [], "raw_projection": []})
            for k in slot:
                slot[k].extend(e[k])
    cov: dict[str, list[int]] = {}
    for r in results:
        for cls, (total, hit) in r.class_coverage.items():
            counts = cov.setdefault(cls, [0, 0])
            counts[0] += total
            counts[1] += hit
    units = [u for r in results for u in r.units]
    return {
        "invocations": inv,
        "camera_minutes": minutes,
        "invocations_per_camera_minute": inv / minutes if minutes else None,
        "events": {
            **ev,
            "precision": _ratio(ev["matched"], ev["predicted"]),
            "recall": _ratio(ev["matched"], ev["gt"]),
        },
        "mot": {"pipeline": mot, "iou_only_baseline": mot_b},
        "rmse_m_by_distance": {
            b: _error_row(e)
            for b, e in sorted(
                errors.items(), key=lambda kv: float(kv[0][1:].split(",")[0])
            )
        },
        "class_coverage": {
            cls: {
                "objects": total,
                "covered": hit,
                "share": _ratio(hit, total),
                "wilson95": list(wilson_interval(hit, total)) if total else None,
            }
            for cls, (total, hit) in sorted(cov.items())
        },
        "bootstrap95_over_gt_tracks": bootstrap(units),
        "bootstrap_units": len(units),
    }


def frontier(
    sources: Sequence[CameraSource],
    person_label: str,
    zones: ZoneConfig,
    backend: str,
    candidates: Sequence[str] = CANDIDATES,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in candidates:
        per_camera: dict[str, Any] = {}
        results = []
        for s in sources:
            runs = schedule(name, s.frames, s.gate_awake, person_label)
            r = evaluate_schedule(s, runs, person_label, zones, backend)
            results.append(r)
            per_camera[s.camera_id] = {**summarise([r]), "event_log": r.event_log}
        out[name] = {"pooled": summarise(results), "cameras": per_camera}
        p = out[name]["pooled"]
        progress(
            f"  {name}: {p['invocations_per_camera_minute']:.0f} inv/cam-min, "
            f"event recall {p['events']['recall']}, "
            f"precision {p['events']['precision']}"
        )
    return out


def decide(result: dict[str, Any]) -> dict[str, Any]:
    """The Day-43 decision rule, fixed before the pretrained numbers existed:
    the candidate with the fewest invocations whose event-recall interval
    overlaps gate-off's; if none does, fixed stride 6. Gate-off itself is not
    a candidate for the choice (it is the reference)."""
    ref = result["gate_off"]["pooled"]["bootstrap95_over_gt_tracks"]["recall"]
    passing = []
    for name, block in result.items():
        if name == "gate_off":
            continue
        lo, hi = block["pooled"]["bootstrap95_over_gt_tracks"]["recall"]
        if lo is None or hi is None or ref[0] is None or ref[1] is None:
            continue
        if lo <= ref[1] and ref[0] <= hi:
            passing.append((block["pooled"]["invocations"], name))
    if passing:
        _, name = min(passing)
        return {
            "chosen": name,
            "rule": "fewest invocations whose recall interval overlaps gate_off's",
            "passing": [n for _, n in sorted(passing)],
        }
    return {
        "chosen": "stride_6",
        "rule": "no candidate overlapped gate_off: fixed stride 6 default",
        "passing": [],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")


__all__ = [
    "CANDIDATES",
    "CameraSource",
    "FrameDets",
    "bootstrap",
    "decide",
    "evaluate_schedule",
    "frontier",
    "schedule",
    "summarise",
    "write_json",
]
