"""The thin end-to-end slice: video in, tracks and events out.

Per camera::

    video -> MotionGate -> DetectorStage(Detector) -> person boxes
          -> Tracker (two-stage IoU/Kalman) -> FloorProjector (bottom-centre)
          -> estimator config A per track -> event compiler -> events

The detector runs once on every frame through the per-frame detection cache
(:mod:`src.pipeline.detcache`); the gate decides which of those frames the
tracker sees. That is exact, not an approximation: the gate's decision is a
function of pixels alone (motion fraction and hysteresis), and the detector
is stateless per frame, so masking all-frame detections with the gate's wake
decisions gives what a gated detector would have produced. Gate modes:

- ``default``: frames the motion gate is awake on.
- ``off``: every frame (the reference that measures what gating costs).
- ``keepalive``: the gate, plus every frame on which any track is alive.

The detections the tracker saw are written to ``<out>/<camera>/detections.npz``
so evaluation can re-run the tracker variants (the IoU-only baseline).

A run refuses a dirty working tree unless ``allow_dirty``: a run is evidence
only if the commit it names is the code that ran.

Output layout (``--out DIR``; the run refuses a DIR that already holds a
manifest, it never overwrites a run)::

    DIR/manifest.json            what ran, on what, with which shas
    DIR/events.parquet           every camera's events, schema v2
    DIR/tracks.parquet           one row per (track, detector run)
    DIR/<camera>/detections.npz  detector output on the frames the tracker saw
    DIR/<camera>/cascade.jsonl   gate and detector wake/latency accounting

Scope stated: per-camera only. Track ids are per camera; the same person seen
by two cameras is two subjects. Cross-camera association is not built.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from src.cascade.motion import MotionGate, MotionGateConfig
from src.cascade.stage import StageAccounting, StageContext
from src.contracts.frames import AffineTransform, FrameGeometry
from src.detect.contract import Detector, FrameInput
from src.estimator.filter import run_single_entity_filter
from src.estimator.measurement_model import measurement_model_for
from src.estimator.motion_model import motion_model_for
from src.model.episode import StateGraph
from src.model.events import write_events_v2_parquet
from src.model.frame_of_reference import FrameOfReference
from src.model.measurement import WorldPositionMeasurement
from src.model.observation import Observation
from src.model.ulid import generate_ulid
from src.pipeline.anchor import Camera, anchor_boxes
from src.model.uncertainty import Uncertainty
from src.pipeline.detcache import CacheKey, DetectionCache, FrameRecord
from src.pipeline.events import Trajectory, ZoneConfig, compile_events
from src.pipeline.floor import FloorProjector
from src.pipeline.tracker import Assignment, Tracker, TrackerConfig

FloatArray = npt.NDArray[np.float64]

ESTIMATOR_CONFIG = "A"
"""ADR 0010: single constant-velocity model, velocity floor disabled."""

GATE_MODES = ("default", "off", "keepalive")

CODE_PATHS = ("src/", "iron/", "configs/")
"""Untracked files here can change what runs; elsewhere they cannot."""


class DirtyTreeError(RuntimeError):
    """The working tree differs from the commit a run would name."""


@dataclass(frozen=True)
class CameraInput:
    camera_id: str
    video: Path
    projector: FloorProjector
    anchor_camera: Camera | None = None
    """The full pinhole camera when the calibration gives one (SmartSpaces):
    lets every track point record its anchor and the foot anchor's implied
    height check (:mod:`src.pipeline.anchor`). ``None`` for a floor-points
    calibration, where only the foot anchor exists."""


@dataclass(frozen=True)
class Detections:
    """What the detector reported on one camera, gate-aware."""

    camera_id: str
    fps: float
    n_frames: int
    width: int
    height: int
    run_frames: npt.NDArray[np.int64]
    """Frames the detector actually ran on (the gate was awake)."""
    frame: npt.NDArray[np.int64]
    boxes: FloatArray
    scores: FloatArray
    labels: npt.NDArray[np.str_]
    producer: tuple[str, str, str]
    backend: str

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            **{k: np.asarray(v) for k, v in asdict(self).items()},
        )

    @classmethod
    def load(cls, path: Path) -> "Detections":
        with np.load(path, allow_pickle=False) as d:
            return cls(
                camera_id=str(d["camera_id"]),
                fps=float(d["fps"]),
                n_frames=int(d["n_frames"]),
                width=int(d["width"]),
                height=int(d["height"]),
                run_frames=d["run_frames"].astype(np.int64),
                frame=d["frame"].astype(np.int64),
                boxes=d["boxes"].astype(np.float64).reshape(-1, 4),
                scores=d["scores"].astype(np.float64),
                labels=d["labels"].astype(str),
                producer=tuple(str(s) for s in d["producer"]),  # type: ignore[arg-type]
                backend=str(d["backend"]),
            )

    def person_runs(
        self, person_label: str
    ) -> list[tuple[int, FloatArray, FloatArray]]:
        """``(frame, boxes, scores)`` for every detector run, person class only.
        A run with no person is present with empty arrays: it is a run."""
        keep = self.labels == person_label
        out = []
        starts = np.searchsorted(self.frame, self.run_frames, side="left")
        ends = np.searchsorted(self.frame, self.run_frames, side="right")
        for f, s, e in zip(self.run_frames, starts, ends):
            k = keep[s:e]
            out.append((int(f), self.boxes[s:e][k], self.scores[s:e][k]))
        return out


def frame_ts_ns(frame_index: int, fps: float) -> int:
    """Nanoseconds from the start of the video. The source carries no wall
    clock, so ``ts_ns`` is video time, stated in the manifest."""
    return int(round(frame_index * 1e9 / fps))


def detect_camera(
    camera: CameraInput,
    detector: Detector,
    video_sha256: str,
    cache_root: Path,
    max_seconds: float | None,
    stats_path: Path,
    gate_mode: str = "default",
    person_label: str = "",
    tracker_config: TrackerConfig | None = None,
    progress: Callable[[str], None] = print,
) -> tuple[Detections, dict[str, Any]]:
    """Decode the video; run the gate on every frame and the detector on
    every frame (through the cache); keep the frames ``gate_mode`` selects.
    Returns the detections and the cache/accounting summary."""
    import cv2

    if gate_mode not in GATE_MODES:
        raise ValueError(f"gate_mode {gate_mode!r} not in {GATE_MODES}")
    if not detector.stateless:
        raise ValueError(
            f"{detector.name} keeps state across frames; the per-frame cache "
            "and the gate-as-mask both assume a stateless detector"
        )
    capture = cv2.VideoCapture(str(camera.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {camera.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not fps > 0:
        raise RuntimeError(f"{camera.video} reports no frame rate")
    max_frames = int(round(max_seconds * fps)) if max_seconds else 2**62
    if max_frames < 1:
        raise ValueError(f"max_seconds={max_seconds} is under one frame at {fps} fps")
    ok, image = capture.read()
    if not ok:
        raise RuntimeError(f"{camera.video} decoded no frames")
    height, width = image.shape[:2]
    producer = detector.producer
    key = CacheKey(
        video_sha256=video_sha256,
        model_sha=producer.model_sha,
        preprocess_sha=producer.preprocess_sha,
        input_hw=detector.input_hw,
    )
    labels_vocab = detector.vocabulary.labels
    gate = MotionGate(MotionGateConfig()) if gate_mode != "off" else None
    shadow = Tracker(tracker_config) if gate_mode == "keepalive" else None
    gate_acc = StageAccounting("motion_gate")
    det_acc = StageAccounting("detector")
    geometry = FrameGeometry(width=width, height=height)
    run_frames: list[int] = []
    frames: list[npt.NDArray[np.int64]] = []
    boxes: list[FloatArray] = []
    scores: list[FloatArray] = []
    labels: list[str] = []
    hits = misses = 0
    index = 0
    started = time.perf_counter()
    with DetectionCache.open(
        cache_root, key, producer.config_sha, labels_vocab, (height, width)
    ) as cache:
        resumed_from = len(cache)
        while index < max_frames:
            if index > 0:
                ok, image = capture.read()
                if not ok:
                    break
            ctx = StageContext(frame_index=index, ts_ns=frame_ts_ns(index, fps))
            awake = True
            if gate is not None:
                with gate_acc.time_wake():
                    output = gate.process(image[None], ctx)
                awake = gate.should_wake_next(output)
                gate_acc.observe(woke=True, woke_next=awake)
            record = cache.get(index)
            if record is None:
                with det_acc.time_wake():
                    found = detector.detect(
                        FrameInput(
                            image_bgr=image,
                            ts_ns=ctx.ts_ns,
                            frame_index=index,
                            geometry=geometry,
                        )
                    )
                det_acc.observe(woke=True, woke_next=True)
                record = FrameRecord(
                    boxes_xyxy=np.asarray(found.boxes_xyxy, np.float64).reshape(-1, 4),
                    scores=found.score_array(),
                    class_ids=np.asarray(found.class_ids, np.int64),
                )
                cache.append(index, record)
                misses += 1
            else:
                hits += 1
            if shadow is not None and not awake and shadow.live_tracks > 0:
                awake = True
            if awake:
                frame_labels = [labels_vocab[int(c)] for c in record.class_ids]
                run_frames.append(index)
                frames.append(np.full(len(frame_labels), index, dtype=np.int64))
                boxes.append(record.boxes_xyxy)
                scores.append(record.scores)
                labels.extend(frame_labels)
                if shadow is not None:
                    keep = np.array([lb == person_label for lb in frame_labels], bool)
                    shadow.step(index, record.boxes_xyxy[keep], record.scores[keep])
            index += 1
            if index % 60 == 0:
                progress(
                    f"  {camera.camera_id}: frame {index}, kept {len(run_frames)}, "
                    f"detected {misses}, cached {hits}, "
                    f"{time.perf_counter() - started:.0f}s"
                )
    capture.release()
    summary = {
        "cache_file": str(cache.path),
        "cache_key": key.as_dict(),
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_resumed_from_frame": resumed_from,
        "gate_mode": gate_mode,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(
            {
                **summary,
                "frames": index,
                "kept_frames": len(run_frames),
                "stages": [
                    acc.snapshot().as_dict()
                    for acc in ((gate_acc,) if gate is not None else ()) + (det_acc,)
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    return (
        Detections(
            camera_id=camera.camera_id,
            fps=fps,
            n_frames=index,
            width=width,
            height=height,
            run_frames=np.array(run_frames, dtype=np.int64),
            frame=np.concatenate(frames) if frames else np.zeros(0, np.int64),
            boxes=np.concatenate(boxes) if boxes else np.zeros((0, 4)),
            scores=np.concatenate(scores) if scores else np.zeros(0),
            labels=np.array(labels, dtype=str),
            producer=producer.as_tuple(),
            backend=detector.name,
        ),
        summary,
    )


def track_camera(
    detections: Detections, person_label: str, config: TrackerConfig
) -> list[Assignment]:
    tracker = Tracker(config)
    for frame, boxes, scores in detections.person_runs(person_label):
        tracker.step(frame, boxes, scores)
    return tracker.finish()


@dataclass(frozen=True)
class TrackPoint:
    """One track at one detector run: the box, its raw floor projection and
    the estimator's filtered position."""

    camera_id: str
    track_id: int
    frame_index: int
    ts_ns: int
    box_xyxy: tuple[float, float, float, float]
    score: float
    raw_xy: tuple[float, float]
    est_xy: tuple[float, float]
    est_sigma_m: float
    """sqrt of the larger floor-axis posterior variance."""
    est_v: tuple[float, float] | None = None
    """The filter's floor velocity (m/s); ``None`` in runs written before it
    was recorded."""
    anchor: str = "foot_unchecked"
    """Which point of the box was put on the floor (:data:`src.pipeline.
    anchor.ANCHORS`). ``"foot"`` only when the implied-height check ran and
    passed; without a full camera (floor-points calibration), and in runs
    written before Day 43, it is ``"foot_unchecked"``."""


def estimate_tracks(
    assignments: Sequence[Assignment],
    projector: FloorProjector,
    fps: float,
    frame: FrameGeometry,
    manifest_sha: str,
    anchor_camera: Camera | None = None,
    anchor_mode: str = "foot",
    label: str = "person",
) -> tuple[list[TrackPoint], int]:
    """Project each assignment to the floor and run estimator config A per
    track. Returns the points and how many assignments could not be projected
    (no anchor's ray meets its plane in front of the camera).

    Without ``anchor_camera`` every box is put on the floor at its
    bottom-centre by ``projector`` and recorded as ``"foot_unchecked"``
    (no camera, no implied-height check). With it, the
    anchor module decides and records the anchor; ``anchor_mode="foot"``
    gives positions identical to ``projector`` (checked, Day 43) and only
    adds the record, ``"fallback"`` uses the head anchor where the foot
    anchor's implied height is implausible for ``label``."""
    motion = motion_model_for("person", velocity_covariance_floor=False)
    measurement = measurement_model_for(sensor=projector.camera_id)
    origin = np.array(projector.camera_xyz_m)
    reference = FrameOfReference(
        geometry=frame,
        to_canonical=AffineTransform.identity(),
        twin_rev=projector.twin_rev,
    )
    by_track: dict[int, list[Assignment]] = {}
    for a in assignments:
        by_track.setdefault(a.track_id, []).append(a)
    points: list[TrackPoint] = []
    unprojectable = 0
    for track_id, items in sorted(by_track.items()):
        boxes = np.array([a.box_xyxy for a in items])
        if anchor_camera is None:
            xy, valid = projector.project_boxes(boxes)
            anchors = np.full(len(items), "foot_unchecked")
        else:
            anchored = anchor_boxes(
                anchor_camera, boxes, [label] * len(items), mode=anchor_mode
            )
            xy, anchors = anchored.xy, anchored.anchor
            valid = anchors != "none"
        unprojectable += int((~valid).sum())
        kept = [(a, p) for a, p, v in zip(items, xy, valid) if v]
        kept_anchors = [str(k) for k, v in zip(anchors, valid) if v]
        if not kept:
            continue
        observations = []
        for a, p in kept:
            ts = frame_ts_ns(a.frame_index, fps)
            distance = float(np.linalg.norm(np.r_[p, 0.0] - origin))
            observations.append(
                Observation(
                    observation_id=generate_ulid(now_ns=max(ts, 1)),
                    sensor_id=projector.camera_id,
                    ts_ns=ts,
                    frame_ref=f"{projector.camera_id}/frame-{a.frame_index}",
                    measurement=WorldPositionMeasurement(
                        x_m=float(p[0]), y_m=float(p[1]), z_m=0.0
                    ),
                    uncertainty=Uncertainty(
                        kind="gaussian_3d",
                        params=(("sigma_m", float(measurement.sigma_m(distance))),),
                    ),
                    frame_of_reference=reference,
                    producer_shas=(manifest_sha,),
                    envelope_status=(
                        "within_envelope"
                        if measurement.is_within_envelope(distance)
                        else "outside_envelope"
                    ),
                )
            )
        graph = StateGraph()
        run_single_entity_filter(
            graph,
            observations,
            motion,
            measurement,
            manifest_sha=manifest_sha,
            sensor_origin_m=(float(origin[0]), float(origin[1]), float(origin[2])),
        )
        estimates = []
        for f in graph.factors_as_of(graph.graph_rev):
            payload = graph.payload_for(f.factor_id)
            if payload is None:
                raise RuntimeError(
                    f"estimator factor {f.factor_id} carries no estimate"
                )
            estimates.append(payload.estimate)
        for (a, p), est, anchor in zip(kept, estimates, kept_anchors):
            mean, cov = est.mean_array(), est.cov_array()
            points.append(
                TrackPoint(
                    camera_id=projector.camera_id,
                    track_id=track_id,
                    frame_index=a.frame_index,
                    ts_ns=est.ts_ns,
                    box_xyxy=a.box_xyxy,
                    score=a.score,
                    raw_xy=(float(p[0]), float(p[1])),
                    est_xy=(float(mean[0]), float(mean[1])),
                    est_sigma_m=float(np.sqrt(max(cov[0, 0], cov[1, 1]))),
                    est_v=(float(mean[3]), float(mean[4])),
                    anchor=anchor,
                )
            )
    return points, unprojectable


def trajectories(
    points: Sequence[TrackPoint], use_estimate: bool = True
) -> list[Trajectory]:
    """Per-track floor trajectories for the event compiler. ``use_estimate=
    False`` gives the raw-projection baseline."""
    by_track: dict[tuple[str, int], list[TrackPoint]] = {}
    for p in points:
        by_track.setdefault((p.camera_id, p.track_id), []).append(p)
    out = []
    for (camera, track_id), items in sorted(by_track.items()):
        items.sort(key=lambda p: p.ts_ns)
        out.append(
            Trajectory(
                subject_id=f"{camera}/t{track_id}",
                ts_ns=np.array([p.ts_ns for p in items], dtype=np.int64),
                xy=np.array([p.est_xy if use_estimate else p.raw_xy for p in items]),
                frame_refs=tuple(f"{camera}/frame-{p.frame_index}" for p in items),
                score=float(np.mean([p.score for p in items])),
            )
        )
    return out


def points_table(points: Sequence[TrackPoint]) -> pa.Table:
    return pa.table(
        {
            "camera_id": [p.camera_id for p in points],
            "track_id": pa.array([p.track_id for p in points], pa.int64()),
            "frame_index": pa.array([p.frame_index for p in points], pa.int64()),
            "ts_ns": pa.array([p.ts_ns for p in points], pa.int64()),
            "x0": [p.box_xyxy[0] for p in points],
            "y0": [p.box_xyxy[1] for p in points],
            "x1": [p.box_xyxy[2] for p in points],
            "y1": [p.box_xyxy[3] for p in points],
            "score": [p.score for p in points],
            "raw_x": [p.raw_xy[0] for p in points],
            "raw_y": [p.raw_xy[1] for p in points],
            "est_x": [p.est_xy[0] for p in points],
            "est_y": [p.est_xy[1] for p in points],
            "est_sigma_m": [p.est_sigma_m for p in points],
            "est_vx": [p.est_v[0] if p.est_v else None for p in points],
            "est_vy": [p.est_v[1] if p.est_v else None for p in points],
            "anchor": [p.anchor for p in points],
        }
    )


def read_points(path: Path) -> list[TrackPoint]:
    rows = pq.read_table(path).to_pylist()
    return [
        TrackPoint(
            camera_id=r["camera_id"],
            track_id=r["track_id"],
            frame_index=r["frame_index"],
            ts_ns=r["ts_ns"],
            box_xyxy=(r["x0"], r["y0"], r["x1"], r["y1"]),
            score=r["score"],
            raw_xy=(r["raw_x"], r["raw_y"]),
            est_xy=(r["est_x"], r["est_y"]),
            est_sigma_m=r["est_sigma_m"],
            est_v=(
                (r["est_vx"], r["est_vy"])
                if r.get("est_vx") is not None and r.get("est_vy") is not None
                else None
            ),
            anchor=r.get("anchor") or "foot_unchecked",
        )
        for r in rows
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def dirty_paths(porcelain_z: str, prefix: str = "") -> list[str]:
    """Paths in ``git status --porcelain=v1 -z`` output that make a run's
    commit a lie: any change to a tracked file, and untracked files under
    :data:`CODE_PATHS`. Untracked files elsewhere (logs, scratch) cannot
    change what runs.

    ``-z`` because without it git quotes paths with spaces or non-ASCII
    bytes, and a quoted ``"src/a b.py"`` would not match ``src/``. Paths come
    relative to the repository top level; ``prefix`` (``git rev-parse
    --show-prefix``) is where this project sits inside it."""
    out = []
    fields = porcelain_z.split("\0")
    i = 0
    while i < len(fields):
        entry = fields[i]
        i += 1
        if len(entry) < 4:
            continue
        status, path = entry[:2], entry[3:]
        if status[0] in "RC":
            i += 1  # the rename/copy source follows as its own field
        if status == "??":
            inside = path[len(prefix) :] if path.startswith(prefix) else None
            if inside is None or not inside.startswith(CODE_PATHS):
                continue
        out.append(path)
    return out


def git_state(root: Path) -> dict[str, Any]:
    """``{"commit", "dirty", "dirty_paths"}``. ``dirty`` is ``None`` when git
    cannot answer, which a run treats as dirty: it cannot prove it is clean."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        prefix = git("rev-parse", "--show-prefix").strip()
        status = git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "commit": f"unknown ({exc.__class__.__name__})",
            "dirty": None,
            "dirty_paths": [],
        }
    paths = dirty_paths(status, prefix)
    return {"commit": commit, "dirty": bool(paths), "dirty_paths": paths}


def run(
    cameras: Sequence[CameraInput],
    detector_factory: Callable[[CameraInput], Detector],
    person_label: str,
    zones: ZoneConfig,
    out_dir: Path,
    max_seconds: float | None,
    tracker_config: TrackerConfig | None = None,
    progress: Callable[[str], None] = print,
    extra_manifest: dict[str, Any] | None = None,
    gate_mode: str = "default",
    cache_root: Path | None = None,
    allow_dirty: bool = False,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run the slice on every camera and write ``out_dir``. Returns the manifest."""
    if gate_mode not in GATE_MODES:
        raise ValueError(f"gate_mode {gate_mode!r} not in {GATE_MODES}")
    if (out_dir / "manifest.json").exists():
        raise FileExistsError(
            f"{out_dir} already holds a run (manifest.json); runs are never "
            "overwritten, pass a new --out"
        )
    root = repo_root or Path(__file__).resolve().parent.parent.parent
    git = git_state(root)
    if git["dirty"] is not False and not allow_dirty:
        what = (
            "git could not report the tree's state"
            if git["dirty"] is None
            else "uncommitted changes: " + ", ".join(git["dirty_paths"][:10])
        )
        raise DirtyTreeError(
            f"refusing to run on a dirty tree ({what}); commit first, or pass "
            "--allow-dirty and the manifest records git_dirty"
        )
    if cache_root is None:
        from src.config import IronConfig

        cache_root = (
            IronConfig.load().paths.resolved_data_dir / "derived" / "detections"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker_config = tracker_config or TrackerConfig()
    video_shas = {c.camera_id: _sha256_file(c.video) for c in cameras}
    gate_identity: Any = (
        "off"
        if gate_mode == "off"
        else asdict(MotionGateConfig())
        if gate_mode == "default"
        else {"keepalive": "any live track", **asdict(MotionGateConfig())}
    )
    identity = {
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "videos": video_shas,
        "twin_revs": {c.camera_id: c.projector.twin_rev for c in cameras},
        "tracker": asdict(tracker_config),
        "zones": json.loads(json.dumps(asdict(zones))),
        "estimator_config": ESTIMATOR_CONFIG,
        "max_seconds": max_seconds,
        "person_label": person_label,
        "motion_gate": gate_identity,
    }
    manifest_sha = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()

    all_points: list[TrackPoint] = []
    per_camera: dict[str, Any] = {}
    detector_name = ""
    for camera in cameras:
        progress(f"{camera.camera_id}: gate + detector on {camera.video.name}")
        started = time.perf_counter()
        detector = detector_factory(camera)
        detector_name = detector.name
        detections, cache_summary = detect_camera(
            camera,
            detector,
            video_shas[camera.camera_id],
            cache_root,
            max_seconds,
            out_dir / camera.camera_id / "cascade.jsonl",
            gate_mode=gate_mode,
            person_label=person_label,
            tracker_config=tracker_config,
            progress=progress,
        )
        detections.save(out_dir / camera.camera_id / "detections.npz")
        detect_s = time.perf_counter() - started
        assignments = track_camera(detections, person_label, tracker_config)
        points, unprojectable = estimate_tracks(
            assignments,
            camera.projector,
            detections.fps,
            FrameGeometry(detections.width, detections.height),
            manifest_sha,
            anchor_camera=camera.anchor_camera,
            anchor_mode="foot",
            label=person_label,
        )
        all_points.extend(points)
        per_camera[camera.camera_id] = {
            "video": str(camera.video),
            "video_sha256": video_shas[camera.camera_id],
            "fps": detections.fps,
            "frames_decoded": detections.n_frames,
            "detector_runs": int(detections.run_frames.size),
            "person_detections": int((detections.labels == person_label).sum()),
            "tracks": len({p.track_id for p in points}),
            "track_points": len(points),
            "unprojectable_assignments": unprojectable,
            "track_point_anchors": dict(
                sorted(Counter(p.anchor for p in points).items())
            ),
            "calibration": camera.projector.source,
            "twin_rev": camera.projector.twin_rev,
            "camera_xyz_m": list(camera.projector.camera_xyz_m),
            "detector_producer_shas": list(detections.producer),
            "wall_s_gate_and_detector": round(detect_s, 1),
            "detection_cache": cache_summary,
        }
        progress(
            f"{camera.camera_id}: {per_camera[camera.camera_id]['tracks']} tracks, "
            f"detector ran on {detections.run_frames.size}/{detections.n_frames} frames"
        )

    pq.write_table(points_table(all_points), out_dir / "tracks.parquet")
    clip_ids = {c.camera_id: (c.camera_id, video_shas[c.camera_id]) for c in cameras}
    _, events = compile_events(
        trajectories(all_points), zones, manifest_sha=manifest_sha, clip_ids=clip_ids
    )
    write_events_v2_parquet(events, out_dir / "events.parquet")

    manifest = {
        "manifest_sha": manifest_sha,
        "identity": identity,
        "detector": detector_name,
        "git_dirty_paths": git["dirty_paths"],
        "cameras": per_camera,
        "events": len(events),
        "events_by_verb": {
            v: sum(1 for e in events if e.verb.value == v)
            for v in ("entered", "exited", "dwelled")
        },
        "conventions": {
            "ts_ns": (
                "nanoseconds from the start of each video; the source has no "
                "wall clock"
            ),
            "world_frame": "scene floor frame, X_Y_Z_UP, metres, floor at z=0",
            "position": "bottom-centre of the person box projected to the floor",
            "confidence": (
                "mean UNCALIBRATED detector score of the track (1.0 for the "
                "oracle); not a probability"
            ),
            "importance": "0.0: no triage policy applied",
            "measurement_noise": (
                "illustrative-unmeasured envelope (src.estimator."
                "measurement_model); no camera has a measured one"
            ),
            "scope": "per-camera; the same person on two cameras is two subjects",
        },
        **(extra_manifest or {}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
