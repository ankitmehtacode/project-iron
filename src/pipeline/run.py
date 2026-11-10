"""The thin end-to-end slice: video in, tracks and events out.

Per camera::

    video -> MotionGate -> DetectorStage(Detector) -> person boxes
          -> Tracker (two-stage IoU/Kalman) -> FloorProjector (bottom-centre)
          -> estimator config A per track -> event compiler -> events

The gate and detector run through :class:`~src.cascade.runner.CascadeRunner`,
so the wake accounting is the product's own. Detections are written to
``<out>/<camera>/detections.npz`` before tracking, so evaluation can re-run
the tracker variants (the IoU-only baseline) without re-running the detector.

Output layout (``--out DIR``; the run refuses a DIR that already holds a
manifest, it never overwrites a run)::

    DIR/manifest.json            what ran, on what, with which shas
    DIR/events.parquet           every camera's events, schema v2
    DIR/tracks.parquet           one row per (track, detector run)
    DIR/<camera>/detections.npz  the detector's raw output, gate-aware
    DIR/<camera>/cascade.jsonl   per-stage wake/latency accounting

Scope stated: per-camera only. Track ids are per camera; the same person seen
by two cameras is two subjects. Cross-camera association is not built.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from src.cascade.motion import MotionGate, MotionGateConfig
from src.cascade.runner import CascadeRunner, JsonlStatsSink
from src.cascade.stage import Stage, StageContext
from src.cascade.stages import DetectionWakePolicy, DetectorStage
from src.contracts.frames import AffineTransform, FrameGeometry
from src.detect.contract import Detector, DetectionSet
from src.estimator.filter import run_single_entity_filter
from src.estimator.measurement_model import measurement_model_for
from src.estimator.motion_model import motion_model_for
from src.model.episode import StateGraph
from src.model.events import write_events_v2_parquet
from src.model.frame_of_reference import FrameOfReference
from src.model.measurement import WorldPositionMeasurement
from src.model.observation import Observation
from src.model.ulid import generate_ulid
from src.model.uncertainty import Uncertainty
from src.pipeline.events import Trajectory, ZoneConfig, compile_events
from src.pipeline.floor import FloorProjector
from src.pipeline.tracker import Assignment, Tracker, TrackerConfig

FloatArray = npt.NDArray[np.float64]

ESTIMATOR_CONFIG = "A"
"""ADR 0010: single constant-velocity model, velocity floor disabled."""


@dataclass(frozen=True)
class CameraInput:
    camera_id: str
    video: Path
    projector: FloorProjector


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
    person_label: str,
    max_seconds: float | None,
    min_score: float,
    stats_path: Path,
    progress: Callable[[str], None] = print,
    gate: bool = True,
) -> Detections:
    """Decode the video and run gate -> detector on every frame. ``gate=
    False`` runs the detector on every frame: the reference that measures
    what the gate's sleeping costs."""
    import cv2

    capture = cv2.VideoCapture(str(camera.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {camera.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not fps > 0:
        raise RuntimeError(f"{camera.video} reports no frame rate")
    max_frames = int(round(max_seconds * fps)) if max_seconds else 2**62
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stage = DetectorStage(
        detector,
        DetectionWakePolicy(classes=frozenset({person_label}), min_score=min_score),
    )
    run_frames: list[int] = []
    frames: list[npt.NDArray[np.int64]] = []
    boxes: list[FloatArray] = []
    scores: list[FloatArray] = []
    labels: list[str] = []
    producer: tuple[str, str, str] | None = None
    width = height = 0
    index = 0
    started = time.perf_counter()
    with open(stats_path, "w") as sink:
        stages: list[Stage] = (
            [MotionGate(MotionGateConfig()), stage] if gate else [stage]
        )
        runner = CascadeRunner(stages, stats_interval_s=60.0, sink=JsonlStatsSink(sink))
        while index < max_frames:
            ok, image = capture.read()
            if not ok:
                break
            height, width = image.shape[:2]
            ctx = StageContext(frame_index=index, ts_ns=frame_ts_ns(index, fps))
            runner.process_frame(image[None], ctx)
            found: DetectionSet | None = ctx.payload.get("detections")
            if found is not None:
                run_frames.append(index)
                producer = found.producer.as_tuple()
                frames.append(np.full(len(found), index, dtype=np.int64))
                boxes.append(found.boxes_xyxy)
                scores.append(found.score_array())
                labels.extend(found.labels())
            index += 1
            if index % 300 == 0:
                progress(
                    f"  {camera.camera_id}: frame {index}, detector ran on "
                    f"{len(run_frames)}, {time.perf_counter() - started:.0f}s"
                )
        runner.flush()
    capture.release()
    if index == 0:
        raise RuntimeError(f"{camera.video} decoded no frames")
    return Detections(
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
        producer=producer or ("none", "none", "none"),
        backend=detector.name,
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


def estimate_tracks(
    assignments: Sequence[Assignment],
    projector: FloorProjector,
    fps: float,
    frame: FrameGeometry,
    manifest_sha: str,
) -> tuple[list[TrackPoint], int]:
    """Project each assignment to the floor and run estimator config A per
    track. Returns the points and how many assignments could not be projected
    (box bottom on or above the horizon)."""
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
        xy, valid = projector.project_boxes(np.array([a.box_xyxy for a in items]))
        unprojectable += int((~valid).sum())
        kept = [(a, p) for a, p, v in zip(items, xy, valid) if v]
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
        for (a, p), est in zip(kept, estimates):
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
        )
        for r in rows
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unknown ({exc.__class__.__name__})"
    return out.stdout.strip()


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
    gate: bool = True,
) -> dict[str, Any]:
    """Run the slice on every camera and write ``out_dir``. Returns the manifest."""
    if (out_dir / "manifest.json").exists():
        raise FileExistsError(
            f"{out_dir} already holds a run (manifest.json); runs are never "
            "overwritten, pass a new --out"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker_config = tracker_config or TrackerConfig()
    root = Path(__file__).resolve().parent.parent.parent
    video_shas = {c.camera_id: _sha256_file(c.video) for c in cameras}
    identity = {
        "git_commit": _git_commit(root),
        "videos": video_shas,
        "twin_revs": {c.camera_id: c.projector.twin_rev for c in cameras},
        "tracker": asdict(tracker_config),
        "zones": json.loads(json.dumps(asdict(zones))),
        "estimator_config": ESTIMATOR_CONFIG,
        "max_seconds": max_seconds,
        "person_label": person_label,
        "motion_gate": asdict(MotionGateConfig()) if gate else "off",
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
        detections = detect_camera(
            camera,
            detector,
            person_label,
            max_seconds,
            tracker_config.low_score,
            out_dir / camera.camera_id / "cascade.jsonl",
            progress,
            gate,
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
            "calibration": camera.projector.source,
            "twin_rev": camera.projector.twin_rev,
            "camera_xyz_m": list(camera.projector.camera_xyz_m),
            "detector_producer_shas": list(detections.producer),
            "wall_s_gate_and_detector": round(detect_s, 1),
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
