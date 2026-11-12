"""Day 43: the recall-vs-invocations frontier over detector schedules.

Replays every candidate in :mod:`src.pipeline.frontier` over per-frame
detections that already exist; no detector runs. Two sources:

- ``--detections-from-run RUN``: the detection cache each camera of RUN
  read (its manifest names the file). Warehouse_022 oracle and pretrained.
- ``--oracle``: the scene's ground-truth boxes (what the oracle detector
  returns), for scenes with no run.

The motion gate's per-frame decision comes from ``--gate-from-run`` (the
``run_frames`` of a gate-default run: the gate is a function of pixels only)
or is computed here by decoding the video with the same gate code
``detect_camera`` uses. ``--verify-gate`` does both on the first camera and
refuses to continue if they differ.

Zones: ``--zones FILE``, or ``--auto-zones OUT.yaml`` for a scene with no zone
file. The auto rule, fixed before any pipeline output on those scenes was
computed: 8 m floor cells on a grid anchored at multiples of 8 m; count GT
person floor samples (visible on the chosen cameras, first ``--seconds``);
take cells by count, skipping any cell within one cell (8-neighbourhood) of
one already taken, until three are taken. Event rules as Warehouse_022's.

Usage::

    python scripts/cascade_frontier.py --scene-dir .../val/Warehouse_022 \
        --detections-from-run outputs/runs/day43-7ed3392/oracle_gate-off \
        --gate-from-run outputs/runs/day43-7ed3392/oracle_gate-default \
        --zones configs/zones/warehouse_022.yaml --out artifacts/day43/x.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cascade.motion import MotionGate, MotionGateConfig  # noqa: E402
from src.cascade.stage import StageContext  # noqa: E402
from src.data.smartspaces import SceneGroundTruth  # noqa: E402
from src.pipeline.events import load_zone_config  # noqa: E402
from src.pipeline.floor import load_projector  # noqa: E402
from src.pipeline.frontier import (  # noqa: E402
    CANDIDATES,
    CameraSource,
    FrameDets,
    decide,
    frontier,
    write_json,
)
from src.pipeline.run import Detections, frame_ts_ns  # noqa: E402
from src.pipeline.tracker import TRACKERS  # noqa: E402
from src.pipeline.truth import load_truth, oracle_boxes, person_truth  # noqa: E402

PERSON_LABEL = {"oracle": "Person", "pretrained": "person"}
CELL_M = 8.0


def read_cache(path: Path, n: int) -> list[FrameDets]:
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])["header"]
    labels = header["labels"]
    out = []
    for line in lines[1 : n + 1]:
        r = json.loads(line)
        out.append(
            FrameDets(
                boxes=np.asarray(r["b"], np.float64).reshape(-1, 4),
                scores=np.asarray(r["s"], np.float64),
                labels=tuple(labels[int(c)] for c in r["c"]),
            )
        )
    if len(out) < n:
        raise SystemExit(f"{path} holds {len(out)} frames, need {n}: cache incomplete")
    return out


def oracle_frames(gt: SceneGroundTruth, camera: str, n: int) -> list[FrameDets]:
    truth = oracle_boxes(gt, camera)
    empty = FrameDets(np.zeros((0, 4)), np.zeros(0), ())
    out = []
    for f in range(n):
        b = truth.boxes.get(f)
        if b is None:
            out.append(empty)
            continue
        out.append(FrameDets(b, np.ones(len(b)), tuple(truth.labels[f])))
    return out


def gate_from_video(video: Path, n: int) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    gate = MotionGate(MotionGateConfig())
    awake = np.zeros(n, dtype=bool)
    for f in range(n):
        ok, image = cap.read()
        if not ok:
            raise SystemExit(f"{video}: decoded {f} frames, need {n}")
        output = gate.process(
            image[None], StageContext(frame_index=f, ts_ns=frame_ts_ns(f, fps))
        )
        awake[f] = gate.should_wake_next(output)
    cap.release()
    return awake


def video_props(video: Path) -> tuple[float, int, int]:
    """``(fps, width, height)`` read from the container, not assumed."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not fps > 0 or width <= 0 or height <= 0:
        raise SystemExit(f"{video} reports fps={fps}, size={width}x{height}")
    return fps, width, height


def gate_from_run(run: Path, camera: str, n: int) -> np.ndarray:
    d = Detections.load(run / camera / "detections.npz")
    if d.n_frames < n:
        raise SystemExit(f"{run}/{camera} covers {d.n_frames} frames, need {n}")
    awake = np.zeros(n, dtype=bool)
    awake[d.run_frames[d.run_frames < n]] = True
    return awake


def objects_by_class(
    gt: SceneGroundTruth, camera: str, n: int
) -> dict[str, list[np.ndarray]]:
    sel = np.flatnonzero(gt.box_camera == gt.camera_names.index(camera))
    rows = gt.box_row[sel]
    frames = gt.frame[rows]
    keep = frames < n
    rows, frames = rows[keep], frames[keep]
    out: dict[str, list[np.ndarray]] = {}
    for oid in np.unique(gt.object_id[rows]):
        m = gt.object_id[rows] == oid
        cls = gt.class_names[int(gt.class_index[rows[m][0]])]
        out.setdefault(cls, []).append(np.unique(frames[m]).astype(np.int64))
    return out


def auto_zones(
    gt: SceneGroundTruth, cameras: list[str], n: int, scene: str, out: Path
) -> Path:
    pts = np.concatenate([person_truth(gt, c, n).floor_xy for c in cameras])
    cells = np.floor(pts / CELL_M).astype(np.int64)
    uniq, counts = np.unique(cells, axis=0, return_counts=True)
    chosen: list[np.ndarray] = []
    for i in np.argsort(-counts, kind="stable"):
        c = uniq[i]
        if all(np.max(np.abs(c - k)) > 1 for k in chosen):
            chosen.append(c)
        if len(chosen) == 3:
            break
    zones = []
    for k, c in enumerate(chosen):
        x0, y0 = float(c[0] * CELL_M), float(c[1] * CELL_M)
        zones.append(
            {
                "name": f"cell_{k}",
                "polygon": [
                    [x0, y0],
                    [x0 + CELL_M, y0],
                    [x0 + CELL_M, y0 + CELL_M],
                    [x0, y0 + CELL_M],
                ],
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    text = (
        f"# Zones for SmartSpaces {scene}, world metres (scene floor frame,\n"
        "# X_Y_Z_UP).\n"
        "# GENERATED by scripts/cascade_frontier.py --auto-zones, by the rule stated\n"
        "# there, fixed before any pipeline output on this scene was computed: the\n"
        f"# three densest {CELL_M:g} m cells of GT person floor samples on cameras\n"
        f"# {', '.join(cameras)} over the first {n} frames, no two within one cell.\n"
        "# Not tuned on any pipeline output. Event rules copied from\n"
        "# warehouse_022.yaml.\n"
    )
    body = {
        "site_id": f"smartspaces-{scene.lower().replace('_', '-')}",
        "zones": zones,
        "rules": {"min_presence_s": 0.5, "dwell_s": 5.0, "max_gap_s": 2.0},
    }
    out.write_text(text + yaml.safe_dump(body, sort_keys=False))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument(
        "--cameras", default="Camera_0000,Camera_0001,Camera_0002,Camera_0003"
    )
    p.add_argument("--seconds", type=float, default=60.0)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--detections-from-run", type=Path)
    src.add_argument("--oracle", action="store_true")
    p.add_argument("--detector", choices=sorted(PERSON_LABEL), default="oracle")
    p.add_argument("--gate-from-run", type=Path)
    p.add_argument("--verify-gate", action="store_true")
    z = p.add_mutually_exclusive_group(required=True)
    z.add_argument("--zones", type=Path)
    z.add_argument("--auto-zones", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--tracker",
        choices=sorted(TRACKERS),
        default="iou",
        help="tracker scored and read by the shadow schedules (default: product)",
    )
    args = p.parse_args(argv)

    started = time.perf_counter()
    scene = args.scene_dir.name
    cameras = args.cameras.split(",")
    gt = load_truth(args.scene_dir / "ground_truth.json")
    props = {
        cam: video_props(args.scene_dir / "videos" / f"{cam}.mp4") for cam in cameras
    }
    rates = {round(v[0], 6) for v in props.values()}
    if len(rates) != 1:
        raise SystemExit(f"cameras disagree on frame rate: {props}")
    fps = props[cameras[0]][0]
    n = int(round(args.seconds * fps))
    zones_path = args.zones or auto_zones(gt, cameras, n, scene, args.auto_zones)
    zones = load_zone_config(zones_path)
    label = PERSON_LABEL[args.detector]
    manifest: dict[str, Any] = {}
    if args.detections_from_run:
        manifest = json.loads((args.detections_from_run / "manifest.json").read_text())
    sources = []
    gate_check: dict[str, Any] = {}
    for i, cam in enumerate(cameras):
        video = args.scene_dir / "videos" / f"{cam}.mp4"
        if args.detections_from_run:
            info = manifest["cameras"][cam]
            if abs(float(info["fps"]) - fps) > 1e-6:
                raise SystemExit(f"{cam}: run fps {info['fps']} != video fps {fps}")
            frames = read_cache(Path(info["detection_cache"]["cache_file"]), n)
        else:
            frames = oracle_frames(gt, cam, n)
        if args.gate_from_run:
            awake = gate_from_run(args.gate_from_run, cam, n)
            if args.verify_gate and i == 0:
                computed = gate_from_video(video, n)
                differ = int((computed != awake).sum())
                gate_check = {"camera": cam, "frames": n, "frames_differing": differ}
                print(f"gate check {cam}: {differ} of {n} frames differ", flush=True)
                if differ:
                    raise SystemExit("gate decoded here differs from the run's gate")
        else:
            awake = gate_from_video(video, n)
        sources.append(
            CameraSource(
                camera_id=cam,
                fps=fps,
                width=props[cam][1],
                height=props[cam][2],
                frames=frames,
                gate_awake=awake,
                projector=load_projector(args.scene_dir / "calibration.json", cam),
                truth=person_truth(gt, cam, n),
                objects=objects_by_class(gt, cam, n),
            )
        )
        print(f"{cam}: {n} frames loaded, gate awake {int(awake.sum())}", flush=True)
    tracker = TRACKERS[args.tracker]
    result = frontier(
        sources,
        label,
        zones,
        args.detector,
        CANDIDATES,
        progress=lambda m: print(m, flush=True),
        tracker_config=tracker,
    )
    payload = {
        "tracker": {"name": args.tracker, **dataclasses.asdict(tracker)},
        "label": (
            "lane S synthetic (NVIDIA PhysicalAI-SmartSpaces), calibration "
            "given; not evidence about real cameras"
        ),
        "scene": scene,
        "cameras": cameras,
        "frames_per_camera": n,
        "detector": args.detector,
        "detections_from": str(args.detections_from_run)
        if args.detections_from_run
        else "ground truth (oracle)",
        "gate_from": str(args.gate_from_run)
        if args.gate_from_run
        else "decoded here (MotionGateConfig defaults)",
        "gate_check": gate_check or None,
        "zones_file": str(zones_path),
        "ground_truth_sha256": gt.source_sha256,
        "candidates": result,
        "decision_rule_applied": decide(result),
        "wall_s": round(time.perf_counter() - started, 1),
    }
    write_json(args.out, payload)
    print(f"wrote {args.out} ({payload['wall_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
