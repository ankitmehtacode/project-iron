"""Iron command line.

    python -m iron run --video V [--video V ...] --calibration C --out DIR
    python -m iron eval --run DIR --ground-truth GT
    python -m iron inspect --run DIR

``run`` is the product path: video in, tracks and events out (see
:mod:`src.pipeline.run`). ``eval`` scores a run against SmartSpaces ground
truth. ``inspect`` serves the Inspector over a run's outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZONES = ROOT / "configs" / "zones" / "warehouse_022.yaml"

PERSON_LABEL = {"oracle": "Person", "rtdetr_v2_s": "person"}
"""The tracked class in each backend's vocabulary (SmartSpaces / COCO)."""


def _camera_id(video: Path, explicit: str | None) -> str:
    return explicit or video.stem


def _cmd_run(args: argparse.Namespace) -> int:
    from src.pipeline.events import load_zone_config
    from src.pipeline.floor import load_projector
    from src.pipeline.run import CameraInput, run

    if args.camera and len(args.video) != 1:
        print("--camera names the camera of a single --video", file=sys.stderr)
        return 2
    cameras = [
        CameraInput(
            camera_id=_camera_id(v, args.camera),
            video=v,
            projector=load_projector(args.calibration, _camera_id(v, args.camera)),
        )
        for v in args.video
    ]
    zones = load_zone_config(args.zones)

    if args.detector == "oracle":
        if args.ground_truth is None:
            print("--detector oracle needs --ground-truth", file=sys.stderr)
            return 2
        from src.detect.backends import OracleDetector
        from src.pipeline.truth import load_truth, oracle_boxes, vocabulary

        gt = load_truth(args.ground_truth)

        def factory(camera: CameraInput) -> Any:
            return OracleDetector(oracle_boxes(gt, camera.camera_id), vocabulary(gt))

    else:
        from src.detect.backends import PretrainedDetector

        shared: list[Any] = []

        def factory(camera: CameraInput) -> Any:
            if not shared:
                shared.append(
                    PretrainedDetector("RT-DETRv2-S-COCO", "rtdetr_v2_r18vd", 0.05)
                )
            return shared[0]

    started = time.perf_counter()
    manifest = run(
        cameras,
        factory,
        PERSON_LABEL[args.detector],
        zones,
        args.out,
        max_seconds=args.seconds or None,
        gate=args.gate == "default",
        extra_manifest={"zones_file": str(args.zones)},
    )
    print(
        f"wrote {args.out}: {manifest['events']} events "
        f"{manifest['events_by_verb']}, {time.perf_counter() - started:.0f}s"
    )
    print(f"view: python -m iron inspect --run {args.out}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from src.runstamp import write_run_stamped
    from src.pipeline.evaluate import evaluate_run

    result = evaluate_run(args.run, args.ground_truth)
    path = write_run_stamped(
        args.run / "eval", "eval", ".json", json.dumps(result, indent=2)
    )
    pooled = result["pooled"]
    print(result["label"])
    print(f"detector: {result['detector']}")
    print(
        "world-position RMSE (m) by camera distance: estimator | raw projection"
        " | n   (median: estimator | raw)"
    )
    for bucket, row in pooled["rmse_m_by_distance"].items():
        est, raw = row["estimator"], row["raw_projection"]
        print(
            f"  {bucket:>10}  {est:.3f} | {raw:.3f} | {row['n']}   "
            f"({row['estimator_p50']:.3f} | {row['raw_projection_p50']:.3f})"
        )
    ours, base = pooled["mot"]["pipeline"], pooled["mot"]["iou_only_baseline"]
    print(
        f"ID switches: pipeline {ours['id_switches']} | IoU-only "
        f"{base['id_switches']}  (GT boxes {ours['gt']}, "
        f"matched {ours['matches']} | {base['matches']})"
    )
    for name, block in pooled["events"].items():
        p = "n/a" if block["precision"] is None else f"{block['precision']:.3f}"
        r = "n/a" if block["recall"] is None else f"{block['recall']:.3f}"
        print(
            f"events ({name}): precision {p} recall {r} "
            f"(predicted {block['predicted']}, GT {block['gt']}, "
            f"matched {block['matched']})"
        )
    print(f"wrote {path}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    from src.inspector.server import main as inspector_main

    return inspector_main(["--run", str(args.run), "--port", str(args.port)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m iron", description=__doc__.split("\n")[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="video in, tracks and events out")
    p_run.add_argument("--video", type=Path, action="append", required=True)
    p_run.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="SmartSpaces calibration.json or a four-point floor file",
    )
    p_run.add_argument("--out", type=Path, required=True)
    p_run.add_argument("--camera", help="camera id (default: the video's file stem)")
    p_run.add_argument("--zones", type=Path, default=DEFAULT_ZONES)
    p_run.add_argument(
        "--detector", choices=sorted(PERSON_LABEL), default="rtdetr_v2_s"
    )
    p_run.add_argument(
        "--ground-truth", type=Path, help="ground_truth.json; the oracle backend only"
    )
    p_run.add_argument(
        "--gate",
        choices=("default", "off"),
        default="default",
        help="off = detector on every frame (the gate-cost reference)",
    )
    p_run.add_argument(
        "--seconds",
        type=float,
        default=60.0,
        help="process the first N seconds (0 = whole video)",
    )
    p_run.set_defaults(func=_cmd_run)

    p_eval = sub.add_parser("eval", help="score a run against SmartSpaces ground truth")
    p_eval.add_argument("--run", type=Path, required=True)
    p_eval.add_argument("--ground-truth", type=Path, required=True)
    p_eval.set_defaults(func=_cmd_eval)

    p_inspect = sub.add_parser("inspect", help="serve the Inspector over a run")
    p_inspect.add_argument("--run", type=Path, required=True)
    p_inspect.add_argument("--port", type=int, default=8899)
    p_inspect.set_defaults(func=_cmd_inspect)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
