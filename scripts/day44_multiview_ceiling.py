"""Day 44: the multi-camera ceiling on oracle boxes (Warehouse_020, 000).

A CEILING: views are paired by ground-truth identity (no cross-camera
association is built), and the boxes are the ground truth's own.

Per GT person and frame, each camera that sees the person contributes its
range interval as a floor segment: from the head point (box top ray on
z = PERSON_MAX_HEIGHT_M, range R_head) to the foot point (box bottom ray on
the floor, range R_foot). Position estimates:

- single view: that camera's fallback anchor (src/pipeline/anchor.py);
  with several views, the one with the narrowest interval ("best single").
- naive mean: the mean of the views' fallback anchors.
- interval fusion: the point minimising the summed squared distance to all
  the views' segments (averaged projections, 200 iterations from the mean
  of the segment midpoints; the objective is convex). With one view this
  is any point on its segment, so one-view rows use the fallback anchor.

Reported by number of views (1, 2, 3+): median / p90 / p99 error against
the GT floor point, per estimator. Events: one trajectory per GT person over
every frame any of cameras 0-3 sees them, positions from each estimator,
compiled with the scene's zones, matched to the events compiled from the
same frames' GT floor points (subject = GT identity; tolerance as the
evaluator's).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from day44_range_interval import person_boxes  # noqa: E402
from src.pipeline.anchor import (  # noqa: E402
    PERSON_MAX_HEIGHT_M,
    anchor_boxes,
    camera_from_smartspaces,
    range_interval,
)
from src.pipeline.evaluate import match_events  # noqa: E402
from src.pipeline.events import (  # noqa: E402
    Trajectory,
    compile_trajectory,
    load_zone_config,
)
from src.pipeline.run import frame_ts_ns  # noqa: E402
from src.pipeline.truth import load_truth  # noqa: E402

CAMERAS = ("Camera_0000", "Camera_0001", "Camera_0002", "Camera_0003")
ESTIMATORS = ("best_single_view", "naive_mean", "interval_fusion")


def project_to_segment(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(
        ((x - a) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-12), 0, 1
    )
    out: np.ndarray = a + t[:, None] * ab
    return out


def fuse(a: np.ndarray, b: np.ndarray, iters: int = 200) -> np.ndarray:
    x = ((a + b) / 2).mean(axis=0, keepdims=True)
    for _ in range(iters):
        x = project_to_segment(np.repeat(x, len(a), axis=0), a, b).mean(
            axis=0, keepdims=True
        )
    out: np.ndarray = x[0]
    return out


def stats(err: np.ndarray) -> dict[str, float | int]:
    if err.size == 0:
        return {"n": 0}
    return {
        "n": int(err.size),
        "median_m": float(np.median(err)),
        "p90_m": float(np.percentile(err, 90)),
        "p99_m": float(np.percentile(err, 99)),
    }


def scene(scene_dir: Path, zones_path: Path, n: int) -> dict[str, Any]:
    gt = load_truth(scene_dir / "ground_truth.json")
    views: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for cam in CAMERAS:
        camera = camera_from_smartspaces(scene_dir / "calibration.json", cam)
        b = person_boxes(gt, cam, n)
        a = anchor_boxes(camera, b["box"], ["Person"] * len(b["box"]), "fallback")
        ri = range_interval(camera, b["box"], PERSON_MAX_HEIGHT_M)
        for i in range(len(b["box"])):
            if a.anchor[i] == "none" or not (
                np.isfinite(ri.r_head[i]) and np.isfinite(ri.r_foot[i])
            ):
                continue
            views.setdefault((int(b["object_id"][i]), int(b["frame"][i])), []).append(
                {
                    "anchor_xy": a.xy[i],
                    "head_xy": ri.head_xy[i],
                    "foot_xy": ri.foot_xy[i],
                    "width": float(ri.width_m[i]),
                    "truth": b["floor"][i, :2],
                }
            )
    err: dict[str, dict[str, list[float]]] = {
        e: {"1": [], "2": [], "3+": []} for e in ESTIMATORS
    }
    traj: dict[str, dict[int, list[tuple[int, np.ndarray]]]] = {
        e: {} for e in ESTIMATORS + ("truth",)
    }
    for (gid, f), vs in sorted(views.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        k = "1" if len(vs) == 1 else "2" if len(vs) == 2 else "3+"
        truth = vs[0]["truth"]
        best = min(vs, key=lambda v: v["width"])["anchor_xy"]
        mean = np.mean([v["anchor_xy"] for v in vs], axis=0)
        fused = (
            vs[0]["anchor_xy"]
            if len(vs) == 1
            else fuse(
                np.array([v["head_xy"] for v in vs]),
                np.array([v["foot_xy"] for v in vs]),
            )
        )
        for name, xy in (
            ("best_single_view", best),
            ("naive_mean", mean),
            ("interval_fusion", fused),
        ):
            err[name][k].append(float(np.linalg.norm(xy - truth)))
            traj[name].setdefault(gid, []).append((f, xy))
        traj["truth"].setdefault(gid, []).append((f, truth))
    zones = load_zone_config(zones_path)

    def compile_all(name: str) -> list[Any]:
        out = []
        for gid, pts in traj[name].items():
            pts.sort(key=lambda p: p[0])
            t = Trajectory(
                subject_id=f"gt/{gid}",
                ts_ns=np.array([frame_ts_ns(f, 30.0) for f, _ in pts], dtype=np.int64),
                xy=np.array([xy for _, xy in pts]),
                frame_refs=tuple(f"scene/frame-{f}" for f, _ in pts),
                score=1.0,
            )
            out += compile_trajectory(t, zones.zones, zones.rules)
        return out

    gt_events = compile_all("truth")
    identity = {f"gt/{g}": f"gt/{g}" for g in traj["truth"]}
    events = {}
    for name in ESTIMATORS:
        m = match_events(compile_all(name), gt_events, identity)
        events[name] = {
            "predicted": m["predicted"],
            "gt": m["gt"],
            "matched": m["matched"],
            "precision": m["matched"] / m["predicted"] if m["predicted"] else None,
            "recall": m["matched"] / m["gt"] if m["gt"] else None,
        }
    counts = {
        k: sum(
            1
            for vs in views.values()
            if (len(vs) == 1 and k == "1")
            or (len(vs) == 2 and k == "2")
            or (len(vs) >= 3 and k == "3+")
        )
        for k in ("1", "2", "3+")
    }
    return {
        "person_frames_by_views": counts,
        "position_error": {
            e: {k: stats(np.array(v)) for k, v in by.items()} for e, by in err.items()
        },
        "events_scene_level": events,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene", action="append", required=True, help="SCENE_DIR=ZONES")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    n = int(round(args.seconds * 30))
    out: dict[str, Any] = {
        "label": (
            "CEILING: lane S synthetic, oracle person boxes, views paired by "
            "ground-truth identity; not a cross-camera association result"
        ),
        "frames_per_camera": n,
        "cameras": list(CAMERAS),
        "scenes": {},
    }
    for spec in args.scene:
        sd, _, zones = spec.partition("=")
        r = scene(Path(sd), Path(zones), n)
        out["scenes"][Path(sd).name] = r
        print(Path(sd).name, r["person_frames_by_views"])
        for e, by in r["position_error"].items():
            print(
                f"  {e:17}",
                {
                    k: (v.get("median_m"), v.get("p90_m"), v.get("p99_m"))
                    for k, v in by.items()
                },
            )
        print("  events", r["events_scene_level"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
