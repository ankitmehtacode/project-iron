"""Day 43: foot vs head anchor on oracle boxes, three SmartSpaces scenes.

No detector runs: the boxes are the ground truth's own (what the oracle
detector returns). For each scene, cameras 0-3, the first ``--seconds``:

1. Position. Every visible GT box of every class, anchored by
   :func:`src.pipeline.anchor.anchor_boxes` in ``foot`` mode (the slice as
   it was) and in ``fallback`` mode. Error = distance to the object's GT
   floor point (centre of the 3D box's bottom face). Per anchor mode and
   class: how often it fires, median / p90 / p99 error, and the share of
   errors inside the stated 95% ellipse (chi-square, 2 dof).
2. Events. The full slice (tracker, estimator, compiler) on every frame
   (gate off), person class, once per anchor mode, through
   :func:`src.pipeline.frontier.evaluate_schedule`: event precision and
   recall as counts, before (foot) and after (fallback).

Lane S synthetic: not evidence about real cameras.
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

from src.data.smartspaces import MEASURED_GEOMETRY  # noqa: E402
from src.pipeline.anchor import (  # noqa: E402
    HEIGHT_PRIORS,
    PIXEL_SIGMA_PX,
    anchor_boxes,
    camera_from_smartspaces,
    covered95,
)
from src.pipeline.events import load_zone_config  # noqa: E402
from src.pipeline.floor import load_projector  # noqa: E402
from src.pipeline.frontier import CameraSource, evaluate_schedule  # noqa: E402
from src.pipeline.truth import load_truth, person_truth  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from cascade_frontier import objects_by_class, oracle_frames  # noqa: E402

CAMERAS = ("Camera_0000", "Camera_0001", "Camera_0002", "Camera_0003")


def all_class_boxes(gt: Any, camera: str, n: int) -> tuple[np.ndarray, ...]:
    """Every visible GT box on ``camera`` with frame < n: boxes, labels,
    floor xy."""
    sel = np.flatnonzero(gt.box_camera == gt.camera_names.index(camera))
    rows = gt.box_row[sel]
    keep = gt.frame[rows] < n
    sel, rows = sel[keep], rows[keep]
    floor = MEASURED_GEOMETRY.floor_point(gt.location[rows], gt.scale[rows])[:, :2]
    labels = np.array([gt.class_names[int(c)] for c in gt.class_index[rows]])
    return gt.box_xyxy[sel].astype(np.float64), labels, floor


def stats(
    err: np.ndarray, cover: np.ndarray, total: int, ll: np.ndarray
) -> dict[str, Any]:
    """``ll``: per-detection log density of the 2-D error under its stated
    covariance. Its margin over a constant isotropic sigma fitted by maximum
    likelihood on these same errors (sigma^2 = mean(|e|^2) / 2) says whether
    the stated covariance is informative, not just whether it covers: a
    constant that never commits passes coverage by declining to compete."""
    if err.size == 0:
        return {"n": 0, "share": 0.0}
    s2 = float(np.mean(err**2) / 2.0)
    const_ll = -np.log(2 * np.pi * s2) - err**2 / (2 * s2)
    return {
        "n": int(err.size),
        "share": err.size / total,
        "median_m": float(np.median(err)),
        "p90_m": float(np.percentile(err, 90)),
        "p99_m": float(np.percentile(err, 99)),
        "covered95": float(cover.mean()),
        "mean_loglik_stated": float(np.mean(ll)),
        "mean_loglik_fitted_constant": float(np.mean(const_ll)),
        "informativeness_nats": float(np.mean(ll) - np.mean(const_ll)),
    }


def position_audit(scene_dir: Path, n: int) -> dict[str, Any]:
    gt = load_truth(scene_dir / "ground_truth.json")
    rows: dict[str, dict[str, list[np.ndarray]]] = {}
    for cam in CAMERAS:
        camera = camera_from_smartspaces(scene_dir / "calibration.json", cam)
        boxes, labels, floor = all_class_boxes(gt, cam, n)
        for mode in ("foot", "fallback"):
            a = anchor_boxes(camera, boxes, labels, mode=mode)
            ok = a.anchor != "none"
            err = np.full(len(boxes), np.nan)
            err[ok] = np.linalg.norm(a.xy[ok] - floor[ok], axis=1)
            cov = np.zeros(len(boxes), dtype=bool)
            cov[ok] = covered95(a.xy[ok] - floor[ok], a.cov[ok])
            ll = np.full(len(boxes), np.nan)
            e = a.xy[ok] - floor[ok]
            inv = np.linalg.inv(a.cov[ok])
            ll[ok] = (
                -np.log(2 * np.pi)
                - 0.5 * np.log(np.linalg.det(a.cov[ok]))
                - 0.5 * np.einsum("ni,nij,nj->n", e, inv, e)
            )
            slot = rows.setdefault(
                mode, {"err": [], "cov": [], "anchor": [], "cls": [], "ll": []}
            )
            slot["err"].append(err)
            slot["cov"].append(cov)
            slot["ll"].append(ll)
            slot["anchor"].append(a.anchor)
            slot["cls"].append(labels)
    out: dict[str, Any] = {}
    for mode, slot in rows.items():
        err = np.concatenate(slot["err"])
        cov = np.concatenate(slot["cov"])
        anchor = np.concatenate(slot["anchor"])
        cls = np.concatenate(slot["cls"])
        ll = np.concatenate(slot["ll"])
        block: dict[str, Any] = {}
        for c in sorted(set(cls.tolist())):
            m = cls == c
            total = int(m.sum())
            ok = m & (anchor != "none")
            per = {"all": stats(err[ok], cov[ok], total, ll[ok])}
            for k in sorted(set(anchor[m].tolist())):
                mm = m & (anchor == k)
                per[k] = (
                    {"n": int(mm.sum()), "share": float(mm.sum() / total)}
                    if k == "none"
                    else stats(err[mm], cov[mm], total, ll[mm])
                )
            block[c] = per
        out[mode] = block
    return out


def event_audit(scene_dir: Path, zones_path: Path, n: int) -> dict[str, Any]:
    gt = load_truth(scene_dir / "ground_truth.json")
    zones = load_zone_config(zones_path)
    out: dict[str, Any] = {}
    for mode in ("foot", "fallback"):
        totals = {"predicted": 0, "gt": 0, "matched": 0}
        anchors: dict[str, int] = {}
        for cam in CAMERAS:
            source = CameraSource(
                camera_id=cam,
                fps=30.0,
                width=1920,
                height=1080,
                frames=oracle_frames(gt, cam, n),
                gate_awake=np.ones(n, dtype=bool),
                projector=load_projector(scene_dir / "calibration.json", cam),
                truth=person_truth(gt, cam, n),
                objects=objects_by_class(gt, cam, n),
            )
            r = evaluate_schedule(
                source,
                np.arange(n, dtype=np.int64),
                "Person",
                zones,
                "oracle",
                anchor_camera=camera_from_smartspaces(
                    scene_dir / "calibration.json", cam
                ),
                anchor_mode=mode,
            )
            for k in totals:
                totals[k] += r.events[k]
            for k, v in r.anchor_counts.items():
                anchors[k] = anchors.get(k, 0) + v
        out[mode] = {
            **totals,
            "precision": totals["matched"] / totals["predicted"]
            if totals["predicted"]
            else None,
            "recall": totals["matched"] / totals["gt"] if totals["gt"] else None,
            "track_point_anchors": anchors,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene", action="append", required=True, help="SCENE_DIR=ZONES")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    n = int(round(args.seconds * 30))
    result: dict[str, Any] = {
        "label": "lane S synthetic; oracle (ground-truth) boxes; not real cameras",
        "frames_per_camera": n,
        "cameras": list(CAMERAS),
        "pixel_sigma_px": PIXEL_SIGMA_PX,
        "priors": {
            k: {
                "mean_m": v.mean_m,
                "sd_m": v.sd_m,
                "range_m": list(v.range_m),
                "source": v.source,
            }
            for k, v in HEIGHT_PRIORS.items()
        },
        "scenes": {},
    }
    for spec in args.scene:
        scene, _, zones = spec.partition("=")
        sd = Path(scene)
        print(f"{sd.name}: positions", flush=True)
        pos = position_audit(sd, n)
        print(f"{sd.name}: events", flush=True)
        ev = event_audit(sd, Path(zones), n)
        result["scenes"][sd.name] = {"zones": zones, "position": pos, "events": ev}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
