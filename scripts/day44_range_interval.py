"""Day 44: the per-detection range interval [R_head, R_foot] on oracle boxes.

For every visible GT person box on cameras 0-3 over the first ``--seconds``
of each scene: R_foot (box bottom-centre ray on the floor) and R_head (box
top-centre ray on the plane z = H_max), from :func:`src.pipeline.anchor.
range_interval`. The true range is the horizontal distance from the camera
to the GT floor point (centre of the 3D box's bottom face).

Assumptions (stated, not tested here): upright person; flat floor at z = 0
with the feet on it; box bottom at or above the feet; box top at or below
the top of the head; head no higher than H_max; camera above H_max.

Reported per scene and pooled, for H_max = PERSON_MAX_HEIGHT_M (2.058 m)
and, as a sensitivity row, HUMAN_RECORD_HEIGHT_M (2.72 m): how often the
true range lies inside the interval, the width distribution (absolute and
relative to the true range), and every violation by cause:

lower end broken (true range < R_head), first cause that holds:
  taller_than_bound      the GT 3D box is taller than H_max
  box_top_above_head     the 2D box top is above the projected GT box top
                         (by more than 1 px)
  reference_offset       the miss is within half the GT box's footprint
                         diagonal: the GT floor point is the box centre,
                         not the point the ray grazes
  other
upper end broken (true range > R_foot):
  box_bottom_below_floor the 2D box bottom is below the projected GT floor
                         point (by more than 1 px)
  reference_offset       as above
  other
  plus a cross-tab of the two tests and the miss in metres and in units of
  half the footprint diagonal: a standing person's box bottom is the near
  foot, so a miss within the footprint is the reference point (box-centre
  floor point), not a box that runs below the feet
A camera at or below H_max has no lower end; those boxes are counted
separately and scored against [0, R_foot].
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.smartspaces import MEASURED_GEOMETRY  # noqa: E402
from src.pipeline.anchor import (  # noqa: E402
    HUMAN_RECORD_HEIGHT_M,
    PERSON_MAX_HEIGHT_M,
    camera_from_smartspaces,
    range_interval,
)
from src.pipeline.truth import PERSON, load_truth  # noqa: E402

CAMERAS = ("Camera_0000", "Camera_0001", "Camera_0002", "Camera_0003")


def person_boxes(gt: Any, camera: str, n: int) -> dict[str, np.ndarray]:
    sel = np.flatnonzero(gt.box_camera == gt.camera_names.index(camera))
    rows = gt.box_row[sel]
    keep = (gt.class_index[rows] == gt.class_names.index(PERSON)) & (gt.frame[rows] < n)
    sel, rows = sel[keep], rows[keep]
    loc, scale = gt.location[rows], gt.scale[rows]
    floor = MEASURED_GEOMETRY.floor_point(loc, scale)
    top = floor.copy()
    top[:, 2] += scale[:, MEASURED_GEOMETRY.height_scale_index]
    return {
        "box": gt.box_xyxy[sel].astype(np.float64),
        "floor": floor,
        "top": top,
        "height": scale[:, MEASURED_GEOMETRY.height_scale_index].astype(np.float64),
        "half_diag": np.hypot(scale[:, 0], scale[:, 1]) / 2.0,
        "half_x": scale[:, 0] / 2.0,
        "half_y": scale[:, 1] / 2.0,
        "yaw": gt.rotation[rows, 2].astype(np.float64),
        "object_id": gt.object_id[rows].astype(np.int64),
        "frame": gt.frame[rows].astype(np.int64),
    }


def pct(x: np.ndarray) -> dict[str, float | None]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}  # type: ignore[dict-item]
    return {
        "n": int(x.size),
        **{f"p{q}": float(np.percentile(x, q)) for q in (10, 50, 90, 99)},
    }


def audit(scene_dir: Path, n: int, h_max: float) -> dict[str, Any]:
    gt = load_truth(scene_dir / "ground_truth.json")
    total = inside = no_lower = no_foot = inside_near = 0
    near_upper_miss: list[float] = []
    widths, rel = [], []
    lower: Counter[str] = Counter()
    upper: Counter[str] = Counter()
    upper_x: Counter[str] = Counter()
    excess: list[float] = []
    excess_rel: list[float] = []
    cams = {}
    for cam in CAMERAS:
        camera = camera_from_smartspaces(scene_dir / "calibration.json", cam)
        b = person_boxes(gt, cam, n)
        ri = range_interval(camera, b["box"], h_max)
        r_true = np.linalg.norm(b["floor"][:, :2] - camera.centre[:2], axis=1)
        ok_foot = np.isfinite(ri.r_foot)
        no_foot += int((~ok_foot).sum())
        # Near edge of the footprint along the camera ray: the support of the
        # yawed rectangle (yaw = rotation[2] about the vertical, measured Day 41,
        # src.data.smartspaces.box_containment) in the camera direction.
        u = b["floor"][:, :2] - camera.centre[:2]
        u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
        c, s_ = np.cos(b["yaw"]), np.sin(b["yaw"])
        extent = b["half_x"] * np.abs(u[:, 0] * c + u[:, 1] * s_) + b[
            "half_y"
        ] * np.abs(-u[:, 0] * s_ + u[:, 1] * c)
        r_near = r_true - extent
        lo = np.where(np.isfinite(ri.r_head), ri.r_head, 0.0)
        no_lower += int((~np.isfinite(ri.r_head) & ok_foot).sum())
        inn = ok_foot & (lo <= r_true) & (r_true <= ri.r_foot)
        inn_near = ok_foot & (lo <= r_true) & (r_near <= ri.r_foot)
        inside_near += int(inn_near.sum())
        bad = ok_foot & (r_near > ri.r_foot)
        near_upper_miss.extend((r_near[bad] - ri.r_foot[bad]).tolist())
        total += int(ok_foot.sum())
        inside += int(inn.sum())
        w = ri.width_m[ok_foot & np.isfinite(ri.r_head)]
        widths.append(w)
        rel.append(w / r_true[ok_foot & np.isfinite(ri.r_head)])
        top_px = camera.project(b["top"])
        floor_px = camera.project(b["floor"])
        for i in np.flatnonzero(ok_foot & ~inn):
            if r_true[i] < lo[i]:
                if b["height"][i] > h_max:
                    lower["taller_than_bound"] += 1
                elif b["box"][i, 1] < top_px[i, 1] - 1.0:
                    lower["box_top_above_head"] += 1
                elif lo[i] - r_true[i] <= b["half_diag"][i]:
                    lower["reference_offset"] += 1
                else:
                    lower["other"] += 1
            else:
                below = b["box"][i, 3] > floor_px[i, 1] + 1.0
                within = r_true[i] - ri.r_foot[i] <= b["half_diag"][i]
                if below:
                    upper["box_bottom_below_floor"] += 1
                elif within:
                    upper["reference_offset"] += 1
                else:
                    upper["other"] += 1
                upper_x[f"below={bool(below)},within_footprint={bool(within)}"] += 1
                excess.append(r_true[i] - ri.r_foot[i])
                excess_rel.append((r_true[i] - ri.r_foot[i]) / b["half_diag"][i])
        cams[cam] = {
            "camera_height_m": float(camera.centre[2]),
            "boxes": int(ok_foot.sum()),
            "inside": int(inn.sum()),
        }
    w_all = np.concatenate(widths)
    r_all = np.concatenate(rel)
    return {
        "h_max_m": h_max,
        "boxes": total,
        "inside": inside,
        "coverage": inside / total if total else None,
        "boxes_without_lower_end": no_lower,
        "boxes_excluded_foot_ray_misses_floor": no_foot,
        "inside_near_edge_reference": inside_near,
        "coverage_near_edge_reference": inside_near / total if total else None,
        "upper_end_misses_vs_near_edge": len(near_upper_miss),
        "upper_end_miss_vs_near_edge_m": pct(np.array(near_upper_miss)),
        "reference_note": (
            "coverage: true range = GT footprint CENTRE; "
            "coverage_near_edge_reference: the upper end scored against the "
            "footprint's near edge along the camera ray (the point a box "
            "bottom on the near foot can reach), the lower end against the "
            "centre; no excuse category"
        ),
        "violations_lower_end": dict(lower.most_common()),
        "violations_upper_end": dict(upper.most_common()),
        "upper_end_crosstab": dict(upper_x.most_common()),
        "upper_end_excess_m": pct(np.array(excess)),
        "upper_end_excess_over_half_footprint": pct(np.array(excess_rel)),
        "width_m": pct(w_all),
        "width_relative_to_true_range": pct(r_all),
        "cameras": cams,
        "gt_person_height_m": pct(
            np.concatenate([person_boxes(gt, c, n)["height"] for c in CAMERAS])
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene", action="append", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    n = int(round(args.seconds * 30))
    out: dict[str, Any] = {
        "label": "lane S synthetic; oracle (ground-truth) person boxes",
        "frames_per_camera": n,
        "assumptions": [
            "upright person",
            "flat floor at z = 0, feet on it",
            "box bottom at or above the feet",
            "box top at or below the top of the head",
            "head no higher than H_max",
            "camera above H_max (else no lower end)",
        ],
        "scenes": {},
    }
    for sd in args.scene:
        out["scenes"][sd.name] = {
            "population_bound": audit(sd, n, PERSON_MAX_HEIGHT_M),
            "record_bound": audit(sd, n, HUMAN_RECORD_HEIGHT_M),
        }
        a = out["scenes"][sd.name]["population_bound"]
        print(
            f"{sd.name}: inside {a['inside']}/{a['boxes']} "
            f"({a['coverage']:.3f}), lower {a['violations_lower_end']}, "
            f"upper {a['violations_upper_end']}, width p50 "
            f"{a['width_m'].get('p50')}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
