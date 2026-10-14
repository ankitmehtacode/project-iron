"""Does SmartSpaces ground truth agree with its own calibration? (Day 41, Obj 2)

For every visible 2D box, project the object's ``3d location`` through the
camera that box belongs to, under each candidate convention, and measure
where the projected point lands relative to the box:

- ``inside``: the fraction of boxes whose rectangle contains the point.
- ``u``, ``v``: the point's normalised position within the box (0 = left/
  top edge, 1 = right/bottom edge). Under the right convention a box centre
  lands near (0.5, 0.5); a bottom-face point lands near (0.5, 1.0).

Conventions are varied ONE AT A TIME from the card's stated reading, and
the pass criterion was fixed before the first run (below), so no number was
tuned into passing:

- projection source: ``cameraMatrix`` / ``K·E`` / ``K·E⁻¹``
- what ``3d location`` names: the box centre, or the bottom face — tested
  by also projecting ``location − h/2·ẑ`` with h taken from each of the
  three ``3d bounding box scale`` indices
- a deliberately wrong axis convention (x and y swapped), which must fail.

Visible boxes are the visible part of an occluded object, so a centre can
legitimately fall outside one; medians and the 90% bar allow for that.

Added AFTER the first run, and labelled as such in the artifact: on
Warehouse_000 the declared point criterion failed under every convention
(best: 76% inside) while the medians sat at (0.52, 0.51). To tell
occlusion from a geometric disagreement, the full 3D box is also projected
(8 corners: centre ± scale/2, rotated by yaw about z; pitch and roll are
<2e-5 rad on disk) and its image bounding rectangle compared with the
visible box. Occlusion only ever removes pixels, so under the right
geometry the visible box lies inside the projected one whatever is in
front of it: ``containment = |visible ∩ projected| / |visible|`` should be
near 1 for occluded and unoccluded objects alike, and a wrong convention
drives it down.

PhysicalAI-SmartSpaces — lane S, synthetic. Not evidence about real crowding.

    python scripts/smartspaces_reprojection.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smartspaces_integrity import (  # noqa: E402
    LANE_LABEL,
    REPO_ROOT,
    load_scene,
    scene_dirs,
)

from src.data.smartspaces import (  # noqa: E402
    ProjectionSource,
    SceneGroundTruth,
    box_containment,
    load_calibration,
    project,
)

ARTIFACT = REPO_ROOT / "artifacts" / "smartspaces" / "day41_reprojection.json"

# Declared before the first run. Do not edit to make a scene pass.
PASS_INSIDE_FRACTION = 0.90
PASS_MEDIAN_UV_TOLERANCE = 0.10
MIN_BOX_SIDE_PX = 4


def _variants() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source in ProjectionSource:
        out.append({"name": f"{source.name}/location", "source": source, "drop": None})
    for k in range(3):
        out.append(
            {
                "name": f"INTRINSIC_EXTRINSIC/location-half-scale{k}",
                "source": ProjectionSource.INTRINSIC_EXTRINSIC,
                "drop": k,
            }
        )
        out.append(
            {
                "name": f"CAMERA_MATRIX/location-half-scale{k}",
                "source": ProjectionSource.CAMERA_MATRIX,
                "drop": k,
            }
        )
    for source in (
        ProjectionSource.INTRINSIC_EXTRINSIC,
        ProjectionSource.CAMERA_MATRIX,
    ):
        out.append(
            {
                "name": f"{source.name}/location-xy-swapped",
                "source": source,
                "drop": None,
                "swap_xy": True,
            }
        )
    return out


def measure_scene(scene: Path, gt: SceneGroundTruth | None = None) -> dict[str, Any]:
    calibration = load_calibration(scene / "calibration.json")
    gt = gt if gt is not None else load_scene(scene)
    boxes = gt.box_xyxy.astype(np.float64)
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    usable = (width >= MIN_BOX_SIDE_PX) & (height >= MIN_BOX_SIDE_PX)

    results: dict[str, Any] = {}
    for variant in _variants():
        u = np.full(boxes.shape[0], np.nan)
        v = np.full(boxes.shape[0], np.nan)
        front = np.zeros(boxes.shape[0], dtype=bool)
        for cam_index, camera_id in enumerate(gt.camera_names):
            sel = np.flatnonzero(gt.box_camera == cam_index)
            if sel.size == 0:
                continue
            rows = gt.box_row[sel]
            world = gt.location[rows].copy()
            if variant["drop"] is not None:
                world[:, 2] -= gt.scale[rows, variant["drop"]] / 2.0
            if variant.get("swap_xy"):
                world[:, [0, 1]] = world[:, [1, 0]]
            pixels, depth = project(
                world, calibration.cameras[camera_id], variant["source"]
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                u[sel] = (pixels[:, 0] - boxes[sel, 0]) / width[sel]
                v[sel] = (pixels[:, 1] - boxes[sel, 1]) / height[sel]
            if variant["source"] is ProjectionSource.CAMERA_MATRIX:
                # cameraMatrix's overall sign is arbitrary (P[2,3] = 1); take
                # "in front" as the sign the majority of visible boxes have.
                sign = np.sign(np.median(depth))
                front[sel] = depth * sign > 0
            else:
                front[sel] = depth > 0
        ok = usable & front & np.isfinite(u) & np.isfinite(v)
        inside = ok & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        n = int(usable.sum())
        results[variant["name"]] = {
            "usable_boxes": n,
            "behind_camera": int((usable & ~front).sum()),
            "inside_fraction": float(inside.sum() / n) if n else float("nan"),
            "median_u": float(np.median(u[ok])) if ok.any() else float("nan"),
            "median_v": float(np.median(v[ok])) if ok.any() else float("nan"),
            "iqr_v": (
                [float(np.percentile(v[ok], 25)), float(np.percentile(v[ok], 75))]
                if ok.any()
                else None
            ),
        }
    return {
        "scene": scene.name,
        "boxes_total": int(boxes.shape[0]),
        "boxes_excluded_smaller_than_min_side": int((~usable).sum()),
        "variants": results,
    }


BOX_VARIANTS = (
    ("CAMERA_MATRIX/box", ProjectionSource.CAMERA_MATRIX, False),
    ("INTRINSIC_EXTRINSIC/box", ProjectionSource.INTRINSIC_EXTRINSIC, False),
    ("INTRINSIC_EXTRINSIC/box-xy-swapped", ProjectionSource.INTRINSIC_EXTRINSIC, True),
)
CONTAINMENT_BAR = 0.9
"""A visible box counts as consistent when at least 90% of its area lies
inside the projected 3D box. Declared with the box metric, after run 1."""


def measure_scene_boxes(scene: Path, gt: SceneGroundTruth) -> dict[str, Any]:
    calibration = load_calibration(scene / "calibration.json")
    boxes = gt.box_xyxy.astype(np.float64)
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    usable = (width >= MIN_BOX_SIDE_PX) & (height >= MIN_BOX_SIDE_PX)
    out: dict[str, Any] = {}
    for name, source, swap in BOX_VARIANTS:
        index = np.flatnonzero(usable)
        location = gt.location[gt.box_row[index]].copy()
        if swap:
            location[:, [0, 1]] = location[:, [1, 0]]
        c, j = box_containment(gt, calibration, source, index, location)
        containment = np.full(boxes.shape[0], np.nan)
        iou = np.full(boxes.shape[0], np.nan)
        containment[index] = c
        iou[index] = j
        scored = np.isfinite(containment)
        by_class = {}
        row_class = gt.class_index[gt.box_row]
        for k, cls in enumerate(gt.class_names):
            m = scored & (row_class == k)
            by_class[cls] = {
                "boxes": int(m.sum()),
                "containment_median": float(np.median(containment[m]))
                if m.any()
                else None,
                "fraction_containment_at_least_bar": (
                    float(np.mean(containment[m] >= CONTAINMENT_BAR))
                    if m.any()
                    else None
                ),
            }
        out[name] = {
            "scored_boxes": int(scored.sum()),
            "not_scored_corner_behind_camera": int((usable & ~scored).sum()),
            "containment_median": float(np.median(containment[scored])),
            "containment_p5": float(np.percentile(containment[scored], 5)),
            "fraction_containment_at_least_bar": float(
                np.mean(containment[scored] >= CONTAINMENT_BAR)
            ),
            "iou_median": float(np.median(iou[scored])),
            "by_class": by_class,
        }
    return out


def passes(result: dict[str, Any], expect_v: float) -> bool:
    return (
        result["inside_fraction"] >= PASS_INSIDE_FRACTION
        and abs(result["median_u"] - 0.5) <= PASS_MEDIAN_UV_TOLERANCE
        and abs(result["median_v"] - expect_v) <= PASS_MEDIAN_UV_TOLERANCE
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", default=[])
    args = parser.parse_args(argv)

    scenes = []
    for scene in scene_dirs():
        if args.scene and scene.name not in args.scene:
            continue
        gt = load_scene(scene)
        result = measure_scene(scene, gt)
        result["box_variants"] = measure_scene_boxes(scene, gt)
        scenes.append(result)
        excluded = result["boxes_excluded_smaller_than_min_side"]
        print(
            f"== {scene.name}  boxes {result['boxes_total']:,} "
            f"(excluded <{MIN_BOX_SIDE_PX}px: {excluded:,})"
        )
        for name, r in result["variants"].items():
            print(
                f"  {name:<44} inside {r['inside_fraction']:.4f}  "
                f"u {r['median_u']:+.3f}  v {r['median_v']:+.3f}  "
                f"behind {r['behind_camera']}"
            )
        for name, r in result["box_variants"].items():
            print(
                f"  {name:<44} containment median {r['containment_median']:.4f} "
                f"p5 {r['containment_p5']:.4f}  >= {CONTAINMENT_BAR}: "
                f"{r['fraction_containment_at_least_bar']:.4f}  "
                f"IoU median {r['iou_median']:.3f}"
                f"  unscored {r['not_scored_corner_behind_camera']}"
            )
            for cls, c in r["by_class"].items():
                print(
                    f"      {cls:<14} n={c['boxes']:<9} containment median "
                    f"{c['containment_median']}  "
                    f">=bar {c['fraction_containment_at_least_bar']}"
                )
    payload = {
        "label": LANE_LABEL,
        "pass_criterion": {
            "inside_fraction_at_least": PASS_INSIDE_FRACTION,
            "median_uv_within": PASS_MEDIAN_UV_TOLERANCE,
            "min_box_side_px": MIN_BOX_SIDE_PX,
            "declared": "before the first run",
        },
        "box_criterion": {
            "containment_bar": CONTAINMENT_BAR,
            "declared": (
                "after the first run, because the point criterion failed on "
                "Warehouse_000 under every convention while its medians were "
                "centred; see the module docstring"
            ),
        },
        "scenes": scenes,
    }
    if not args.scene:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Written: {ARTIFACT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
