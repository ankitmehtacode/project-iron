"""The validity-gate battery on SmartSpaces frames (Day 42, Objective B7).

lane S synthetic (SmartSpaces) — not a product claim; NVIDIA terms
unresolved. Frames are read from the B4 export under data/ and never written
anywhere else.

Every capability registered in ``validity.GATES`` (read live, as
``scripts/validity_matrix.py`` does) is evaluated per exported camera on the
first 4 of that camera's exported frames (sampled, not consecutive: the
gates read appearance statistics, and only ``motion_geometry`` /
``state_estimation`` need more than one frame). Detection gets the split's GT
instance counts per class per 1080p height bucket. Depth gets no GT depth
(SmartSpaces depth maps were deliberately not fetched), so it refuses.

Each camera is evaluated twice: at native 1920x1080, and downscaled
(INTER_AREA) to 320x240, the QVGA raster ``MIN_TRACKABLE_CORNER_DENSITY`` was
set on. The point-tracking gate caps corners at 1024, so at 1080p its density
cannot exceed 1024 / (1920 * 1080) = 4.9e-4, under its 1e-3 floor: it cannot
pass at native resolution whatever the content. No floor is changed here.

    python scripts/smartspaces_validity_row.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smartspaces_integrity import REPO_ROOT  # noqa: E402
from smartspaces_to_coco import HEIGHT_EDGES  # noqa: E402
from smartspaces_to_coco import SUMMARY as EXPORT_SUMMARY  # noqa: E402

from src.data import validity  # noqa: E402
from src.data.coco_export import height_bucket  # noqa: E402

ARTIFACT = REPO_ROOT / "artifacts" / "smartspaces" / "day42_validity_row.json"
LABEL = "lane S synthetic (SmartSpaces) — not a product claim; NVIDIA terms unresolved"
RASTERS = {"native_1920x1080": None, "qvga_320x240": (320, 240)}


def instance_counts(coco: dict[str, Any]) -> dict[str, dict[str, int]]:
    names = {c["id"]: c["name"] for c in coco["categories"]}
    counts: dict[str, Counter[str]] = {}
    for a in coco["annotations"]:
        b = height_bucket(a["height_native"], HEIGHT_EDGES)
        label = f"[{HEIGHT_EDGES[b]:g},{HEIGHT_EDGES[b + 1]:g})"
        counts.setdefault(names[a["category_id"]], Counter())[label] += 1
    return {k: dict(v) for k, v in sorted(counts.items())}


def main() -> int:
    export = REPO_ROOT / json.loads(EXPORT_SUMMARY.read_text())["export_dir"]
    capabilities = tuple(validity.GATES)
    rows: dict[str, Any] = {}
    for split in ("val", "train"):
        coco = json.loads((export / f"{split}.json").read_text())
        counts = instance_counts(coco)
        by_camera: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for image in coco["images"]:
            by_camera.setdefault((image["scene"], image["camera"]), []).append(image)
        per_raster: dict[str, Any] = {}
        for raster, size in RASTERS.items():
            results: dict[str, list[dict[str, Any]]] = {c: [] for c in capabilities}
            for (scene, camera), images in sorted(by_camera.items()):
                images = sorted(images, key=lambda i: i["frame_index"])[:4]
                stack = []
                for image in images:
                    bgr = cv2.imread(str(export / "images" / image["file_name"]))
                    if size is not None:
                        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
                    stack.append(bgr[:, :, ::-1])
                frames = np.stack(stack)
                for capability in capabilities:
                    result = validity.evaluate(
                        capability,
                        f"SmartSpaces/{split}/{scene}/{camera}",
                        frames=frames,
                        instance_counts=counts,
                    )
                    results[capability].append(
                        {
                            "camera": f"{scene}/{camera}",
                            "passed": result.passed,
                            "reason": result.reason,
                            "evidence": result.evidence,
                        }
                    )
            per_raster[raster] = {
                capability: {
                    "cameras": len(r),
                    "passed": sum(x["passed"] for x in r),
                    "verdict": (
                        "measurable"
                        if r and all(x["passed"] for x in r)
                        else "unmeasurable_here"
                        if not any(x["passed"] for x in r)
                        else "mixed_by_camera"
                    ),
                    "per_camera": r,
                }
                for capability, r in results.items()
            }
        rows[split] = {"instance_counts": counts, "rasters": per_raster}
    payload = {
        "label": LABEL,
        "capabilities": list(capabilities),
        "frames_per_camera": 4,
        "notes": [
            "Depth: no GT depth supplied (SmartSpaces depth maps not fetched).",
            "point_tracking at native 1080p is capped by maxCorners=1024: density "
            "<= 4.9e-4 < floor 1e-3 for ANY content; the QVGA row is the one the "
            "floor was set on.",
            "No floor was changed.",
        ],
        "splits": rows,
    }
    ARTIFACT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Written: {ARTIFACT.relative_to(REPO_ROOT)}")
    for split, row in rows.items():
        for raster, caps in row["rasters"].items():
            print(
                split,
                raster,
                {
                    c: f"{v['passed']}/{v['cameras']} {v['verdict']}"
                    for c, v in caps.items()
                },
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
