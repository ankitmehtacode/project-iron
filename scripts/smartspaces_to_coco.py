"""SmartSpaces frames to COCO, split rules enforced (Day 42, Objective B4).

PhysicalAI-SmartSpaces — lane S, synthetic. Not evidence about real crowding.
NVIDIA's terms-of-service clause is unresolved: frames are written ONLY under
data/ (gitignored), never into docs, fixtures, artifacts or the Inspector.

Sampling rule (fixed before running, seed 42): the 46 cameras of the Day-41
gate-parity sample (so B5 can join every frame to stored gate decisions), the
first 1800 frames of each. Split placement per
:func:`src.data.coco_export.assign_splits`; cameras of excluded scenes are
dropped. Per kept camera, frames [warmup, 1800) are cut into six 300-frame
blocks and 2 frames are drawn from each block with numpy default_rng(42):
12 frames per camera. Annotations: every visible 2D GT box
(``2d bounding box visible``) on that camera and frame.

Output (content-addressed): data/derived/PhysicalAI-SmartSpaces/<commit>/
coco_day42/<sha12>/{images/*.png, train.json, val.json, manifest.json}, where
the sha covers the rule, the frame list and the GT source hashes. A summary
with per-split counts goes to artifacts/smartspaces/day42_coco_export.json.

    python scripts/smartspaces_to_coco.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smartspaces_integrity import (  # noqa: E402
    CACHE_ROOT,
    COMMIT,
    CT25_RERENDER_OF,
    LANE_LABEL,
    REPO_ROOT,
    load_scene,
    scene_dirs,
)
from smartspaces_rerender_transient import cache_file, detect_cuts  # noqa: E402

from src.cascade.motion import MotionGateConfig  # noqa: E402
from src.data.coco_export import (  # noqa: E402
    FrameRecord,
    ScenePlacement,
    assign_splits,
    height_bucket,
    validate_split_plan,
)

PARITY = REPO_ROOT / "artifacts" / "smartspaces" / "day41_gate_parity.json"
TRANSIENT = REPO_ROOT / "artifacts" / "smartspaces" / "day42_rerender_transient.json"
SUMMARY = REPO_ROOT / "artifacts" / "smartspaces" / "day42_coco_export.json"
EXPORT_ROOT = CACHE_ROOT / "coco_day42"
SEED = 42
WINDOW = 1800
BLOCK = 300
PER_BLOCK = 2
HEIGHT_EDGES = (0.0, 32.0, 64.0, 128.0, 256.0, float("inf"))
"""Native-pixel height buckets; the same edges are applied to the scaled
heights at 720p and 360p."""
RESOLUTIONS = {"1080p": 1.0, "720p": 720 / 1080, "360p": 360 / 1080}
MIN_INSTANCES = 30
LABEL = (
    "lane S synthetic (SmartSpaces) — zero-shot, not a product claim; "
    "NVIDIA terms unresolved"
)


def _scene(name: str) -> Path:
    return next(p for p in scene_dirs() if p.name == name)


def plan() -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
    placements = [
        ScenePlacement(
            scene=p.name,
            dataset_split=p.parent.name,  # type: ignore[arg-type]
            rerender_of=CT25_RERENDER_OF.get(p.name),
        )
        for p in scene_dirs()
    ]
    splits, excluded = assign_splits(placements)
    cameras = json.loads(PARITY.read_text())["cameras"]
    kept = [c for c in cameras if c["scene"] in splits]
    return splits, excluded, kept


def sample_frames(seed: int = SEED) -> list[tuple[str, str, int]]:
    """``(scene, camera, frame)``, the rule in the module docstring."""
    _, _, cameras = plan()
    rng = np.random.default_rng(seed)
    warm = MotionGateConfig().warmup_frames
    out = []
    for cam in sorted(cameras, key=lambda c: (c["scene"], c["camera"])):
        for start in range(0, WINDOW, BLOCK):
            lo = max(start, warm)
            picks = rng.choice(np.arange(lo, start + BLOCK), PER_BLOCK, replace=False)
            out.extend((cam["scene"], cam["camera"], int(f)) for f in sorted(picks))
    return out


def transient_frames() -> int:
    return int(
        json.loads(TRANSIENT.read_text())["windows"]["post_hoc_excess_end"][
            "frames_after_cut"
        ]
    )


def main() -> int:
    splits, excluded, cameras = plan()
    frames = sample_frames()
    classes = sorted(
        {
            name
            for scene in {s for s, _, _ in frames}
            for name in load_scene(_scene(scene)).class_names
        }
    )
    categories = [{"id": i + 1, "name": c} for i, c in enumerate(classes)]
    window = transient_frames()

    gt_shas = {
        s: load_scene(_scene(s)).source_sha256
        for s in sorted({s for s, _, _ in frames})
    }
    rule = {
        "seed": SEED,
        "window_frames": WINDOW,
        "block": BLOCK,
        "per_block": PER_BLOCK,
        "cameras": "day41_gate_parity sample minus excluded scenes",
        "transient_frames_after_cut": window,
    }
    sha = hashlib.sha256(
        json.dumps(
            {"rule": rule, "frames": frames, "gt": gt_shas, "commit": COMMIT}
        ).encode()
    ).hexdigest()
    out = EXPORT_ROOT / sha[:12]
    if (out / "manifest.json").exists():
        print(f"already exported: {out}")
    else:
        staging = out.with_name(out.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)
        (staging / "images").mkdir(parents=True)
        _write(staging, frames, splits, categories, window)
        manifest = {
            "label": LABEL,
            "lane": LANE_LABEL,
            "commit": COMMIT,
            "sha": sha,
            "rule": rule,
            "splits": splits,
            "excluded_scenes": excluded,
            "gt_source_sha256": gt_shas,
            "categories": categories,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(out)
    summary = _summary(out, splits, excluded, sha, rule)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Written: {SUMMARY.relative_to(REPO_ROOT)}; export at {out}")
    for split, row in summary["splits"].items():
        print(split, row["images"], "images", row["instances_per_class"])
    return 0


def _write(
    staging: Path,
    frames: list[tuple[str, str, int]],
    splits: dict[str, str],
    categories: list[dict[str, Any]],
    window: int,
) -> None:
    cat_id = {c["name"]: c["id"] for c in categories}
    coco: dict[str, dict[str, Any]] = {
        s: {"images": [], "annotations": [], "categories": categories}
        for s in ("train", "val")
    }
    records: list[dict[str, Any]] = []
    by_camera: dict[tuple[str, str], list[int]] = {}
    for scene, camera, f in frames:
        by_camera.setdefault((scene, camera), []).append(f)
    image_id = ann_id = 0
    for (scene, camera), wanted in sorted(by_camera.items()):
        gt = load_scene(_scene(scene))
        k = gt.camera_names.index(camera) if camera in gt.camera_names else -1
        rerender = scene in CT25_RERENDER_OF
        cuts = (
            detect_cuts(
                np.load(cache_file({"scene": scene, "camera": camera}))["mean_bgr"]
            )
            if rerender
            else None
        )
        capture = cv2.VideoCapture(str(_scene(scene) / "videos" / f"{camera}.mp4"))
        index = -1
        for f in sorted(wanted):
            while index < f:
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f"{scene}/{camera} ended before frame {f}")
                index += 1
            image_id += 1
            name = f"{scene}__{camera}__{f:05d}.png"
            cv2.imwrite(str(staging / "images" / name), image)
            since = None
            if cuts is not None:
                before = cuts[(cuts > 0) & (cuts <= f)]
                since = int(f - before.max()) if before.size else None
            record = FrameRecord(
                image_id=image_id,
                scene=scene,
                camera=camera,
                frame_index=f,
                split=splits[scene],  # type: ignore[arg-type]
                ct25_rerender=rerender,
                frames_since_cut=since,
                in_transient=since is not None and since < window,
                file_name=name,
            )
            records.append(record.__dict__)
            coco[record.split]["images"].append(
                {
                    "id": image_id,
                    "file_name": name,
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "scene": scene,
                    "camera": camera,
                    "frame_index": f,
                    "ct25_rerender": rerender,
                    "frames_since_cut": since,
                    "in_transient": record.in_transient,
                }
            )
            sel = np.flatnonzero((gt.box_camera == k) & (gt.frame[gt.box_row] == f))
            for j in sel:
                x0, y0, x1, y1 = (float(v) for v in gt.box_xyxy[j])
                x0, y0 = max(0.0, x0), max(0.0, y0)
                x1, y1 = min(float(image.shape[1]), x1), min(float(image.shape[0]), y1)
                if x1 <= x0 or y1 <= y0:
                    continue
                ann_id += 1
                cls = gt.class_names[int(gt.class_index[gt.box_row[j]])]
                coco[record.split]["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cat_id[cls],
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "area": (x1 - x0) * (y1 - y0),
                        "iscrowd": 0,
                        "height_native": y1 - y0,
                        "object_id": int(gt.object_id[gt.box_row[j]]),
                    }
                )
        capture.release()
    validate_split_plan(
        {r["scene"]: r["split"] for r in records},
        {s: CT25_RERENDER_OF.get(s) for s in {r["scene"] for r in records}},
    )
    for split, payload in coco.items():
        (staging / f"{split}.json").write_text(json.dumps(payload) + "\n")


def _summary(
    out: Path,
    splits: dict[str, str],
    excluded: dict[str, str],
    sha: str,
    rule: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "label": LABEL,
        "export_dir": str(out.relative_to(REPO_ROOT)),
        "sha": sha,
        "rule": rule,
        "splits_by_scene": splits,
        "excluded_scenes": excluded,
        "height_edges_px": [e if np.isfinite(e) else "inf" for e in HEIGHT_EDGES],
        "min_instances_to_report": MIN_INSTANCES,
        "splits": {},
    }
    for split in ("train", "val"):
        coco = json.loads((out / f"{split}.json").read_text())
        names = {c["id"]: c["name"] for c in coco["categories"]}
        anns = coco["annotations"]
        rerender_imgs = {i["id"] for i in coco["images"] if i["ct25_rerender"]}
        transient_imgs = {i["id"] for i in coco["images"] if i["in_transient"]}
        per_class: dict[str, int] = {}
        buckets: dict[str, dict[str, dict[str, Any]]] = {}
        for a in anns:
            per_class[names[a["category_id"]]] = (
                per_class.get(names[a["category_id"]], 0) + 1
            )
        for res, scale in RESOLUTIONS.items():
            counts: dict[str, dict[str, Any]] = {}
            for a in anns:
                b = height_bucket(a["height_native"] * scale, HEIGHT_EDGES)
                label = f"[{HEIGHT_EDGES[b]:g},{HEIGHT_EDGES[b + 1]:g})"
                row = counts.setdefault(label, {"instances": 0, "person": 0})
                row["instances"] += 1
                row["person"] += names[a["category_id"]] == "Person"
            for row in counts.values():
                row["too_thin"] = row["instances"] < MIN_INSTANCES
                row["person_too_thin"] = row["person"] < MIN_INSTANCES
            buckets[res] = dict(
                sorted(counts.items(), key=lambda kv: float(kv[0][1:].split(",")[0]))
            )
        result["splits"][split] = {
            "images": len(coco["images"]),
            "rerender_images": len(rerender_imgs),
            "images_in_transient": len(transient_imgs),
            "instances": len(anns),
            "instances_per_class": dict(sorted(per_class.items())),
            "classes_too_thin": sorted(
                c for c in names.values() if per_class.get(c, 0) < MIN_INSTANCES
            ),
            "instances_per_height_bucket": buckets,
        }
    return result


if __name__ == "__main__":
    raise SystemExit(main())
