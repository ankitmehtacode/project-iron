"""Data-integrity report for the fetched PhysicalAI-SmartSpaces subset (Day 41).

HuggingFace's hash check proved transport integrity: the bytes are the ones
published. It did not prove the files contain what the card says, that the
three places a camera can be named agree, or that videos decode. This
script checks those, per scene, and writes:

- ``artifacts/smartspaces/day41_integrity.json`` — every number below.
- ``docs/smartspaces_integrity.md`` — the same, as tables.
- ``data/derived/PhysicalAI-SmartSpaces/<commit>/<scene>.npz`` — the
  strictly-parsed ground truth as columns, for the later Day-41 scripts
  (ignored; rebuilt whenever the source bytes change).

Every reader raises on schema drift (:mod:`src.data.smartspaces`), so a
scene that parses at all parsed completely.

PhysicalAI-SmartSpaces — lane S, synthetic. Not evidence about real crowding.

    python scripts/smartspaces_integrity.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.data.smartspaces import (
    CARD_DOCUMENTED_GT_KEYS,
    GT_OBJECT_KEYS,
    SceneGroundTruth,
    load_calibration,
    load_scene_ground_truth,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
COMMIT = "2cbe9563cbe9f47f846e5c871ee994572bbbc60e"
LABEL = "paths-0deaa0fe541fbaf1"
DATA_ROOT = (
    REPO_ROOT
    / "data"
    / "raw"
    / "PhysicalAI-SmartSpaces"
    / COMMIT
    / LABEL
    / "MTMC_Tracking_2026"
)
CACHE_ROOT = REPO_ROOT / "data" / "derived" / "PhysicalAI-SmartSpaces" / COMMIT
ARTIFACT = REPO_ROOT / "artifacts" / "smartspaces" / "day41_integrity.json"
REPORT = REPO_ROOT / "docs" / "smartspaces_integrity.md"

LANE_LABEL = (
    "PhysicalAI-SmartSpaces — lane S, synthetic. Not evidence about real crowding."
)

CT25_RERENDER_OF = {
    "Warehouse_001": "Warehouse_000",
    "Warehouse_003": "Warehouse_002",
    "Warehouse_005": "Warehouse_004",
    "Warehouse_007": "Warehouse_006",
    "Warehouse_021": "Warehouse_020",
}
"""From the card (raw README at 2cbe9563, the scene table's footnote and the
2026-04-29 changelog): these are Cosmos Transfer 2.5 augmented
re-renderings of their source scene — same cameras, calibration and GT,
augmented RGB. ``Warehouse_024`` is the sixth, in test/, not fetched. The
script checks the "same calibration and GT" half byte for byte."""


def scene_dirs() -> list[Path]:
    return sorted(DATA_ROOT.glob("*/Warehouse_*"))


def cache_path(scene: Path) -> Path:
    return CACHE_ROOT / f"{scene.name}.npz"


def load_scene(scene: Path) -> SceneGroundTruth:
    return load_scene_ground_truth(scene / "ground_truth.json", cache_path(scene))


def probe_video(path: Path) -> dict[str, Any]:
    """Container metadata, plus a decode at the first, middle and last
    frame by seek. Not a full sequential decode — see the report."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"opened": False}
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    result: dict[str, Any] = {
        "opened": True,
        "container_frame_count": count,
        "fps": capture.get(cv2.CAP_PROP_FPS),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fourcc": "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)),
        "decoded": {},
    }
    for index in sorted({0, count // 2, max(count - 1, 0)}):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        result["decoded"][str(index)] = list(frame.shape) if ok else None
    capture.release()
    return result


def _dist(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "p1": float(np.percentile(values, 1)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def check_scene(scene: Path) -> dict[str, Any]:
    calibration = load_calibration(scene / "calibration.json")
    gt = load_scene(scene)
    videos = sorted(p.stem for p in (scene / "videos").glob("*.mp4"))

    cal_ids = sorted(calibration.cameras)
    gt_ids = sorted(gt.camera_names)
    camera_sets = {
        "calibration": cal_ids,
        "ground_truth_visible_keys": gt_ids,
        "videos": videos,
        "in_calibration_not_video": sorted(set(cal_ids) - set(videos)),
        "in_video_not_calibration": sorted(set(videos) - set(cal_ids)),
        "in_gt_not_calibration": sorted(set(gt_ids) - set(cal_ids)),
        "in_calibration_never_in_gt": sorted(set(cal_ids) - set(gt_ids)),
    }
    camera_sets["agree"] = set(cal_ids) == set(videos) and set(gt_ids) <= set(cal_ids)

    # H3: cameraMatrix vs intrinsic @ extrinsic. Raw, and after scaling K·E
    # so its [2, 3] entry is 1 — cameraMatrix is stored with P[2, 3] == 1,
    # and a projection matrix is only defined up to scale. Also: is the
    # homography exactly cameraMatrix's columns 0, 1, 3 (the z = 0 plane)?
    h3 = {}
    h3_scaled = {}
    homography_is_z0_plane = {}
    for cid, cam in calibration.cameras.items():
        product = cam.intrinsic @ cam.extrinsic
        norm = float(np.max(np.abs(cam.camera_matrix)))
        h3[cid] = float(np.max(np.abs(cam.camera_matrix - product)) / norm)
        h3_scaled[cid] = float(
            np.max(np.abs(cam.camera_matrix - product / product[2, 3])) / norm
        )
        homography_is_z0_plane[cid] = bool(
            np.allclose(
                cam.homography / cam.homography[2, 2],
                cam.camera_matrix[:, [0, 1, 3]] / cam.camera_matrix[2, 3],
                rtol=1e-9,
                atol=0.0,
            )
        )

    frames = gt.frame_ids_in_file_order
    frame_range = {
        "first": int(frames.min()),
        "last": int(frames.max()),
        "count": int(frames.size),
        "contiguous": bool(
            np.array_equal(np.sort(frames), np.arange(frames.min(), frames.max() + 1))
        ),
        "file_order_sorted": bool(np.all(np.diff(frames) > 0)),
    }

    # Entities by class: unique object ids; an id under two classes is a defect.
    by_class: dict[str, int] = {}
    ids_per_class = {}
    for k, name in enumerate(gt.class_names):
        ids_per_class[name] = set(np.unique(gt.object_id[gt.class_index == k]).tolist())
        by_class[name] = len(ids_per_class[name])
    names = list(ids_per_class)
    shared = sorted(
        {
            i
            for a in range(len(names))
            for b in range(a + 1, len(names))
            for i in ids_per_class[names[a]] & ids_per_class[names[b]]
        }
    )
    per_frame = np.bincount(
        np.searchsorted(np.sort(frames), gt.frame), minlength=frames.size
    )

    # H6/H7 raw evidence: box bottom under each "which scale index is height"
    # reading, per class. Interpreted in the report, not here.
    z = gt.location[:, 2]
    vertical = {}
    for k, name in enumerate(gt.class_names):
        rows = gt.class_index == k
        vertical[name] = {
            "z_centre": _dist(z[rows]),
            "z_minus_half_scale0": _dist(z[rows] - gt.scale[rows, 0] / 2),
            "z_minus_half_scale1": _dist(z[rows] - gt.scale[rows, 1] / 2),
            "z_minus_half_scale2": _dist(z[rows] - gt.scale[rows, 2] / 2),
            "rotation_abs_max_per_axis": np.abs(gt.rotation[rows]).max(axis=0).tolist(),
        }

    video_probes = {v: probe_video(scene / "videos" / f"{v}.mp4") for v in videos}

    return {
        "scene": scene.name,
        "split": scene.parent.name,
        "calibration_sha256": calibration.sha256,
        "twin_rev": calibration.twin_rev,
        "ground_truth_sha256": gt.source_sha256,
        "camera_sets": camera_sets,
        "calibration_fps": sorted({c.fps for c in calibration.cameras.values()}),
        "calibration_frame_size": sorted(
            {(c.frame_width, c.frame_height) for c in calibration.cameras.values()}
        ),
        "h3_relative_max_abs_diff": h3,
        "h3_relative_max_abs_diff_after_scale": h3_scaled,
        "homography_is_camera_matrix_z0_plane": homography_is_z0_plane,
        "gt_frame_range": frame_range,
        "gt_object_rows": int(gt.frame.size),
        "gt_visible_boxes": int(gt.box_row.size),
        "entities_by_class": by_class,
        "object_ids_under_two_classes": shared,
        "entities_per_frame": _dist(per_frame.astype(np.float64)),
        "vertical": vertical,
        "videos": video_probes,
    }


def summarise(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {s["scene"]: s for s in scenes}
    rerenders = {}
    for copy, source in CT25_RERENDER_OF.items():
        if copy in by_name and source in by_name:
            rerenders[copy] = {
                "source": source,
                "calibration_identical": by_name[copy]["calibration_sha256"]
                == by_name[source]["calibration_sha256"],
                "ground_truth_identical": by_name[copy]["ground_truth_sha256"]
                == by_name[source]["ground_truth_sha256"],
            }
    video_problems = []
    for s in scenes:
        n_gt = s["gt_frame_range"]["count"]
        for vid, probe in s["videos"].items():
            bad = (
                not probe.get("opened")
                or probe["width"] != 1920
                or probe["height"] != 1080
                or abs(probe["fps"] - 30.0) > 1e-6
                or any(shape != [1080, 1920, 3] for shape in probe["decoded"].values())
                or probe["container_frame_count"] != n_gt
            )
            if bad:
                video_problems.append(
                    {"scene": s["scene"], "video": vid, **probe, "gt_frames": n_gt}
                )
    classes: dict[str, int] = {}
    for s in scenes:
        if s["scene"] in CT25_RERENDER_OF:
            continue  # same GT as its source; counting it twice double-counts
        for name, n in s["entities_by_class"].items():
            classes[name] = classes.get(name, 0) + n
    return {
        "scenes": len(scenes),
        "videos": sum(len(s["videos"]) for s in scenes),
        "camera_set_disagreements": [
            s["scene"] for s in scenes if not s["camera_sets"]["agree"]
        ],
        "h3_worst_relative": max(
            v for s in scenes for v in s["h3_relative_max_abs_diff"].values()
        ),
        "h3_worst_relative_after_scale": max(
            v
            for s in scenes
            for v in s["h3_relative_max_abs_diff_after_scale"].values()
        ),
        "h3_best_relative_after_scale": min(
            v
            for s in scenes
            for v in s["h3_relative_max_abs_diff_after_scale"].values()
        ),
        "homography_is_z0_plane_everywhere": all(
            v
            for s in scenes
            for v in s["homography_is_camera_matrix_z0_plane"].values()
        ),
        "ct25_rerenders": rerenders,
        "video_problems": video_problems,
        "entities_by_class_unique_scenes": classes,
        "gt_keys_on_disk": sorted(GT_OBJECT_KEYS),
        "gt_keys_on_card": sorted(CARD_DOCUMENTED_GT_KEYS),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# PhysicalAI-SmartSpaces — data integrity (Day 41)",
        "",
        f"*{LANE_LABEL}*",
        "",
        f"Source: `nvidia/PhysicalAI-SmartSpaces@{COMMIT}`, `{LABEL}` "
        "(MTMC_Tracking_2026 train + val, no depth, nothing from test/). "
        "Generated by `python scripts/smartspaces_integrity.py`; numbers "
        "from `artifacts/smartspaces/day41_integrity.json`.",
        "",
        "## Summary",
        "",
        f"- Scenes: {summary['scenes']}; videos: {summary['videos']}.",
        "- Ground-truth keys on disk use spaces "
        f"({', '.join('`' + k + '`' for k in summary['gt_keys_on_disk'])}); "
        "the card documents underscores. The strict reader accepts only the "
        "on-disk spelling and raises on anything else.",
        "- Camera-set disagreements (calibration vs videos, or GT naming a "
        "camera with no calibration): "
        f"{summary['camera_set_disagreements'] or 'none'}.",
        "- H3: worst relative |cameraMatrix − K·E| "
        f"{summary['h3_worst_relative']:.3e}; "
        "after scaling K·E to cameraMatrix's normalisation (P[2,3] = 1) the "
        f"relative difference is {summary['h3_best_relative_after_scale']:.3e}–"
        f"{summary['h3_worst_relative_after_scale']:.3e} across cameras — "
        "proportional only approximately, so the two are not interchangeable "
        "until the reprojection check says which one the ground truth follows.",
        f"- Homography equals cameraMatrix's columns 0, 1, 3 (the z = 0 plane) "
        f"on every camera: {summary['homography_is_z0_plane_everywhere']}.",
        f"- Video problems (open, 1920×1080, 30 fps, decode at first/mid/last "
        f"frame, container frame count = GT frame count): "
        f"{len(summary['video_problems'])}.",
        "",
        "## Re-render pair map",
        "",
        "Each Cosmos Transfer 2.5 re-render renders its source scene's "
        "trajectories again with augmented RGB: one ground-truth trajectory "
        "set, two videos per camera. **How the pairs were identified:** "
        "named by the dataset card (raw README at `2cbe9563`: the scene "
        "table's footnote, \"Scenes 001 / 003 / 005 / 007 / 021 / 024 are "
        "Cosmos Transfer 2.5 (CT2.5) augmented re-renderings of 000 / 002 / "
        '004 / 006 / 020 / 023 respectively", and the 2026-04-29 changelog '
        "entry), then **checked byte for byte** here: each pair's "
        "`calibration.json` and `ground_truth.json` have equal sha256. The "
        "check was made only for the pairs the card names; no other pair of "
        "fetched scenes shares a ground-truth sha256 (all 23 hashes "
        "compared). Warehouse_023/024 are test scenes and were not fetched.",
        "",
        "| re-render | split | source | split | calibration identical "
        "| GT identical | straddles train/val |",
        "|---|---|---|---|---|---|---|",
    ]
    split_of = {sc["scene"]: sc["split"] for sc in payload["scenes"]}
    straddling = []
    for copy, r in summary["ct25_rerenders"].items():
        straddles = split_of[copy] != split_of[r["source"]]
        if straddles:
            straddling.append(copy)
        lines.append(
            f"| {copy} | {split_of[copy]} | {r['source']} | "
            f"{split_of[r['source']]} | {r['calibration_identical']} | "
            f"{r['ground_truth_identical']} | {'**YES**' if straddles else 'no'} |"
        )
    gt_shas: dict[str, list[str]] = {}
    for sc in payload["scenes"]:
        gt_shas.setdefault(sc["ground_truth_sha256"], []).append(sc["scene"])
    unnamed = [
        v
        for v in gt_shas.values()
        if len(v) > 1 and not set(v[1:]) <= set(CT25_RERENDER_OF)
    ]
    lines += [
        "",
        (
            f"**Pairs straddling train/val: {', '.join(straddling)}.**"
            if straddling
            else "**No pair straddles train/val**: the four train pairs are "
            "train on both sides, and Warehouse_020/021 are val on both sides "
            "— the card's split already keeps each re-render with its source."
        ),
        f"Scenes sharing a GT sha256 that the card does not name as a pair: "
        f"{unnamed or 'none'}.",
        "",
        "Entities (unique object ids) by class, re-renders excluded so no "
        "scene is counted twice:",
        "",
        "| class | entities |",
        "|---|---:|",
    ]
    for name, n in sorted(summary["entities_by_class_unique_scenes"].items()):
        lines.append(f"| {name} | {n} |")
    lines += [
        "",
        "## Per scene",
        "",
        "| scene | split | cams (cal / GT / video) | camera sets agree "
        "| GT frames | video frames (container) | fps | size | entities "
        "| per-frame p50 / p95 / max |",
        "|---|---|---|---|---|---|---|---|---:|---|",
    ]
    for s in payload["scenes"]:
        cs = s["camera_sets"]
        counts = sorted({v.get("container_frame_count") for v in s["videos"].values()})
        fps = sorted({round(v.get("fps", 0), 3) for v in s["videos"].values()})
        sizes = sorted(
            {f"{v.get('width')}×{v.get('height')}" for v in s["videos"].values()}
        )
        fr = s["gt_frame_range"]
        pf = s["entities_per_frame"]
        lines.append(
            f"| {s['scene']} | {s['split']} | {len(cs['calibration'])} / "
            f"{len(cs['ground_truth_visible_keys'])} / {len(cs['videos'])} | "
            f"{cs['agree']} | {fr['first']}–{fr['last']} ({fr['count']}"
            f"{', contiguous' if fr['contiguous'] else ', GAPS'}) | "
            f"{', '.join(map(str, counts))} | {', '.join(map(str, fps))} | "
            f"{', '.join(sizes)} | {sum(s['entities_by_class'].values())} | "
            f"{pf['p50']:.0f} / {pf['p95']:.0f} / {pf['max']:.0f} |"
        )
    lines += ["", "## Camera ids", ""]
    for s in payload["scenes"]:
        cs = s["camera_sets"]
        notes = [
            f"{k}: {', '.join(v)}" for k, v in cs.items() if k.startswith("in_") and v
        ]
        ids = ", ".join(c.removeprefix("Camera_") for c in cs["videos"])
        lines.append(
            f"- **{s['scene']}**: {ids}" + (f" — {'; '.join(notes)}" if notes else "")
        )
    lines += ["", "## Entities by class, per scene", ""]
    for s in payload["scenes"]:
        parts = ", ".join(f"{k} {v}" for k, v in sorted(s["entities_by_class"].items()))
        dup = s["object_ids_under_two_classes"]
        lines.append(
            f"- **{s['scene']}**: {parts}"
            + (f" — ids under two classes: {dup}" if dup else "")
        )
    if summary["video_problems"]:
        lines += ["", "## Video problems", ""]
        for p in summary["video_problems"]:
            detail = {k: v for k, v in p.items() if k not in ("scene", "video")}
            lines.append(f"- {p['scene']}/{p['video']}: {json.dumps(detail)}")
    lines += [
        "",
        "## What this does not check",
        "",
        "- Video decode is checked at three seek points per video, not by a "
        "full sequential decode. The Day-41 gate-parity run decodes its "
        "sampled cameras in full.",
        "- Whether the ground truth agrees with the calibration geometrically: "
        "that is the reprojection check (Objective 2), reported separately.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="re-render the markdown from the existing artifact, no rescan",
    )
    args = parser.parse_args(argv)
    if args.render_only:
        REPORT.write_text(render_markdown(json.loads(ARTIFACT.read_text())))
        print(f"Written: {REPORT.relative_to(REPO_ROOT)}")
        return 0

    scenes = []
    for scene in scene_dirs():
        if args.scene and scene.name not in args.scene:
            continue
        print(f"{scene.parent.name}/{scene.name} ...", flush=True)
        scenes.append(check_scene(scene))

    payload = {
        "label": LANE_LABEL,
        "commit": COMMIT,
        "summary": summarise(scenes),
        "scenes": scenes,
    }
    if args.scene:
        print(json.dumps(payload["summary"], indent=2, default=str))
        return 0
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    REPORT.write_text(render_markdown(payload))
    print(
        f"Written: {ARTIFACT.relative_to(REPO_ROOT)}, {REPORT.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
