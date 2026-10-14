"""Day-41 follow-ups to the gate-parity run.

Reads ``artifacts/smartspaces/day41_gate_parity.json``.

PhysicalAI-SmartSpaces — lane S, synthetic. No sensor-noise model (Day 17).
No timing figure is measured or reported.

1. Frames where the full-resolution gate woke and the 320x180 gate did not:
   what the ground truth says was happening on them (the observability
   partition the scorecard used), and a contact sheet of 20 of them.
2. The re-render/source pair whose wake decisions differ most: which scene
   and camera, and a contact sheet of 20 frames where they differ, source
   and re-render side by side.
3. Found while doing 2: each Cosmos re-render changes its whole look every
   900 frames (30 s). Measured on every camera of every re-render and of
   one source scene (control): the mean-colour jump between frames
   900k - 1 and 900k, against the jump between two consecutive frames in
   the middle of the same chunk.

Nothing is re-gated: both use the per-frame decisions stored in the parity
artifact, recompute the partition from GT (no video needed), and decode
only the frames drawn. Contact sheets go under
``artifacts/smartspaces/contact_sheets/`` (rendered dataset frames, kept
out of git); the counts go to ``artifacts/smartspaces/day41_gate_followup.json``.

    python scripts/smartspaces_gate_followup.py
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

from smartspaces_gate_parity import (  # noqa: E402
    ARTIFACT as PARITY_ARTIFACT,
    _box_area_partition,
)
from smartspaces_integrity import (  # noqa: E402
    LANE_LABEL,
    REPO_ROOT,
    load_scene,
    scene_dirs,
)  # noqa: E402

from src.cascade.envelope import DEFAULT_ENVELOPE_PATH, MeasuredEnvelope  # noqa: E402
from src.cascade.motion import MotionGateConfig  # noqa: E402
from src.data.scorecard import Observability  # noqa: E402
from src.data.smartspaces import SceneGroundTruth, load_calibration  # noqa: E402

OUT = REPO_ROOT / "artifacts" / "smartspaces" / "day41_gate_followup.json"
SHEETS = REPO_ROOT / "artifacts" / "smartspaces" / "contact_sheets"
SEED = 41
SAMPLES = 20
TILE = (480, 270)
LABEL_NAMES = {int(o): o.name for o in Observability}


def _bits(text: str) -> np.ndarray:
    return np.frombuffer(text.encode(), dtype=np.uint8) == ord("1")


def _scene(name: str) -> Path:
    return next(p for p in scene_dirs() if p.name == name)


def _frame(scene: Path, camera: str, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(scene / "videos" / f"{camera}.mp4"))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not decode {scene.name}/{camera} frame {index}")
    return image


def _draw(
    image: np.ndarray, gt: SceneGroundTruth, camera: str, index: int, caption: str
) -> np.ndarray:
    out = image.copy()
    if camera in gt.camera_names:
        k = gt.camera_names.index(camera)
        sel = np.flatnonzero((gt.box_camera == k) & (gt.frame[gt.box_row] == index))
        for i in sel:
            x1, y1, x2, y2 = (int(v) for v in gt.box_xyxy[i])
            name = gt.class_names[gt.class_index[gt.box_row[i]]]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(out, name, (x1, max(20, y1 - 6)), 0, 0.9, (0, 255, 0), 2)
    out = cv2.resize(out, TILE, interpolation=cv2.INTER_AREA)
    cv2.rectangle(out, (0, 0), (TILE[0], 24), (0, 0, 0), -1)
    cv2.putText(out, caption, (4, 17), 0, 0.45, (255, 255, 255), 1)
    return out


def _grid(tiles: list[np.ndarray], columns: int) -> np.ndarray:
    blank = np.zeros_like(tiles[0])
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        row += [blank] * (columns - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


STYLE_CUT_JUMP = 10.0
"""A boundary counts as a style cut when the frame's mean BGR moves by more
than 10 grey levels in one frame; mid-chunk consecutive frames move by
0.1 (median) and at most 2.0 in every scene measured."""


def style_cuts(scene_names: list[str]) -> dict[str, dict[str, Any]]:
    """Mean-colour jump across every 900-frame boundary, per scene."""

    def mean_at(capture: Any, index: int) -> np.ndarray:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = capture.read()
        if not ok:
            raise RuntimeError(f"could not decode frame {index}")
        return np.asarray(image.reshape(-1, 3).astype(np.float64).mean(axis=0))

    out = {}
    for name in scene_names:
        scene = _scene(name)
        boundary, control = [], []
        videos = sorted((scene / "videos").glob("*.mp4"))
        for video in videos:
            capture = cv2.VideoCapture(str(video))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            for b in range(900, frames, 900):
                boundary.append(
                    np.linalg.norm(mean_at(capture, b) - mean_at(capture, b - 1))
                )
                control.append(
                    np.linalg.norm(
                        mean_at(capture, b - 450) - mean_at(capture, b - 451)
                    )
                )
            capture.release()
        bnd, ctl = np.array(boundary), np.array(control)
        out[name] = {
            "cameras": len(videos),
            "boundaries": int(bnd.size),
            "boundaries_with_cut": int((bnd > STYLE_CUT_JUMP).sum()),
            "boundary_jump_median": float(np.median(bnd)),
            "boundary_jump_max": float(bnd.max()),
            "mid_chunk_jump_median": float(np.median(ctl)),
            "mid_chunk_jump_max": float(ctl.max()),
        }
    return out


def main() -> int:
    parity = json.loads(PARITY_ARTIFACT.read_text())
    cameras = parity["cameras"]
    config = MotionGateConfig()
    envelope = MeasuredEnvelope.load(REPO_ROOT / DEFAULT_ENVELOPE_PATH)
    gt_cache: dict[str, SceneGroundTruth] = {}

    def gt_for(name: str) -> SceneGroundTruth:
        if name not in gt_cache:
            gt_cache[name] = load_scene(_scene(name))
        return gt_cache[name]

    # -- 1. full-resolution-only wakes --------------------------------------
    labels_by_cam: dict[tuple[str, str], np.ndarray] = {}
    counts: Counter[str] = Counter()
    per_camera = []
    candidates: list[tuple[str, str, int]] = []
    for r in cameras:
        scene = _scene(r["scene"])
        gt = gt_for(r["scene"])
        small = _bits(r["wake_bits"]["320x180"])
        full = _bits(r["wake_bits"]["full"])
        partition = _box_area_partition(
            gt,
            load_calibration(scene / "calibration.json"),
            r["camera"],
            small.size,
            config,
            envelope,
        )
        labels = partition.labels
        labels_by_cam[(r["scene"], r["camera"])] = labels
        only_full = np.flatnonzero(full & ~small)
        in_warmup = only_full < partition.warm
        row = Counter(LABEL_NAMES[int(labels[i])] for i in only_full[~in_warmup])
        row["IN_WARMUP"] = int(in_warmup.sum())
        counts.update(row)
        per_camera.append(
            {
                "scene": r["scene"],
                "camera": r["camera"],
                "full_only_wakes": int(only_full.size),
                **row,
            }
        )
        candidates += [(r["scene"], r["camera"], int(i)) for i in only_full]

    rng = np.random.default_rng(SEED)
    picks = [
        candidates[i]
        for i in sorted(rng.choice(len(candidates), SAMPLES, replace=False))
    ]
    tiles = []
    for scene_name, camera, index in picks:
        label = LABEL_NAMES[int(labels_by_cam[(scene_name, camera)][index])]
        image = _frame(_scene(scene_name), camera, index)
        tiles.append(
            _draw(
                image,
                gt_for(scene_name),
                camera,
                index,
                f"{scene_name[-3:]}/{camera} f{index} full-only wake: {label}",
            )
        )
    SHEETS.mkdir(parents=True, exist_ok=True)
    sheet1 = SHEETS / "day41_full_res_only_wakes.png"
    cv2.imwrite(str(sheet1), _grid(tiles, 4))

    # -- 2. the pair that differs most --------------------------------------
    pairs = parity["rerender_vs_source"]
    worst = max(pairs, key=lambda p: p["320x180"]["wake_decisions_differ"])
    by_key = {(r["scene"], r["camera"]): r for r in cameras}
    rr = by_key[(worst["rerender"], worst["camera"])]
    src = by_key[(worst["source"], worst["camera"])]
    a = _bits(rr["wake_bits"]["320x180"])
    b = _bits(src["wake_bits"]["320x180"])
    differ = np.flatnonzero(a != b)
    pair_labels = labels_by_cam[(worst["source"], worst["camera"])]
    pair_counts = {
        "rerender_woke_source_slept": Counter(
            LABEL_NAMES[int(pair_labels[i])] for i in differ if a[i] and not b[i]
        ),
        "source_woke_rerender_slept": Counter(
            LABEL_NAMES[int(pair_labels[i])] for i in differ if b[i] and not a[i]
        ),
    }
    pair_picks = sorted(rng.choice(differ, min(SAMPLES, differ.size), replace=False))
    gt = gt_for(worst["source"])
    tiles = []
    for index in pair_picks:
        index = int(index)
        label = LABEL_NAMES[int(pair_labels[index])]
        for scene_name, woke in (
            (worst["source"], b[index]),
            (worst["rerender"], a[index]),
        ):
            image = _frame(_scene(scene_name), worst["camera"], index)
            tiles.append(
                _draw(
                    image,
                    gt,
                    worst["camera"],
                    index,
                    f"{scene_name[-3:]} f{index} "
                    f"{'WOKE' if woke else 'slept'} ({label})",
                )
            )
    sheet2 = SHEETS / "day41_rerender_vs_source_differ.png"
    cv2.imwrite(str(sheet2), _grid(tiles, 4))

    payload: dict[str, Any] = {
        "label": LANE_LABEL,
        "full_res_only_wakes": {
            "frames": int(sum(c["full_only_wakes"] for c in per_camera)),
            "by_partition_label": dict(counts),
            "meaning": {
                "ABOVE_ENVELOPE": (
                    "a GT event the 320x180 gate slept through and full res "
                    "caught (counted in the 320x180 miss_cost)"
                ),
                "BELOW_ENVELOPE": (
                    "GT motion visible but below the measured envelope (not a "
                    "defect either way)"
                ),
                "NOT_OBSERVABLE": "GT motion with no visible box on this camera",
                "NO_MOTION": "no resolvable GT motion: a full-res wake on nothing",
                "IN_WARMUP": "first frames, not scored by either gate's scorecard",
            },
            "per_camera": per_camera,
            "contact_sheet": str(sheet1.relative_to(REPO_ROOT)),
            "contact_sheet_frames": [list(p) for p in picks],
        },
        "largest_rerender_difference": {
            **worst,
            "fraction_320x180": worst["320x180"]["wake_decisions_differ"]
            / worst["320x180"]["frames"],
            "by_partition_label": {k: dict(v) for k, v in pair_counts.items()},
            "contact_sheet": str(sheet2.relative_to(REPO_ROOT)),
            "contact_sheet_frames": [int(i) for i in pair_picks],
        },
        "style_cuts_every_900_frames": {
            "threshold_mean_bgr_jump": STYLE_CUT_JUMP,
            "scenes": style_cuts(
                [
                    "Warehouse_001",
                    "Warehouse_003",
                    "Warehouse_005",
                    "Warehouse_007",
                    "Warehouse_021",
                    "Warehouse_000",
                ]
            ),
            "control_scene": "Warehouse_000 (a source scene)",
        },
        "seed": SEED,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in payload["full_res_only_wakes"].items()
                if k != "per_camera"
            },
            indent=2,
        )
    )
    print(json.dumps(payload["largest_rerender_difference"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
