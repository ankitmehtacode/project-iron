"""Motion-gate miss rate as a function of object size (Day 42, Objective A3).

PhysicalAI-SmartSpaces — lane S, synthetic. Not evidence about real crowding.

Day 41 labelled 1,581 frames "visible motion below the gate's measured
detection limit". That limit (``configs/envelope/gate_320x180.envelope.json``)
was measured on Kubric silhouettes at 320x180. This script does not use it to
classify anything. It asks the data directly: for each GT event frame, how
tall is the moving target, and did each gate wake?

No gate is re-run. The per-frame decisions are the ones Day 41's parity run
stored (``artifacts/smartspaces/day41_gate_parity.json``: 46 cameras, the
first 1,800 frames, both rasters), so the curve describes exactly the
decisions behind the Day-41 numbers.

Definitions (fixed before running):

- **event frame**: after the gate's warmup, at least one entity moves in the
  world (``world_motion``, the scorecard's rule) and has a visible 2D box on
  this camera.
- **target height**: the tallest visible box among the moving entities on
  that frame, in native pixels (1080p); in gate pixels it is that times
  ``gate_height / 1080`` (180/1080 for the product gate, 1 for full res).
  The tallest is used because a gate wakes on any mover; a frame with a large
  mover and a small one is not evidence about small objects.
- **miss**: the gate's ``wake_next`` is False on an event frame (the same
  convention as ``gate_counts``).

Every bin is reported with its count and a 95% Wilson interval. The heights
at which the 320x180 gate's recall drops under 0.9, 0.8 and 0.5 are read off
the curve (first bin, from the top, whose point estimate falls under the
level); they are descriptions of this data, not design thresholds.

NO timing, CPU or latency figure is measured or reported (Day 19).

    python scripts/smartspaces_gate_miss_by_size.py
"""

from __future__ import annotations

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
)  # noqa: E402

from src.cascade.motion import MotionGateConfig  # noqa: E402
from src.data.scorecard import world_motion  # noqa: E402
from src.data.smartspaces import load_calibration  # noqa: E402
from src.eval.intervals import wilson_interval  # noqa: E402

PARITY = REPO_ROOT / "artifacts" / "smartspaces" / "day41_gate_parity.json"
FOLLOWUP = REPO_ROOT / "artifacts" / "smartspaces" / "day41_gate_followup.json"
ARTIFACT = REPO_ROOT / "artifacts" / "smartspaces" / "day42_gate_miss_by_size.json"
NATIVE_HEIGHT = 1080
HEIGHT_EDGES_NATIVE = [0, 20, 30, 40, 50, 60, 80, 100, 130, 160, 200, 260, 340, 1081]
"""Native-pixel bin edges. Finer at the small end, where the curve moves."""
RECALL_LEVELS = (0.9, 0.8, 0.5)
MIN_BIN = 30
"""A bin with fewer event frames is reported but marked too thin to read."""


def _bits(text: str) -> np.ndarray:
    return np.frombuffer(text.encode(), dtype=np.uint8) == ord("1")


def camera_events(scene_name: str, camera: str, frames: int) -> dict[str, np.ndarray]:
    """Per frame: is it an event frame, and the tallest moving visible box."""
    scene = next(p for p in scene_dirs() if p.name == scene_name)
    gt = load_scene(scene)
    cam = load_calibration(scene / "calibration.json").cameras[camera]
    ids = np.unique(gt.object_id)
    column = {int(o): i for i, o in enumerate(ids)}
    xyz = np.full((frames, ids.size, 3), np.nan)
    rows = np.flatnonzero(gt.frame < frames)
    xyz[gt.frame[rows], [column[int(o)] for o in gt.object_id[rows]]] = gt.location[
        rows
    ]
    intrinsics = np.array(
        [
            cam.intrinsic[0, 0],
            cam.intrinsic[1, 1],
            cam.intrinsic[0, 2],
            cam.intrinsic[1, 2],
        ]
    )
    extrinsics = np.vstack([cam.extrinsic, [0.0, 0.0, 0.0, 1.0]])
    with np.errstate(invalid="ignore"):
        moved = world_motion(xyz, intrinsics, extrinsics)

    height = np.zeros((frames, ids.size))
    centre = np.full((frames, ids.size, 2), np.nan)
    klass = np.full(ids.size, -1)
    klass[[column[int(o)] for o in gt.object_id]] = gt.class_index
    if camera in gt.camera_names:
        k = gt.camera_names.index(camera)
        sel = np.flatnonzero((gt.box_camera == k) & (gt.frame[gt.box_row] < frames))
        box = gt.box_xyxy[sel].astype(np.float64)
        r = gt.box_row[sel]
        cols = [column[int(o)] for o in gt.object_id[r]]
        height[gt.frame[r], cols] = np.clip(box[:, 3] - box[:, 1], 0, None)
        centre[gt.frame[r], cols] = (box[:, :2] + box[:, 2:]) / 2
    moving_visible = moved & (height > 0)
    masked = np.where(moving_visible, height, 0.0)
    which = masked.argmax(axis=1)
    tallest = masked.max(axis=1)
    # Image speed of that mover: box-centre displacement since the previous
    # frame, native px/frame (NaN where it was not visible on both frames).
    step = np.full(frames, np.nan)
    rows_t = np.arange(1, frames)
    step[1:] = np.linalg.norm(
        centre[rows_t, which[1:]] - centre[rows_t - 1, which[1:]], axis=1
    )
    return {
        "event": moving_visible.any(axis=1),
        "tallest_native": tallest,
        "tallest_class": np.array(gt.class_names + ("?",))[klass[which]],
        "tallest_speed_native": step,
    }


def curve(
    heights: np.ndarray, missed: np.ndarray, edges: list[float]
) -> list[dict[str, Any]]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (heights >= lo) & (heights < hi)
        n = int(sel.sum())
        m = int((missed & sel).sum())
        row: dict[str, Any] = {
            "height_lo": lo,
            "height_hi": hi,
            "events": n,
            "misses": m,
        }
        if n:
            row["miss_rate"] = m / n
            row["miss_rate_wilson95"] = list(wilson_interval(m, n))
            row["recall"] = 1 - m / n
        else:
            row["miss_rate"] = None
        row["too_thin"] = n < MIN_BIN
        out.append(row)
    return out


def recall_crossings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Walking down from the tallest bin, the first bin whose recall point
    estimate falls under each level (thin bins skipped and counted)."""
    out = {}
    for level in RECALL_LEVELS:
        hit = None
        for row in reversed(rows):
            if row["too_thin"] or row["miss_rate"] is None:
                continue
            if row["recall"] < level:
                hit = {
                    "bin_native_px": [row["height_lo"], row["height_hi"]],
                    "recall": row["recall"],
                    "events": row["events"],
                }
                break
        out[f"recall_below_{level}"] = hit
    return out


def main() -> int:
    parity = json.loads(PARITY.read_text())
    warm = MotionGateConfig().warmup_frames
    per_raster: dict[str, dict[str, list[np.ndarray]]] = {
        r: {"h": [], "miss": [], "rerender": [], "cls": [], "speed": [], "cam": []}
        for r in ("320x180", "full")
    }
    for cam in parity["cameras"]:
        frames = cam["frames_decoded"]
        ev = camera_events(cam["scene"], cam["camera"], frames)
        event = ev["event"].copy()
        event[:warm] = False
        for raster in ("320x180", "full"):
            wake = _bits(cam["wake_bits"][raster])[:frames]
            per_raster[raster]["h"].append(ev["tallest_native"][event])
            per_raster[raster]["miss"].append(~wake[event])
            per_raster[raster]["rerender"].append(
                np.full(int(event.sum()), cam["ct25_rerender"])
            )
            per_raster[raster]["cls"].append(ev["tallest_class"][event])
            per_raster[raster]["speed"].append(ev["tallest_speed_native"][event])
            per_raster[raster]["cam"].append(
                np.full(int(event.sum()), f"{cam['scene']}/{cam['camera']}")
            )
        print(f"  {cam['scene']}/{cam['camera']}: {int(event.sum())} event frames")

    out: dict[str, Any] = {}
    for raster, parts in per_raster.items():
        h = np.concatenate(parts["h"])
        miss = np.concatenate(parts["miss"])
        rer = np.concatenate(parts["rerender"])
        cls = np.concatenate(parts["cls"])
        speed = np.concatenate(parts["speed"])
        cams = np.concatenate(parts["cam"])
        scale = (
            (MotionGateConfig().gate_height / NATIVE_HEIGHT)
            if raster == "320x180"
            else 1.0
        )
        strata = {"all": np.ones(h.size, bool), "source": ~rer, "ct25_rerender": rer}
        out[raster] = {
            "gate_px_per_native_px": scale,
            "strata": {
                name: {
                    "events": int(sel.sum()),
                    "misses": int((miss & sel).sum()),
                    "curve": curve(h[sel], miss[sel], HEIGHT_EDGES_NATIVE),
                }
                for name, sel in strata.items()
            },
        }
        for name in strata:
            rows = out[raster]["strata"][name]["curve"]
            for row in rows:
                row["height_lo_gate_px"] = row["height_lo"] * scale
                row["height_hi_gate_px"] = row["height_hi"] * scale
            out[raster]["strata"][name]["recall_crossings"] = recall_crossings(rows)
        out[raster]["by_class"] = {
            str(c): curve(h[cls == c], miss[cls == c], HEIGHT_EDGES_NATIVE)
            for c in np.unique(cls)
        }
        finite = np.isfinite(speed)
        terciles = np.percentile(speed[finite], [100 / 3, 200 / 3])
        out[raster]["speed_terciles_native_px_per_frame"] = terciles.tolist()
        bands = np.digitize(np.nan_to_num(speed, nan=-1.0), terciles)
        out[raster]["by_speed_tercile"] = {
            name: curve(h[sel], miss[sel], HEIGHT_EDGES_NATIVE)
            for name, sel in {
                "slow": finite & (bands == 0),
                "medium": finite & (bands == 1),
                "fast": finite & (bands == 2),
                "speed_unknown": ~finite,
            }.items()
        }
        mid = (h >= 60) & (h < 130)
        out[raster]["cameras_in_60_130_bin"] = sorted(
            (
                {
                    "camera": str(c),
                    "events": int((mid & (cams == c)).sum()),
                    "misses": int((mid & (cams == c) & miss).sum()),
                }
                for c in np.unique(cams[mid])
            ),
            key=lambda r: -r["misses"],
        )[:12]

    followup = json.loads(FOLLOWUP.read_text())
    payload = {
        "label": LANE_LABEL,
        "caveats": [
            LANE_LABEL,
            "Decisions are Day 41's stored wake bits (46 cameras, first 1800 frames).",
            "Target height is the tallest moving visible GT box; boxes of loaded "
            "Transporters and the Warehouse_019 stacker include their payload.",
            "The Kubric-measured envelope is NOT used to classify any frame here.",
            "No timing, CPU or latency figure is measured or reported (Day 19).",
        ],
        "definitions": {
            "event_frame": "post-warmup frame where >=1 entity moves in the world "
            "and has a visible box on this camera",
            "target_height": "tallest visible box among moving entities, native px",
            "miss": "wake_next False on an event frame",
            "height_edges_native_px": HEIGHT_EDGES_NATIVE,
            "min_events_per_bin": MIN_BIN,
        },
        "rasters": out,
        "day41_below_envelope_question": _below_envelope(followup),
    }
    ARTIFACT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Written: {ARTIFACT.relative_to(REPO_ROOT)}")
    for raster in ("320x180", "full"):
        print(raster, json.dumps(out[raster]["strata"]["all"]["recall_crossings"]))
    return 0


def _below_envelope(followup: dict[str, Any]) -> dict[str, Any]:
    """The Day-41 1,581 frames, by the partition both gates were scored on."""
    parity = json.loads(PARITY.read_text())
    below = caught_320 = caught_full = 0
    for cam in parity["cameras"]:
        below += cam["counts"]["below_envelope_frames"]
        caught_320 += (
            cam["counts"]["below_envelope_frames"]
            - cam["counts"]["envelope_limited_misses"]
        )
        f = cam["full_res_decisions_against_same_partition"]["counts"]
        caught_full += f["below_envelope_frames"] - f["envelope_limited_misses"]
    return {
        "below_envelope_frames_all_46_cameras": below,
        "woken_by_320x180": caught_320,
        "woken_by_full_res": caught_full,
        "day41_only_full_res_woke_and_below": 1581,
        "note": (
            "The 1,581 are by construction frames the full-resolution gate woke "
            "on and the 320x180 gate did not; the question that matters is the "
            "whole below-envelope population, reported above."
        ),
        "day41_followup_partition_counts": followup.get("full_res_only_wakes", {}),
    }


if __name__ == "__main__":
    raise SystemExit(main())
