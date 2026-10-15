"""Motion-gate wake parity on real-codec 1080p video (Day 41, Objective 5).

PhysicalAI-SmartSpaces — lane S, synthetic. Not evidence about real crowding.

Day 7/8 left full-resolution vs 320x180 gate parity without valid evidence:
the only real clip was 320x176, below the gate raster, so no downscale ran.
These videos are 1920x1080 H.264, so the downscale path genuinely runs.

Per sampled camera, the same decoded frames go through two gates that
differ only in raster — ``MotionGateConfig()`` (320x180, the product
default) and ``MotionGateConfig(gate_width=0, gate_height=0)`` (full
1080p) — and two things are reported, separately (Day 7):

- **wake-decision parity**: frames whose ``wake_next`` differs;
- **intermediate values**: the foreground fractions each gate computed,
  which can differ without changing any decision.

Sampling rule (fixed before running, seed 41): per SOURCE scene (a
distinct ground-truth trajectory set — Cosmos Transfer re-renders excluded),
cameras are ranked by GT busyness (mean visible boxes per frame over the
window) and split into a quiet half and a busy half; one camera is drawn
from each half. Each re-render then takes exactly its source's cameras, so
render and source are compared on the same camera and the same frames: the
same trajectories under different RGB. The first ``WINDOW_FRAMES`` frames
of each video are used. Results are grouped by trajectory set (a re-render
and its source are one), and the re-render pairs are their own stratum.

The scorecard scores the 320x180 gate with the Day-14/15 frame
(``assemble_gate_scorecard``: wake_fraction, recall_retained, miss_cost,
dataset.moving_frame_fraction, always/never-wake baselines). Motion is the
existing derivation (``world_motion``: world displacement above what one
pixel resolves at the entity's depth). Observability uses the visible 2D
box area as a proxy for silhouette area (``OBSERVABILITY_BOX_AREA``),
because SmartSpaces ships boxes, not instance masks; the scorecard says so
and ``require_comparable`` refuses to compare it with v3–v6.

Caveats carried in every artifact: rendered video has H.264 compression but
no sensor-noise model, so quiet-scene results are optimistic by an unknown
margin (Day 17); this is not the Tier-1 economic claim. NO timing, CPU or
latency figure is measured or reported (Day 19 lockout).

    python scripts/smartspaces_gate_parity.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smartspaces_integrity import (  # noqa: E402
    COMMIT,
    CT25_RERENDER_OF,
    LANE_LABEL,
    REPO_ROOT,
    load_scene,
    scene_dirs,
)

from src.cascade import MotionGate, StageContext  # noqa: E402
from src.cascade.envelope import DEFAULT_ENVELOPE_PATH, MeasuredEnvelope  # noqa: E402
from src.cascade.motion import MotionGateConfig  # noqa: E402
from src.data.scorecard import (  # noqa: E402
    OBSERVABILITY_BOX_AREA,
    Observability,
    Partition,
    Scorecard,
    assemble_gate_scorecard,
    current_measurement_environment,
    empty_gate_totals,
    gate_counts,
    world_motion,
)
from src.data.smartspaces import (  # noqa: E402
    MEASURED_GEOMETRY,
    SceneGroundTruth,
    load_calibration,
    project,
)

ARTIFACT = REPO_ROOT / "artifacts" / "smartspaces" / "day41_gate_parity.json"
SCORECARD = REPO_ROOT / "artifacts" / "smartspaces" / "day41_gate_scorecard.json"
SEED = 41
WINDOW_FRAMES = 1800
NATIVE = (1080, 1920)

CAVEATS = [
    LANE_LABEL,
    "Rendered video: H.264 compression, but no sensor-noise model. Quiet-scene "
    "results are optimistic by an unknown margin (Day 17).",
    "Not the Tier-1 economic claim: 60 s windows of authored scenes, not 24 h "
    "of real footage.",
    "Observability proxy: visible 2D box area stands in for silhouette area. A "
    "box overstates a silhouette, so more frames land above the envelope than "
    "an instance mask would put there. NOT comparable with v3-v6 scorecards.",
    "Loaded Transporters (Warehouse_010-016) and the Warehouse_019 stacker have "
    "2D boxes that include their payload (Day 41 reprojection); their box "
    "area overstates the 3D object's.",
    "No timing, CPU or latency figure is measured or reported (Day 19).",
]


def sample_cameras(seed: int = SEED) -> list[dict[str, Any]]:
    """The fixed, seeded sampling rule in the module docstring."""
    rng = np.random.default_rng(seed)
    scenes = scene_dirs()
    chosen_by_source: dict[str, list[dict[str, Any]]] = {}
    for scene in scenes:
        if scene.name in CT25_RERENDER_OF:
            continue
        gt = load_scene(scene)
        in_window = gt.frame[gt.box_row] < WINDOW_FRAMES
        busyness = {
            name: float(np.sum(in_window & (gt.box_camera == k)) / WINDOW_FRAMES)
            for k, name in enumerate(gt.camera_names)
        }
        videos = sorted(p.stem for p in (scene / "videos").glob("*.mp4"))
        for camera in videos:
            busyness.setdefault(camera, 0.0)
        ranked = sorted(videos, key=lambda c: (busyness[c], c))
        half = len(ranked) // 2
        strata = {"quiet": ranked[:half], "busy": ranked[half:]}
        picks = []
        for stratum, cameras in strata.items():
            if not cameras:
                continue
            camera = cameras[int(rng.integers(len(cameras)))]
            picks.append(
                {
                    "scene": scene.name,
                    "trajectory_set": scene.name,
                    "split": scene.parent.name,
                    "camera": camera,
                    "busyness_stratum": stratum,
                    "visible_boxes_per_frame": busyness[camera],
                    "ct25_rerender": False,
                }
            )
        chosen_by_source[scene.name] = picks
    chosen = [c for picks in chosen_by_source.values() for c in picks]
    for scene in scenes:
        source = CT25_RERENDER_OF.get(scene.name)
        if source is None:
            continue
        for pick in chosen_by_source[source]:
            chosen.append(
                {
                    **pick,
                    "scene": scene.name,
                    "split": scene.parent.name,
                    "ct25_rerender": True,
                }
            )
    return chosen


def compare_pairs(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render vs source, same camera, same frames: how often the wake
    decision differs, at each gate raster."""
    by_key = {(r["scene"], r["camera"]): r for r in results}
    out = []
    for r in results:
        if not r["ct25_rerender"]:
            continue
        src = by_key[(r["trajectory_set"], r["camera"])]
        row = {
            "rerender": r["scene"],
            "source": src["scene"],
            "camera": r["camera"],
            "split_rerender": r["split"],
            "split_source": src["split"],
        }
        for raster in ("320x180", "full"):
            a = np.frombuffer(r["wake_bits"][raster].encode(), dtype=np.uint8) - 48
            b = np.frombuffer(src["wake_bits"][raster].encode(), dtype=np.uint8) - 48
            n = min(a.size, b.size)
            row[raster] = {
                "frames": int(n),
                "wake_decisions_differ": int(np.sum(a[:n] != b[:n])),
                "wake_fraction_rerender": float(a[:n].mean()),
                "wake_fraction_source": float(b[:n].mean()),
            }
        out.append(row)
    return out


def by_trajectory_set(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Parity grouped per distinct ground-truth scene (re-render + source)."""
    groups: dict[str, dict[str, Any]] = {}
    for r in results:
        g = groups.setdefault(
            r["trajectory_set"],
            {"scenes": [], "cameras": 0, "frames": 0, "wake_decisions_differ": 0},
        )
        if r["scene"] not in g["scenes"]:
            g["scenes"].append(r["scene"])
        g["cameras"] += 1
        g["frames"] += r["frames_decoded"]
        g["wake_decisions_differ"] += r["wake_decisions_differ"]
    return groups


def _box_area_partition(
    gt: SceneGroundTruth,
    calibration: Any,
    camera: str,
    frames: int,
    gate_config: MotionGateConfig,
    envelope: MeasuredEnvelope,
) -> Partition:
    """The Day-6/15 observability partition with box area in place of the
    instance mask; same labels, same envelope, same world-motion rule."""
    cam = calibration.cameras[camera]
    entity_ids = np.unique(gt.object_id)
    column = {int(oid): i for i, oid in enumerate(entity_ids)}
    xyz = np.full((frames, entity_ids.size, 3), np.nan)
    rows = np.flatnonzero(gt.frame < frames)
    cols = np.array([column[int(o)] for o in gt.object_id[rows]])
    xyz[gt.frame[rows], cols] = gt.location[rows]

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
        world_moved = world_motion(xyz, intrinsics, extrinsics)

    native_px = NATIVE[0] * NATIVE[1]
    gate_px = gate_config.gate_width * gate_config.gate_height
    to_gate = gate_px / native_px
    area = np.zeros((frames, entity_ids.size))
    cam_index = gt.camera_names.index(camera) if camera in gt.camera_names else -1
    sel = np.flatnonzero((gt.box_camera == cam_index) & (gt.frame[gt.box_row] < frames))
    box = gt.box_xyxy[sel].astype(np.float64)
    box_rows = gt.box_row[sel]
    area[
        gt.frame[box_rows], [column[int(o)] for o in gt.object_id[box_rows]]
    ] = np.clip(box[:, 2] - box[:, 0], 0, None) * np.clip(
        box[:, 3] - box[:, 1], 0, None
    )
    area *= to_gate

    flat = xyz.reshape(-1, 3)
    finite = np.isfinite(flat).all(axis=1)
    uv = np.full((flat.shape[0], 2), np.nan)
    uv[finite] = project(flat[finite], cam, MEASURED_GEOMETRY.projection)[0]
    uv = uv.reshape(frames, entity_ids.size, 2)
    speed = np.zeros((frames, entity_ids.size))
    speed[1:] = np.nan_to_num(
        np.linalg.norm(np.diff(uv, axis=0), axis=2)
        * (gate_config.gate_width / NATIVE[1]),
        nan=0.0,
        posinf=0.0,
    )

    labels = np.full(frames, Observability.NO_MOTION, dtype=np.int8)
    for a in range(entity_ids.size):
        moving = world_moved[:, a]
        blind = moving & (area[:, a] <= 0)
        labels = np.where(
            blind, np.maximum(labels, int(Observability.NOT_OBSERVABLE)), labels
        ).astype(np.int8)
        if not moving.any():
            continue
        thresholds = np.array([envelope.wake_threshold_px(s) for s in speed[:, a]])
        rank = np.where(
            area[:, a] >= thresholds,
            Observability.ABOVE_ENVELOPE,
            np.where(
                area[:, a] > 0,
                Observability.BELOW_ENVELOPE,
                Observability.NOT_OBSERVABLE,
            ),
        )
        labels = np.where(moving, np.maximum(labels, rank), labels).astype(np.int8)
    return Partition(
        labels=labels,
        warm=gate_config.warmup_frames,
        envelope_gate_px=gate_config.min_foreground_fraction * gate_px,
        world_moved_any=np.any(world_moved, axis=1),
    )


def run_camera(sample: dict[str, Any]) -> dict[str, Any]:
    scene = next(p for p in scene_dirs() if p.name == sample["scene"])
    video = scene / "videos" / f"{sample['camera']}.mp4"
    product = MotionGateConfig()
    full = MotionGateConfig(gate_width=0, gate_height=0)
    gates = {"320x180": MotionGate(product), "full": MotionGate(full)}
    wake = {k: [] for k in gates}
    fraction = {k: [] for k in gates}
    shapes = {k: None for k in gates}
    capture = cv2.VideoCapture(str(video))
    for index in range(WINDOW_FRAMES):
        ok, frame = capture.read()
        if not ok:
            break
        for key, gate in gates.items():
            out = gate.process(frame[np.newaxis, ...], StageContext(index, index))
            wake[key].append(bool(out.wake_next))
            fraction[key].append(float(out.detail["foreground_fraction"]))
            shapes[key] = out.detail["gate_shape"]
    capture.release()
    frames = len(wake["full"])

    a = np.array(wake["320x180"])
    b = np.array(wake["full"])
    fa = np.array(fraction["320x180"])
    fb = np.array(fraction["full"])
    diff = np.abs(fa - fb)
    differing = np.flatnonzero(a != b)

    gt = load_scene(scene)
    calibration = load_calibration(scene / "calibration.json")
    envelope = MeasuredEnvelope.load(REPO_ROOT / DEFAULT_ENVELOPE_PATH)
    partition = _box_area_partition(
        gt, calibration, sample["camera"], frames, product, envelope
    )
    counts, rates = gate_counts(partition, a)
    # Cross-check only, labelled as such: the full-resolution decisions
    # scored against the SAME partition (whose envelope is the 320x180
    # one). It answers "did the frames the two gates disagree on matter",
    # not "what does a full-resolution gate score".
    counts_full, rates_full = gate_counts(partition, b)
    return {
        **sample,
        "frames_decoded": frames,
        "gate_shapes": shapes,
        "backend": gates["full"].backend,
        "wake_decisions_differ": int(differing.size),
        "first_differing_frames": differing[:20].tolist(),
        "wake_bits": {
            "320x180": "".join("1" if w else "0" for w in a),
            "full": "".join("1" if w else "0" for w in b),
        },
        "wake_fraction_320x180": float(a.mean()) if frames else None,
        "wake_fraction_full": float(b.mean()) if frames else None,
        "foreground_fraction_abs_diff": {
            "p50": float(np.percentile(diff, 50)),
            "p95": float(np.percentile(diff, 95)),
            "max": float(diff.max()),
            "frames_nonzero": int(np.count_nonzero(diff)),
        },
        "frames_moving_one_gate_only": int(
            np.sum(
                (fa >= product.min_foreground_fraction)
                != (fb >= full.min_foreground_fraction)
            )
        ),
        "counts": counts,
        "rates": rates,
        "full_res_decisions_against_same_partition": {
            "counts": counts_full,
            "rates": rates_full,
        },
        # Condition bucket: did GT put anything in this camera's view at all
        # during the window ("occupied") or not ("empty").
        "occupied": sample["visible_boxes_per_frame"] > 0,
    }


def build_scorecard(results: list[dict[str, Any]], label: str) -> Scorecard:
    envelope = MeasuredEnvelope.load(REPO_ROOT / DEFAULT_ENVELOPE_PATH)
    identity = [
        (r["scene"], r["camera"], r["frames_decoded"])
        for r in sorted(results, key=lambda r: (r["scene"], r["camera"]))
    ]
    set_sha = hashlib.sha256(
        json.dumps({"commit": COMMIT, "cameras": identity}).encode()
    ).hexdigest()
    card = Scorecard(
        golden_set_version=f"smartspaces-{COMMIT[:8]}-day41-{label}",
        golden_set_sha=set_sha,
        domain="indoor",
        clips_scored=len(results),
        observability_proxy=OBSERVABILITY_BOX_AREA,
        envelope=envelope.provenance(),
        measurement_environment=current_measurement_environment(),
        caveats=list(CAVEATS),
        includes_rerender_frames=label == "ct25",
        rerender_exclusion=(
            {
                "frames_after_cut": 0,
                "source": "Day 41 parity run: nothing excluded; the post-cut "
                "transient was first measured on Day 42",
            }
            if label == "ct25"
            else None
        ),
    )
    totals = empty_gate_totals()
    condition_totals = {
        b: empty_gate_totals() for b in ("occupied", "empty", "night", "degenerate")
    }
    condition_clips = {b: 0 for b in condition_totals}
    for r in results:
        bucket = "occupied" if r["occupied"] else "empty"
        for key in totals:
            totals[key] += r["counts"][key]
            condition_totals[bucket][key] += r["counts"][key]
        condition_clips[bucket] += 1
        card.per_clip[f"{r['scene']}/{r['camera']}"] = {
            **{k: float(v) for k, v in r["counts"].items()},
            **{k: float(v) for k, v in r["rates"].items()},
        }
    return assemble_gate_scorecard(
        card, totals, condition_totals, condition_clips, MotionGateConfig()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="first N samples only")
    args = parser.parse_args(argv)

    samples = sample_cameras()
    if args.limit:
        samples = samples[: args.limit]
    print(f"{len(samples)} cameras sampled (seed {SEED}, {WINDOW_FRAMES} frames each)")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_camera, samples))
    for r in results:
        print(
            f"  {r['scene']}/{r['camera']:<12} {r['busyness_stratum']:<5} "
            f"frames {r['frames_decoded']}  wake differ {r['wake_decisions_differ']}  "
            f"wake {r['wake_fraction_320x180']:.3f}/{r['wake_fraction_full']:.3f}  "
            f"|dfrac| max {r['foreground_fraction_abs_diff']['max']:.5f}"
        )
    source = [r for r in results if not r["ct25_rerender"]]
    rerender = [r for r in results if r["ct25_rerender"]]
    cards = {
        "source_scenes": build_scorecard(source, "source"),
        "ct25_rerenders": build_scorecard(rerender, "ct25"),
    }
    payload = {
        "label": LANE_LABEL,
        "caveats": CAVEATS,
        "sampling_rule": {
            "seed": SEED,
            "window_frames": WINDOW_FRAMES,
            "rule": (
                "per scene: rank cameras by mean GT-visible boxes per frame over "
                "the window, split into quiet/busy halves, draw one from each "
                "half with numpy default_rng(seed)"
            ),
        },
        "gate_configs": {
            "320x180": "MotionGateConfig() — product default",
            "full": "MotionGateConfig(gate_width=0, gate_height=0)",
        },
        "parity": {
            "cameras": len(results),
            "frames": sum(r["frames_decoded"] for r in results),
            "frames_with_differing_wake_decision": sum(
                r["wake_decisions_differ"] for r in results
            ),
            "cameras_with_any_difference": [
                f"{r['scene']}/{r['camera']}"
                for r in results
                if r["wake_decisions_differ"]
            ],
            "by_trajectory_set": by_trajectory_set(results),
        },
        "rerender_vs_source": compare_pairs(results),
        "cameras": results,
        "scorecards": {k: c.as_dict() for k, c in cards.items()},
    }
    if not args.limit:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        SCORECARD.write_text(
            json.dumps(
                {k: c.as_dict() for k, c in cards.items()}, indent=2, default=str
            )
            + "\n"
        )
        written = [p.relative_to(REPO_ROOT) for p in (ARTIFACT, SCORECARD)]
        print(f"Written: {written[0]}, {written[1]}")
    for key, card in cards.items():
        print(card.render())
    print(json.dumps(payload["parity"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
