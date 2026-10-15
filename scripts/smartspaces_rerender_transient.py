"""Cut-aware handling of the Cosmos Transfer re-renders (Day 42, Objective A2).

PhysicalAI-SmartSpaces — lane S, synthetic. Not evidence about real crowding.

Day 41 found that every re-render changes its whole look every 30 s. A
background-subtraction gate does not recover from a global change in a few
frames: MOG2 here is built with ``history=250`` and ``apply()`` is called
with no learning rate (``src/cascade/motion.py``), so OpenCV's automatic
rate 1/min(n, 250) applies and the model re-adapts over hundreds of frames.
This script MEASURES that transient instead of assuming a window.

Two passes, both resumable (per-camera arrays cached under ``data/derived``):

1. **Per-frame pass**, every camera of every fetched re-render (001, 003,
   005, 007, 021) and of its source (000, 002, 004, 006, 020), whole video:
   the frame's mean BGR (for cut detection) and the product 320x180 gate's
   foreground fraction, raw motion bit and ``wake_next``. The 20 cameras
   Day 41 sampled for these pairs also get the full-resolution gate.
2. **Analysis** over the cache:

   - *Cuts from pixels.* ``jump[t] = ||meanBGR[t] - meanBGR[t-1]||``; a cut
     is ``jump > 10`` (Day 41: mid-chunk consecutive frames move by at most
     2.0 in every scene). Cuts are found at whatever frame they occur, not
     assumed at multiples of 900; boundaries at multiples of 900 with no cut
     are counted, and so are cuts anywhere else and cuts in source scenes.
   - *Transient curve.* For every detected cut ``c > 0`` in a re-render
     camera, the gate's wake indicator at ``c + k`` for ``k`` up to the next
     cut, paired with the SOURCE scene's same camera at the same frame index
     (same trajectories, same calibration, no cut). Binned in 30-frame bins
     over k in [0, 900); intervals are mean +/- z * SE over (camera, cut)
     units.
   - *Transient end*, defined before running. Two readings, both reported:
     ``returns_to_source``: the first bin from which every later bin's
     simultaneous (Bonferroni, 30 bins) 95% interval of the paired
     difference (re-render minus source) contains 0.
     ``reaches_steady_state``: the first bin from which every later bin's
     simultaneous interval contains the late-chunk difference (mean over k
     in [600, 900)). Revised from pointwise 95% intervals before the real
     data was analysed: the unit test showed the pointwise "every later
     bin" rule fails by chance on a flat curve (see ``Z_SIMULTANEOUS``).
     The exclusion window is the steady-state reading — the part of the
     difference that decays with time since the cut. A difference that
     never decays is not adaptation: it is the re-render's appearance, and
     is reported as what remains. If no bin satisfies a reading, the
     reading is ``None`` with the bound (900 frames searched).

NO timing, CPU or latency figure is measured or reported (Day 19).

    python scripts/smartspaces_rerender_transient.py           # both passes
    python scripts/smartspaces_rerender_transient.py --analyse  # cache only
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

from smartspaces_gate_parity import _box_area_partition  # noqa: E402
from smartspaces_integrity import (  # noqa: E402
    CACHE_ROOT,
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
)
from src.data.smartspaces import load_calibration  # noqa: E402

ARTIFACT = REPO_ROOT / "artifacts" / "smartspaces" / "day42_rerender_transient.json"
PARITY = REPO_ROOT / "artifacts" / "smartspaces" / "day41_gate_parity.json"
FRAME_CACHE = CACHE_ROOT / "day42_rerender_transient"
STYLE_CUT_JUMP = 10.0
CHUNK = 900
BIN = 30
LATE = (600, 900)
Z_POINTWISE = 1.959963984540054
Z_SIMULTANEOUS = 3.1355
"""Two-sided Bonferroni over the 30 bins of one curve: alpha = 0.05 / 30,
z = Phi^-1(1 - alpha / 2). The transient-end rule asks that EVERY later bin
contain the target; with pointwise 95% intervals that fails by chance on a
flat curve about 70% of the time (0.95 ** 25), so it is read against
simultaneous intervals. Chosen before the real data was analysed, after
``tests/test_rerender_transient.py`` exposed the pointwise version."""
SEED = 42
DAY41_WINDOW = 1800
MOG2_HISTORY = 250
"""Read from ``src/cascade/motion.py`` (``createBackgroundSubtractorMOG2(
history=250, ...)``); recorded here so the artifact names the gate it
measured. ``test_rerender_transient.py`` checks the two agree."""


def _scene(name: str) -> Path:
    return next(p for p in scene_dirs() if p.name == name)


def jobs() -> list[dict[str, Any]]:
    parity = json.loads(PARITY.read_text())
    sampled = {
        (c["trajectory_set"], c["camera"])
        for c in parity["cameras"]
        if c["ct25_rerender"]
    }
    out = []
    for rerender, source in sorted(CT25_RERENDER_OF.items()):
        if not any(p.name == rerender for p in scene_dirs()):
            continue
        for scene in (rerender, source):
            for video in sorted((_scene(scene) / "videos").glob("*.mp4")):
                out.append(
                    {
                        "scene": scene,
                        "camera": video.stem,
                        "full_res": (source, video.stem) in sampled,
                    }
                )
    return out


def cache_file(job: dict[str, Any]) -> Path:
    return FRAME_CACHE / f"{job['scene']}__{job['camera']}.npz"


def run_job(job: dict[str, Any]) -> str:
    """Decode one whole video once; write per-frame arrays. Idempotent."""
    out = cache_file(job)
    if out.exists():
        return f"cached {out.name}"
    # One OpenCV thread per worker process: the pool supplies the parallelism,
    # and per-pixel MOG2 output does not depend on the thread count.
    cv2.setNumThreads(1)
    video = _scene(job["scene"]) / "videos" / f"{job['camera']}.mp4"
    gates = {"320x180": MotionGate(MotionGateConfig())}
    if job["full_res"]:
        gates["full"] = MotionGate(MotionGateConfig(gate_width=0, gate_height=0))
    means: list[np.ndarray] = []
    frac: dict[str, list[float]] = {k: [] for k in gates}
    moving: dict[str, list[bool]] = {k: [] for k in gates}
    wake: dict[str, list[bool]] = {k: [] for k in gates}
    capture = cv2.VideoCapture(str(video))
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        means.append(np.asarray(cv2.mean(frame)[:3], dtype=np.float64))
        for key, gate in gates.items():
            result = gate.process(frame[np.newaxis, ...], StageContext(index, index))
            frac[key].append(float(result.detail["foreground_fraction"]))
            moving[key].append(bool(result.detail["moving"]))
            wake[key].append(bool(result.wake_next))
        index += 1
    capture.release()
    arrays: dict[str, np.ndarray] = {"mean_bgr": np.array(means)}
    for key in gates:
        arrays[f"fraction_{key}"] = np.array(frac[key], dtype=np.float32)
        arrays[f"moving_{key}"] = np.array(moving[key], dtype=bool)
        arrays[f"wake_{key}"] = np.array(wake[key], dtype=bool)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.rename(out)
    return f"wrote {out.name} ({index} frames)"


# --- analysis -------------------------------------------------------------


def detect_cuts(mean_bgr: np.ndarray) -> np.ndarray:
    jump = np.linalg.norm(np.diff(mean_bgr, axis=0), axis=1)
    return np.flatnonzero(jump > STYLE_CUT_JUMP) + 1


def _ci(units: np.ndarray, z: float) -> tuple[float, float, float]:
    """Mean and ``mean +/- z * standard error`` over (camera, cut) units."""
    units = units[np.isfinite(units)]
    if units.size < 2:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(units.mean())
    half = z * float(units.std(ddof=1)) / float(np.sqrt(units.size))
    return (mean, mean - half, mean + half)


def cut_segments(cuts: np.ndarray, frames: int) -> list[tuple[int, int]]:
    """(cut, end) for every cut after frame 0; end is the next cut or EOF."""
    bounds = [int(c) for c in cuts if c > 0] + [frames]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def transient_curve(
    pairs: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]],
    key: str,
) -> dict[str, Any]:
    """Per 30-frame bin of frames-since-cut: re-render, source and paired
    difference, each a mean over (camera, cut) units with a bootstrap CI."""
    bins = []
    unit_rows: list[np.ndarray] = []
    for rer, src, cuts in pairs:
        r = rer[key].astype(np.float64)
        s = src[key].astype(np.float64)
        n = min(r.size, s.size)
        for start, end in cut_segments(cuts, n):
            row = np.full((3, CHUNK), np.nan)
            span = min(end - start, CHUNK)
            row[0, :span] = r[start : start + span]
            row[1, :span] = s[start : start + span]
            row[2, :span] = row[0, :span] - row[1, :span]
            unit_rows.append(row)
    stack = np.stack(unit_rows)  # (units, 3, CHUNK)
    for b0 in range(0, CHUNK, BIN):
        per_unit = np.nanmean(stack[:, :, b0 : b0 + BIN], axis=2)
        bins.append(
            {
                "k_start": b0,
                "rerender": _ci(per_unit[:, 0], Z_POINTWISE),
                "source": _ci(per_unit[:, 1], Z_POINTWISE),
                "difference": _ci(per_unit[:, 2], Z_POINTWISE),
                "difference_simultaneous": _ci(per_unit[:, 2], Z_SIMULTANEOUS),
            }
        )
    late = np.nanmean(stack[:, 2, LATE[0] : LATE[1]], axis=1)
    late_ci = _ci(late, Z_POINTWISE)

    def first_stable(target: float) -> int | None:
        for i in range(len(bins)):
            if all(
                b["difference_simultaneous"][1]
                <= target
                <= b["difference_simultaneous"][2]
                for b in bins[i:]
            ):
                return bins[i]["k_start"]
        return None

    return {
        "units": int(stack.shape[0]),
        "bins": bins,
        "late_chunk_difference": late_ci,
        "returns_to_source_k": first_stable(0.0),
        # Every bin whose simultaneous interval lies wholly above or below 0,
        # so a later excess or deficit is visible whatever the rules return.
        "significant_bins": [
            {"k_start": b["k_start"], "difference": b["difference"][0]}
            for b in bins
            if b["difference_simultaneous"][1] > 0.0
            or b["difference_simultaneous"][2] < 0.0
        ],
        "reaches_steady_state_k": first_stable(late_ci[0]),
        "searched_k": CHUNK,
    }


def excess_ends_k(curve: dict[str, Any]) -> int | None:
    """POST HOC (Day 42, after viewing the curve): the first bin whose
    simultaneous interval of the paired difference reaches zero — where the
    IMMEDIATE post-cut excess ends. A first crossing only: later bins can be
    significant again (the curve's ``significant_bins`` lists them), so this
    is not a claim that the difference is gone after it. Kept separate from
    the pre-registered rule, which is also reported: the two answer different
    questions."""
    for b in curve["bins"]:
        if b["difference_simultaneous"][1] <= 0.0:
            return int(b["k_start"])
    return None


def exclusion_mask(cuts: np.ndarray, frames: int, window: int) -> np.ndarray:
    keep = np.ones(frames, dtype=bool)
    for c in cuts:
        if c > 0:
            keep[c : c + window] = False
    return keep


def _partition(scene: str, camera: str, frames: int) -> Partition:
    """The Day-41 box-area partition. A re-render shares its source's GT and
    calibration byte for byte, so one partition serves both."""
    gt = load_scene(_scene(scene))
    calibration = load_calibration(_scene(scene) / "calibration.json")
    envelope = MeasuredEnvelope.load(REPO_ROOT / DEFAULT_ENVELOPE_PATH)
    return _box_area_partition(
        gt, calibration, camera, frames, MotionGateConfig(), envelope
    )


def _subset(partition: Partition, keep: np.ndarray) -> Partition:
    """The partition restricted to kept frames. Cuts are never inside the
    warmup (the first cut is at frame 900), so the warmup prefix survives."""
    if not keep[: partition.warm].all():
        raise ValueError("exclusion window reaches into the gate warmup")
    return Partition(
        labels=partition.labels[keep],
        warm=partition.warm,
        envelope_gate_px=partition.envelope_gate_px,
        world_moved_any=partition.world_moved_any[keep],
    )


def gate_scorecard(
    rows: list[tuple[str, Partition, np.ndarray, np.ndarray]],
    label: str,
    rerender: bool,
    window: dict[str, Any] | None,
) -> Scorecard:
    """320x180 gate scorecard over whole videos, Day-41 arithmetic, with the
    exclusion window recorded on the card itself."""
    envelope = MeasuredEnvelope.load(REPO_ROOT / DEFAULT_ENVELOPE_PATH)
    identity = sorted((name, int(keep.sum())) for name, _, _, keep in rows)
    set_sha = hashlib.sha256(
        json.dumps({"commit": COMMIT, "cameras": identity}).encode()
    ).hexdigest()
    card = Scorecard(
        golden_set_version=f"smartspaces-{COMMIT[:8]}-day42-{label}",
        golden_set_sha=set_sha,
        domain="indoor",
        clips_scored=len(rows),
        observability_proxy=OBSERVABILITY_BOX_AREA,
        envelope=envelope.provenance(),
        measurement_environment=current_measurement_environment(),
        caveats=[
            LANE_LABEL,
            "Whole 9000-frame videos of the five fetched re-renders and their "
            "sources, every camera; 320x180 product gate.",
            "Observability proxy: visible 2D box area (Day 41); not comparable "
            "with v3-v6.",
            "No timing, CPU or latency figure is measured or reported (Day 19).",
        ],
        includes_rerender_frames=rerender,
        rerender_exclusion=window,
    )
    totals = empty_gate_totals()
    condition_totals = {
        b: empty_gate_totals() for b in ("occupied", "empty", "night", "degenerate")
    }
    condition_clips = {b: 0 for b in condition_totals}
    for name, partition, wake, keep in rows:
        counts, rates = gate_counts(_subset(partition, keep), wake[keep])
        occupied = bool(np.any(partition.labels[keep] != Observability.NO_MOTION))
        bucket = "occupied" if occupied else "empty"
        for key in totals:
            totals[key] += counts[key]
            condition_totals[bucket][key] += counts[key]
        condition_clips[bucket] += 1
        card.per_clip[name] = {
            **{k: float(v) for k, v in counts.items()},
            **{k: float(v) for k, v in rates.items()},
        }
    return assemble_gate_scorecard(
        card, totals, condition_totals, condition_clips, MotionGateConfig()
    )


def analyse() -> dict[str, Any]:
    all_jobs = jobs()
    data = {(j["scene"], j["camera"]): dict(np.load(cache_file(j))) for j in all_jobs}

    # 1. cuts, every camera
    cut_rows: dict[str, Any] = {}
    cuts_by_cam: dict[tuple[str, str], np.ndarray] = {}
    for (scene, camera), arrays in data.items():
        cuts = detect_cuts(arrays["mean_bgr"])
        cuts_by_cam[(scene, camera)] = cuts
        frames = arrays["mean_bgr"].shape[0]
        boundaries = np.arange(CHUNK, frames, CHUNK)
        jump = np.linalg.norm(np.diff(arrays["mean_bgr"], axis=0), axis=1)
        row = cut_rows.setdefault(
            scene,
            {
                "ct25_rerender": scene in CT25_RERENDER_OF,
                "cameras": 0,
                "boundaries": 0,
                "boundaries_with_cut": 0,
                "boundaries_without_cut": [],
                "cuts_off_boundary": [],
                "cut_jump_median": [],
                "non_cut_jump_max": 0.0,
            },
        )
        row["cameras"] += 1
        row["boundaries"] += int(boundaries.size)
        hit = np.isin(boundaries, cuts)
        row["boundaries_with_cut"] += int(hit.sum())
        for b, j in zip(boundaries[~hit], jump[boundaries[~hit] - 1]):
            row["boundaries_without_cut"].append(
                {"camera": camera, "frame": int(b), "jump": round(float(j), 3)}
            )
        for c in cuts[~np.isin(cuts, boundaries)]:
            row["cuts_off_boundary"].append(
                {"camera": camera, "frame": int(c), "jump": float(jump[c - 1])}
            )
        row["cut_jump_median"].extend(jump[cuts - 1].tolist())
        quiet = np.ones(jump.size, dtype=bool)
        quiet[cuts - 1] = False
        row["non_cut_jump_max"] = max(row["non_cut_jump_max"], float(jump[quiet].max()))
    for row in cut_rows.values():
        values = row.pop("cut_jump_median")
        row["cut_jump_median"] = float(np.median(values)) if values else None

    # 2. transient curves (re-render cut vs source at the same frame index)
    pairs_320 = []
    pairs_full = []
    for rerender, source in CT25_RERENDER_OF.items():
        for (scene, camera), arrays in data.items():
            if scene != rerender:
                continue
            src = data[(source, camera)]
            cuts = cuts_by_cam[(scene, camera)]
            pairs_320.append(
                (
                    {
                        "wake": arrays["wake_320x180"],
                        "moving": arrays["moving_320x180"],
                        "fraction": arrays["fraction_320x180"],
                    },
                    {
                        "wake": src["wake_320x180"],
                        "moving": src["moving_320x180"],
                        "fraction": src["fraction_320x180"],
                    },
                    cuts,
                )
            )
            if "wake_full" in arrays:
                pairs_full.append(
                    (
                        {"wake": arrays["wake_full"], "moving": arrays["moving_full"]},
                        {"wake": src["wake_full"], "moving": src["moving_full"]},
                        cuts,
                    )
                )
    curves = {
        "320x180": {k: transient_curve(pairs_320, k) for k in ("wake", "moving")},
        "320x180_foreground_fraction": transient_curve(pairs_320, "fraction"),
        "full": {k: transient_curve(pairs_full, k) for k in ("wake", "moving")},
    }
    window = curves["320x180"]["wake"]["reaches_steady_state_k"]
    window_bound = CHUNK if window is None else window
    excess_end = excess_ends_k(curves["320x180"]["wake"])
    windows = {
        "preregistered_steady_state": window_bound,
        "post_hoc_excess_end": CHUNK if excess_end is None else excess_end,
    }

    # 3. render-vs-source wake difference, with and without the window
    parity = json.loads(PARITY.read_text())
    pair_rows = []
    card_rows_rer: dict[str, list[tuple[str, Partition, np.ndarray, np.ndarray]]] = {}
    card_rows_rer_all: list[tuple[str, Partition, np.ndarray, np.ndarray]] = []
    card_rows_src: list[tuple[str, Partition, np.ndarray, np.ndarray]] = []
    for rerender, source in sorted(CT25_RERENDER_OF.items()):
        for (scene, camera), arrays in sorted(data.items()):
            if scene != rerender:
                continue
            src = data[(source, camera)]
            n = min(arrays["wake_320x180"].size, src["wake_320x180"].size)
            cuts = cuts_by_cam[(scene, camera)]
            keeps = {w: exclusion_mask(cuts, n, f) for w, f in windows.items()}
            partition = _partition(source, camera, n)
            no_motion = np.asarray(partition.labels == Observability.NO_MOTION)
            for w, keep in keeps.items():
                card_rows_rer.setdefault(w, []).append(
                    (
                        f"{rerender}/{camera}",
                        partition,
                        arrays["wake_320x180"][:n],
                        keep,
                    )
                )
            card_rows_rer_all.append(
                (
                    f"{rerender}/{camera}",
                    partition,
                    arrays["wake_320x180"][:n],
                    np.ones(n, dtype=bool),
                )
            )
            card_rows_src.append(
                (
                    f"{source}/{camera}",
                    partition,
                    src["wake_320x180"][:n],
                    np.ones(n, dtype=bool),
                )
            )
            rasters = ["320x180"] + (["full"] if "wake_full" in arrays else [])
            row: dict[str, Any] = {
                "rerender": rerender,
                "source": source,
                "camera": camera,
                "day41_sampled": "wake_full" in arrays,
                "cuts": cuts.tolist(),
            }
            for raster in rasters:
                a = arrays[f"wake_{raster}"][:n]
                b = src[f"wake_{raster}"][:n]
                differ = a != b
                w41 = np.arange(n) < DAY41_WINDOW
                row[raster] = {
                    "day41_window": {
                        "frames": int(w41.sum()),
                        "differ_all": float(differ[w41].mean()),
                    },
                    "whole_video": {
                        "frames": int(n),
                        "differ_all": float(differ.mean()),
                    },
                    "gt_no_motion_frames": {
                        "frames": int(no_motion.sum()),
                        "wake_rerender_all": _rate(a, no_motion),
                        "wake_source_all": _rate(b, no_motion),
                    },
                }
                for w, keep in keeps.items():
                    row[raster]["day41_window"][w] = {
                        "frames_kept": int((w41 & keep).sum()),
                        "differ": _rate(differ, w41 & keep),
                    }
                    row[raster]["whole_video"][w] = {
                        "frames_kept": int(keep.sum()),
                        "differ": _rate(differ, keep),
                    }
                    row[raster]["gt_no_motion_frames"][w] = {
                        "frames_kept": int((no_motion & keep).sum()),
                        "wake_rerender": _rate(a, no_motion & keep),
                        "wake_source": _rate(b, no_motion & keep),
                    }
            pair_rows.append(row)

    day41_pairs = {
        (p["rerender"], p["camera"]): p["320x180"]["wake_decisions_differ"]
        / p["320x180"]["frames"]
        for p in parity["rerender_vs_source"]
    }
    for row in pair_rows:
        key = (row["rerender"], row["camera"])
        if key in day41_pairs:
            row["day41_reported_differ_320x180"] = day41_pairs[key]

    window_source = {
        "preregistered_steady_state": "reaches_steady_state_k of the 320x180 "
        "wake curve (pre-registered rule)"
        + ("" if window is not None else "; never reached, bound used"),
        "post_hoc_excess_end": "POST HOC, chosen after viewing the curve: "
        "first bin whose simultaneous interval of the 320x180 wake difference "
        "reaches 0 (end of the post-cut excess only)",
    }
    scorecards = {
        "sources": gate_scorecard(card_rows_src, "sources", False, None),
        "ct25_rerenders_no_exclusion": gate_scorecard(
            card_rows_rer_all,
            "ct25-noexcl",
            True,
            {"frames_after_cut": 0, "source": "explicitly no exclusion"},
        ),
    }
    for w, frames in windows.items():
        scorecards[f"ct25_rerenders_excluding_{w}"] = gate_scorecard(
            card_rows_rer[w],
            f"ct25-excl-{w}",
            True,
            {
                "frames_after_cut": int(frames),
                "source": "artifacts/smartspaces/day42_rerender_transient.json: "
                + window_source[w],
            },
        )

    return {
        "label": LANE_LABEL,
        "scorecards": {k: c.as_dict() for k, c in scorecards.items()},
        "caveats": [
            LANE_LABEL,
            "No timing, CPU or latency figure is measured or reported (Day 19).",
            "Cut threshold is the Day-41 one (mean-BGR jump > 10 grey levels); "
            "its margin is reported per scene as the largest non-cut jump.",
        ],
        "gate": {
            "backend": "MOG2",
            "history": MOG2_HISTORY,
            "learning_rate": "OpenCV automatic, 1/min(frames_seen, history)",
            "configs": {
                "320x180": "MotionGateConfig() — product default",
                "full": "MotionGateConfig(gate_width=0, gate_height=0)",
            },
        },
        "rules": {
            "style_cut_jump": STYLE_CUT_JUMP,
            "bin_frames": BIN,
            "late_chunk": list(LATE),
            "interval": "mean +/- z * SE over (camera, cut) units; pointwise "
            f"z={Z_POINTWISE}, simultaneous (Bonferroni over 30 bins) "
            f"z={Z_SIMULTANEOUS}",
            "seed": SEED,
            "transient_end": (
                "reaches_steady_state_k of the 320x180 wake curve: the first "
                "30-frame bin from which every later bin's 95% CI of "
                "(re-render minus source) contains the late-chunk difference"
            ),
        },
        "cuts": cut_rows,
        "curves": curves,
        "exclusion_window_frames": window,
        "window_used_downstream": {
            "name": "post_hoc_excess_end",
            "by": "scripts/smartspaces_to_coco.py (in_transient flag) and every "
            "B5/B6 metric",
        },
        "exclusion_window_bound_used": window_bound,
        "windows": {
            w: {"frames_after_cut": f, "rule": window_source[w]}
            for w, f in windows.items()
        },
        "pairs": pair_rows,
    }


def _rate(bits: np.ndarray, mask: np.ndarray) -> float | None:
    return float(bits[mask].mean()) if mask.any() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--analyse", action="store_true", help="cache only")
    args = parser.parse_args(argv)
    if not args.analyse:
        todo = jobs()
        # full-res jobs are the long ones; start them first
        todo.sort(key=lambda j: not j["full_res"])
        print(f"{len(todo)} videos ({sum(j['full_res'] for j in todo)} with full-res)")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for message in pool.map(run_job, todo):
                print(message, flush=True)
    payload = analyse()
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Written: {ARTIFACT.relative_to(REPO_ROOT)}")
    print("exclusion window (frames after a cut):", payload["exclusion_window_frames"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
