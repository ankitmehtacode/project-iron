"""Score the site-zero-solo capture: one function for the real clips and the
synthetic stand-in (Day 46).

    data_dir/
      A.mp4 B.mp4 C.mp4   the clips named in notes.yaml
      marks.csv           name, x_m, y_m, role (calib | test)
      notes.yaml          camera_id; per clip: file, segments, expected_events
      zones.yaml          the site's zones (written before any run)

``notes.yaml`` segments are ``{kind: empty | on_mark, start_s, end_s[, mark]}``
in video seconds; ``expected_events`` is ``{zone: {entered: n, exited: n}}``.

:func:`run_site_zero` is the whole path: for each clip, the consent gate,
``iron run``'s own camera construction and :func:`src.pipeline.run.run`
(product defaults: motion gate on, IoU-only tracker, detector every frame),
then :func:`site_zero_report`. The only thing the caller chooses is the
detector: RT-DETRv2-S for the real capture, the oracle over the stand-in's
known boxes for the synthetic one. Nothing else differs, and
``tests/test_site_zero.py`` fails if the two entry points stop sharing this
function.

Rules fixed before any data exists:

- a clip with on-mark segments runs with the motion gate OFF: it measures
  localisation, and the gate stops waking the detector a few frames after a
  person stands still (``stay_awake_frames``), which would leave the marks
  with no samples. Every other clip runs at the product default (gate on),
  because false wakes and events are what it measures;
- per-mark error uses the samples in ``[start_s + TRIM_S, end_s - TRIM_S]``
  of each on-mark segment, so walking onto and off the mark is excluded;
- an empty segment's "false wakes" are the frames on which the detector's
  output was kept (the gate was awake); person detections and track points
  inside it are counted too, and none of them is excused;
- entries and exits are the run's own ``entered``/``exited`` events per zone,
  against the counts written in the notes.

Each number comes with a trivial baseline: per-mark error against predicting
the centroid of the test marks for every sample; false wakes against an
always-awake gate (every frame in the segment).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

TRIM_S = 1.0
"""Seconds dropped at each end of an on-mark segment (arrival, departure)."""


def load_marks(path: Path) -> dict[str, dict[str, Any]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        r["name"]: {
            "xy": (float(r["x_m"]), float(r["y_m"])),
            "role": r["role"].strip(),
        }
        for r in rows
    }


def load_notes(path: Path) -> dict[str, Any]:
    notes = yaml.safe_load(path.read_text())
    if not isinstance(notes, dict) or {"camera_id", "clips"} - set(notes):
        raise ValueError(f"{path}: needs camera_id and clips")
    for name, clip in notes["clips"].items():
        if "file" not in clip:
            raise ValueError(f"{path}: clip {name} has no file")
        for seg in clip.get("segments", []):
            if seg.get("kind") not in ("empty", "on_mark"):
                raise ValueError(f"{path}: clip {name}: segment kind {seg.get('kind')}")
            if not float(seg["end_s"]) > float(seg["start_s"]):
                raise ValueError(f"{path}: clip {name}: empty interval {seg}")
            if seg["kind"] == "on_mark" and "mark" not in seg:
                raise ValueError(f"{path}: clip {name}: on_mark without a mark")
    return dict(notes)


def gate_mode_for(clip: Mapping[str, Any]) -> str:
    """``off`` for a clip that measures localisation, else the product's."""
    on_mark = any(s["kind"] == "on_mark" for s in clip.get("segments", []))
    return "off" if on_mark else "default"


def run_site_zero(
    data_dir: Path,
    calibration: Path,
    out_root: Path,
    detector_factory: Callable[[Any], Any],
    person_label: str,
    cache_root: Path | None = None,
    allow_dirty: bool = False,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run every clip in ``notes.yaml`` and score the runs. Writes
    ``out_root/<clip>/`` (a normal run folder) and returns the report."""
    from src.data.footage import require_consented
    from src.main import camera_input
    from src.pipeline.events import load_zone_config
    from src.pipeline.run import run

    notes = load_notes(data_dir / "notes.yaml")
    zones = load_zone_config(data_dir / "zones.yaml")
    run_dirs: dict[str, Path] = {}
    for name, clip in notes["clips"].items():
        video = data_dir / clip["file"]
        require_consented([video])
        camera = camera_input(video, calibration, notes["camera_id"])
        run(
            [camera],
            detector_factory,
            person_label,
            zones,
            out_root / name,
            max_seconds=None,
            gate_mode=gate_mode_for(clip),
            progress=lambda message: None,
            cache_root=cache_root,
            allow_dirty=allow_dirty,
            repo_root=repo_root,
        )
        run_dirs[name] = out_root / name
    return site_zero_report(run_dirs, notes, load_marks(data_dir / "marks.csv"))


def _summary(values: npt.NDArray[np.float64]) -> dict[str, Any]:
    if values.size == 0:
        return {"n": 0, "p50": None, "p90": None, "max": None}
    return {
        "n": int(values.size),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def site_zero_report(
    run_dirs: Mapping[str, Path],
    notes: Mapping[str, Any],
    marks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Per-mark position error, empty-stretch false wakes, entry/exit counts."""
    import pyarrow.parquet as pq

    tests = {n: m["xy"] for n, m in marks.items() if m["role"] == "test"}
    centroid = np.mean(np.array(list(tests.values())), axis=0) if tests else None
    per_mark: list[dict[str, Any]] = []
    empty: list[dict[str, Any]] = []
    events_out: dict[str, Any] = {}
    manifests: dict[str, str] = {}
    for clip_name, clip in notes["clips"].items():
        run_dir = run_dirs[clip_name]
        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifests[clip_name] = manifest["manifest_sha"]
        (camera_id,) = manifest["cameras"]
        fps = float(manifest["cameras"][camera_id]["fps"])
        det = np.load(run_dir / camera_id / "detections.npz")
        run_frames = det["run_frames"]
        det_frames = det["frame"][det["labels"] == manifest["identity"]["person_label"]]
        points = pq.read_table(run_dir / "tracks.parquet").to_pydict()
        p_frame = np.asarray(points["frame_index"], np.int64)
        raw = np.column_stack([points["raw_x"], points["raw_y"]]).astype(np.float64)
        est = np.column_stack([points["est_x"], points["est_y"]]).astype(np.float64)
        for seg in clip.get("segments", []):
            if seg["kind"] == "on_mark":
                lo = int(np.ceil((float(seg["start_s"]) + TRIM_S) * fps))
                hi = int(np.floor((float(seg["end_s"]) - TRIM_S) * fps))
                sel = (p_frame >= lo) & (p_frame <= hi)
                mark = np.array(tests[seg["mark"]])
                row = {
                    "clip": clip_name,
                    "mark": seg["mark"],
                    "window_frames": max(0, hi - lo + 1),
                    "frames_with_a_sample": int(np.unique(p_frame[sel]).size),
                    "raw_error_m": _summary(np.linalg.norm(raw[sel] - mark, axis=1)),
                    "estimator_error_m": _summary(
                        np.linalg.norm(est[sel] - mark, axis=1)
                    ),
                    "baseline_centroid_error_m": float(np.linalg.norm(centroid - mark)),
                }
                per_mark.append(row)
            else:
                lo = int(np.ceil(float(seg["start_s"]) * fps))
                hi = int(np.floor(float(seg["end_s"]) * fps)) - 1
                inside = (run_frames >= lo) & (run_frames <= hi)
                empty.append(
                    {
                        "clip": clip_name,
                        "start_s": float(seg["start_s"]),
                        "end_s": float(seg["end_s"]),
                        "frames": hi - lo + 1,
                        "gate_on": manifest["identity"]["motion_gate"] != "off",
                        "detector_kept_frames": int(inside.sum()),
                        "baseline_always_awake_frames": hi - lo + 1,
                        "person_detections": int(
                            ((det_frames >= lo) & (det_frames <= hi)).sum()
                        ),
                        "track_points": int(((p_frame >= lo) & (p_frame <= hi)).sum()),
                    }
                )
        rows = pq.read_table(run_dir / "events.parquet").to_pylist()
        counted: dict[str, dict[str, int]] = {}
        for r in rows:
            if r["verb"] in ("entered", "exited") and r["zone"]:
                zone = counted.setdefault(r["zone"]["id"], {"entered": 0, "exited": 0})
                zone[r["verb"]] += 1
        expected = clip.get("expected_events") or {}
        events_out[clip_name] = {
            "expected": expected,
            "observed": counted,
            "matches": (
                all(
                    counted.get(z, {}).get(v, 0) == n
                    for z, verbs in expected.items()
                    for v, n in verbs.items()
                )
                if expected
                else None
            ),
            "events_total": len(rows),
        }
    return {
        "rules": {"trim_s": TRIM_S},
        "manifest_shas": manifests,
        "per_mark": per_mark,
        "empty_stretches": empty,
        "entries_exits": events_out,
    }
