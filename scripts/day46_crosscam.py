"""Day 46, Objective 3: real cross-camera association, scored against two
baselines. Lane S synthetic (SmartSpaces), calibration given.

Per scene and detection source (oracle boxes, or a pretrained detection
cache), cameras 0-3, the first ``--seconds``, detector frames every
``--step``-th frame (1 = every frame):

1. the product path per camera: tracker (IoU-only), floor projection, foot
   anchor (rule R3); each track becomes a :class:`~src.pipeline.crosscam.
   Tracklet` whose sigma is the fitted error model (rule R4);
2. :func:`src.pipeline.crosscam.associate` over all tracklets;
3. three position estimates per GT person-frame, bucketed by how many cameras
   have a track point matched to that person (the true number of views):
   - ``single_camera``: the matched point with the smallest stated sigma
     (the floor: no cross-camera information);
   - ``gt_pairing_ceiling``: least-squares ray intersection of every matched
     view (views paired by ground-truth identity: not achievable);
   - ``association``: the fused identity of the best view's tracklet: the
     ray intersection of its members present at that frame on two or more
     cameras, else the single foot point. Wrong members make it worse.
4. association precision/recall over tracklet pairs (different cameras, at
   least one shared frame) against the pairs whose tracklets have the same
   majority GT identity;
5. scene-level events (counts): GT events per GT person over the frames any
   camera matched them; single camera = every tracklet its own subject (the
   product today); ceiling = one trajectory per GT person; association = one
   per fused identity. Subjects map to GT by majority vote.

The coupling-distance prediction: for every candidate pair the association
judged (decisive or ambiguous), the GT distance from that pair's person to the
nearest other visible GT person, median over the shared frames, against the
coupling distance (median stated 95% radius, REGION_K * sigma, over all
tracklet points). Error = accepted wrong pair or true pair not accepted.
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
sys.path.insert(0, str(ROOT / "scripts"))

from cascade_frontier import oracle_frames, video_props  # noqa: E402
from src.contracts.frames import FrameGeometry  # noqa: E402
from src.pipeline.anchor import (  # noqa: E402
    camera_from_smartspaces,
    fitted_sigma_m,
    range_interval,
)
from src.pipeline.crosscam import (  # noqa: E402
    REGION_K,
    Tracklet,
    associate,
    ls_intersection,
)
from src.pipeline.evaluate import match_events, match_frames  # noqa: E402
from src.pipeline.events import (  # noqa: E402
    Trajectory,
    compile_trajectory,
    load_zone_config,
)
from src.pipeline.floor import load_projector  # noqa: E402
from src.pipeline.frontier import (  # noqa: E402
    CameraSource,
    FrameDets,
    detections_for,
)
from src.pipeline.run import estimate_tracks, frame_ts_ns, track_camera  # noqa: E402
from src.pipeline.tracker import PRODUCT_TRACKER  # noqa: E402
from src.pipeline.truth import load_truth, person_truth  # noqa: E402

CAMERAS = ("Camera_0000", "Camera_0001", "Camera_0002", "Camera_0003")
ESTIMATES = ("single_camera", "gt_pairing_ceiling", "association")


def read_cache_by_frame(path: Path, n: int, step: int) -> list[FrameDets]:
    """Detections by frame index from a detection cache, every-frame or
    strided (``frame_step`` k: lines for frames 0, k, 2k, ...). Frames the
    cache does not hold stay empty and are never scheduled (``--step`` must be
    a multiple of the cache's step); a scheduled frame the cache lacks is an
    error, never an empty frame."""
    lines = path.read_text().splitlines()
    labels = json.loads(lines[0])["header"]["labels"]
    empty = FrameDets(np.zeros((0, 4)), np.zeros(0), ())
    out = [empty] * n
    held = set()
    for line in lines[1:]:
        r = json.loads(line)
        f = int(r["f"])
        if f < n:
            out[f] = FrameDets(
                boxes=np.asarray(r["b"], np.float64).reshape(-1, 4),
                scores=np.asarray(r["s"], np.float64),
                labels=tuple(labels[int(c)] for c in r["c"]),
            )
            held.add(f)
    missing = [f for f in range(0, n, step) if f not in held]
    if missing:
        raise SystemExit(f"{path}: {len(missing)} scheduled frames not cached")
    return out


def stats(err: list[float]) -> dict[str, Any]:
    if not err:
        return {"n": 0}
    a = np.array(err)
    return {
        "n": int(a.size),
        "p50_m": float(np.percentile(a, 50)),
        "p90_m": float(np.percentile(a, 90)),
        "p99_m": float(np.percentile(a, 99)),
    }


def views_key(k: int) -> str:
    return "1" if k == 1 else "2" if k == 2 else "3+"


def scene_eval(
    scene_dir: Path,
    zones_path: Path,
    detector: str,
    run_dir: Path | None,
    seconds: float,
    step: int,
) -> dict[str, Any]:
    gt = load_truth(scene_dir / "ground_truth.json")
    fps, width, height = video_props(scene_dir / "videos" / f"{CAMERAS[0]}.mp4")
    n = int(round(seconds * fps))
    label = "Person" if detector == "oracle" else "person"
    manifest = json.loads((run_dir / "manifest.json").read_text()) if run_dir else {}
    tracklets: list[Tracklet] = []
    gt_of_point: dict[tuple[str, int], tuple[int, np.ndarray]] = {}
    majority: dict[str, int] = {}
    point_of: dict[str, dict[int, np.ndarray]] = {}
    sigmas: list[float] = []
    for cam in CAMERAS:
        frames = (
            oracle_frames(gt, cam, n)
            if detector == "oracle"
            else read_cache_by_frame(
                Path(manifest["cameras"][cam]["detection_cache"]["cache_file"]), n, step
            )
        )
        projector = load_projector(scene_dir / "calibration.json", cam)
        camera = camera_from_smartspaces(scene_dir / "calibration.json", cam)
        truth = person_truth(gt, cam, n)
        source = CameraSource(
            camera_id=cam,
            fps=fps,
            width=width,
            height=height,
            frames=frames,
            gate_awake=np.zeros(n, bool),
            projector=projector,
            truth=truth,
            objects={},
        )
        dets = detections_for(source, np.arange(0, n, step, dtype=np.int64), detector)
        assignments = track_camera(dets, label, PRODUCT_TRACKER)
        points, _ = estimate_tracks(
            assignments,
            projector,
            fps,
            FrameGeometry(width, height),
            "day46-crosscam",
            anchor_camera=camera,
            anchor_mode="foot",
            label=label,
        )
        points = [p for p in points if np.all(np.isfinite(p.raw_xy))]
        boxes = np.array([p.box_xyxy for p in points]).reshape(-1, 4)
        _, matches = match_frames(
            truth,
            [p.frame_index for p in points],
            [p.track_id for p in points],
            boxes,
            n,
        )
        votes: dict[int, Counter[int]] = {}
        for f, gt_row, tid, _ in matches:
            gid = int(truth.object_id[gt_row])
            votes.setdefault(tid, Counter())[gid] += 1
            gt_of_point[(f"{cam}/t{tid}", int(f))] = (gid, truth.floor_xy[gt_row])
        for tid, c in votes.items():
            majority[f"{cam}/t{tid}"] = c.most_common(1)[0][0]
        c_xy = np.array(projector.camera_xyz_m[:2])
        ri = range_interval(camera, boxes)
        raw = np.array([p.raw_xy for p in points]).reshape(-1, 2)
        rng = np.linalg.norm(raw - c_xy, axis=1)
        sig = fitted_sigma_m(rng, ri.width_m, ["foot"] * len(points))
        by_track: dict[int, list[int]] = {}
        for i, p in enumerate(points):
            by_track.setdefault(p.track_id, []).append(i)
        for tid, rows in by_track.items():
            rows.sort(key=lambda i: points[i].frame_index)
            key = f"{cam}/t{tid}"
            fr = np.array([points[i].frame_index for i in rows], dtype=np.int64)
            tracklets.append(
                Tracklet(key, cam, c_xy, fr, raw[rows], sig[rows].astype(np.float64))
            )
            point_of[key] = {int(f): raw[i] for f, i in zip(fr, rows)}
            sigmas.extend(float(s) for s in sig[rows])
    camera_xy = {t.camera_id: t.camera_xy for t in tracklets}
    camera_of = {t.key: t.camera_id for t in tracklets}
    sigma_of = {
        (t.key, int(f)): float(s)
        for t in tracklets
        for f, s in zip(t.frames, t.sigma_m)
    }
    assoc = associate(tracklets)
    members: dict[str, list[str]] = {}
    for key, comp in assoc.components.items():
        members.setdefault(comp, []).append(key)

    # association precision / recall over tracklet pairs
    judged = {
        (a, b): ev for (a, b), ev in assoc.evidence.items() if a < b and ev.overlap > 0
    }
    true_pairs = {
        p
        for p in judged
        if p[0] in majority and majority.get(p[0]) == majority.get(p[1])
    }
    # scored at identity level: a pair is "predicted same" when both tracklets
    # end in one fused identity (directly accepted or joined through others)
    accepted = {p for p in judged if assoc.components[p[0]] == assoc.components[p[1]]}
    direct = set(assoc.accepted)
    correct = accepted & true_pairs

    # position error by true number of views
    by_person_frame: dict[tuple[int, int], list[tuple[str, np.ndarray]]] = {}
    for (key, f), (gid, floor) in gt_of_point.items():
        by_person_frame.setdefault((gid, f), []).append((key, floor))
    err: dict[str, dict[str, list[float]]] = {
        e: {"1": [], "2": [], "3+": []} for e in ESTIMATES
    }
    traj: dict[str, dict[str, list[tuple[int, np.ndarray]]]] = {
        e: {} for e in ESTIMATES + ("truth",)
    }

    def fused_at(comp: str, f: int) -> np.ndarray:
        here = [k for k in members[comp] if f in point_of[k]]
        cams: dict[str, str] = {}
        for k in here:
            cams.setdefault(camera_of[k], k)
        if len(cams) >= 2:
            ks = list(cams.values())
            x = ls_intersection(
                np.array([camera_xy[camera_of[k]] for k in ks]),
                np.array([point_of[k][f] for k in ks]),
            )
            if x is not None:
                return x
        best = min(here, key=lambda k: sigma_of[(k, f)])
        return point_of[best][f]

    for (gid, f), views in sorted(by_person_frame.items(), key=lambda kv: kv[0][1]):
        per_cam: dict[str, str] = {}
        for key, _ in views:
            per_cam.setdefault(camera_of[key], key)
        keys = list(per_cam.values())
        truth_xy = views[0][1]
        k = views_key(len(keys))
        best = min(keys, key=lambda kk: sigma_of[(kk, f)])
        single = point_of[best][f]
        ceiling = single
        if len(keys) >= 2:
            x = ls_intersection(
                np.array([camera_xy[camera_of[kk]] for kk in keys]),
                np.array([point_of[kk][f] for kk in keys]),
            )
            ceiling = single if x is None else x
        fused = fused_at(assoc.components[best], f)
        for name, xy in (
            ("single_camera", single),
            ("gt_pairing_ceiling", ceiling),
            ("association", fused),
        ):
            err[name][k].append(float(np.linalg.norm(xy - truth_xy)))
        traj["gt_pairing_ceiling"].setdefault(f"gt/{gid}", []).append((f, ceiling))
        traj["truth"].setdefault(f"gt/{gid}", []).append((f, truth_xy))
    for t in tracklets:
        traj["single_camera"][t.key] = [
            (int(f), xy) for f, xy in zip(t.frames, t.foot_xy)
        ]
    for comp, keys in members.items():
        frames = sorted({f for k in keys for f in point_of[k]})
        traj["association"][comp] = [(f, fused_at(comp, f)) for f in frames]

    zones = load_zone_config(zones_path)

    def compile_all(name: str) -> list[Any]:
        out = []
        for subject, pts in traj[name].items():
            pts.sort(key=lambda p: p[0])
            out += compile_trajectory(
                Trajectory(
                    subject_id=subject,
                    ts_ns=np.array([frame_ts_ns(f, fps) for f, _ in pts], np.int64),
                    xy=np.array([xy for _, xy in pts]),
                    frame_refs=tuple(f"scene/frame-{f}" for f, _ in pts),
                    score=1.0,
                ),
                zones.zones,
                zones.rules,
            )
        return out

    gt_events = compile_all("truth")
    subject_of: dict[str, dict[str, str]] = {
        "gt_pairing_ceiling": {s: s for s in traj["truth"]},
        "single_camera": {k: f"gt/{g}" for k, g in majority.items()},
    }
    comp_votes: dict[str, Counter[int]] = {}
    for (key, _f), (gid, _) in gt_of_point.items():
        comp_votes.setdefault(assoc.components[key], Counter())[gid] += 1
    subject_of["association"] = {
        c: f"gt/{v.most_common(1)[0][0]}" for c, v in comp_votes.items()
    }
    events = {}
    for name in ESTIMATES:
        m = match_events(compile_all(name), gt_events, subject_of[name])
        events[name] = {k: m[k] for k in ("predicted", "gt", "matched")}

    # coupling-distance prediction
    others: dict[int, dict[int, np.ndarray]] = {}
    for (gid, f), views in by_person_frame.items():
        others.setdefault(f, {})[gid] = views[0][1]
    coupling = float(np.median(sigmas)) * REGION_K if sigmas else float("nan")
    near, far = {"pairs": 0, "errors": 0}, {"pairs": 0, "errors": 0}
    for pair, ev in judged.items():
        if ev.compatible == 0 or pair[0] not in majority:
            continue
        gid = majority[pair[0]]
        shared = sorted(set(point_of[pair[0]]) & set(point_of[pair[1]]))
        d = []
        for f in shared:
            me = others.get(f, {}).get(gid)
            rest = [xy for g, xy in others.get(f, {}).items() if g != gid]
            if me is not None and rest:
                d.append(min(float(np.linalg.norm(me - xy)) for xy in rest))
        if not d:
            continue
        is_error = (pair in accepted) != (pair in true_pairs)
        bucket = near if float(np.median(d)) < coupling else far
        bucket["pairs"] += 1
        bucket["errors"] += int(is_error)

    kinds = Counter(str(v["kind"]) for v in assoc.verdicts)
    return {
        "detector": detector,
        "detections_from": str(run_dir) if run_dir else "ground truth (oracle)",
        "frames_per_camera": n,
        "detector_step": step,
        "tracklets": len(tracklets),
        "association": {
            "pairs_judged_with_shared_frames": len(judged),
            "true_pairs": len(true_pairs),
            "accepted_directly": len(direct),
            "same_fused_identity": len(accepted),
            "same_fused_identity_correct": len(correct),
            "precision": len(correct) / len(accepted) if accepted else None,
            "recall": len(correct) / len(true_pairs) if true_pairs else None,
            "verdicts": dict(kinds),
            "fused_identities_with_2plus_tracklets": sum(
                1 for v in members.values() if len(v) > 1
            ),
        },
        "position_error_by_views": {
            e: {k: stats(v) for k, v in by.items()} for e, by in err.items()
        },
        "events": events,
        "coupling_distance": {
            "coupling_distance_m": coupling,
            "definition": "median over tracklet points of REGION_K * stated sigma",
            "nearest_other_below": near,
            "nearest_other_at_or_above": far,
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene", action="append", required=True, help="SCENE_DIR=ZONES")
    p.add_argument("--detector", choices=("oracle", "pretrained"), default="oracle")
    p.add_argument("--run", action="append", default=[], help="SCENE=RUN_DIR")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    runs = dict(r.split("=", 1) for r in args.run)
    out: dict[str, Any] = {
        "label": (
            "lane S synthetic (SmartSpaces), calibration given; geometry-only "
            "association; not evidence about real cameras"
        ),
        "scenes": {},
    }
    for spec in args.scene:
        sd, _, zones = spec.partition("=")
        name = Path(sd).name
        run_dir = Path(runs[name]) if args.detector == "pretrained" else None
        r = scene_eval(
            Path(sd), Path(zones), args.detector, run_dir, args.seconds, args.step
        )
        out["scenes"][name] = r
        print(name, json.dumps(r["association"]), flush=True)
        for e, by in r["position_error_by_views"].items():
            print(
                " ",
                e,
                {
                    k: (v.get("n"), v.get("p50_m"), v.get("p90_m"))
                    for k, v in by.items()
                },
            )
        print("  events", r["events"])
        print("  coupling", r["coupling_distance"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
