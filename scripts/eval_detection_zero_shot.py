"""Zero-shot detection on SmartSpaces (Day 42, Objective B5).

lane S synthetic (SmartSpaces) — zero-shot, not a product claim; NVIDIA terms
unresolved. NO timing, throughput, CPU or latency figure is measured or
reported (Day 19).

Two phases, both resumable:

1. ``run``: per backend and resolution, detections on every frame of the B4
   export (export 66c683ffa66f; the frame count is the export's recorded
   rule, 456 frames), appended one JSON line per frame under the export's
   ``runs/`` directory, so an interrupted run resumes where it stopped.
   - ``rtdetr_v2_s``: the signed RT-DETRv2-S checkpoint (score floor 0.05).
   - ``motion_blob``: the gate's MOG2 blobs; it replays each camera's video
     from frame 0 so its background model is what it would be live.
   Resolutions: native 1920x1080, and the same decoded frame downscaled
   (cv2 INTER_AREA, no re-compression: optimistic against a low-bitrate
   camera) to 1280x720 and 640x360. Boxes come back in that raster's own
   canonical frame and are scaled to 1080p for scoring against native GT.
2. ``score``: pycocotools only (validated by ``tests/test_coco_metrics.py``).
   Height buckets are pycocotools area ranges over ``height**2`` (every GT and
   detection ``area`` is overwritten with its box height squared), so the
   library's own range logic does the bucketing. COCO's range rule applies:
   an unmatched detection whose own height is outside a bucket is ignored
   there, so a bucket's AP is not charged for false positives of other sizes
   (``tests/test_coco_metrics.py`` checks this by hand). Frames inside a re-render's
   post-cut transient are excluded. Every metric is reported as
   ``value (baseline, margin)`` against an empty predictor and against
   ``motion_blob``.

    python scripts/eval_detection_zero_shot.py run
    python scripts/eval_detection_zero_shot.py score
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smartspaces_integrity import REPO_ROOT, scene_dirs  # noqa: E402
from smartspaces_to_coco import SUMMARY as EXPORT_SUMMARY  # noqa: E402

from src.contracts.frames import FrameGeometry  # noqa: E402
from src.detect.contract import FrameInput  # noqa: E402

LABEL = (
    "lane S synthetic (SmartSpaces) — zero-shot, not a product claim; "
    "NVIDIA terms unresolved"
)
ARTIFACT = REPO_ROOT / "artifacts" / "smartspaces" / "day42_detection_zero_shot.json"
PARITY = REPO_ROOT / "artifacts" / "smartspaces" / "day41_gate_parity.json"
RESOLUTIONS = {"1080p": (1920, 1080), "720p": (1280, 720), "360p": (640, 360)}
HEIGHT_EDGES = (0.0, 32.0, 64.0, 128.0, 256.0, 1e5)
"""Native-1080p pixel heights, the B4 buckets. Bucket membership is by the
height AT THE EVALUATED RESOLUTION (native x scale), so a 360p row's [0,32)
bucket holds objects that are under 32 px in the 360p image."""
SCORE_FLOOR = 0.05
HALF_OPEN = 1e-6
"""Subtracted from each bucket's upper area edge (height**2) so pycocotools'
closed ranges behave as the half-open buckets the counts use."""
IOU_RECALL = 0.5
BACKENDS = ("rtdetr_v2_s", "motion_blob")


def export_dir() -> Path:
    return REPO_ROOT / json.loads(EXPORT_SUMMARY.read_text())["export_dir"]


def load_coco(split: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((export_dir() / f"{split}.json").read_text())
    return payload


def run_path(backend: str, resolution: str) -> Path:
    return export_dir() / "runs" / f"{backend}__{resolution}.jsonl"


def _rows(path: Path) -> list[dict[str, Any]]:
    """Every complete row. A crash mid-write can leave a truncated LAST line;
    that one is dropped (and re-run). A bad line anywhere else raises."""
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for n, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if n != len(lines) - 1:
                raise
    return rows


def _done(path: Path) -> set[int]:
    return {r["image_id"] for r in _rows(path)}


def producer_identity(backend: str, resolution: str) -> dict[str, Any]:
    """What a run file's rows were produced by. Recorded beside the file and
    checked on every resume and every score, so changed weights, settings or
    resolution can never resume onto, or be scored with, someone else's rows."""
    if backend == "rtdetr_v2_s":
        from src.data.registry import DatasetRegistry
        from src.detect.preprocess import CANONICAL_SPEC_DIR, DetectorPreprocessSpec

        entry = DatasetRegistry.load(REPO_ROOT / "configs" / "datasets.yaml").get(
            "RT-DETRv2-S-COCO"
        )
        spec = DetectorPreprocessSpec.load(
            CANONICAL_SPEC_DIR / "rtdetr_v2_r18vd.preprocess.json"
        )
        detail: dict[str, Any] = {
            "model_sha": entry.content_sha,
            "preprocess_sha": spec.preprocess_sha(),
            "score_floor": SCORE_FLOOR,
        }
    else:
        from src.detect.backends import MotionBlobConfig

        detail = {"config": MotionBlobConfig().__dict__, "mog2": "history=250"}
    return {"backend": backend, "resolution": resolution, **detail}


def _producer_path(path: Path) -> Path:
    return path.with_name(path.stem + ".producer.json")


def check_producer(backend: str, resolution: str, adopt_legacy: bool = False) -> None:
    """Refuse a run file whose recorded producer differs from the current one,
    or that has rows and no record at all (unless explicitly adopted)."""
    path = run_path(backend, resolution)
    sidecar = _producer_path(path)
    current = producer_identity(backend, resolution)
    if sidecar.exists():
        recorded = json.loads(sidecar.read_text())
        if recorded != current:
            raise SystemExit(
                f"{path.name} was produced by {recorded}, not {current}: refusing "
                "to resume or score it. Move it aside and re-run."
            )
        return
    if _rows(path) and not adopt_legacy:
        raise SystemExit(
            f"{path.name} has rows but no producer record; re-run it, or pass "
            "--adopt-legacy if you know it was produced by the current settings"
        )
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(current, indent=2) + "\n")


def _resize(image: np.ndarray, resolution: str) -> np.ndarray:
    w, h = RESOLUTIONS[resolution]
    if image.shape[1] == w:
        return image
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def _record(image_id: int, detections: Any, scale: float) -> dict[str, Any]:
    boxes = detections.boxes_xyxy * scale
    return {
        "image_id": image_id,
        "boxes_xyxy_1080p": boxes.round(3).tolist(),
        "scores": detections.score_array().round(6).tolist(),
        "labels": detections.labels(),
    }


def run(backend: str, resolution: str) -> None:
    images = [i for s in ("train", "val") for i in load_coco(s)["images"]]
    out = run_path(backend, resolution)
    out.parent.mkdir(parents=True, exist_ok=True)
    check_producer(backend, resolution)
    done = _done(out)
    todo = [i for i in images if i["id"] not in done]
    print(f"{backend} @ {resolution}: {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    scale = 1080 / RESOLUTIONS[resolution][1]
    if backend == "rtdetr_v2_s":
        from src.detect.backends import PretrainedDetector

        detector = PretrainedDetector(
            "RT-DETRv2-S-COCO", "rtdetr_v2_r18vd", SCORE_FLOOR
        )
        with open(out, "a") as sink:
            for meta in todo:
                image = cv2.imread(str(export_dir() / "images" / meta["file_name"]))
                small = _resize(image, resolution)
                geometry = FrameGeometry(width=small.shape[1], height=small.shape[0])
                det = detector.detect(
                    FrameInput(small, 0, meta["frame_index"], geometry)
                )
                sink.write(json.dumps(_record(meta["id"], det, scale)) + "\n")
                sink.flush()
        return

    run_motion_blob([resolution])


def run_motion_blob(resolutions: list[str]) -> None:
    """Decode each camera's video ONCE and feed one MotionBlobDetector per
    resolution still pending for it. Resumable per (resolution, image)."""
    from src.detect.backends import MotionBlobDetector

    images = [i for s in ("train", "val") for i in load_coco(s)["images"]]
    for r in resolutions:
        check_producer("motion_blob", r)
    done = {r: _done(run_path("motion_blob", r)) for r in resolutions}
    by_camera: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for meta in images:
        by_camera.setdefault((meta["scene"], meta["camera"]), []).append(meta)
    sinks = {}
    for r in resolutions:
        run_path("motion_blob", r).parent.mkdir(parents=True, exist_ok=True)
        sinks[r] = open(run_path("motion_blob", r), "a")
    try:
        for (scene, camera), metas in sorted(by_camera.items()):
            pending = {
                r: {m["frame_index"]: m for m in metas if m["id"] not in done[r]}
                for r in resolutions
            }
            pending = {r: w for r, w in pending.items() if w}
            if not pending:
                continue
            print(f"motion_blob {scene}/{camera}: {sorted(pending)}", flush=True)
            video = next(p for p in scene_dirs() if p.name == scene) / "videos"
            capture = cv2.VideoCapture(str(video / f"{camera}.mp4"))
            detectors = {r: MotionBlobDetector() for r in pending}
            last = max(max(w) for w in pending.values())
            for index in range(last + 1):
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f"{scene}/{camera} ended at {index}")
                for r, detector in detectors.items():
                    small = _resize(image, r)
                    geometry = FrameGeometry(
                        width=small.shape[1], height=small.shape[0]
                    )
                    det = detector.detect(FrameInput(small, 0, index, geometry))
                    if index in pending[r]:
                        scale = 1080 / RESOLUTIONS[r][1]
                        sinks[r].write(
                            json.dumps(_record(pending[r][index]["id"], det, scale))
                            + "\n"
                        )
                        sinks[r].flush()
            capture.release()
    finally:
        for sink in sinks.values():
            sink.close()


# --- scoring ---------------------------------------------------------------


def _load_run(backend: str, resolution: str) -> dict[int, dict[str, Any]]:
    check_producer(backend, resolution)
    return {r["image_id"]: r for r in _rows(run_path(backend, resolution))}


def _gate_bits() -> dict[tuple[str, str], dict[str, np.ndarray]]:
    parity = json.loads(PARITY.read_text())
    return {
        (c["scene"], c["camera"]): {
            raster: np.frombuffer(c["wake_bits"][raster].encode(), np.uint8) == 49
            for raster in ("320x180", "full")
        }
        for c in parity["cameras"]
    }


def _cocoeval(
    gt_payload: dict[str, Any],
    dets: list[dict[str, Any]],
    scale: float,
    use_cats: bool,
) -> Any:
    """pycocotools over height buckets (area := height**2)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    payload = copy.deepcopy(gt_payload)
    for a in payload["annotations"]:
        a["area"] = (a["bbox"][3] * scale) ** 2
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = payload
        gt.createIndex()
        dt = gt.loadRes(dets) if dets else None
    if dt is None:
        return None
    for d in dt.dataset["annotations"]:
        d["area"] = (d["bbox"][3] * scale) ** 2
    dt.createIndex()
    ev = COCOeval(gt, dt, "bbox")
    ev.params.useCats = int(use_cats)
    # pycocotools' ranges are CLOSED ([lo, hi] by `<` / `>`); the buckets are
    # half-open [lo, hi). Integer GT heights sit exactly on the edges, so the
    # upper edge is pulled in by HALF_OPEN to count each box once.
    ev.params.areaRng = [[0, 1e10]] + [
        [(lo * 1.0) ** 2, (hi * 1.0) ** 2 - HALF_OPEN]
        for lo, hi in zip(HEIGHT_EDGES[:-1], HEIGHT_EDGES[1:])
    ]
    ev.params.areaRngLbl = ["all"] + [
        f"[{lo:g},{hi:g})" if hi < 1e5 else f"[{lo:g},inf)"
        for lo, hi in zip(HEIGHT_EDGES[:-1], HEIGHT_EDGES[1:])
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
    return ev


def _ap_recall(ev: Any, n_gt: dict[str, int]) -> dict[str, dict[str, Any]]:
    """AP@[.5:.95], AP50 and recall@IoU0.5 (maxDets 100) per area range."""
    out = {}
    iou50 = int(np.argmin(np.abs(ev.params.iouThrs - IOU_RECALL)))
    for a, label in enumerate(ev.params.areaRngLbl):
        if n_gt.get(label, 0) == 0:
            out[label] = {"gt": 0}
            continue
        p = ev.eval["precision"][:, :, :, a, -1]
        r = ev.eval["recall"][:, :, a, -1]

        def mean(x: np.ndarray) -> float | None:
            x = x[x > -1]
            return float(x.mean()) if x.size else None

        out[label] = {
            "gt": n_gt[label],
            "AP": mean(p),
            "AP50": mean(p[iou50]),
            "recall_iou50": mean(r[iou50]),
        }
    return out


def _gt_counts(payload: dict[str, Any], scale: float) -> dict[str, int]:
    counts = {"all": len(payload["annotations"])}
    for lo, hi in zip(HEIGHT_EDGES[:-1], HEIGHT_EDGES[1:]):
        label = f"[{lo:g},{hi:g})" if hi < 1e5 else f"[{lo:g},inf)"
        counts[label] = sum(
            lo <= a["bbox"][3] * scale < hi for a in payload["annotations"]
        )
    return counts


def _subset(
    coco: dict[str, Any], image_ids: set[int], category: str | None
) -> dict[str, Any]:
    cats = coco["categories"]
    keep_cat = {c["id"] for c in cats if category is None or c["name"] == category}
    return {
        "images": [i for i in coco["images"] if i["id"] in image_ids],
        "annotations": [
            a
            for a in coco["annotations"]
            if a["image_id"] in image_ids and a["category_id"] in keep_cat
        ],
        "categories": [c for c in cats if c["id"] in keep_cat],
    }


def _dets(
    run_rows: dict[int, dict[str, Any]],
    image_ids: set[int],
    category_id: int,
    label_filter: set[str] | None,
) -> list[dict[str, Any]]:
    out = []
    for image_id in image_ids:
        row = run_rows.get(image_id)
        if row is None:
            continue
        for box, score, label in zip(
            row["boxes_xyxy_1080p"], row["scores"], row["labels"]
        ):
            if label_filter is not None and label not in label_filter:
                continue
            x0, y0, x1, y1 = box
            out.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "score": score,
                }
            )
    return out


def _with_baselines(
    value: dict[str, dict[str, Any]], blob: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """``value (baseline, margin)`` for each metric, against empty and blob."""
    out: dict[str, dict[str, Any]] = {}
    for bucket, row in value.items():
        entry: dict[str, Any] = {"gt": row.get("gt", 0)}
        for metric in ("AP", "AP50", "recall_iou50"):
            v = row.get(metric)
            b = blob.get(bucket, {}).get(metric)
            entry[metric] = {
                "value": v,
                "vs_empty": {"baseline": 0.0, "margin": v},
                "vs_motion_blob": {
                    "baseline": b,
                    "margin": None if v is None or b is None else v - b,
                },
            }
        out[bucket] = entry
    return out


def _blob_coverage(
    c: dict[str, Any], ids: set[int], run_rows: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """DIAGNOSTIC, not a metric: is the motion-blob baseline weak because it
    finds nothing, or because its boxes do not fit people? Blobs per frame,
    and the share of Person GT >= 64 px whose box centre lies inside a blob."""
    person = {x["id"] for x in c["categories"] if x["name"] == "Person"}
    per_frame = [len(run_rows[i]["scores"]) for i in ids if i in run_rows]
    hit = n = 0
    for a in c["annotations"]:
        if a["image_id"] not in ids or a["category_id"] not in person:
            continue
        x, y, w, h = a["bbox"]
        if h < 64:
            continue
        n += 1
        cx, cy = x + w / 2, y + h / 2
        boxes = run_rows.get(a["image_id"], {}).get("boxes_xyxy_1080p", [])
        hit += any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes)
    return {
        "blobs_per_frame_median": float(np.median(per_frame)) if per_frame else None,
        "blobs_per_frame_max": max(per_frame) if per_frame else None,
        "person_ge_64px": n,
        "person_centre_inside_a_blob": hit / n if n else None,
    }


HUMANOIDS = ("FourierGR1T2", "AgilityDigit")
CONFIDENT = 0.6


def _humanoid_confusion(
    c: dict[str, Any],
    ids: set[int],
    run_rows: dict[int, dict[str, Any]],
    person_id: int,
    scale: float,
) -> dict[str, Any]:
    """How often the COCO ``person`` label lands on a humanoid robot.

    pycocotools matching (IoU 0.5, area range all): GT = the humanoid-robot
    boxes relabelled to the person category, detections = person-labelled
    boxes scoring >= 0.6. ``humanoids_called_person`` is that recall;
    ``confident_person_dets_on_humanoids`` counts matched detections."""
    humanoid_cats = {x["id"] for x in c["categories"] if x["name"] in HUMANOIDS}
    gt = {
        "images": [i for i in c["images"] if i["id"] in ids],
        "annotations": [
            {**a, "category_id": person_id}
            for a in c["annotations"]
            if a["image_id"] in ids and a["category_id"] in humanoid_cats
        ],
        "categories": [x for x in c["categories"] if x["id"] == person_id],
    }
    dets = [
        d
        for d in _dets(run_rows, ids, person_id, {"person"})
        if d["score"] >= CONFIDENT
    ]
    out: dict[str, Any] = {
        "humanoid_gt": len(gt["annotations"]),
        "confident_person_dets": len(dets),
        "score_threshold": CONFIDENT,
    }
    if not gt["annotations"] or not dets:
        out["humanoids_called_person"] = None
        out["confident_person_dets_on_humanoids"] = 0
        return out
    ev = _cocoeval(gt, dets, scale, use_cats=True)
    t = int(np.argmin(np.abs(ev.params.iouThrs - IOU_RECALL)))
    matched = sum(
        int(np.count_nonzero(e["dtMatches"][t]))
        for e in ev.evalImgs
        if e is not None and e["aRng"] == ev.params.areaRng[0]
    )
    out["humanoids_called_person"] = _ap_recall(ev, {"all": 1})["all"]["recall_iou50"]
    out["confident_person_dets_on_humanoids"] = matched
    return out


def score() -> dict[str, Any]:
    gates = _gate_bits()
    coco = {s: load_coco(s) for s in ("train", "val")}
    person_id = next(
        c["id"] for c in coco["val"]["categories"] if c["name"] == "Person"
    )
    result: dict[str, Any] = {
        "label": LABEL,
        "caveats": [
            LABEL,
            "Zero-shot COCO weights; SmartSpaces is rendered (no sensor noise model).",
            "720p/360p are downscales of the decoded 1080p frame with no "
            "re-compression: optimistic against a low-bitrate camera.",
            "Height buckets use the height at the evaluated resolution; a bucket "
            "with fewer than 30 GT instances is marked too_thin.",
            "Frames inside a re-render's post-cut transient (A2, 60 frames) are "
            "excluded from every metric.",
            "Only RT-DETRv2-S was evaluated; YOLOX-S is signed but has no backend "
            "yet, so no candidate selection took place.",
            "No timing, throughput, CPU or latency figure is measured or reported.",
        ],
        "score_floor": SCORE_FLOOR,
        "producers": {
            f"{b}__{r}": producer_identity(b, r) for b in BACKENDS for r in RESOLUTIONS
        },
        "height_edges_px_at_evaluated_resolution": list(HEIGHT_EDGES[:-1]) + ["inf"],
        "splits": {},
    }
    for split in ("val", "train"):
        c = coco[split]
        eligible = {i["id"] for i in c["images"] if not i["in_transient"]}
        strata = {
            "all": eligible,
            "source": {i["id"] for i in c["images"] if not i["ct25_rerender"]}
            & eligible,
            "ct25_rerender": {i["id"] for i in c["images"] if i["ct25_rerender"]}
            & eligible,
        }
        meta = {i["id"]: i for i in c["images"]}
        split_out: dict[str, Any] = {"images_scored": len(eligible), "strata": {}}
        for stratum, ids in strata.items():
            if not ids:
                continue
            res_out: dict[str, Any] = {}
            for resolution, (_, h) in RESOLUTIONS.items():
                scale = h / 1080
                runs = {b: _load_run(b, resolution) for b in BACKENDS}
                person_gt = _subset(c, ids, "Person")
                n_gt = _gt_counts(person_gt, scale)
                person = {}
                for backend, labels in (
                    ("rtdetr_v2_s", {"person"}),
                    ("motion_blob", None),
                ):
                    ev = _cocoeval(
                        person_gt,
                        _dets(runs[backend], ids, person_id, labels),
                        scale,
                        use_cats=True,
                    )
                    person[backend] = _ap_recall(ev, n_gt) if ev is not None else {}
                block: dict[str, Any] = {
                    "person": _with_baselines(
                        person["rtdetr_v2_s"], person["motion_blob"]
                    ),
                }
                for b in block["person"].values():
                    b["too_thin"] = b["gt"] < 30
                # The same physical objects across resolutions: buckets by
                # NATIVE 1080p height (scale 1.0), for prediction P2.
                native = {}
                for backend, labels in (
                    ("rtdetr_v2_s", {"person"}),
                    ("motion_blob", None),
                ):
                    ev = _cocoeval(
                        person_gt,
                        _dets(runs[backend], ids, person_id, labels),
                        1.0,
                        use_cats=True,
                    )
                    native[backend] = (
                        _ap_recall(ev, _gt_counts(person_gt, 1.0)) if ev else {}
                    )
                block["person_by_native_height"] = _with_baselines(
                    native["rtdetr_v2_s"], native["motion_blob"]
                )
                block["motion_blob_coverage_diagnostic"] = _blob_coverage(
                    c, ids, runs["motion_blob"]
                )
                block["person_label_on_humanoid_robots"] = _humanoid_confusion(
                    c, ids, runs["rtdetr_v2_s"], person_id, scale
                )
                # per class, class-agnostic recall (no COCO class for most)
                per_class = {}
                for cat in c["categories"]:
                    gt_c = _subset(c, ids, cat["name"])
                    if not gt_c["annotations"]:
                        continue
                    row = {}
                    for backend in BACKENDS:
                        ev = _cocoeval(
                            gt_c,
                            _dets(runs[backend], ids, cat["id"], None),
                            scale,
                            use_cats=False,
                        )
                        r = _ap_recall(ev, _gt_counts(gt_c, scale)) if ev else {}
                        row[backend] = r.get("all", {}).get("recall_iou50")
                    per_class[cat["name"]] = {
                        "gt": len(gt_c["annotations"]),
                        "too_thin": len(gt_c["annotations"]) < 30,
                        "in_detector_vocabulary": cat["name"] == "Person",
                        "class_agnostic_recall_iou50": row,
                        "AP": (
                            block["person"]["all"]["AP"]["value"]
                            if cat["name"] == "Person"
                            else "undefined: no COCO class"
                        ),
                    }
                block["per_class"] = per_class
                # cascade-level recall for Person
                cascade = {}
                for gate in ("none", "full", "320x180"):
                    kept = {
                        i
                        for i in ids
                        if gate == "none"
                        or gates[(meta[i]["scene"], meta[i]["camera"])][gate][
                            meta[i]["frame_index"]
                        ]
                    }
                    ev = _cocoeval(
                        person_gt,
                        _dets(runs["rtdetr_v2_s"], kept, person_id, {"person"}),
                        scale,
                        use_cats=True,
                    )
                    r = _ap_recall(ev, n_gt) if ev else {}
                    cascade[gate] = {
                        "frames_detector_ran": len(kept),
                        "recall_iou50": {
                            k: v.get("recall_iou50") for k, v in r.items()
                        },
                    }
                block["cascade_person_recall"] = cascade
                res_out[resolution] = block
            split_out["strata"][stratum] = res_out
        result["splits"][split] = split_out
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "score"))
    parser.add_argument("--backend", choices=BACKENDS, action="append")
    parser.add_argument("--resolution", choices=tuple(RESOLUTIONS), action="append")
    parser.add_argument(
        "--adopt-legacy",
        action="store_true",
        help="record the CURRENT producer for run files written before producer "
        "records existed; only when you know they were produced by it",
    )
    args = parser.parse_args(argv)
    if args.adopt_legacy:
        for backend in BACKENDS:
            for resolution in RESOLUTIONS:
                check_producer(backend, resolution, adopt_legacy=True)
        return 0
    if args.phase == "run":
        resolutions = args.resolution or list(RESOLUTIONS)
        for backend in args.backend or BACKENDS:
            if backend == "motion_blob":
                run_motion_blob(resolutions)
                continue
            for resolution in resolutions:
                run(backend, resolution)
        return 0
    payload = score()
    ARTIFACT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Written: {ARTIFACT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
