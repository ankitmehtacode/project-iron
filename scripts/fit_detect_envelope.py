"""Fit the detect envelope and the score calibration (Day 42, Objective B6).

lane S synthetic (SmartSpaces) — zero-shot, not a product claim; NVIDIA terms
unresolved. synthetic-derived: not valid for a real camera until validated on
real footage. No timing figure anywhere.

Inputs are B5's per-frame detections (``scripts/eval_detection_zero_shot.py
run``) for RT-DETRv2-S, class Person, frames inside a re-render's post-cut
transient excluded. Matching is pycocotools' own (``COCOeval.evalImgs`` at
IoU 0.5, area range "all"); nothing here re-implements matching.

- Envelope, per resolution: over height bins holding >= 30 GT instances,
  recall@IoU0.5 (pycocotools), and for matched pairs the robust sigma
  (1.4826 x MAD) of centre residuals (dx and dy pooled) and of size
  residuals (dw and dh pooled), in pixels AT THAT RESOLUTION. A curve point's
  x is the median GT height in the bin. Fit on SmartSpaces val (ADR 0013
  rule 2); train is reported alongside as a held-out check.
- Calibration: reliability bins, score bin x the detection's own height
  bucket, empirical precision = TP / (TP + FP) with Wilson intervals. Fit on
  val, checked on train. Recorded as a CalibrationRecord (method
  histogram_binning) valid only for this model, preprocessing, resolution
  and class.

    python scripts/fit_detect_envelope.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_detection_zero_shot import (  # noqa: E402
    HEIGHT_EDGES,
    LABEL,
    RESOLUTIONS,
    _ap_recall,
    _cocoeval,
    _dets,
    _gt_counts,
    _load_run,
    _subset,
    load_coco,
    run_path,
)
from smartspaces_integrity import REPO_ROOT  # noqa: E402

from src.detect.envelope import SYNTHETIC_DERIVED  # noqa: E402
from src.eval.intervals import wilson_interval  # noqa: E402
from src.model.evidence import CalibrationRecord  # noqa: E402

ARTIFACT = REPO_ROOT / "artifacts" / "smartspaces" / "day42_detect_envelope.json"
SCORE_BINS = (0.05, 0.2, 0.4, 0.6, 0.8, 1.0000001)
MIN_PER_BIN = 30
IOU = 0.5


def _robust_sigma(values: list[float]) -> float | None:
    """``values`` holds two residuals per matched pair (x and y, or w and
    h); the point needs MIN_PER_BIN PAIRS, not pooled values."""
    if len(values) // 2 < MIN_PER_BIN:
        return None
    x = np.asarray(values, dtype=np.float64)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _bucket(height: float) -> int:
    for i in range(len(HEIGHT_EDGES) - 1):
        if HEIGHT_EDGES[i] <= height < HEIGHT_EDGES[i + 1]:
            return i
    return len(HEIGHT_EDGES) - 2


def _bucket_label(i: int) -> str:
    lo, hi = HEIGHT_EDGES[i], HEIGHT_EDGES[i + 1]
    return f"[{lo:g},{hi:g})" if hi < 1e5 else f"[{lo:g},inf)"


HUMANOIDS = ("FourierGR1T2", "AgilityDigit")


def _humanoid_matched(
    coco: dict[str, Any], ids: set[int], dets: list[dict[str, Any]], scale: float
) -> set[int]:
    """Indices into ``dets`` that pycocotools matches (IoU 0.5) to a humanoid
    robot box relabelled as the detections' category."""
    cats = {c["id"] for c in coco["categories"] if c["name"] in HUMANOIDS}
    if not dets:
        return set()
    cat = dets[0]["category_id"]
    gt = {
        "images": [i for i in coco["images"] if i["id"] in ids],
        "annotations": [
            {**a, "category_id": cat}
            for a in coco["annotations"]
            if a["image_id"] in ids and a["category_id"] in cats
        ],
        "categories": [c for c in coco["categories"] if c["id"] == cat],
    }
    if not gt["annotations"]:
        return set()
    ev = _cocoeval(gt, dets, scale, use_cats=True)
    t = int(np.argmin(np.abs(ev.params.iouThrs - IOU)))
    matched: set[int] = set()
    for e in ev.evalImgs:
        if e is None or e["aRng"] != ev.params.areaRng[0]:
            continue
        for k, dt_id in enumerate(e["dtIds"]):
            if e["dtMatches"][t][k]:
                matched.add(int(dt_id) - 1)  # loadRes numbers dets 1..N in order
    return matched


def measure(
    split: str, resolution: str, drop_humanoid_matches: bool = False
) -> dict[str, Any]:
    coco = load_coco(split)
    scale = RESOLUTIONS[resolution][1] / 1080
    ids = {i["id"] for i in coco["images"] if not i["in_transient"]}
    person = next(c["id"] for c in coco["categories"] if c["name"] == "Person")
    gt = _subset(coco, ids, "Person")
    run = _load_run("rtdetr_v2_s", resolution)
    dets = _dets(run, ids, person, {"person"})
    dropped = 0
    if drop_humanoid_matches:
        drop = _humanoid_matched(coco, ids, dets, scale)
        dropped = len(drop)
        dets = [d for k, d in enumerate(dets) if k not in drop]
    ev = _cocoeval(gt, dets, scale, use_cats=True)
    recall = _ap_recall(ev, _gt_counts(gt, scale))
    t = int(np.argmin(np.abs(ev.params.iouThrs - IOU)))
    all_rng = ev.params.areaRng[0]

    centre: dict[int, list[float]] = {}
    size: dict[int, list[float]] = {}
    heights: dict[int, list[float]] = {}
    reliability: dict[tuple[int, int], list[int]] = {}
    for a in gt["annotations"]:
        h = a["bbox"][3] * scale
        heights.setdefault(_bucket(h), []).append(h)
    for e in ev.evalImgs:
        if e is None or e["aRng"] != all_rng:
            continue
        for k, dt_id in enumerate(e["dtIds"]):
            if e["dtIgnore"][t][k]:
                continue
            match = int(e["dtMatches"][t][k])
            d = ev.cocoDt.anns[dt_id]["bbox"]
            dh = d[3] * scale
            score_bin = int(np.searchsorted(SCORE_BINS, e["dtScores"][k], "right")) - 1
            key = (max(0, score_bin), _bucket(dh))
            reliability.setdefault(key, [0, 0])[0 if match else 1] += 1
            if match:
                g = ev.cocoGt.anns[match]["bbox"]
                b = _bucket(g[3] * scale)
                gx, gy = g[0] + g[2] / 2, g[1] + g[3] / 2
                dx, dy = d[0] + d[2] / 2, d[1] + d[3] / 2
                centre.setdefault(b, []).extend([(dx - gx) * scale, (dy - gy) * scale])
                size.setdefault(b, []).extend(
                    [(d[2] - g[2]) * scale, (d[3] - g[3]) * scale]
                )
    bins = []
    for b in range(len(HEIGHT_EDGES) - 1):
        label = _bucket_label(b)
        n = len(heights.get(b, []))
        bins.append(
            {
                "bucket": label,
                "gt": n,
                "median_height_px": float(np.median(heights[b])) if n else None,
                "recall_iou50": recall.get(label, {}).get("recall_iou50"),
                "matched_pairs": len(centre.get(b, [])) // 2,
                "centre_sigma_px": _robust_sigma(centre.get(b, [])),
                "size_sigma_px": _robust_sigma(size.get(b, [])),
                "too_thin": n < MIN_PER_BIN,
            }
        )
    table = []
    for (s, b), (tp, fp) in sorted(reliability.items()):
        n = tp + fp
        low, high = wilson_interval(tp, n)
        table.append(
            {
                "score_bin": [SCORE_BINS[s], min(1.0, SCORE_BINS[s + 1])],
                "height_bucket": _bucket_label(b),
                "detections": n,
                "precision": tp / n,
                "precision_wilson95": [low, high],
                "too_thin": n < MIN_PER_BIN,
            }
        )
    return {"bins": bins, "reliability": table, "humanoid_matched_dropped": dropped}


def _transfer(fit: list[dict[str, Any]], check: list[dict[str, Any]]) -> dict[str, Any]:
    """Does the val-fitted reliability table hold on train? Per (score bin,
    height bucket) populated on BOTH sides (>= 30 detections each)."""
    key = lambda r: (tuple(r["score_bin"]), r["height_bucket"])  # noqa: E731
    held = {key(r): r for r in check if not r["too_thin"]}
    rows = []
    for r in fit:
        other = held.get(key(r))
        if r["too_thin"] or other is None:
            continue
        overlap = not (
            r["precision_wilson95"][1] < other["precision_wilson95"][0]
            or other["precision_wilson95"][1] < r["precision_wilson95"][0]
        )
        rows.append(
            {
                "score_bin": r["score_bin"],
                "height_bucket": r["height_bucket"],
                "precision_val": r["precision"],
                "precision_train": other["precision"],
                "intervals_overlap": overlap,
            }
        )
    return {
        "comparable_bins": len(rows),
        "bins_where_intervals_disjoint": sum(not r["intervals_overlap"] for r in rows),
        "rows": rows,
        "verdict": (
            "no comparable bin"
            if not rows
            else "does not transfer"
            if any(not r["intervals_overlap"] for r in rows)
            else "consistent on comparable bins"
        ),
    }


def _curve(bins: list[dict[str, Any]], field: str) -> list[list[float]] | None:
    pts = [
        [b["median_height_px"], b[field]]
        for b in bins
        if not b["too_thin"] and b[field] is not None and b["median_height_px"]
    ]
    return pts if len(pts) >= 2 else None


def main() -> int:
    run_shas = {
        res: hashlib.sha256(run_path("rtdetr_v2_s", res).read_bytes()).hexdigest()
        for res in RESOLUTIONS
    }
    manifest_sha = hashlib.sha256(
        json.dumps({"runs": run_shas, "script": Path(__file__).read_text()}).encode()
    ).hexdigest()
    registry_model_sha = _model_sha()
    out: dict[str, Any] = {
        "label": LABEL,
        "envelope_label": SYNTHETIC_DERIVED,
        "class": "Person",
        "fit_on": "SmartSpaces val, export 66c683ffa66f, transient frames excluded",
        "check_on": "SmartSpaces train (held out from the fit)",
        "manifest_sha": manifest_sha,
        "run_shas": run_shas,
        "producer_model_sha": registry_model_sha,
        "producer_preprocess_sha": _preprocess_sha(),
        "resolutions": {},
        "calibration_records": {},
    }
    for res in RESOLUTIONS:
        fit = measure("val", res)
        check = measure("train", res)
        check_clean = measure("train", res, drop_humanoid_matches=True)
        curves = {
            f: _curve(fit["bins"], f)
            for f in ("recall_iou50", "centre_sigma_px", "size_sigma_px")
        }
        out["resolutions"][res] = {
            "calibration_transfer": _transfer(fit["reliability"], check["reliability"]),
            "calibration_transfer_without_humanoid_matches": _transfer(
                fit["reliability"], check_clean["reliability"]
            ),
            "humanoid_matched_person_dets_dropped": check_clean[
                "humanoid_matched_dropped"
            ],
            "fit_val": fit,
            "check_train": check,
            "envelope_curves": curves,
            "envelope_status": (
                "measured" if all(curves.values()) else "too few populated bins"
            ),
        }
        record = CalibrationRecord(
            calibration_id=f"rtdetr_v2_s-person-{res}-{manifest_sha[:12]}",
            method="histogram_binning (score bin x detection height bucket)",
            fit_on=out["fit_on"],
            valid_for=f"RT-DETRv2-S-COCO@{registry_model_sha[:12]} person @ {res}",
            manifest_sha=manifest_sha,
        )
        block = out["resolutions"][res]
        transfer = block["calibration_transfer"]
        clean = block["calibration_transfer_without_humanoid_matches"]
        consistent = transfer["verdict"] == "consistent on comparable bins"
        out["calibration_records"][res] = {
            **record.__dict__,
            # The record exists; whether a score may be SHOWN as a probability
            # is a separate verdict, from the held-out check.
            "show_as_probability": consistent,
            "why_not": None
            if consistent
            else (
                f"{transfer['verdict']} to train: "
                f"{transfer['bins_where_intervals_disjoint']} of "
                f"{transfer['comparable_bins']} comparable bins disjoint; with the "
                f"{block['humanoid_matched_person_dets_dropped']} person detections "
                "that match a humanoid-robot box removed, "
                f"{clean['bins_where_intervals_disjoint']} of "
                f"{clean['comparable_bins']} remain disjoint, in buckets "
                + ", ".join(
                    sorted(
                        {
                            r["height_bucket"]
                            for r in clean["rows"]
                            if not r["intervals_overlap"]
                        }
                    )
                )
            ),
        }
    ARTIFACT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Written: {ARTIFACT.relative_to(REPO_ROOT)}")
    return 0


def _preprocess_sha() -> str:
    from src.detect.preprocess import CANONICAL_SPEC_DIR, DetectorPreprocessSpec

    return DetectorPreprocessSpec.load(
        CANONICAL_SPEC_DIR / "rtdetr_v2_r18vd.preprocess.json"
    ).preprocess_sha()


def _model_sha() -> str:
    from src.data.registry import DatasetRegistry

    entry = DatasetRegistry.load(REPO_ROOT / "configs" / "datasets.yaml").get(
        "RT-DETRv2-S-COCO"
    )
    if not entry.content_sha:
        raise SystemExit("RT-DETRv2-S-COCO has no content_sha: fetch it first")
    return entry.content_sha


if __name__ == "__main__":
    raise SystemExit(main())
