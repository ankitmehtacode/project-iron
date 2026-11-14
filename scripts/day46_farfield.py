"""Day 46, Objective 5: far-field person detection, full frame vs tiles.

Run through scripts/on_ac.py; every detector output is appended (fsynced) to
a resumable JSONL cache keyed by (scene, camera, frame, mode), so a power stop
resumes at the first missing key:

    python scripts/on_ac.py -- python scripts/day46_farfield.py \\
        --cache data/derived/farfield/day46.jsonl --out artifacts/day46/farfield.json

Sample (fixed before any detector output was read): validation scenes whose
ground truth is their own (Warehouse_020: 16 cameras, Warehouse_022: 4;
Warehouse_021 is excluded because its ground_truth.json is byte-identical to
020's while its videos differ, so its boxes cannot be trusted to describe its
frames); 15 frames per (scene, camera) drawn with numpy seed 46 from the
first 1800 frames: 300 frames.

Modes (RT-DETRv2-S, its own preprocessing, nothing tuned here):
- ``full``: the whole frame, stretched to the model's 640x640 (1 invocation);
- ``tiles``: 2x2 tiles cut from the native frame, each 1/1.8 of the frame's
  width and height (20% of a tile overlaps its neighbour), each stretched to
  640x640 (4 invocations); merged by mapping boxes back to the frame and
  class-wise NMS at IoU 0.5, higher score kept;
- ``full_1280``: the whole frame at 1280x1280 if the model accepts that input
  (checked on the first frame; RT-DETR builds its anchors from the feature
  map size), else not run and said so (1 invocation, larger).

Scoring: person detections with score >= 0.5 (the product tracker's existing
``high_score``; not chosen here) matched one-to-one to visible GT Person boxes
at IoU >= 0.5 (Hungarian). Recall per GT box height bucket, precision per
detection height bucket (source pixels), with 95% intervals from 2000
bootstrap resamples of scenes (two scenes: the interval is close to
degenerate and is reported anyway) and of (scene, camera) clusters. Person AP
at IoU 0.5 (all-point interpolation, every score >= 0.05) twice: humanoid
robots as ignore regions and as false positives; the ground truth's classes
are listed so the reader can see whether a humanoid class exists.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.contracts.frames import FrameGeometry  # noqa: E402
from src.detect.contract import FrameInput  # noqa: E402
from src.pipeline.tracker import PRODUCT_TRACKER, iou_matrix  # noqa: E402
from src.pipeline.truth import load_truth, oracle_boxes  # noqa: E402

SCENES = {
    "Warehouse_020": "val/Warehouse_020",
    "Warehouse_022": "val/Warehouse_022",
}
DATA = (
    ROOT
    / "data/raw/PhysicalAI-SmartSpaces/2cbe9563cbe9f47f846e5c871ee994572bbbc60e"
    / "paths-0deaa0fe541fbaf1/MTMC_Tracking_2026"
)
PER_STRATUM = 15
SEED = 46
FIRST_FRAMES = 1800
TILE_FRACTION = 1 / 1.8
NMS_IOU = 0.5
MATCH_IOU = 0.5
SCORE = PRODUCT_TRACKER.high_score
AP_FLOOR = 0.05
BUCKETS = (0, 40, 80, 160, 320, 10**9)
HUMANOID_WORDS = ("humanoid", "gr1", "digit", "robot")


def bucket_names() -> list[str]:
    return [
        f"[{lo},{hi})" if hi < 10**9 else f"[{lo},inf)"
        for lo, hi in zip(BUCKETS, BUCKETS[1:])
    ]


def bucket_of(h: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.array(BUCKETS[1:]), h, side="right")


def sample() -> list[tuple[str, str, int]]:
    rng = np.random.default_rng(SEED)
    out = []
    for scene, rel in SCENES.items():
        cams = sorted(p.stem for p in (DATA / rel / "videos").glob("Camera_*.mp4"))
        for cam in cams:
            frames = sorted(rng.choice(FIRST_FRAMES, PER_STRATUM, replace=False))
            out += [(scene, cam, int(f)) for f in frames]
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, labels: list[str]) -> np.ndarray:
    keep: list[int] = []
    for lab in set(labels):
        idx = [i for i, lb in enumerate(labels) if lb == lab]
        idx.sort(key=lambda i: -scores[i])
        chosen: list[int] = []
        for i in idx:
            if chosen and iou_matrix(boxes[[i]], boxes[chosen]).max() >= NMS_IOU:
                continue
            chosen.append(i)
        keep += chosen
    return np.array(sorted(keep), dtype=np.int64)


def run_mode(detector: Any, image: np.ndarray, mode: str) -> dict[str, Any]:
    h, w = image.shape[:2]
    vocab = detector.vocabulary.labels

    def one(
        img: np.ndarray, ox: int, oy: int
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        g = FrameGeometry(width=img.shape[1], height=img.shape[0])
        d = detector.detect(
            FrameInput(
                image_bgr=np.ascontiguousarray(img), ts_ns=0, frame_index=0, geometry=g
            )
        )
        b = np.asarray(d.boxes_xyxy, np.float64).reshape(-1, 4) + [ox, oy, ox, oy]
        return b, d.score_array(), [vocab[int(c)] for c in d.class_ids]

    if mode in ("full", "full_1280"):
        b, s, lab = one(image, 0, 0)
        invocations = 1
    else:
        tw, th = int(round(w * TILE_FRACTION)), int(round(h * TILE_FRACTION))
        parts = [
            one(image[oy : oy + th, ox : ox + tw], ox, oy)
            for oy in (0, h - th)
            for ox in (0, w - tw)
        ]
        b = np.concatenate([p[0] for p in parts])
        s = np.concatenate([p[1] for p in parts])
        lab = [x for p in parts for x in p[2]]
        k = nms(b, s, lab)
        b, s, lab = b[k], s[k], [lab[i] for i in k]
        invocations = 4
    return {
        "b": b.round(2).tolist(),
        "s": s.round(5).tolist(),
        "l": lab,
        "inv": invocations,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--score-only", action="store_true", help="no detector; score the cache"
    )
    args = p.parse_args(argv)
    frames = sample()
    done: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    if args.cache.exists():
        for line in args.cache.read_text().splitlines():
            r = json.loads(line)
            done[(r["scene"], r["cam"], r["frame"], r["mode"])] = r
    modes = ["full", "tiles", "full_1280"]
    accepts_1280: dict[str, Any] = {}
    if not args.score_only:
        import cv2

        from src.main import pretrained_detector_factory

        detector = pretrained_detector_factory()(None)
        big = replace(detector.spec, input_hw=(1280, 1280))
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        with open(args.cache, "a") as cache:
            by_video: dict[tuple[str, str], list[int]] = {}
            for scene, cam, f in frames:
                by_video.setdefault((scene, cam), []).append(f)
            for (scene, cam), fs in by_video.items():
                need = [
                    f for f in fs if any((scene, cam, f, m) not in done for m in modes)
                ]
                if not need:
                    continue
                cap = cv2.VideoCapture(
                    str(DATA / SCENES[scene] / "videos" / f"{cam}.mp4")
                )
                index = 0
                for f in sorted(need):
                    while index <= f:
                        ok, image = cap.read()
                        if not ok:
                            raise SystemExit(
                                f"{scene}/{cam}: cannot decode frame {index}"
                            )
                        index += 1
                    for mode in modes:
                        if (scene, cam, f, mode) in done:
                            continue
                        if mode == "full_1280":
                            normal = detector._spec
                            detector._spec = big
                            try:
                                r = run_mode(detector, image, mode)
                            except Exception as exc:  # recorded, not swallowed
                                accepts_1280 = {"accepted": False, "error": repr(exc)}
                                raise SystemExit(f"full_1280 refused: {exc!r}")
                            finally:
                                detector._spec = normal
                        else:
                            r = run_mode(detector, image, mode)
                        r.update(scene=scene, cam=cam, frame=f, mode=mode)
                        cache.write(json.dumps(r) + "\n")
                        cache.flush()
                        os.fsync(cache.fileno())
                        done[(scene, cam, f, mode)] = r
                    print(
                        f"{scene}/{cam} frame {f}: {len(done)} of {len(frames) * len(modes)}",
                        flush=True,
                    )
                cap.release()
    result = score(frames, done, modes)
    result["accepts_1280"] = accepts_1280 or {
        "accepted": any(k[3] == "full_1280" for k in done),
        "how": "the model ran on a 1280x1280 input and returned boxes; see full_1280 rows",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {m: result["modes"][m]["recall_by_gt_height"] for m in result["modes"]},
            indent=1,
        )
    )
    return 0


def score(
    frames: list[tuple[str, str, int]],
    done: dict[tuple[str, str, int, str], dict[str, Any]],
    modes: list[str],
) -> dict[str, Any]:
    gts = {s: load_truth(DATA / rel / "ground_truth.json") for s, rel in SCENES.items()}
    classes = {s: list(g.class_names) for s, g in gts.items()}
    humanoid = {
        s: [c for c in cl if any(w in c.lower() for w in HUMANOID_WORDS)]
        for s, cl in classes.items()
    }
    boxes_cache: dict[tuple[str, str], Any] = {}
    names = bucket_names()
    out: dict[str, Any] = {
        "sample": {"frames": len(frames), "per_stratum": PER_STRATUM, "seed": SEED},
        "gt_classes": classes,
        "humanoid_classes_found": humanoid,
        "gt_person_boxes_by_height": {n: 0 for n in names},
        "modes": {},
    }
    rows: dict[str, list[dict[str, Any]]] = {m: [] for m in modes}
    counted = False
    for mode in modes:
        for scene, cam, f in frames:
            r = done.get((scene, cam, f, mode))
            if r is None:
                continue
            if (scene, cam) not in boxes_cache:
                boxes_cache[(scene, cam)] = oracle_boxes(gts[scene], cam)
            truth = boxes_cache[(scene, cam)]
            gb = np.asarray(truth.boxes.get(f, np.zeros((0, 4))), np.float64).reshape(
                -1, 4
            )
            gl = list(truth.labels.get(f, []))
            person = np.array([lb == "Person" for lb in gl], bool)
            ignore = np.array([lb in humanoid[scene] for lb in gl], bool)
            g = gb[person]
            b = np.asarray(r["b"], np.float64).reshape(-1, 4)
            s = np.asarray(r["s"], np.float64)
            isp = np.array([lb == "person" for lb in r["l"]], bool)
            b, s = b[isp], s[isp]
            rows[mode].append(
                {
                    "scene": scene,
                    "cam": cam,
                    "g": g,
                    "ign": gb[ignore],
                    "b": b,
                    "s": s,
                    "inv": r["inv"],
                }
            )
            if not counted:
                for k in bucket_of(g[:, 3] - g[:, 1]):
                    out["gt_person_boxes_by_height"][names[k]] += 1
        counted = counted or bool(rows[mode])
    for mode in modes:
        if not rows[mode]:
            out["modes"][mode] = {"ran": False}
            continue
        out["modes"][mode] = summarise(rows[mode], names)
    return out


def _match(g: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mg = np.zeros(len(g), bool)
    mb = np.zeros(len(b), bool)
    if len(g) and len(b):
        iou = iou_matrix(g, b)
        r, c = linear_sum_assignment(1 - iou)
        ok = iou[r, c] >= MATCH_IOU
        mg[r[ok]] = True
        mb[c[ok]] = True
    return mg, mb


def summarise(rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    nb = len(names)
    per: list[dict[str, Any]] = []
    for row in rows:
        keep = row["s"] >= SCORE
        b = row["b"][keep]
        mg, mb = _match(row["g"], b)
        gk = bucket_of(row["g"][:, 3] - row["g"][:, 1])
        bk = bucket_of(b[:, 3] - b[:, 1])
        per.append(
            {
                "scene": row["scene"],
                "cluster": f"{row['scene']}/{row['cam']}",
                "gt": np.bincount(gk, minlength=nb),
                "tp_gt": np.bincount(gk[mg], minlength=nb),
                "det": np.bincount(bk, minlength=nb),
                "tp_det": np.bincount(bk[mb], minlength=nb),
            }
        )

    def totals(items: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        return {
            k: sum((it[k] for it in items), np.zeros(nb, int))
            for k in ("gt", "tp_gt", "det", "tp_det")
        }

    t = totals(per)

    def boot(key: str) -> dict[str, list[list[float | None]]]:
        rng = np.random.default_rng(0)
        groups: dict[str, list[dict[str, Any]]] = {}
        for it in per:
            groups.setdefault(it[key], []).append(it)
        keys = sorted(groups)
        rec, prec = [], []
        for _ in range(2000):
            pick = rng.choice(len(keys), len(keys), replace=True)
            tt = totals([it for i in pick for it in groups[keys[i]]])
            with np.errstate(divide="ignore", invalid="ignore"):
                rec.append(tt["tp_gt"] / tt["gt"])
                prec.append(tt["tp_det"] / tt["det"])

        def ci(a: list[np.ndarray]) -> list[list[float | None]]:
            m = np.array(a)
            out = []
            for j in range(nb):
                col = m[:, j][np.isfinite(m[:, j])]
                out.append(
                    [float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5))]
                    if col.size
                    else [None, None]
                )
            return out

        return {"recall": ci(rec), "precision": ci(prec)}

    by_scene, by_cluster = boot("scene"), boot("cluster")
    return {
        "ran": True,
        "frames": len(rows),
        "invocations_per_frame": rows[0]["inv"],
        "recall_by_gt_height": {
            n: {
                "matched": int(t["tp_gt"][j]),
                "gt": int(t["gt"][j]),
                "ci95_scene": by_scene["recall"][j],
                "ci95_camera": by_cluster["recall"][j],
            }
            for j, n in enumerate(names)
        },
        "precision_by_det_height": {
            n: {
                "matched": int(t["tp_det"][j]),
                "detections": int(t["det"][j]),
                "ci95_scene": by_scene["precision"][j],
                "ci95_camera": by_cluster["precision"][j],
            }
            for j, n in enumerate(names)
        },
        "ap50_humanoid_ignored": ap(rows, ignore=True),
        "ap50_humanoid_false_positive": ap(rows, ignore=False),
    }


def ap(rows: list[dict[str, Any]], ignore: bool) -> float | None:
    scored: list[tuple[float, int]] = []
    n_gt = 0
    for row in rows:
        keep = row["s"] >= AP_FLOOR
        b, s = row["b"][keep], row["s"][keep]
        order = np.argsort(-s)
        b, s = b[order], s[order]
        g = row["g"]
        n_gt += len(g)
        used = np.zeros(len(g), bool)
        for i in range(len(b)):
            hit = -1
            if len(g):
                iou = iou_matrix(b[[i]], g)[0]
                iou[used] = -1
                j = int(iou.argmax())
                if iou[j] >= MATCH_IOU:
                    hit = j
            if hit >= 0:
                used[hit] = True
                scored.append((float(s[i]), 1))
                continue
            if (
                ignore
                and len(row["ign"])
                and iou_matrix(b[[i]], row["ign"]).max() >= MATCH_IOU
            ):
                continue
            scored.append((float(s[i]), 0))
    if not n_gt:
        return None
    scored.sort(key=lambda x: -x[0])
    tp = np.cumsum([x[1] for x in scored])
    fp = np.cumsum([1 - x[1] for x in scored])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate([[0], recall, [1]])
    mpre = np.concatenate([[0], precision, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


if __name__ == "__main__":
    raise SystemExit(main())
