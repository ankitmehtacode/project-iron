"""Day 44: a fitted per-detection position-error model vs a single constant.

Data: GT person boxes (oracle), cameras 0-3, first ``--seconds`` of each
scene, anchored in ``fallback`` mode (src/pipeline/anchor.py). Error = the
anchored floor point minus the GT floor point (2-D, metres).

Features, every one available at inference (none uses ground truth):
  log of the estimated range (camera to the anchored point), log of the
  range-interval width at PERSON_MAX_HEIGHT_M (+0.1 m; a box with no width
  gets the training median and an indicator), and the anchor mode
  (foot / head / foot_flagged one-hot; foot is the reference level).

Model: isotropic 2-D Gaussian, log sigma = X @ w, fitted by maximum
likelihood on the TRAIN scenes. Baseline: one constant sigma fitted by
maximum likelihood on the same train data (sigma^2 = mean |e|^2 / 2).
Also scored, unfitted: the analytic ellipse (anchor.py covariance).

Held out: the val scenes. Reported per held-out scene and pooled: mean log
density (nats per detection) of each model, the margin over the constant
with a 95% bootstrap clustered by GT person, and coverage of the 68% and
95% regions. The model replaces the analytic ellipse ONLY if its held-out
margin over the constant is positive with an interval excluding zero.

Train scenes (fixed before fitting): Warehouse_000, 002, 004, 006, the
first four train SOURCE scenes (001/003/005/007 are Cosmos re-renders of
them). Held out: Warehouse_020 and 022 (val).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.smartspaces import MEASURED_GEOMETRY  # noqa: E402
from src.pipeline.anchor import (  # noqa: E402
    PERSON_MAX_HEIGHT_M,
    anchor_boxes,
    camera_from_smartspaces,
    range_interval,
)
from src.pipeline.frontier import BOOTSTRAP_ROUNDS, BOOTSTRAP_SEED  # noqa: E402
from src.pipeline.truth import PERSON, load_truth  # noqa: E402

CAMERAS = ("Camera_0000", "Camera_0001", "Camera_0002", "Camera_0003")
ANCHOR_LEVELS = ("head", "foot_flagged")
CHI2_2 = {"68": 2.2788, "95": 5.9915}


def collect(scene_dir: Path, n: int) -> dict[str, np.ndarray]:
    gt = load_truth(scene_dir / "ground_truth.json")
    rows: dict[str, list[np.ndarray]] = {
        k: [] for k in ("err", "cov", "range", "width", "anchor", "person")
    }
    for cam in CAMERAS:
        camera = camera_from_smartspaces(scene_dir / "calibration.json", cam)
        sel = np.flatnonzero(gt.box_camera == gt.camera_names.index(cam))
        r = gt.box_row[sel]
        keep = (gt.class_index[r] == gt.class_names.index(PERSON)) & (gt.frame[r] < n)
        sel, r = sel[keep], r[keep]
        boxes = gt.box_xyxy[sel].astype(np.float64)
        floor = MEASURED_GEOMETRY.floor_point(gt.location[r], gt.scale[r])[:, :2]
        a = anchor_boxes(camera, boxes, ["Person"] * len(boxes), mode="fallback")
        ri = range_interval(camera, boxes, PERSON_MAX_HEIGHT_M)
        ok = a.anchor != "none"
        rows["err"].append(a.xy[ok] - floor[ok])
        rows["cov"].append(a.ray_cov[ok])
        rows["range"].append(np.linalg.norm(a.xy[ok] - camera.centre[:2], axis=1))
        rows["width"].append(ri.width_m[ok])
        rows["anchor"].append(a.anchor[ok])
        rows["person"].append(
            np.array([f"{scene_dir.name}/{int(o)}" for o in gt.object_id[r][ok]])
        )
    return {k: np.concatenate(v) for k, v in rows.items()}


def design(d: dict[str, np.ndarray], width_fill: float) -> np.ndarray:
    w = d["width"].copy()
    missing = ~np.isfinite(w) | (w < 0)
    w[missing] = width_fill
    cols = [
        np.ones(len(w)),
        np.log(np.maximum(d["range"], 0.1)),
        np.log(w + 0.1),
        missing.astype(float),
    ]
    cols += [(d["anchor"] == lvl).astype(float) for lvl in ANCHOR_LEVELS]
    return np.stack(cols, axis=1)


def loglik_iso(e2: np.ndarray, log_sigma: np.ndarray) -> np.ndarray:
    s2 = np.exp(2 * log_sigma)
    out: np.ndarray = -np.log(2 * np.pi * s2) - e2 / (2 * s2)
    return out


def loglik_cov(err: np.ndarray, cov: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(cov)
    m2 = np.einsum("ni,nij,nj->n", err, inv, err)
    out: np.ndarray = -np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(cov)) - 0.5 * m2
    return out


def fit(x: np.ndarray, e2: np.ndarray) -> np.ndarray:
    def nll(w: np.ndarray) -> float:
        return float(-np.mean(loglik_iso(e2, x @ w)))

    def grad(w: np.ndarray) -> np.ndarray:
        ls = x @ w
        g = -2.0 + e2 * np.exp(-2 * ls)  # d loglik / d log_sigma
        out: np.ndarray = -(x * g[:, None]).mean(axis=0)
        return out

    w0 = np.zeros(x.shape[1])
    w0[0] = 0.5 * np.log(np.mean(e2) / 2)
    res = minimize(nll, w0, jac=grad, method="L-BFGS-B")
    if not res.success:
        raise SystemExit(f"fit failed: {res.message}")
    out: np.ndarray = res.x
    return out


def clustered_margin(diff: np.ndarray, person: np.ndarray) -> tuple[float, list[float]]:
    keys, inv = np.unique(person, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    counts = np.bincount(inv).astype(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(keys), size=(BOOTSTRAP_ROUNDS, len(keys)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(diff.mean()), [
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    ]


def score(
    d: dict[str, np.ndarray], w: np.ndarray, sigma_c: float, width_fill: float
) -> dict[str, Any]:
    e2 = (d["err"] ** 2).sum(axis=1)
    ls = design(d, width_fill) @ w
    ll_model = loglik_iso(e2, ls)
    ll_const = loglik_iso(e2, np.full(len(e2), np.log(sigma_c)))
    ll_ellipse = loglik_cov(d["err"], d["cov"])
    out: dict[str, Any] = {"n": int(len(e2)), "persons": int(len(set(d["person"])))}
    for name, ll in (
        ("fitted", ll_model),
        ("constant", ll_const),
        ("analytic_ellipse", ll_ellipse),
    ):
        out[f"mean_loglik_{name}"] = float(ll.mean())
    for name, ll in (("fitted", ll_model), ("analytic_ellipse", ll_ellipse)):
        m, ci = clustered_margin(ll - ll_const, d["person"])
        out[f"margin_over_constant_{name}"] = {"nats": m, "ci95_by_person": ci}
    for level, q in CHI2_2.items():
        out[f"coverage{level}_fitted"] = float((e2 <= q * np.exp(2 * ls)).mean())
        out[f"coverage{level}_constant"] = float((e2 <= q * sigma_c**2).mean())
        inv = np.linalg.inv(d["cov"])
        m2 = np.einsum("ni,nij,nj->n", d["err"], inv, d["err"])
        out[f"coverage{level}_analytic_ellipse"] = float((m2 <= q).mean())
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--train", action="append", type=Path, required=True)
    p.add_argument("--heldout", action="append", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    n = int(round(args.seconds * 30))
    train_parts = [collect(s, n) for s in args.train]
    train = {k: np.concatenate([t[k] for t in train_parts]) for k in train_parts[0]}
    wfin = train["width"][np.isfinite(train["width"]) & (train["width"] >= 0)]
    width_fill = float(np.median(wfin))
    e2 = (train["err"] ** 2).sum(axis=1)
    w = fit(design(train, width_fill), e2)
    sigma_c = float(np.sqrt(e2.mean() / 2))
    names = ["intercept", "log_range", "log_width", "width_missing"] + [
        f"anchor_{lvl}" for lvl in ANCHOR_LEVELS
    ]
    result: dict[str, Any] = {
        "label": "lane S synthetic; oracle person boxes; fallback anchor",
        "train_scenes": [s.name for s in args.train],
        "heldout_scenes": [s.name for s in args.heldout],
        "frames_per_camera": n,
        "coefficients_log_sigma": dict(zip(names, map(float, w))),
        "train_rows_by_anchor": {
            str(k): int(v)
            for k, v in zip(*np.unique(train["anchor"], return_counts=True))
        },
        "unidentified": [
            f"anchor_{lvl}"
            for lvl in ANCHOR_LEVELS
            if not (train["anchor"] == lvl).any()
        ],
        "unidentified_note": (
            "a level with no training rows keeps its starting coefficient 0: "
            "it is NOT fitted, and such detections get the foot-level sigma"
        ),
        "constant_sigma_m": sigma_c,
        "width_fill_m": width_fill,
        "train": score(train, w, sigma_c, width_fill),
        "heldout": {},
    }
    held = []
    for s in args.heldout:
        d = collect(s, n)
        held.append(d)
        result["heldout"][s.name] = score(d, w, sigma_c, width_fill)
    pooled = {k: np.concatenate([h[k] for h in held]) for k in held[0]}
    result["heldout"]["pooled"] = score(pooled, w, sigma_c, width_fill)
    m = result["heldout"]["pooled"]["margin_over_constant_fitted"]
    result["replaces_analytic_ellipse"] = bool(m["ci95_by_person"][0] > 0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["coefficients_log_sigma"], indent=1))
    for k, v in result["heldout"].items():
        print(
            k,
            {
                x: (round(y, 3) if isinstance(y, float) else y)
                for x, y in v.items()
                if not isinstance(y, dict)
            },
        )
        print(
            "   margin fitted",
            v["margin_over_constant_fitted"],
            "ellipse",
            v["margin_over_constant_analytic_ellipse"],
        )
    print("replaces_analytic_ellipse:", result["replaces_analytic_ellipse"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
