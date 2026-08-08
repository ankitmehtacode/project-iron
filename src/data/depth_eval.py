"""First evaluation of the depth stage against exact ground truth.

Depth-Anything-V2 emits **relative inverse depth**: unknown scale *and* unknown
shift, per frame. No constant converts it to metres. Everything here follows
from that one fact.

Alignment is explicit and reported
----------------------------------
Metrics are computed twice. **Unaligned** takes the model's output at face
value as metres — which is exactly what the code did before the ``DepthField``
type existed, and what any consumer does who reads a float and assumes a unit.
**Aligned** fits a per-frame scale and shift against ground truth first. The
gap between the two is the honest size of the claim "we produce metric depth",
and reporting only the aligned number would hide the entire problem.

The fit is least-squares in *disparity* space with Huber re-weighting, matching
how DA-V2 and MiDaS are conventionally evaluated: ``a * pred + b ~ 1 / gt``.
Fitting in depth space instead would let the far field, where inverse depth is
compressed, dominate the residual.

What synthetic ground truth is and is not
-----------------------------------------
These scenes are analytic primitives: matte, untextured, perfectly Lambertian,
with no glass, no monitors, no polished floor and no specular anything. That is
the friendliest possible input to a monocular depth model. **Every number here
is a ceiling, not an estimate.** Real office footage will be worse and the
report says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# Distance buckets, metres. One aggregate number hides far-field failure, which
# is the regime the product actually cares about — a person at the end of a
# corridor is the case a coverage advisor has to reason about.
DISTANCE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0-3m", 0.0, 3.0),
    ("3-8m", 3.0, 8.0),
    ("8m+", 8.0, float("inf")),
)

HUBER_DELTA = 1.0
"""Huber cutoff in units of the robust residual scale.

Present because a least-squares fit on a scene containing a wall at 15 m and a
person at 3 m is dominated by whichever occupies more pixels, not by whichever
matters. Re-weighting keeps a large flat background from setting the scale for
everything in front of it."""


@dataclass(frozen=True)
class Alignment:
    """Per-frame scale and shift taking predicted disparity to GT disparity."""

    scale: float
    shift: float
    inliers: int

    def apply_to_depth(self, disparity: FloatArray) -> FloatArray:
        """Aligned metric depth, in metres, from predicted disparity.

        Non-positive aligned disparity is returned as ``inf`` rather than a
        negative depth: a point the model places behind the camera is missing
        information, not a measurement, and silently taking its reciprocal
        would put a negative metre value into the metrics.
        """
        aligned = self.scale * disparity + self.shift
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(aligned > 1e-8, 1.0 / aligned, np.inf)
        return depth


def fit_alignment(
    disparity: FloatArray, gt_depth: FloatArray, valid: npt.NDArray[np.bool_]
) -> Alignment:
    """Least-squares scale/shift in disparity space, Huber re-weighted."""
    usable = valid & np.isfinite(disparity) & (gt_depth > 1e-6)
    if usable.sum() < 16:
        return Alignment(scale=float("nan"), shift=float("nan"), inliers=0)

    x = disparity[usable].astype(np.float64)
    y = (1.0 / gt_depth[usable]).astype(np.float64)

    weights = np.ones_like(x)
    scale, shift = 1.0, 0.0
    for _ in range(8):
        design = np.stack([x, np.ones_like(x)], axis=1) * weights[:, None]
        target = y * weights
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        scale, shift = float(solution[0]), float(solution[1])

        residual = np.abs(scale * x + shift - y)
        sigma = np.median(residual) + 1e-12
        normalised = residual / (HUBER_DELTA * sigma * 1.4826)
        weights = np.where(normalised <= 1.0, 1.0, 1.0 / np.sqrt(normalised))

    return Alignment(scale=scale, shift=shift, inliers=int(usable.sum()))


def depth_metrics(
    predicted_m: FloatArray, gt_m: FloatArray, valid: npt.NDArray[np.bool_]
) -> dict[str, float]:
    """Standard monocular depth metrics. Protocol, not invention.

    AbsRel, RMSE, delta<1.25 and scale-invariant log error, as defined in the
    depth literature. ``nan`` where a bucket has no valid pixels — never 0,
    which would read as a perfect score.
    """
    usable = (
        valid
        & np.isfinite(predicted_m)
        & np.isfinite(gt_m)
        & (predicted_m > 1e-6)
        & (gt_m > 1e-6)
    )
    count = int(usable.sum())
    if count == 0:
        return {
            "absrel": float("nan"),
            "rmse_m": float("nan"),
            "delta_1_25": float("nan"),
            "silog": float("nan"),
            "pixels": 0,
        }

    pred = predicted_m[usable].astype(np.float64)
    truth = gt_m[usable].astype(np.float64)

    ratio = np.maximum(pred / truth, truth / pred)
    log_diff = np.log(pred) - np.log(truth)

    return {
        "absrel": float(np.mean(np.abs(pred - truth) / truth)),
        "rmse_m": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "delta_1_25": float(np.mean(ratio < 1.25)),
        "silog": float(
            np.sqrt(np.mean(log_diff**2) - np.mean(log_diff) ** 2 + 1e-12)
        ),
        "pixels": count,
    }


def static_point_z_std(
    depths_m: list[FloatArray], gt_depth: FloatArray, instances: npt.NDArray[np.integer[Any]]
) -> dict[str, float]:
    """Temporal flicker on points ground truth proves are static.

    Almost nobody measures this and it decides whether any downstream velocity
    estimate is usable: a static point whose estimated Z wanders by half a
    metre between frames produces motion that is not there.

    A point qualifies only if GT depth is identical across every frame and no
    agent ever covers it — so any variation in the estimate is the estimator,
    with nothing in the world to explain it.
    """
    stack = np.stack(depths_m)
    agent_free = np.all(instances < 100, axis=0)
    gt_static = np.all(np.abs(gt_depth - gt_depth[0]) < 1e-4, axis=0)
    qualifying = agent_free & gt_static & np.all(np.isfinite(stack), axis=0)

    if qualifying.sum() < 32:
        return {"points": 0, "z_std_m": float("nan"), "z_std_p95_m": float("nan")}

    per_point_std = np.std(stack[:, qualifying], axis=0)
    return {
        "points": int(qualifying.sum()),
        "z_std_m": float(np.mean(per_point_std)),
        "z_std_p95_m": float(np.percentile(per_point_std, 95)),
    }


MIN_RANK_CORRELATION = 0.30
"""Rank correlation with GT disparity below which metrics are not reported.

A scale-and-shift fit can make almost any prediction look reasonable against
the *dominant* depth in a scene. If a model predicts a near-constant value and
the scene is mostly a wall at 15 m, the fit lands on 15 m, the background
scores beautifully, and the aggregate AbsRel looks respectable while the model
knows nothing.

The check that survives alignment is ordering: does the model rank pixels by
depth the way the world does? Below this floor it does not, and any metric
computed on top is a property of the fit rather than of the model.
"""


def rank_correlation(disparity: FloatArray, gt_depth: FloatArray) -> float:
    """Spearman correlation between predicted and true disparity.

    Alignment-invariant by construction, which is the point: it cannot be
    rescued by a good scale/shift fit.
    """
    usable = np.isfinite(disparity) & (gt_depth > 1e-6)
    if usable.sum() < 64:
        return float("nan")
    predicted = disparity[usable].ravel()
    truth = (1.0 / gt_depth[usable]).ravel()
    if predicted.size > 200_000:
        stride = predicted.size // 200_000 + 1
        predicted, truth = predicted[::stride], truth[::stride]
    pred_rank = np.argsort(np.argsort(predicted))
    truth_rank = np.argsort(np.argsort(truth))
    return float(np.corrcoef(pred_rank, truth_rank)[0, 1])


@dataclass
class ClipDepthResult:
    clip_id: str
    frames_scored: int
    rank_correlation: float = float("nan")
    aligned: dict[str, float] = field(default_factory=dict)
    unaligned: dict[str, float] = field(default_factory=dict)
    by_bucket: dict[str, dict[str, float]] = field(default_factory=dict)
    flicker: dict[str, float] = field(default_factory=dict)
    alignment_drift: dict[str, float] = field(default_factory=dict)


def score_clip(clip_path: Path, wrapper: Any, frame_stride: int) -> ClipDepthResult:
    """Score one clip's depth against its exact GT."""
    with np.load(clip_path) as data:
        rgb = np.asarray(data["rgb"])
        gt_depth = np.asarray(data["depth_m"], dtype=np.float64)
        instances = np.asarray(data["instances"])

    indices = list(range(0, rgb.shape[0], frame_stride))
    aligned_frames: list[FloatArray] = []
    alignments: list[Alignment] = []
    gt_frames: list[FloatArray] = []
    raw_frames: list[FloatArray] = []

    for index in indices:
        prediction = wrapper.predict({"image": rgb[index]})["depth"]
        disparity = np.asarray(prediction.data, dtype=np.float64)
        valid = np.asarray(prediction.valid_mask)

        alignment = fit_alignment(disparity, gt_depth[index], valid)
        alignments.append(alignment)
        aligned_frames.append(alignment.apply_to_depth(disparity))
        gt_frames.append(gt_depth[index])
        raw_frames.append(disparity)

    result = ClipDepthResult(clip_id=clip_path.stem, frames_scored=len(indices))
    result.rank_correlation = float(
        np.mean(
            [rank_correlation(raw, truth) for raw, truth in zip(raw_frames, gt_frames)]
        )
    )

    all_aligned = np.concatenate([f.ravel() for f in aligned_frames])
    all_gt = np.concatenate([f.ravel() for f in gt_frames])
    all_raw = np.concatenate([f.ravel() for f in raw_frames])
    finite = np.isfinite(all_aligned)

    result.aligned = depth_metrics(all_aligned, all_gt, finite)
    # The model's own output read as metres — the pre-DepthField failure mode,
    # measured rather than described.
    result.unaligned = depth_metrics(all_raw, all_gt, np.isfinite(all_raw))

    for name, low, high in DISTANCE_BUCKETS:
        in_bucket = finite & (all_gt >= low) & (all_gt < high)
        result.by_bucket[name] = depth_metrics(all_aligned, all_gt, in_bucket)

    result.flicker = static_point_z_std(
        aligned_frames, gt_depth[indices], instances[indices]
    )

    scales = np.array([a.scale for a in alignments], dtype=np.float64)
    shifts = np.array([a.shift for a in alignments], dtype=np.float64)
    good = np.isfinite(scales)
    result.alignment_drift = {
        "scale_mean": float(np.mean(scales[good])) if good.any() else float("nan"),
        # Relative spread, because absolute scale is arbitrary. If this is not
        # near zero the per-frame fit is chasing the image, which is the
        # scale-flicker problem the audit predicted.
        "scale_cv": (
            float(np.std(scales[good]) / (abs(np.mean(scales[good])) + 1e-12))
            if good.any()
            else float("nan")
        ),
        "shift_mean": float(np.mean(shifts[good])) if good.any() else float("nan"),
        "shift_std": float(np.std(shifts[good])) if good.any() else float("nan"),
    }
    return result


def aggregate(results: list[ClipDepthResult]) -> dict[str, Any]:
    """Pool clip results. Means over clips, so one long clip cannot dominate."""

    def pooled(select: Callable[[ClipDepthResult], dict[str, float]]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key in ("absrel", "rmse_m", "delta_1_25", "silog"):
            values = [
                select(r)[key]
                for r in results
                if select(r) and np.isfinite(select(r).get(key, float("nan")))
            ]
            out[key] = float(np.mean(values)) if values else float("nan")
        return out

    def _bucket_selector(
        name: str,
    ) -> Callable[[ClipDepthResult], dict[str, float]]:
        # A closure over a plain lambda default-arg (`lambda r, n=name: ...`)
        # is a common way to avoid the late-binding loop-variable trap, but
        # mypy cannot infer the lambda's type against the Callable[[X], Y] it
        # is passed to here (misc: "Cannot infer type of lambda"). A named
        # function with an explicit return type sidesteps that inference gap
        # with no change in behaviour.
        return lambda r: r.by_bucket.get(name, {})

    summary: dict[str, Any] = {
        "clips": len(results),
        "frames_scored": sum(r.frames_scored for r in results),
        "aligned": pooled(lambda r: r.aligned),
        "unaligned": pooled(lambda r: r.unaligned),
        "by_bucket": {
            name: pooled(_bucket_selector(name)) for name, _, _ in DISTANCE_BUCKETS
        },
    }

    flicker = [r.flicker["z_std_m"] for r in results if r.flicker.get("points")]
    summary["static_point_z_std_m"] = (
        float(np.mean(flicker)) if flicker else float("nan")
    )
    ranks = [r.rank_correlation for r in results if np.isfinite(r.rank_correlation)]
    summary["rank_correlation"] = float(np.mean(ranks)) if ranks else float("nan")
    # The gate. Reported before any metric, because it decides whether the
    # metrics below mean anything at all.
    summary["valid_for_depth_scoring"] = bool(
        ranks and np.mean(ranks) >= MIN_RANK_CORRELATION
    )
    summary["min_rank_correlation"] = MIN_RANK_CORRELATION

    cvs = [
        r.alignment_drift["scale_cv"]
        for r in results
        if np.isfinite(r.alignment_drift.get("scale_cv", float("nan")))
    ]
    summary["alignment_scale_cv"] = float(np.mean(cvs)) if cvs else float("nan")
    return summary
