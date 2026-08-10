"""Can this dataset meaningfully score this capability?

Day 9 asked that question about depth on v3-indoor and the answer was no: the
model ordered pixels backwards against ground truth (rank correlation −0.5924)
while a scale-and-shift fit produced a respectable-looking AbsRel of 0.1538.
The number existed, was reproducible, and described the fit rather than the
model.

That was caught by one bespoke check. This module makes it a mechanism, because
the failure is not specific to depth — it is what happens whenever a fixture is
asked to score a capability it cannot express.

The partition it encodes
------------------------
**Geometry-derived** capabilities — motion, occlusion topology, coverage — are
synthetically evaluable. Where an object is, whether it moved, and what hides
it are all exactly computable from scene description, and an analytic renderer
knows them perfectly.

**Appearance-learned** capabilities — depth, semantics, re-identification — are
not. Those models run on shading gradients, texture statistics and object
recognition, and analytic primitives delete precisely those cues. A flat matte
wall at 15 m is not an easy depth target; it is an absent one.

A gate is not a quality score. It answers whether a measurement taken here
would mean anything, and it is checked *before* the metric is computed so that
a refusal is reported instead of a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

FrameArray = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class GateResult:
    """Whether a capability can be scored on a dataset, and the evidence."""

    capability: str
    dataset: str
    passed: bool
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "dataset": self.dataset,
            "passed": self.passed,
            "reason": self.reason,
            "evidence": self.evidence,
        }


# --- thresholds, each with the measurement that set it ---------------------

MIN_DEPTH_RANK_CORRELATION = 0.30
"""Below this, a depth metric describes its alignment fit and not the model.

v3-indoor measured −0.5924. Rank correlation is used because it is the one
thing a scale-and-shift alignment cannot rescue."""

MIN_DEPTH_DYNAMIC_RANGE = 3.0
"""Predicted disparity spread, below which the model is near-constant.

Real footage through the same weights spread 4.44; v3's analytic renders
spread 2.23. A near-constant prediction fits whatever depth dominates the
frame and scores well there."""

MIN_TEXTURE_ENERGY = 12.0
"""Mean local gradient magnitude, below which there is no appearance to learn.

Appearance models key on texture and shading gradients. Procedurally-noised
flat surfaces have a little; real materials have a great deal more."""

MIN_MATERIAL_DIVERSITY = 24.0
"""Std of per-region mean intensity: how many distinguishable surfaces exist.

A scene rendered from four flat greys cannot exercise re-identification or
same-object retrieval however many agents walk through it."""

MAX_BAND_CONCENTRATION = 0.90
"""Share of pixels one distance band may hold before scores stop being pooled.

v3: 95% of pixels sat at 8 m+, so the aggregate reported that band and the
3-8 m band containing every agent scored delta<1.25 of exactly 0.0000 without
moving the headline."""


MIN_TRACKABLE_CORNER_DENSITY = 1.0e-3
"""Shi-Tomasi corners per pixel, below which point tracking is starving.

Point tracking sits between geometry and appearance: it estimates a geometric
quantity (2D location over time) but the estimator keys on local texture to
find something to track. Analytic primitives shaded with a little procedural
noise give the corner detector almost nothing, and a tracker fed nothing
produces a track anyway — it just extrapolates. The floor is deliberately
loose: on v2 real corridor footage a Shi-Tomasi detector at
``maxCorners=1024, qualityLevel=0.01, minDistance=3`` returns hundreds of
strong corners per QVGA frame (density ~5e-3). Anything an order of
magnitude below that starves the matcher, and the tracker's numbers describe
its extrapolator rather than its accuracy."""


# --- gates -----------------------------------------------------------------


def gate_motion_geometry(
    frames: FrameArray, **_: Any
) -> tuple[bool, str, dict[str, Any]]:
    """Motion is geometry, so a synthetic renderer can score it exactly.

    The positive case in this registry, and it is here on purpose: a validity
    mechanism that only ever refuses is indistinguishable from a broken one.
    Ground truth for "did this move" comes from the scene description rather
    than from anything the model must infer, so an analytic fixture is not a
    compromise here — it is better than real footage, which needs annotation.

    Also registered as ``"state_estimation"`` (Day 20): position and
    velocity are the same kind of exactly-known scene-description quantity
    as "did this move" — v3-indoor's ``agent_xyz`` is analytic GT, not an
    annotation — so the same reasoning that authorizes scoring motion here
    authorizes scoring the state estimator's accuracy here too.
    """
    if frames.ndim != 4 or frames.shape[0] < 2:
        return False, "need at least two frames to observe motion at all", {}
    return (
        True,
        "motion is derived from scene geometry, which an analytic renderer "
        "knows exactly; no appearance cue is required",
        {"frames": int(frames.shape[0])},
    )


def gate_depth(
    frames: FrameArray,
    gt_depth: npt.NDArray[np.float64] | None = None,
    predictor: Callable[[FrameArray], npt.NDArray[np.float64]] | None = None,
    **_: Any,
) -> tuple[bool, str, dict[str, Any]]:
    """Depth is appearance-learned, so the fixture has to be checked.

    Three questions, all of which v3-indoor failed or would have hidden:

    1. Does the model *order* pixels by depth the way the world does? This is
       the only check a scale-and-shift alignment cannot launder.
    2. Does it produce any dynamic range, or is it near-constant?
    3. How are pixels distributed across distance bands? A band holding
       almost everything sets the aggregate on its own.
    """
    if gt_depth is None or predictor is None:
        return (
            False,
            "depth validity needs both ground-truth depth and a predictor; "
            "without them the gate would be asserting rather than measuring",
            {},
        )

    from src.data.depth_eval import DISTANCE_BUCKETS, rank_correlation

    correlations, ranges = [], []
    for index in range(min(4, frames.shape[0])):
        disparity = np.asarray(predictor(frames[index]), dtype=np.float64)
        correlations.append(rank_correlation(disparity, gt_depth[index]))
        finite = disparity[np.isfinite(disparity)]
        ranges.append(float(finite.max() - finite.min()) if finite.size else 0.0)

    correlation = float(np.nanmean(correlations))
    spread = float(np.mean(ranges))

    truth = gt_depth[: min(4, gt_depth.shape[0])]
    total = int(np.isfinite(truth).sum())
    populations = {}
    for name, low, high in DISTANCE_BUCKETS:
        count = int(((truth >= low) & (truth < high)).sum())
        populations[name] = round(count / total, 4) if total else 0.0
    concentration = max(populations.values()) if populations else 1.0

    evidence = {
        "rank_correlation": round(correlation, 4),
        "rank_correlation_floor": MIN_DEPTH_RANK_CORRELATION,
        "dynamic_range": round(spread, 4),
        "dynamic_range_floor": MIN_DEPTH_DYNAMIC_RANGE,
        "band_populations": populations,
        "max_band_concentration": round(concentration, 4),
    }

    failures = []
    if not np.isfinite(correlation) or correlation < MIN_DEPTH_RANK_CORRELATION:
        failures.append(
            f"rank correlation {correlation:+.4f} is below "
            f"{MIN_DEPTH_RANK_CORRELATION:+.2f} — the model does not order "
            "these pixels by depth, so any aligned metric describes the fit"
        )
    if spread < MIN_DEPTH_DYNAMIC_RANGE:
        failures.append(
            f"predicted spread {spread:.2f} is below {MIN_DEPTH_DYNAMIC_RANGE} "
            "— the prediction is near-constant and will fit whatever depth "
            "dominates the frame"
        )
    if concentration > MAX_BAND_CONCENTRATION:
        failures.append(
            f"{concentration:.0%} of pixels sit in one distance band, so a "
            "pooled score reports that band and hides the others"
        )

    if failures:
        return False, "; ".join(failures), evidence
    return (
        True,
        "the model orders pixels by depth, produces real dynamic range, and "
        "the distance bands are populated well enough to pool",
        evidence,
    )


def gate_appearance_semantics(
    frames: FrameArray, **_: Any
) -> tuple[bool, str, dict[str, Any]]:
    """Appearance-learned capabilities need appearance to be present.

    Measures what the fixture actually contains rather than assuming: local
    gradient energy (is there texture?) and per-region intensity spread (are
    there distinguishable materials?). Analytic primitives shaded with a little
    procedural noise have almost neither.
    """
    if frames.ndim != 4:
        return False, "expected [T, H, W, C] frames", {}

    sample = frames[: min(4, frames.shape[0])].astype(np.float64)
    grey = sample.mean(axis=3)

    gradient_y = np.abs(np.diff(grey, axis=1)).mean()
    gradient_x = np.abs(np.diff(grey, axis=2)).mean()
    texture_energy = float((gradient_x + gradient_y) / 2.0)

    # Coarse tiles stand in for "surfaces": if their mean intensities barely
    # differ, the scene has almost one material in it.
    height, width = grey.shape[1], grey.shape[2]
    tiles = grey[:, : height // 8 * 8, : width // 8 * 8].reshape(
        grey.shape[0], 8, height // 8, 8, width // 8
    )
    material_diversity = float(np.mean(np.std(tiles.mean(axis=(2, 4)), axis=(1, 2))))

    evidence = {
        "texture_energy": round(texture_energy, 3),
        "texture_energy_floor": MIN_TEXTURE_ENERGY,
        "material_diversity": round(material_diversity, 3),
        "material_diversity_floor": MIN_MATERIAL_DIVERSITY,
    }

    failures = []
    if texture_energy < MIN_TEXTURE_ENERGY:
        failures.append(
            f"texture energy {texture_energy:.2f} is below {MIN_TEXTURE_ENERGY} "
            "— there is almost no local gradient for an appearance model to key on"
        )
    if material_diversity < MIN_MATERIAL_DIVERSITY:
        failures.append(
            f"material diversity {material_diversity:.2f} is below "
            f"{MIN_MATERIAL_DIVERSITY} — too few distinguishable surfaces to "
            "exercise identity or retrieval"
        )

    if failures:
        return False, "; ".join(failures), evidence
    return True, "the frames carry texture and distinguishable materials", evidence


def gate_point_tracking(
    frames: FrameArray, **_: Any
) -> tuple[bool, str, dict[str, Any]]:
    """Point tracking needs something to track.

    Sits between geometry and appearance. The output is geometric (2D
    location, per frame, per query point), but the estimator finds and
    matches local texture patches — corners, edges, gradient junctions. A
    fixture with no local structure gives the matcher no lock, and the
    tracker returns an extrapolation while the metric averages it into a
    number that looks like accuracy.

    The measurement here is Shi-Tomasi corner density, in corners per pixel,
    averaged over the first few frames. A single number, with the same
    "measure before scoring" stance every other gate in this file takes: run
    it and refuse rather than compute an accuracy number on a starved
    matcher and hide the fact by pooling.
    """
    if frames.ndim != 4:
        return False, "expected [T, H, W, C] frames", {}
    try:
        import cv2
    except ImportError:
        return (
            False,
            "opencv is required to measure corner density and is not "
            "available; refuse rather than assume the fixture is trackable",
            {},
        )

    sample = frames[: min(4, frames.shape[0])]
    densities: list[float] = []
    for index in range(sample.shape[0]):
        frame = sample[index]
        grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        corners = cv2.goodFeaturesToTrack(
            grey, maxCorners=1024, qualityLevel=0.01, minDistance=3
        )
        found = 0 if corners is None else int(corners.shape[0])
        densities.append(found / float(grey.shape[0] * grey.shape[1]))

    density = float(np.mean(densities)) if densities else 0.0
    evidence = {
        "corner_density_per_pixel": round(density, 6),
        "corner_density_floor": MIN_TRACKABLE_CORNER_DENSITY,
        "frames_sampled": len(densities),
    }
    if density < MIN_TRACKABLE_CORNER_DENSITY:
        return (
            False,
            f"corner density {density:.3e} is below "
            f"{MIN_TRACKABLE_CORNER_DENSITY:.0e} corners/pixel — the matcher "
            "has almost nothing to lock onto, and any tracking accuracy "
            "reported here would describe extrapolation, not correspondence",
            evidence,
        )
    return (
        True,
        "the frames carry enough local structure for a corner-based tracker "
        "to have something to lock onto",
        evidence,
    )


GATES: dict[str, Callable[..., tuple[bool, str, dict[str, Any]]]] = {
    "motion_geometry": gate_motion_geometry,
    "state_estimation": gate_motion_geometry,
    "depth": gate_depth,
    "appearance_semantics": gate_appearance_semantics,
    "point_tracking": gate_point_tracking,
}


def evaluate(capability: str, dataset: str, **inputs: Any) -> GateResult:
    """Run one gate. Unknown capabilities refuse rather than default to pass."""
    gate = GATES.get(capability)
    if gate is None:
        return GateResult(
            capability=capability,
            dataset=dataset,
            passed=False,
            reason=(
                f"no validity gate is registered for {capability!r}. A "
                "capability with no gate has not been shown to be measurable "
                "here, and defaulting to pass is how an unmeasurable one gets "
                "a number."
            ),
        )
    passed, reason, evidence = gate(**inputs)
    return GateResult(
        capability=capability,
        dataset=dataset,
        passed=passed,
        reason=reason,
        evidence=evidence,
    )
