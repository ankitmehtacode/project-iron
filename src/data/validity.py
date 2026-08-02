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


# --- gates -----------------------------------------------------------------


def gate_motion_geometry(frames: np.ndarray, **_: Any) -> tuple[bool, str, dict]:
    """Motion is geometry, so a synthetic renderer can score it exactly.

    The positive case in this registry, and it is here on purpose: a validity
    mechanism that only ever refuses is indistinguishable from a broken one.
    Ground truth for "did this move" comes from the scene description rather
    than from anything the model must infer, so an analytic fixture is not a
    compromise here — it is better than real footage, which needs annotation.
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
    frames: np.ndarray,
    gt_depth: np.ndarray | None = None,
    predictor: Callable[[np.ndarray], np.ndarray] | None = None,
    **_: Any,
) -> tuple[bool, str, dict]:
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


def gate_appearance_semantics(frames: np.ndarray, **_: Any) -> tuple[bool, str, dict]:
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


GATES: dict[str, Callable[..., tuple[bool, str, dict]]] = {
    "motion_geometry": gate_motion_geometry,
    "depth": gate_depth,
    "appearance_semantics": gate_appearance_semantics,
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
