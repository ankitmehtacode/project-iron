"""Cross-camera association by geometry, and fusion of associated views
(Day 46, Objective 3).

No appearance features and no re-ID model: two tracklets on different cameras
are the same person only if their floor rays agree.

Each tracklet point is a floor ray from the camera's floor position through
the box's foot point (the foot anchor kept by rule R3: the range interval was
not adopted), with the stated per-detection uncertainty of rule R4, the
fitted error model (:func:`src.pipeline.anchor.fitted_sigma_m`), which is an
isotropic floor sigma. A point's *region* is therefore the stated 95% disk
around the foot point, radius ``REGION_K * sigma``.

- **Compatible at a step**: the two regions intersect, ``|p_a - p_b| <=
  REGION_K * (sigma_a + sigma_b)``. The gate is the stated region, not a
  tuned number. (A first version gated the range residuals where the two
  rays meet; near-parallel views made that meeting point unstable and a
  few heavy-tailed steps vetoed true pairs: Warehouse_022 oracle, recall
  13 of 67.)
- **Evidence over the overlap**: per shared step, the log-likelihood ratio
  of "same person" (``p_a - p_b`` Gaussian with variance ``sigma_a^2 +
  sigma_b^2`` per axis) against "unrelated" (uniform over the gate disk),
  with the distance clipped at the gate radius, so one step outside the
  stated region costs a bounded amount (R4's model is heavy-tailed, Day 44);
  summed over every step the two tracklets share.
- **Solving**: for each tracklet and each other camera, the candidates are
  every tracklet there compatible on at least one shared step, plus "no
  partner" (log-likelihood ratio 0). :func:`src.estimator.joint.
  resolve_data_association` (budget = all candidates) returns Decisive or
  Ambiguous. A pair is accepted only if each tracklet is the other's
  Decisive winner. An Ambiguous verdict is recorded and never yields a fused
  identity.
- **Fused identity**: the connected components of accepted pairs.
- **Fused position**: at a step where members on two or more cameras are
  present, the least-squares intersection of their rays; with one camera,
  that camera's foot point.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from src.estimator.joint import (
    Ambiguous,
    AssociationBudgetConfig,
    AssociationCandidate,
    Decisive,
    resolve_data_association,
)
from src.model.hypothesis import HypothesisStore

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

GATE_CHI2_2DOF_95 = 5.991464547107979
"""chi-square(2) 0.95 quantile: the stated 95% region of an isotropic 2-D
Gaussian is a disk of radius sqrt(5.991) sigma."""
REGION_K = math.sqrt(GATE_CHI2_2DOF_95)
NO_PARTNER = "no_partner"


@dataclass(frozen=True)
class Tracklet:
    """One per-camera track, as floor rays."""

    key: str
    camera_id: str
    camera_xy: FloatArray
    frames: IntArray
    """Strictly increasing."""
    foot_xy: FloatArray
    sigma_m: FloatArray

    def __post_init__(self) -> None:
        n = self.frames.shape[0]
        if self.foot_xy.shape != (n, 2) or self.sigma_m.shape != (n,):
            raise ValueError(f"{self.key}: frames, foot_xy, sigma_m disagree")
        if n and np.any(np.diff(self.frames) <= 0):
            raise ValueError(f"{self.key}: frames must be strictly increasing")
        if np.any(~(self.sigma_m > 0)):
            raise ValueError(f"{self.key}: sigma must be positive")


@dataclass(frozen=True)
class PairEvidence:
    a: str
    b: str
    overlap: int
    compatible: int
    llr: float


def pair_evidence(a: Tracklet, b: Tracklet) -> PairEvidence:
    if (
        not a.frames.size
        or not b.frames.size
        or a.frames[-1] < b.frames[0]
        or b.frames[-1] < a.frames[0]
    ):
        return PairEvidence(a.key, b.key, 0, 0, 0.0)
    shared, ia, ib = np.intersect1d(a.frames, b.frames, return_indices=True)
    if shared.size == 0:
        return PairEvidence(a.key, b.key, 0, 0, 0.0)
    d = np.linalg.norm(a.foot_xy[ia] - b.foot_xy[ib], axis=1)
    sa, sb = a.sigma_m[ia], b.sigma_m[ib]
    gate = REGION_K * (sa + sb)
    compatible = d <= gate
    var = sa**2 + sb**2
    dc = np.minimum(d, gate)
    same = -0.5 * dc**2 / var - np.log(2 * math.pi * var)
    unrelated = -np.log(math.pi * gate**2)
    llr = float((same - unrelated).sum())
    return PairEvidence(a.key, b.key, int(shared.size), int(compatible.sum()), llr)


@dataclass
class Association:
    accepted: list[tuple[str, str]] = field(default_factory=list)
    verdicts: list[dict[str, object]] = field(default_factory=list)
    """One per (tracklet, other camera) question: kind, winner, margin."""
    evidence: dict[tuple[str, str], PairEvidence] = field(default_factory=dict)
    components: dict[str, str] = field(default_factory=dict)
    """tracklet key -> fused identity id (a component of accepted pairs);
    a tracklet with no accepted pair is its own component."""


def associate(tracklets: Sequence[Tracklet]) -> Association:
    out = Association()
    by_camera: dict[str, list[Tracklet]] = {}
    for t in tracklets:
        by_camera.setdefault(t.camera_id, []).append(t)
    for a in tracklets:
        for cam, others in by_camera.items():
            if cam == a.camera_id:
                continue
            for b in others:
                if (b.key, a.key) in out.evidence:
                    ev = out.evidence[(b.key, a.key)]
                    out.evidence[(a.key, b.key)] = PairEvidence(
                        a.key, b.key, ev.overlap, ev.compatible, ev.llr
                    )
                else:
                    out.evidence[(a.key, b.key)] = pair_evidence(a, b)
    winner: dict[tuple[str, str], str] = {}
    for a in tracklets:
        for cam, others in by_camera.items():
            if cam == a.camera_id:
                continue
            cands = [
                AssociationCandidate(
                    NO_PARTNER, f"{a.key} has no partner on {cam}", 0.0
                )
            ]
            for b in others:
                ev = out.evidence[(a.key, b.key)]
                if ev.compatible > 0:
                    cands.append(
                        AssociationCandidate(b.key, f"{a.key} is {b.key}", ev.llr)
                    )
            if len(cands) == 1:
                continue
            res = resolve_data_association(
                HypothesisStore(),
                f"{a.key}->{cam}",
                cands,
                hard_violation_check=lambda _c: None,
                budget_config=AssociationBudgetConfig(len(cands)),
            )
            v = res.verdict
            if isinstance(v, Decisive):
                win = v.winner.id.split("::", 1)[1]
                winner[(a.key, cam)] = win
                out.verdicts.append(
                    {
                        "tracklet": a.key,
                        "camera": cam,
                        "kind": "decisive",
                        "winner": win,
                        "margin": v.margin,
                        "candidates": len(cands),
                    }
                )
            elif isinstance(v, Ambiguous):
                out.verdicts.append(
                    {
                        "tracklet": a.key,
                        "camera": cam,
                        "kind": "ambiguous",
                        "winner": None,
                        "margin": v.margin,
                        "candidates": len(cands),
                    }
                )
    camera_of = {t.key: t.camera_id for t in tracklets}
    for (a_key, cam), b_key in winner.items():
        if b_key == NO_PARTNER or a_key > b_key:
            continue
        if winner.get((b_key, camera_of[a_key])) == a_key:
            out.accepted.append((a_key, b_key))
    parent = {t.key: t.key for t in tracklets}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for a_key, b_key in out.accepted:
        ra, rb = find(a_key), find(b_key)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    out.components = {t.key: f"fused/{find(t.key)}" for t in tracklets}
    return out


def ls_intersection(centres: FloatArray, points: FloatArray) -> FloatArray | None:
    """Least-squares point closest to the lines ``centres[k] -> points[k]``
    (two or more). ``None`` when the lines are (numerically) parallel."""
    a = np.zeros((2, 2))
    rhs = np.zeros(2)
    for c, p in zip(centres, points):
        d = p - c
        norm = float(np.linalg.norm(d))
        if not norm > 0:
            continue
        d = d / norm
        proj = np.eye(2) - np.outer(d, d)
        a += proj
        rhs += proj @ c
    if np.linalg.eigvalsh(a)[0] < 1e-9:
        return None
    out: FloatArray = np.linalg.solve(a, rhs)
    return out
