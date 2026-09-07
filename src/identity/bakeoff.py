"""Backbone bake-off harness (Day 36, Objective 4) — BUILT, not run for real.

Four capability probes any :class:`~src.identity.backbone.FrozenBackbone`
can be scored on: same-object retrieval mAP (cross-boundary queries only,
position-only baseline — Day 12's rule and the Day-11 repair, generalized
from ``scripts/eval_semantics.py``), temporal embedding stability,
patch-boundary discontinuity, and clothing-change robustness (CHIRLA's
specific contribution, once fetchable — see
``docs/chirla_verification_checklist.md``). Every probe goes through
:mod:`src.eval.baselines`'s registry (``compute_baselines``/``margin``) —
the SAME registered baseline computers ``scripts/eval_semantics.py`` uses
for ``semantics.mAP``/``semantics.temporal_cosine``/
``semantics.patch_boundary_l2``, not copies of them (see that module's
``_register_defaults`` for the ``identity.bakeoff.*`` registrations).

Real data, not run today
-------------------------
:func:`open_bakeoff_eval_set` is wired to accept any lane-R dataset through
``src.data.registry.DatasetRegistry.open_for_eval`` — the eval-only gate,
never ``open_for_training``, matching ADR-established lane rules (lane R:
model selection and eval only, never trained on). It is not exercised
against real data today: CHIRLA has no ``license_snapshot`` (Objective 1),
and no other registered dataset fills the ``multi_camera``/
``reappearance``/clothing-change cells this harness exists to probe.

Every function below accepts ``self_test: bool`` and stamps
:data:`~src.identity.selftest.SELF_TEST_LABEL` onto its result when set —
this harness does not select a backbone, does not run against real data,
and a self-test result must never be read as a bake-off answer. The
decision ledger's "Dense-semantics encoder: open" line stays open.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.data.registry import DatasetEntry, DatasetRegistry
from src.eval.baselines import Baseline, compute_baselines
from src.eval.baselines import margin as _margin
from src.eval.retrieval import average_precision
from src.identity.backbone import FrozenBackbone, extract_features_no_grad
from src.identity.selftest import SELF_TEST_LABEL

# Deliberately NO `IRON_TRAINING_PATH = True` here (contrast
# src/identity/lane_gate.py): this module is an eval path. Its absence is
# what lets DatasetRegistry.open_for_eval's stack walk refuse a lane-R
# dataset if this module is ever reached from something that IS marked
# training — see open_bakeoff_eval_set below.


@dataclass(frozen=True)
class BakeoffProbeResult:
    metric_name: str
    value: float
    baselines: tuple[Baseline, ...]
    margin: float
    self_test_label: str | None = None

    def as_evidence(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": round(self.value, 6) if np.isfinite(self.value) else None,
            "baselines": [b.as_dict() for b in self.baselines],
            "margin": round(self.margin, 6) if np.isfinite(self.margin) else None,
            "self_test_label": self.self_test_label,
        }

    def render(self) -> str:
        flag_worthy = [
            b for b in self.baselines if b.flag_worthy and np.isfinite(b.value)
        ]
        strongest_val = max((b.value for b in flag_worthy), default=float("nan"))
        strongest_name = next(
            (b.name for b in flag_worthy if b.value == strongest_val), "n/a"
        )
        prefix = f"[{self.self_test_label}] " if self.self_test_label else ""
        return (
            f"{prefix}{self.metric_name} = {self.value:.4f} "
            f"(baseline {strongest_name}: {strongest_val:.4f}, margin: "
            f"{self.margin:+.4f})"
        )


def _probe_result(
    metric_name: str,
    value: float,
    *,
    higher_is_better: bool = True,
    self_test: bool = False,
    **baseline_context: Any,
) -> BakeoffProbeResult:
    baselines = tuple(compute_baselines(metric_name, **baseline_context))
    computed_margin = _margin(value, list(baselines), higher_is_better=higher_is_better)
    return BakeoffProbeResult(
        metric_name=metric_name,
        value=value,
        baselines=baselines,
        margin=computed_margin,
        self_test_label=SELF_TEST_LABEL if self_test else None,
    )


def _pooled(backbone: FrozenBackbone, clips: torch.Tensor) -> torch.Tensor:
    features = extract_features_no_grad(backbone, clips)
    return features.data.mean(dim=1) if features.data.ndim == 3 else features.data


def same_object_retrieval_map(
    *,
    backbone: FrozenBackbone,
    clips: torch.Tensor,
    identity_labels: "list[int]",
    pos0: "list[tuple[int, int]]",
    posk: "list[tuple[int, int]]",
    self_test: bool = False,
) -> BakeoffProbeResult:
    """Same-object retrieval mAP, cross-boundary queries only.

    Every sample is a query (``pos0[i] != posk[i]`` by construction of the
    synthetic-stand-in position generator — see
    :func:`src.identity.selftest.make_synthetic_patch_positions` — or, for
    real GT tracks, ``scripts/eval_semantics.py``'s ``crossed_boundary``).
    The pool is every sample's backbone embedding. ``position_only`` ranks
    the pool by proximity of ``posk`` to the QUERY's OWN ``pos0`` — the
    "assume it barely moved" strategy that ignores the encoder entirely,
    identical in shape to ``scripts/eval_semantics.py``'s position-only
    baseline.
    """
    pooled = _pooled(backbone, clips)
    normalized = torch.nn.functional.normalize(pooled, dim=-1).detach().cpu().numpy()
    sim = normalized @ normalized.T

    labels = np.asarray(identity_labels)
    pos0_arr = np.asarray(pos0, dtype=np.float64)
    posk_arr = np.asarray(posk, dtype=np.float64)

    aps: list[float] = []
    position_only_aps: list[float] = []
    for i in range(len(labels)):
        if np.array_equal(pos0_arr[i], posk_arr[i]):
            continue  # not a cross-boundary query; excluded, not counted as 0

        row = sim[i].copy()
        row[i] = -np.inf  # exclude self-match
        ap = average_precision(i, row, labels, labels[i])
        if ap is not None:
            aps.append(ap)

        dist = -np.linalg.norm(posk_arr - pos0_arr[i], axis=1)
        dist[i] = -np.inf
        pos_ap = average_precision(i, dist, labels, labels[i])
        if pos_ap is not None:
            position_only_aps.append(pos_ap)

    if not aps:
        raise ValueError(
            "no cross-boundary query survived (every sample had pos0 == "
            "posk); this harness refuses to report a retrieval mAP over an "
            "empty query set rather than emit a silent NaN — see Day 9's "
            "Undefined-vs-NaN rule"
        )

    retrieval_map = float(np.mean(aps))
    position_only_map = (
        float(np.mean(position_only_aps)) if position_only_aps else float("nan")
    )

    return _probe_result(
        "identity.bakeoff.retrieval_map",
        retrieval_map,
        self_test=self_test,
        n_tracks=len(labels),
        position_only_map=position_only_map,
    )


def temporal_embedding_stability(
    *,
    backbone: FrozenBackbone,
    clip_sequence: torch.Tensor,
    self_test: bool = False,
) -> BakeoffProbeResult:
    """Mean cosine similarity between consecutive embeddings of the SAME
    tracked entity across ``clip_sequence`` (``[T, ...]``, one clip per
    timestep). Baseline (``constant_embedding``, cosine 1.0 by
    construction) is the "return the same vector every frame" strategy —
    see ``_semantics_temporal_cosine_baselines``: a real encoder's temporal
    stability shows only in the gap between 1.0 and its own number, so a
    HIGHER stability is not automatically better in isolation, but a
    collapsing (near-0 or negative) value signals the encoder is not
    tracking a stable identity across time at all.
    """
    pooled = _pooled(backbone, clip_sequence)
    normalized = torch.nn.functional.normalize(pooled, dim=-1)
    if normalized.shape[0] < 2:
        raise ValueError("temporal_embedding_stability needs at least 2 timesteps")
    consecutive_cosine = (normalized[:-1] * normalized[1:]).sum(dim=-1)
    stability = float(consecutive_cosine.mean().item())

    return _probe_result(
        "identity.bakeoff.temporal_stability",
        stability,
        higher_is_better=True,
        self_test=self_test,
    )


def patch_boundary_discontinuity(
    *,
    backbone: FrozenBackbone,
    clips_a: torch.Tensor,
    clips_b: torch.Tensor,
    self_test: bool = False,
) -> BakeoffProbeResult:
    """Mean L2 distance between embeddings of adjacent-patch pairs
    (``clips_a[i]``, ``clips_b[i]`` — a sample and its synthetic neighbour
    just across a patch-grid boundary). No trivial ceiling is defined for
    this one (``_semantics_patch_boundary_baselines``: "no adversary
    strategy defined yet") — reported as a set/encoder descriptor, not a
    pass/fail gate, same as ``semantics.patch_boundary_l2``.
    """
    a = _pooled(backbone, clips_a)
    b = _pooled(backbone, clips_b)
    discontinuity = float(torch.linalg.norm(a - b, dim=-1).mean().item())

    return _probe_result(
        "identity.bakeoff.patch_boundary_l2", discontinuity, self_test=self_test
    )


def clothing_change_robustness(
    *,
    backbone: FrozenBackbone,
    clean_clips: torch.Tensor,
    perturbed_clips: torch.Tensor,
    identity_labels: "list[int]",
    self_test: bool = False,
) -> BakeoffProbeResult:
    """Retrieval mAP on a clothing-changed gallery (CHIRLA's specific
    contribution, once fetchable) — query the perturbed embeddings against
    a pool of clean embeddings from the SAME identities, same-identity
    retrieval. Today this can only be exercised against
    :func:`src.identity.selftest.apply_synthetic_clothing_change`'s
    synthetic perturbation; CHIRLA itself has no ``license_snapshot``
    (Objective 1) and is not fetchable.
    """
    clean = torch.nn.functional.normalize(_pooled(backbone, clean_clips), dim=-1)
    perturbed = torch.nn.functional.normalize(
        _pooled(backbone, perturbed_clips), dim=-1
    )
    labels = np.asarray(identity_labels)

    cross_sim = (perturbed @ clean.T).detach().cpu().numpy()
    clean_sim = (clean @ clean.T).detach().cpu().numpy()

    perturbed_aps: list[float] = []
    clean_aps: list[float] = []
    for i in range(len(labels)):
        row = cross_sim[i].copy()
        row[i] = -np.inf
        ap = average_precision(i, row, labels, labels[i])
        if ap is not None:
            perturbed_aps.append(ap)

        clean_row = clean_sim[i].copy()
        clean_row[i] = -np.inf
        clean_ap = average_precision(i, clean_row, labels, labels[i])
        if clean_ap is not None:
            clean_aps.append(clean_ap)

    perturbed_map = float(np.mean(perturbed_aps)) if perturbed_aps else float("nan")
    clean_map = float(np.mean(clean_aps)) if clean_aps else float("nan")
    chance_map = 1.0 / len(set(identity_labels)) if identity_labels else float("nan")

    return _probe_result(
        "identity.bakeoff.clothing_change_map",
        perturbed_map,
        self_test=self_test,
        chance_map=chance_map,
        clean_map=clean_map,
    )


def open_bakeoff_eval_set(registry: DatasetRegistry, name: str) -> DatasetEntry:
    """Resolve a real eval dataset for this harness — lane R permitted,
    never lane C_pending_consent without a record, and NEVER reachable
    from a training-marked module (``open_for_eval``'s stack walk). Not
    called against real data by anything in this repository today: see
    module docstring."""
    return registry.open_for_eval(name)
