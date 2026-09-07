"""Promotion gate: "no number, no merge" applied to model adoption.

A trained checkpoint may be marked ``promoted`` only if its identity
retrieval mAP clears the frozen backbone's own raw cosine similarity —
computed with NO adapter applied at all — by a positive margin. This is
Day 12's baseline rule (:mod:`src.eval.baselines`), applied to a model
artifact instead of a scorecard metric: the adapter has to beat its own
backbone, not just chance, or it has not earned promotion. Reuses
``src.eval.baselines``'s ``Baseline``/``margin`` primitives directly (the
same way ``scripts/eval_semantics.py`` does for its own per-gap sweep)
rather than re-deriving "value (baseline, margin)" arithmetic a second
time, and reuses ``scripts/eval_semantics.py``'s ``average_precision`` for
the mAP computation itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.eval.baselines import Baseline
from src.eval.baselines import margin as _margin
from src.eval.retrieval import average_precision
from src.identity.adapter import Adapter
from src.identity.backbone import FrozenBackbone, extract_features_no_grad
from src.identity.selftest import SELF_TEST_LABEL

# A checkpoint is promoted only on a STRICT positive margin over the
# strongest flag-worthy baseline (raw_backbone_cosine, or chance if that is
# somehow absent) — Day 12's rule: margin <= 0 means the metric is not
# distinguishing the adapter from a trivial strategy.
PROMOTION_MARGIN_THRESHOLD = 0.0


@dataclass(frozen=True)
class PromotionResult:
    metric_name: str
    value: float
    baselines: tuple[Baseline, ...]
    margin: float
    promoted: bool
    self_test_label: str | None = None
    """None for a real evaluate_promotion() call. Equal to SELF_TEST_LABEL,
    verbatim, when self_test=True was passed. Carried on the result itself
    — not only added by a caller's print statement — so that .render() and
    .as_evidence() are self-labelling wherever they end up (a log line, a
    notebook cell, a different script) rather than depending on every call
    site remembering to prefix it. Day 36's own audit found this field
    missing (see FOUNDATION_REPORT.md's Day-36 section) while
    BakeoffProbeResult already had it; this closes that gap."""

    def as_evidence(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": round(self.value, 6),
            "baselines": [b.as_dict() for b in self.baselines],
            "margin": round(self.margin, 6) if np.isfinite(self.margin) else None,
            "promoted": self.promoted,
            "self_test_label": self.self_test_label,
        }

    def render(self) -> str:
        flag_worthy = [
            b for b in self.baselines if b.flag_worthy and np.isfinite(b.value)
        ]
        strongest = max((b.value for b in flag_worthy), default=float("nan"))
        strongest_name = next(
            (b.name for b in flag_worthy if b.value == strongest), "n/a"
        )
        verdict = "PROMOTED" if self.promoted else "NOT PROMOTED"
        prefix = f"[{self.self_test_label}] " if self.self_test_label else ""
        return (
            f"{prefix}{self.metric_name} = {self.value:.4f} "
            f"(baseline {strongest_name}: {strongest:.4f}, margin: "
            f"{self.margin:+.4f}) -> {verdict}"
        )


def _retrieval_map(embeddings: torch.Tensor, identity_labels: "list[int]") -> float:
    """Same-identity retrieval mAP by cosine-similarity ranking.

    Uses ``src.eval.retrieval.average_precision`` — the same AP computation
    ``scripts/eval_semantics.py`` uses for its own same-object retrieval
    mAP, extracted to ``src.eval`` (Day 36) specifically so this promotion
    gate could import it as library code instead of reaching into a
    ``scripts/`` entrypoint.
    """
    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    matrix = (normalized @ normalized.T).detach().cpu().numpy()
    labels = np.asarray(identity_labels)

    aps: list[float] = []
    for i in range(len(labels)):
        row = matrix[i].copy()
        row[i] = -np.inf  # exclude self-match
        ap = average_precision(i, row, labels, labels[i])
        if ap is not None:
            aps.append(ap)
    return float(np.mean(aps)) if aps else float("nan")


def evaluate_promotion(
    *,
    backbone: FrozenBackbone,
    adapter: Adapter,
    clips: torch.Tensor,
    identity_labels: "list[int]",
    self_test: bool = False,
) -> PromotionResult:
    """Score both the adapter and the raw-backbone baseline on the same
    gallery, and decide promotion.

    Args:
        clips: Raw backbone input for every gallery member, batched.
        identity_labels: Ground-truth identity per row of ``clips``.
        self_test: Pass True for any call whose ``clips``/``identity_labels``
            are synthetic — stamps :data:`SELF_TEST_LABEL` onto the returned
            result so ``.render()``/``.as_evidence()`` cannot be read as a
            real result out of context, regardless of what the caller does
            with them afterward.
    """
    features = extract_features_no_grad(backbone, clips)

    # Raw baseline: pool tokens the same trivial way Adapter.forward does,
    # but apply NO learned transform — this is what "no adapter applied"
    # (the objective's own phrase) means concretely.
    raw = features.data.mean(dim=1) if features.data.ndim == 3 else features.data
    raw_map = _retrieval_map(raw, identity_labels)

    with torch.no_grad():
        embedding = adapter(features)
    adapter_map = _retrieval_map(embedding.data, identity_labels)

    n_identities = len(set(identity_labels))
    chance_map = 1.0 / n_identities if n_identities > 0 else float("nan")

    baselines = (
        Baseline("chance", chance_map, "1 / n_identities — uniform-random ranker"),
        Baseline(
            "raw_backbone_cosine",
            raw_map,
            "cosine similarity ranking directly on frozen backbone "
            "features, no adapter applied",
        ),
    )
    computed_margin = _margin(adapter_map, list(baselines), higher_is_better=True)
    promoted = bool(
        np.isfinite(computed_margin) and computed_margin > PROMOTION_MARGIN_THRESHOLD
    )

    return PromotionResult(
        metric_name="identity.retrieval_map",
        value=adapter_map,
        baselines=baselines,
        margin=computed_margin,
        promoted=promoted,
        self_test_label=SELF_TEST_LABEL if self_test else None,
    )
