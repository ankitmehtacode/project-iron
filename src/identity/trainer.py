"""Adapter trainer: standard supervised metric-learning shape.

Loss: triplet margin loss on cosine distance
----------------------------------------------
Chosen over a plain pairwise-contrastive loss because triplet loss directly
optimizes the same quantity the promotion gate measures
(:mod:`src.identity.promotion`) — the RELATIVE ranking of same-identity vs
different-identity cosine similarity — rather than an absolute distance
threshold a contrastive loss would need tuned separately and which the
retrieval-mAP metric does not actually care about. Swappable: ``loss_fn`` is
a constructor parameter to :func:`train_adapter`, defaulting to
``torch.nn.TripletMarginWithDistanceLoss`` with cosine distance, so a future
InfoNCE/contrastive variant is a one-argument change here, not a rewrite of
the loop.

Every training step calls the backbone through
:func:`src.identity.backbone.extract_features_no_grad` — see that function
and ``docs/adr/0001-identity-adapter-architecture.md`` for why. Nothing in
this module resolves a dataset name; that is
:mod:`src.identity.lane_gate`'s job, called by whatever entrypoint invokes
this trainer with real data, never bypassed here.
"""

from __future__ import annotations

import hashlib
import json

import torch
from pydantic import BaseModel, ConfigDict, Field

from src.identity.adapter import Adapter
from src.identity.backbone import FrozenBackbone, extract_features_no_grad
from src.identity.selftest import TripletSampler


class TrainingConfig(BaseModel):
    """Everything that determines a training run's optimization behaviour.

    Frozen and hashable (:meth:`config_sha`), same discipline as
    :class:`src.identity.adapter.AdapterConfig` — a checkpoint manifest
    names this hash, not the individual hyperparameters, so two runs are
    comparable iff their configs are byte-identical.
    """

    model_config = ConfigDict(frozen=True)

    epochs: int = Field(gt=0, default=20)
    learning_rate: float = Field(gt=0, default=1e-3)
    triplet_margin: float = Field(gt=0, default=0.2)
    batch_size: int = Field(gt=0, default=16)
    seed: int = 20260907

    def config_sha(self) -> str:
        payload = self.model_dump(mode="json")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    similarity: torch.Tensor = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    distance: torch.Tensor = torch.ones_like(similarity) - similarity
    return distance


def train_adapter(
    *,
    backbone: FrozenBackbone,
    adapter: Adapter,
    triplets: TripletSampler,
    config: TrainingConfig,
) -> list[float]:
    """Run ``config.epochs`` optimizer steps of triplet-margin training.

    Args:
        backbone: Reached only through :func:`extract_features_no_grad`.
        adapter: The only module whose parameters this function optimizes.
        triplets: Called once per epoch; must return three same-shape
            batches of RAW CLIPS (anchor, positive, negative) — backbone
            input, not features — so every step exercises the full
            backbone-then-adapter path rather than a shortcut around it.
        config: Hyperparameters; see :class:`TrainingConfig`.

    Returns:
        Per-epoch scalar loss, for a caller to plot or sanity-check
        monotonic decrease.
    """
    loss_fn = torch.nn.TripletMarginWithDistanceLoss(
        distance_function=cosine_distance, margin=config.triplet_margin
    )
    optimizer = torch.optim.Adam(adapter.parameters(), lr=config.learning_rate)

    losses: list[float] = []
    for _ in range(config.epochs):
        anchor_clip, positive_clip, negative_clip = triplets()

        anchor_emb = adapter(extract_features_no_grad(backbone, anchor_clip)).data
        positive_emb = adapter(extract_features_no_grad(backbone, positive_clip)).data
        negative_emb = adapter(extract_features_no_grad(backbone, negative_clip)).data

        loss = loss_fn(anchor_emb, positive_emb, negative_emb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return losses
