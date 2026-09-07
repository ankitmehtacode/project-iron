"""SELF_TEST-only stand-ins: NOT a backbone bake-off answer, NOT a trained
capability, NOT to be quoted as re-ID performance.

Objective 4's backbone bake-off harness is built to run against a real
:class:`~src.identity.backbone.FrozenBackbone` and real lane-R eval data,
but does not select a backbone today, and Objective 3's trainer/promotion
gate is run end to end today ONLY against these stand-ins. Every output
produced through this module must carry :data:`SELF_TEST_LABEL` verbatim —
see ``tests/test_identity_adapter.py`` and
``tests/test_identity_trainer.py`` for the tests that pin this.

:class:`SyntheticStandInBackbone` deliberately does NOT set
``requires_grad = False`` on its own parameters — that is the point of it.
Objective 2's frozen-backbone test needs a backbone that would leak a
gradient if the training entrypoint's ``torch.no_grad()`` wrapper
(:func:`src.identity.backbone.extract_features_no_grad`) were not doing real
work, not a backbone that is safe by accident.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from src.identity.contracts import FeatureTensor

SELF_TEST_LABEL = (
    "SELF_TEST — not a trained model, not to be quoted as re-ID performance"
)

TripletSampler = Callable[[], "tuple[torch.Tensor, torch.Tensor, torch.Tensor]"]


class SyntheticStandInBackbone:
    """A deterministic, NOT-a-real-encoder projection.

    It has learned nothing about appearance, identity, or anything else — it
    is a fixed random linear projection, seeded for reproducibility. Its
    only job is to give the adapter/trainer/bake-off pipeline something with
    the right shape and a real ``backbone_sha`` to run against before a real
    backbone is chosen.
    """

    def __init__(
        self, input_dim: int = 32, feature_dim: int = 64, seed: int = 20260907
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self._projection = nn.Linear(input_dim, feature_dim, bias=False)
        with torch.no_grad():
            self._projection.weight.copy_(
                torch.randn(feature_dim, input_dim, generator=generator)
            )
        # Deliberately NOT frozen here — see module docstring.
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.backbone_sha = self._compute_sha()

    def _compute_sha(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"SyntheticStandInBackbone-v1")
        digest.update(
            self._projection.weight.detach().cpu().contiguous().numpy().tobytes()
        )
        return digest.hexdigest()

    def extract_features(self, clip: torch.Tensor) -> FeatureTensor:
        out = self._projection(clip)
        return FeatureTensor(data=out, backbone_sha=self.backbone_sha)

    def parameters(self) -> "list[nn.Parameter]":
        """Exposed for the frozen-gradient test only — not part of
        :class:`~src.identity.backbone.FrozenBackbone`'s protocol."""
        return list(self._projection.parameters())

    def save_weights(self, path: Path) -> Path:
        """Write this backbone's weights as raw bytes to ``path`` — a
        stand-in "backbone artifact" file for
        :mod:`src.identity.erasure_drill` to hash with
        :func:`src.provenance.sha256_file`, exactly mirroring how a real
        exported backbone (``.xml``/``.bin``) would be checked. Same byte
        serialization as :meth:`current_weights_sha`, so the file's hash
        and a live re-hash of the tensor are directly comparable.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            self._projection.weight.detach().cpu().contiguous().numpy().tobytes()
        )
        return path

    def current_weights_sha(self) -> str:
        """Hash of this backbone's CURRENT in-memory weights, recomputed
        live from the tensor every call.

        Unlike :attr:`backbone_sha` (fixed once at construction, the same
        role a real backbone's content hash plays), this catches an
        in-place mutation even if nothing ever refreshed ``backbone_sha``
        to reflect it — the property the erasure drill actually needs to
        verify.
        """
        digest = hashlib.sha256()
        digest.update(
            self._projection.weight.detach().cpu().contiguous().numpy().tobytes()
        )
        return digest.hexdigest()


def make_synthetic_identity_gallery(
    n_identities: int = 8,
    samples_per_identity: int = 12,
    input_dim: int = 32,
    seed: int = 20260907,
    cluster_scale: float = 3.0,
    noise_scale: float = 1.0,
) -> "tuple[torch.Tensor, list[int]]":
    """Synthetic stand-in "clips": one gaussian cluster per identity in
    backbone-input space, spaced ``cluster_scale`` apart with
    ``noise_scale`` intra-identity spread — a problem that is genuinely
    learnable by a real metric-learning loss, not pure noise, so a passing
    self-test demonstrates the plumbing actually moves gradients rather
    than merely running without crashing.

    NOT real identities, NOT real appearance, NOT lane C, NOT lane S
    (does not touch the dataset registry at all — there is nothing here
    for it to check). See module docstring: every result derived from this
    must carry :data:`SELF_TEST_LABEL`.
    """
    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_identities, input_dim, generator=generator) * cluster_scale
    clips = []
    labels: list[int] = []
    for identity in range(n_identities):
        noise = (
            torch.randn(samples_per_identity, input_dim, generator=generator)
            * noise_scale
        )
        clips.append(centers[identity] + noise)
        labels.extend([identity] * samples_per_identity)
    return torch.cat(clips, dim=0), labels


def make_triplet_sampler(
    clips: torch.Tensor, labels: "list[int]", batch_size: int, seed: int = 0
) -> TripletSampler:
    """A callable that draws a fresh (anchor, positive, negative) batch of
    RAW CLIPS each call — same-identity anchor/positive, different-identity
    negative — for :func:`src.identity.trainer.train_adapter`."""
    rng = random.Random(seed)
    by_identity: dict[int, list[int]] = {}
    for i, identity in enumerate(labels):
        by_identity.setdefault(identity, []).append(i)
    identities = list(by_identity)
    if len(identities) < 2:
        raise ValueError("need at least 2 identities to sample a negative")

    def sample() -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        anchors, positives, negatives = [], [], []
        for _ in range(batch_size):
            pos_id, neg_id = rng.sample(identities, 2)
            pool = by_identity[pos_id]
            a_idx, p_idx = (
                rng.sample(pool, 2) if len(pool) >= 2 else (pool[0], pool[0])
            )
            n_idx = rng.choice(by_identity[neg_id])
            anchors.append(clips[a_idx])
            positives.append(clips[p_idx])
            negatives.append(clips[n_idx])
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

    return sample


def make_synthetic_patch_positions(
    n_samples: int, grid_size: int = 4, seed: int = 0
) -> "tuple[list[tuple[int, int]], list[tuple[int, int]]]":
    """A synthetic ``(pos0, posk)`` patch-grid coordinate pair per sample,
    with ``posk`` always different from ``pos0`` — every sample "crosses a
    boundary" by construction, standing in for
    ``scripts/eval_semantics.py``'s ``crossed_boundary`` GT-position filter
    without needing real GT tracks. See
    :func:`src.identity.bakeoff.same_object_retrieval_map`.
    """
    rng = random.Random(seed)
    cells = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    pos0: list[tuple[int, int]] = []
    posk: list[tuple[int, int]] = []
    for _ in range(n_samples):
        start = rng.choice(cells)
        remaining = [cell for cell in cells if cell != start]
        end = rng.choice(remaining)
        pos0.append(start)
        posk.append(end)
    return pos0, posk


def apply_synthetic_clothing_change(
    clips: torch.Tensor, labels: "list[int]", seed: int = 0, shift_scale: float = 1.5
) -> torch.Tensor:
    """Add a fixed per-identity "clothing" shift vector to every sample of
    that identity — a synthetic stand-in for CHIRLA's real clothing-change
    scenario (Objective 1; CHIRLA has no license_snapshot and is not
    fetchable today). An identity-preserving, appearance-perturbing
    transform: same shift for every sample of one identity, different
    identities get different (and therefore separable) shifts, exactly the
    property a real clothing change has (same person, different visual
    appearance) that a naive appearance-only encoder would confuse for a
    different identity.
    """
    generator = torch.Generator().manual_seed(seed)
    n_identities = len(set(labels))
    shifts = torch.randn(n_identities, clips.shape[-1], generator=generator) * shift_scale
    return torch.stack([clips[i] + shifts[labels[i]] for i in range(clips.shape[0])])
