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

import torch
from torch import nn

from src.identity.contracts import FeatureTensor

SELF_TEST_LABEL = (
    "SELF_TEST — not a trained model, not to be quoted as re-ID performance"
)


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
