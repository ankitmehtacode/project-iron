"""Identity-adapter architecture per ADR 0001
(``docs/adr/0001-identity-adapter-architecture.md``): person-specific
capability lives in a small, deletable adapter over a frozen, backbone-
agnostic encoder — never in fine-tuned backbone weights.

Nothing in this package is trainable today. Lane C (the only lane eligible
to train or calibrate anything) has zero clips, and lane S's appearance-
learned validity gate has refused every synthetic set this project owns
since Day 9/11. See ``FOUNDATION_REPORT.md``'s Day-36 section for the full
deterministic argument. This package exists so that the day real lane-C
data arrives, no further engineering stands between it and a first trained
checkpoint — see :mod:`src.identity.selftest` for what is exercised today
instead, and its ``SELF_TEST_LABEL``.
"""

from src.identity.adapter import Adapter, AdapterConfig, AdapterConfigError
from src.identity.backbone import FrozenBackbone, extract_features_no_grad
from src.identity.contracts import ContractError, FeatureTensor, IdentityEmbedding
from src.identity.lane_gate import require_training_dataset

__all__ = [
    "Adapter",
    "AdapterConfig",
    "AdapterConfigError",
    "ContractError",
    "FeatureTensor",
    "FrozenBackbone",
    "IdentityEmbedding",
    "extract_features_no_grad",
    "require_training_dataset",
]
