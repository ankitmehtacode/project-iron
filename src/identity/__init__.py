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
from src.identity.bakeoff import (
    BakeoffProbeResult,
    clothing_change_robustness,
    open_bakeoff_eval_set,
    patch_boundary_discontinuity,
    same_object_retrieval_map,
    temporal_embedding_stability,
)
from src.identity.checkpoint import (
    CheckpointManifest,
    write_checkpoint_manifest,
    write_selftest_checkpoint_manifest,
)
from src.identity.contracts import ContractError, FeatureTensor, IdentityEmbedding
from src.identity.erasure_drill import ErasureDrillResult, run_erasure_drill
from src.identity.lane_gate import require_training_dataset
from src.identity.promotion import PromotionResult, evaluate_promotion
from src.identity.trainer import TrainingConfig, train_adapter

__all__ = [
    "Adapter",
    "AdapterConfig",
    "AdapterConfigError",
    "BakeoffProbeResult",
    "CheckpointManifest",
    "ContractError",
    "ErasureDrillResult",
    "FeatureTensor",
    "FrozenBackbone",
    "IdentityEmbedding",
    "PromotionResult",
    "TrainingConfig",
    "clothing_change_robustness",
    "evaluate_promotion",
    "extract_features_no_grad",
    "open_bakeoff_eval_set",
    "patch_boundary_discontinuity",
    "require_training_dataset",
    "run_erasure_drill",
    "same_object_retrieval_map",
    "temporal_embedding_stability",
    "train_adapter",
    "write_checkpoint_manifest",
    "write_selftest_checkpoint_manifest",
]
