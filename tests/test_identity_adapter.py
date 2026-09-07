"""Tests for the identity-adapter architecture (Day 36, Objective 2).

ADR 0001's whole argument rests on two properties this file exists to prove
rather than assume: the backbone truly never receives a gradient from an
identity objective, and the training path truly cannot reach lane-R or
unconsented data by any route. Everything else here (contract validation,
parameter bounds) is the usual boundary discipline this codebase applies to
every new envelope.
"""

from __future__ import annotations

import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.data.registry import (
    ConsentRecord,
    DatasetEntry,
    DatasetRegistry,
    LaneViolation,
    LicenseSnapshot,
    normalise,
)
from src.identity.adapter import Adapter, AdapterConfig, AdapterConfigError
from src.identity.backbone import FrozenBackbone, extract_features_no_grad
from src.identity.contracts import ContractError, FeatureTensor, IdentityEmbedding
from src.identity.lane_gate import require_training_dataset
from src.identity.selftest import SyntheticStandInBackbone

EXAMPLES = settings(
    max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


# -- FeatureTensor / IdentityEmbedding contracts -----------------------------


def test_feature_tensor_accepts_pooled_and_token_shapes() -> None:
    FeatureTensor(data=torch.randn(4, 32), backbone_sha="abc")
    FeatureTensor(data=torch.randn(4, 8, 32), backbone_sha="abc")


def test_feature_tensor_rejects_bad_rank() -> None:
    import pytest

    with pytest.raises(ContractError, match="batch, dim"):
        FeatureTensor(data=torch.randn(32), backbone_sha="abc")


def test_feature_tensor_requires_backbone_sha() -> None:
    import pytest

    with pytest.raises(ContractError, match="backbone_sha is required"):
        FeatureTensor(data=torch.randn(4, 32), backbone_sha="")


def test_identity_embedding_requires_both_shas() -> None:
    import pytest

    with pytest.raises(ContractError, match="backbone_sha"):
        IdentityEmbedding(data=torch.randn(4, 16), backbone_sha="", adapter_sha="x")
    with pytest.raises(ContractError, match="adapter_sha"):
        IdentityEmbedding(data=torch.randn(4, 16), backbone_sha="x", adapter_sha="")


@given(
    batch=st.integers(min_value=1, max_value=16),
    dim=st.integers(min_value=1, max_value=128),
)
@EXAMPLES
def test_feature_tensor_dim_and_batch_properties(batch: int, dim: int) -> None:
    ft = FeatureTensor(data=torch.randn(batch, dim), backbone_sha="abc")
    assert ft.dim == dim
    assert ft.batch == batch


# -- Adapter ------------------------------------------------------------------


def test_adapter_forward_produces_labelled_l2_normalized_embedding() -> None:
    config = AdapterConfig(input_dim=32, hidden_dim=16, output_dim=8, num_layers=2)
    adapter = Adapter(config)
    features = FeatureTensor(data=torch.randn(5, 32), backbone_sha="backbone-x")

    out = adapter(features)

    assert isinstance(out, IdentityEmbedding)
    assert out.data.shape == (5, 8)
    assert out.backbone_sha == "backbone-x"
    assert out.adapter_sha == adapter.adapter_sha
    norms = out.data.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-5)


def test_adapter_pools_token_features_before_the_head() -> None:
    config = AdapterConfig(input_dim=32, output_dim=8)
    adapter = Adapter(config)
    features = FeatureTensor(data=torch.randn(3, 10, 32), backbone_sha="b")
    out = adapter(features)
    assert out.data.shape == (3, 8)


def test_adapter_rejects_mismatched_feature_dim() -> None:
    import pytest

    config = AdapterConfig(input_dim=32, output_dim=8)
    adapter = Adapter(config)
    features = FeatureTensor(data=torch.randn(3, 16), backbone_sha="b")
    with pytest.raises(AdapterConfigError, match="input_dim"):
        adapter(features)


def test_adapter_is_parameter_bounded() -> None:
    """ADR 0001's erasure argument depends on the adapter being orders of
    magnitude smaller than the backbone. This must be a refusal, not a
    style guideline."""
    import pytest

    huge = AdapterConfig(input_dim=4096, hidden_dim=4096, output_dim=4096, num_layers=4)
    with pytest.raises(AdapterConfigError, match="exceeding the"):
        Adapter(huge)


def test_adapter_param_count_within_declared_cap() -> None:
    from src.identity.adapter import MAX_ADAPTER_PARAMETERS

    config = AdapterConfig(input_dim=768, hidden_dim=256, output_dim=128, num_layers=2)
    adapter = Adapter(config)
    assert 0 < adapter.param_count <= MAX_ADAPTER_PARAMETERS


def test_adapter_sha_changes_when_weights_change() -> None:
    """Objective 5 depends on this: two adapters trained differently must
    have distinct adapter_sha. Pinned here at the unit level first."""
    config = AdapterConfig(input_dim=16, output_dim=8)
    adapter = Adapter(config)
    sha_before = adapter.adapter_sha
    with torch.no_grad():
        for p in adapter.parameters():
            p.add_(1.0)
    assert adapter.adapter_sha != sha_before


def test_two_adapters_same_config_different_init_have_different_shas() -> None:
    config = AdapterConfig(input_dim=16, output_dim=8)
    a = Adapter(config)
    b = Adapter(config)
    # Random init (no seed pinned) makes collision astronomically unlikely;
    # this is what adapter_sha is FOR.
    assert a.adapter_sha != b.adapter_sha


# -- Frozen-backbone gradient isolation (Objective 2's central claim) --------


def test_synthetic_stand_in_satisfies_the_frozen_backbone_protocol() -> None:
    backbone = SyntheticStandInBackbone(input_dim=16, feature_dim=32)
    assert isinstance(backbone, FrozenBackbone)


def test_frozen_backbone_has_no_gradient_path() -> None:
    """The whole point of Objective 2: no gradient from the adapter's loss
    ever reaches backbone parameters.

    Deliberately uses a backbone that did NOT set requires_grad=False on
    its own weights (see SyntheticStandInBackbone's docstring) — the
    property under test is that extract_features_no_grad's torch.no_grad()
    + detach() isolation holds regardless of backbone hygiene, not that a
    conveniently-frozen backbone happens to be safe.
    """
    backbone = SyntheticStandInBackbone(input_dim=16, feature_dim=32)
    assert any(p.requires_grad for p in backbone.parameters()), (
        "test setup check: the stand-in must start un-frozen for this test "
        "to prove anything"
    )

    adapter = Adapter(AdapterConfig(input_dim=32, output_dim=8))

    clip = torch.randn(6, 16, requires_grad=False)
    features = extract_features_no_grad(backbone, clip)
    embedding = adapter(features)

    loss = embedding.data.sum()
    loss.backward()

    for name, p in backbone._projection.named_parameters():
        assert p.grad is None, (
            f"backbone parameter {name!r} received a gradient — ADR 0001's "
            "frozen-backbone constraint is violated"
        )
    assert any(p.grad is not None for p in adapter.parameters()), (
        "adapter parameters received no gradient at all; the test setup "
        "is broken, not passing for the right reason"
    )


def test_extract_features_no_grad_detaches_even_a_grad_carrying_input() -> None:
    """features.data must not require grad after the no_grad boundary, even
    when nothing about the backbone itself would have prevented it."""
    backbone = SyntheticStandInBackbone(input_dim=8, feature_dim=8)
    clip = torch.randn(2, 8, requires_grad=True)
    features = extract_features_no_grad(backbone, clip)
    assert not features.data.requires_grad


# -- Lane enforcement reuse (no second lane check) ---------------------------


def registry_with(*entries: DatasetEntry) -> DatasetRegistry:
    return DatasetRegistry({normalise(e.name): e for e in entries})


def snapshot() -> LicenseSnapshot:
    from datetime import date

    return LicenseSnapshot(
        url="https://example.org/license",
        verified_date=date(2026, 9, 7),
        text_sha256="0" * 64,
        verified_by="test",
        verified_class="research-only",
    )


def test_require_training_dataset_refuses_lane_r() -> None:
    import pytest

    entry = DatasetEntry(name="lane-r-set", lane="R", license_snapshot=snapshot())
    registry = registry_with(entry)
    with pytest.raises(LaneViolation, match="never enter a training"):
        require_training_dataset(registry, "lane-r-set")


def test_require_training_dataset_refuses_pending_consent_without_a_record() -> None:
    import pytest

    entry = DatasetEntry(
        name="unconsented-archive",
        lane="C_pending_consent",
        license_snapshot=snapshot(),
    )
    registry = registry_with(entry)
    with pytest.raises(LaneViolation, match="C_pending_consent"):
        require_training_dataset(registry, "unconsented-archive")


def test_require_training_dataset_accepts_lane_c_with_consent_record() -> None:
    from datetime import date

    entry = DatasetEntry(
        name="site-zero-adapter-set",
        lane="C_pending_consent",
        license_snapshot=snapshot(),
        consent_record=ConsentRecord(
            path="data/site-zero/session-001",
            sha="1" * 64,
            subjects=3,
            captured_on=date(2026, 9, 1),
            recorded_by="test",
            purpose_note="adapter training, same purpose as capture",
        ),
    )
    registry = registry_with(entry)
    resolved = require_training_dataset(registry, "site-zero-adapter-set")
    assert resolved.name == "site-zero-adapter-set"


def test_require_training_dataset_does_not_duplicate_registry_messages() -> None:
    """The reuse requirement made concrete: the LaneViolation raised through
    require_training_dataset must be byte-identical to the one
    open_for_training raises directly — proof this is a pass-through, not a
    parallel implementation with its own wording."""
    entry = DatasetEntry(name="lane-r-set", lane="R", license_snapshot=snapshot())
    registry = registry_with(entry)

    direct_message = None
    try:
        registry.open_for_training("lane-r-set")
    except LaneViolation as exc:
        direct_message = str(exc)

    wrapped_message = None
    try:
        require_training_dataset(registry, "lane-r-set")
    except LaneViolation as exc:
        wrapped_message = str(exc)

    assert direct_message is not None
    assert direct_message == wrapped_message


def test_lane_gate_module_is_marked_for_the_stack_walk() -> None:
    """src.data.registry's stack-walk looks for this exact module-global —
    a typo here would silently disable the protection."""
    import src.identity.lane_gate as lane_gate
    from src.data.registry import TRAINING_PATH_MARKER

    assert getattr(lane_gate, TRAINING_PATH_MARKER) is True
