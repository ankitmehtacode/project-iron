"""Objective 1 — core primitives: Observation, Entity, Envelope.

Every STRUCTURAL rule gets a test that attempts the violation and asserts
the failure, per the Day-13 quality bar.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.contracts.frames import AffineTransform, FrameGeometry
from src.model.entity import (
    AnonymousSessionEntity,
    EntityError,
    EnrolledEntity,
    FixtureEntity,
    RegisteredAssetEntity,
    TimeVaryingAttribute,
)
from src.model.envelope import Envelope, EnvelopeCurve, EnvelopeError
from src.model.frame_of_reference import FrameOfReference
from src.model.measurement import BadgeSwipeMeasurement, CameraFrameMeasurement
from src.model.observation import Observation, ObservationError
from src.model.uncertainty import (
    ALL_UNCERTAINTY_KINDS,
    Uncertainty,
    UncertaintyError,
)
from src.model.ulid import ULID, InvalidULID, generate_ulid


def _frame_of_reference(twin_rev: int = 1) -> FrameOfReference:
    return FrameOfReference(
        geometry=FrameGeometry(1920, 1080),
        to_canonical=AffineTransform.identity(),
        twin_rev=twin_rev,
    )


def _observation(**overrides: object) -> Observation:
    kwargs: dict[str, object] = dict(
        observation_id=generate_ulid(),
        sensor_id="cam-lobby-01",
        ts_ns=1_785_000_000_000_000_000,
        frame_ref="cam-lobby-01/frame-000123",
        measurement=CameraFrameMeasurement(bbox_px=(10.0, 10.0, 50.0, 90.0)),
        uncertainty=Uncertainty.gaussian_px(2.0, 2.0),
        frame_of_reference=_frame_of_reference(),
        producer_shas=("abc123",),
        envelope_status="within_envelope",
    )
    kwargs.update(overrides)
    return Observation(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ULID
# ---------------------------------------------------------------------------


def test_ulid_generates_valid_sortable_ids() -> None:
    a = generate_ulid(now_ns=1_700_000_000_000_000_000)
    b = generate_ulid(now_ns=1_700_000_000_001_000_000)
    assert len(a.value) == 26
    assert str(a) < str(b)
    assert a.timestamp_ms == 1_700_000_000_000


def test_ulid_monotonic_within_same_millisecond() -> None:
    ids = [generate_ulid(now_ns=1_700_000_000_000_000_000) for _ in range(50)]
    assert ids == sorted(ids, key=str)
    assert len(set(str(i) for i in ids)) == len(ids)


def test_ulid_rejects_malformed_string() -> None:
    with pytest.raises(InvalidULID):
        ULID("not-a-ulid")
    with pytest.raises(InvalidULID):
        ULID("I" * 26)  # 'I' is not a Crockford Base32 character


# ---------------------------------------------------------------------------
# Observation — STRUCTURAL: uncertainty is required
# ---------------------------------------------------------------------------


def test_observation_without_uncertainty_kwarg_is_unconstructable() -> None:
    fields = {f.name for f in dataclasses.fields(Observation)}
    assert "uncertainty" in fields
    with pytest.raises(TypeError):
        Observation(  # type: ignore[call-arg]
            observation_id=generate_ulid(),
            sensor_id="cam-1",
            ts_ns=1,
            frame_ref="f-1",
            measurement=CameraFrameMeasurement(),
            frame_of_reference=_frame_of_reference(),
            producer_shas=(),
            envelope_status="within_envelope",
        )


def test_observation_with_none_uncertainty_raises() -> None:
    with pytest.raises(ObservationError, match="uncertainty is required"):
        _observation(uncertainty=None)


def test_observation_rejects_non_uncertainty_value() -> None:
    with pytest.raises(ObservationError):
        _observation(uncertainty="0.1")  # type: ignore[arg-type]


def test_observation_valid_construction_round_trips_fields() -> None:
    obs = _observation()
    assert obs.sensor_id == "cam-lobby-01"
    assert obs.uncertainty.kind == "gaussian_px"


def test_observation_rejects_unknown_envelope_status() -> None:
    with pytest.raises(ObservationError):
        _observation(envelope_status="probably_fine")


def test_observation_non_camera_sensor_is_first_class() -> None:
    obs = _observation(
        sensor_id="badge-reader-03",
        measurement=BadgeSwipeMeasurement(
            badge_id="b-42", door_id="lobby-main", granted=True
        ),
        uncertainty=Uncertainty.categorical(
            false_accept_rate=1e-6, false_reject_rate=1e-3
        ),
        producer_shas=(),
    )
    assert obs.measurement.sensor_kind == "badge"


def test_observation_is_frozen() -> None:
    obs = _observation()
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.sensor_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


def test_uncertainty_requires_kind_specific_params() -> None:
    with pytest.raises(UncertaintyError):
        Uncertainty(kind="gaussian_px", params=(("sigma_x", 1.0),))


def test_uncertainty_interval_rejects_low_above_high() -> None:
    with pytest.raises(UncertaintyError):
        Uncertainty(kind="interval", params=(("low", 5.0), ("high", 1.0)))


def test_uncertainty_rejects_negative_sigma() -> None:
    with pytest.raises(UncertaintyError):
        Uncertainty.gaussian_px(-1.0, 1.0)


def test_required_params_covers_every_uncertainty_kind() -> None:
    """Day 24, Objective 3 audit: _REQUIRED_PARAMS is a hand-written dict
    keyed by the UncertaintyKind Literal, the same shape as the
    CAPABILITIES/GATES defect (Day 16, Day 23) -- a kind present in
    UncertaintyKind but missing from _REQUIRED_PARAMS would raise a bare
    KeyError from Uncertainty.__post_init__ instead of a clear
    UncertaintyError. No drift exists today; this is the cheap structural
    guard against it happening the moment a sixth kind is added."""
    from src.model import uncertainty as uncertainty_module

    assert set(uncertainty_module._REQUIRED_PARAMS) == set(ALL_UNCERTAINTY_KINDS)


# ---------------------------------------------------------------------------
# Entity — STRUCTURAL: identity_class enforced at the type level
# ---------------------------------------------------------------------------


def test_anonymous_entity_has_no_persistence_field_to_populate() -> None:
    field_names = {f.name for f in dataclasses.fields(AnonymousSessionEntity)}
    assert "persistent_identity_ref" not in field_names
    with pytest.raises(TypeError):
        AnonymousSessionEntity(  # type: ignore[call-arg]
            entity_id="sess-1",
            kind="person",
            first_seen_ns=0,
            last_seen_ns=10,
            persistent_identity_ref="employee-42",
        )


def test_enrolled_entity_requires_persistence_ref() -> None:
    with pytest.raises(EntityError):
        EnrolledEntity(entity_id="e-1", kind="person", first_seen_ns=0, last_seen_ns=10)
    ok = EnrolledEntity(
        entity_id="e-1",
        kind="person",
        first_seen_ns=0,
        last_seen_ns=10,
        persistent_identity_ref="employee-42",
    )
    assert ok.identity_class == "enrolled"


def test_anonymous_entity_constructs_without_persistence() -> None:
    session = AnonymousSessionEntity(
        entity_id="sess-1", kind="person", first_seen_ns=0, last_seen_ns=10
    )
    assert session.identity_class == "anonymous_session"


def test_entity_rejects_last_seen_before_first_seen() -> None:
    with pytest.raises(EntityError):
        AnonymousSessionEntity(
            entity_id="sess-1", kind="person", first_seen_ns=10, last_seen_ns=0
        )


def test_entity_rejects_unknown_kind() -> None:
    bad_kind: str = "spaceship"
    with pytest.raises(EntityError):
        AnonymousSessionEntity(
            entity_id="sess-1",
            kind=bad_kind,  # type: ignore[arg-type]
            first_seen_ns=0,
            last_seen_ns=1,
        )


def test_registered_asset_and_fixture_construct() -> None:
    asset = RegisteredAssetEntity(
        entity_id="asset-1",
        kind="asset",
        first_seen_ns=0,
        last_seen_ns=1,
        asset_tag="laptop-114",
    )
    fixture = FixtureEntity(
        entity_id="lobby", kind="zone", first_seen_ns=0, last_seen_ns=1
    )
    assert asset.identity_class == "registered_asset"
    assert fixture.identity_class == "fixture"


def test_time_varying_attribute_validity_interval() -> None:
    attr = TimeVaryingAttribute(
        name="role", value="visitor", valid_from_ns=0, valid_to_ns=100
    )
    assert attr.valid_to_ns == 100
    with pytest.raises(EntityError):
        TimeVaryingAttribute(
            name="role", value="visitor", valid_from_ns=100, valid_to_ns=0
        )


# ---------------------------------------------------------------------------
# Envelope — STRUCTURAL: curves, not scalars
# ---------------------------------------------------------------------------


def test_envelope_curve_rejects_single_point() -> None:
    with pytest.raises(EnvelopeError, match="at least 2 measured points"):
        EnvelopeCurve(independent_variable="speed_mps", points=((1.0, 0.01),))


def test_envelope_curve_rejects_non_increasing_x() -> None:
    with pytest.raises(EnvelopeError):
        EnvelopeCurve(
            independent_variable="speed_mps", points=((1.0, 0.01), (1.0, 0.02))
        )
    with pytest.raises(EnvelopeError):
        EnvelopeCurve(
            independent_variable="speed_mps", points=((2.0, 0.01), (1.0, 0.02))
        )


def test_envelope_curve_interpolates_and_clamps() -> None:
    curve = EnvelopeCurve(
        independent_variable="speed_mps", points=((0.2, 0.002), (0.98, 0.0098))
    )
    assert curve.value_at(0.2) == pytest.approx(0.002)
    assert curve.value_at(0.98) == pytest.approx(0.0098)
    mid = curve.value_at((0.2 + 0.98) / 2)
    assert 0.002 < mid < 0.0098
    assert curve.value_at(0.0) == pytest.approx(0.002)  # clamped, not extrapolated
    assert curve.value_at(10.0) == pytest.approx(0.0098)


def test_envelope_reproduces_day7_49x_speed_dependent_threshold() -> None:
    curve = EnvelopeCurve(
        independent_variable="speed_mps", points=((0.2, 0.002), (0.98, 0.0098))
    )
    ratio = curve.value_at(0.98) / curve.value_at(0.2)
    assert ratio == pytest.approx(4.9, abs=0.01)
    envelope = Envelope(
        capability="motion_gate",
        camera_id="cam-lobby-01",
        twin_rev=1,
        curve=curve,
        sample_count=42,
        manifest_sha="deadbeef",
    )
    assert envelope.curve.value_at(0.5) > 0


def test_envelope_rejects_low_sample_count() -> None:
    curve = EnvelopeCurve(
        independent_variable="speed_mps", points=((0.2, 0.002), (0.98, 0.0098))
    )
    with pytest.raises(EnvelopeError):
        Envelope(
            capability="motion_gate",
            camera_id="cam-1",
            twin_rev=1,
            curve=curve,
            sample_count=1,
            manifest_sha="deadbeef",
        )


# ---------------------------------------------------------------------------
# FrameOfReference — reuses contracts, adds twin_rev
# ---------------------------------------------------------------------------


def test_frame_of_reference_rejects_negative_twin_rev() -> None:
    with pytest.raises(ValueError):
        FrameOfReference(
            geometry=FrameGeometry(10, 10),
            to_canonical=AffineTransform.identity(),
            twin_rev=-1,
        )


def test_frame_of_reference_is_current_for() -> None:
    ref = _frame_of_reference(twin_rev=3)
    assert ref.is_current_for(3)
    assert not ref.is_current_for(4)
