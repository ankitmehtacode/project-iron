"""Objective 1 (Day 14) — closing falsification test 3, the twin_rev hole.

STRUCTURAL rules under test:
  - WorldPosition is unconstructable without twin_rev (4 required,
    undefaulted fields, mirroring Day 13's Relationship).
  - Cross-twin_rev distance/reprojection raises without an explicit
    TwinRevTransform.
  - TwinRevTransformRegistry.resolve() raises for an unregistered pair —
    never defaults to identity.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.model.world import (
    RigidTransform3D,
    TwinRevError,
    TwinRevTransform,
    TwinRevTransformRegistry,
    WorldPosition,
)


# ---------------------------------------------------------------------------
# STRUCTURAL: WorldPosition unconstructable without twin_rev
# ---------------------------------------------------------------------------


def test_world_position_all_four_fields_are_required_undefaulted() -> None:
    for f in dataclasses.fields(WorldPosition):
        no_factory: object = f.default_factory
        assert f.default is dataclasses.MISSING
        assert no_factory is dataclasses.MISSING


def test_world_position_missing_twin_rev_is_unconstructable() -> None:
    with pytest.raises(TypeError):
        WorldPosition(x_m=1.0, y_m=2.0, z_m=3.0)  # type: ignore[call-arg]


def test_world_position_rejects_non_finite_coordinates() -> None:
    with pytest.raises(ValueError):
        WorldPosition(x_m=float("nan"), y_m=0.0, z_m=0.0, twin_rev=1)
    with pytest.raises(ValueError):
        WorldPosition(x_m=float("inf"), y_m=0.0, z_m=0.0, twin_rev=1)


def test_world_position_rejects_negative_twin_rev() -> None:
    with pytest.raises(ValueError):
        WorldPosition(x_m=0.0, y_m=0.0, z_m=0.0, twin_rev=-1)


def test_world_position_is_frozen() -> None:
    pos = WorldPosition(x_m=1.0, y_m=2.0, z_m=3.0, twin_rev=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        pos.x_m = 9.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# No arithmetic operators exist at all — the safest way to "raise"
# ---------------------------------------------------------------------------


def test_world_position_has_no_arithmetic_operators() -> None:
    a = WorldPosition(x_m=0.0, y_m=0.0, z_m=0.0, twin_rev=1)
    b = WorldPosition(x_m=1.0, y_m=1.0, z_m=1.0, twin_rev=1)
    with pytest.raises(TypeError):
        a - b  # type: ignore[operator]
    with pytest.raises(TypeError):
        a + b  # type: ignore[operator]
    with pytest.raises(TypeError):
        a < b  # type: ignore[operator]


# ---------------------------------------------------------------------------
# distance_to — same-rev works directly, cross-rev requires a transform
# ---------------------------------------------------------------------------


def test_distance_to_same_rev_computes_directly() -> None:
    a = WorldPosition(x_m=0.0, y_m=0.0, z_m=0.0, twin_rev=1)
    b = WorldPosition(x_m=3.0, y_m=4.0, z_m=0.0, twin_rev=1)
    assert a.distance_to(b) == pytest.approx(5.0)


def test_distance_to_cross_rev_without_transform_raises() -> None:
    a = WorldPosition(x_m=0.0, y_m=0.0, z_m=0.0, twin_rev=1)
    b = WorldPosition(x_m=0.0, y_m=0.0, z_m=0.0, twin_rev=2)
    with pytest.raises(TwinRevError):
        a.distance_to(b)


def test_distance_to_cross_rev_with_transform_succeeds() -> None:
    # rev 2 is rev 1 translated by +10m in x (e.g. a remount).
    transform = TwinRevTransform(
        from_twin_rev=2,
        to_twin_rev=1,
        transform=RigidTransform3D(
            rotation=(1, 0, 0, 0, 1, 0, 0, 0, 1), translation=(10.0, 0.0, 0.0)
        ),
    )
    a = WorldPosition(x_m=0.0, y_m=0.0, z_m=0.0, twin_rev=1)
    b = WorldPosition(x_m=0.0, y_m=0.0, z_m=0.0, twin_rev=2)  # -> (10,0,0) in rev 1
    assert a.distance_to(b, transform=transform) == pytest.approx(10.0)


def test_reproject_wrong_source_rev_raises() -> None:
    transform = TwinRevTransform(
        from_twin_rev=2, to_twin_rev=1, transform=RigidTransform3D.identity()
    )
    pos = WorldPosition(x_m=0.0, y_m=0.0, z_m=0.0, twin_rev=1)  # not rev 2
    with pytest.raises(TwinRevError):
        pos.reproject(transform)


def test_reproject_historical_position_remains_interpretable() -> None:
    """Falsification test 3's core claim: re-versioning does not silently
    change what a stored position means — it stays interpretable via an
    explicit transform, and the original record is untouched.
    """
    transform = TwinRevTransform(
        from_twin_rev=1,
        to_twin_rev=2,
        transform=RigidTransform3D(
            rotation=(1, 0, 0, 0, 1, 0, 0, 0, 1), translation=(0.0, 0.0, 0.5)
        ),
    )
    historical = WorldPosition(x_m=1.0, y_m=2.0, z_m=0.0, twin_rev=1)
    reprojected = historical.reproject(transform)

    assert reprojected.twin_rev == 2
    assert reprojected.z_m == pytest.approx(0.5)
    # The original, frozen record is completely unchanged.
    assert historical.twin_rev == 1
    assert historical.z_m == 0.0


# ---------------------------------------------------------------------------
# TwinRevTransform — same-rev must be identity
# ---------------------------------------------------------------------------


def test_same_rev_transform_must_be_identity() -> None:
    with pytest.raises(TwinRevError):
        TwinRevTransform(
            from_twin_rev=1,
            to_twin_rev=1,
            transform=RigidTransform3D(
                rotation=(1, 0, 0, 0, 1, 0, 0, 0, 1), translation=(1.0, 0.0, 0.0)
            ),
        )
    ok = TwinRevTransform.identity_for(1)
    assert ok.transform == RigidTransform3D.identity()


# ---------------------------------------------------------------------------
# STRUCTURAL: registry never defaults to identity across differing revs
# ---------------------------------------------------------------------------


def test_registry_resolve_same_rev_needs_no_registration() -> None:
    registry = TwinRevTransformRegistry()
    resolved = registry.resolve(5, 5)
    assert resolved == TwinRevTransform.identity_for(5)


def test_registry_resolve_unregistered_pair_raises_not_identity() -> None:
    registry = TwinRevTransformRegistry()
    with pytest.raises(TwinRevError):
        registry.resolve(1, 2)


def test_registry_register_and_resolve_round_trip() -> None:
    registry = TwinRevTransformRegistry()
    transform = TwinRevTransform(
        from_twin_rev=1,
        to_twin_rev=2,
        transform=RigidTransform3D(
            rotation=(1, 0, 0, 0, 1, 0, 0, 0, 1), translation=(5.0, 0.0, 0.0)
        ),
    )
    registry.register(transform)
    resolved = registry.resolve(1, 2)
    assert resolved == transform


def test_registry_register_also_registers_inverse() -> None:
    registry = TwinRevTransformRegistry()
    transform = TwinRevTransform(
        from_twin_rev=1,
        to_twin_rev=2,
        transform=RigidTransform3D(
            rotation=(1, 0, 0, 0, 1, 0, 0, 0, 1), translation=(5.0, 0.0, 0.0)
        ),
    )
    registry.register(transform)
    inverse = registry.resolve(2, 1)
    pos_rev2 = WorldPosition(x_m=0.0, y_m=0.0, z_m=0.0, twin_rev=2)
    back = pos_rev2.reproject(inverse)
    assert back.twin_rev == 1
    assert back.x_m == pytest.approx(-5.0)


def test_registry_refuses_to_register_a_same_rev_transform() -> None:
    registry = TwinRevTransformRegistry()
    with pytest.raises(TwinRevError):
        registry.register(TwinRevTransform.identity_for(3))


# ---------------------------------------------------------------------------
# RigidTransform3D
# ---------------------------------------------------------------------------


def test_rigid_transform_identity_apply_is_noop() -> None:
    p = (1.0, 2.0, 3.0)
    assert RigidTransform3D.identity().apply(p) == pytest.approx(p)


def test_rigid_transform_inverse_round_trips() -> None:
    # 90-degree rotation about Z: (x, y, z) -> (-y, x, z), plus a translation.
    rot90z = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    t = RigidTransform3D(rotation=rot90z, translation=(2.0, -3.0, 1.0))
    p = (1.0, 0.0, 5.0)
    forward = t.apply(p)
    back = t.inverse().apply(forward)
    assert back[0] == pytest.approx(p[0], abs=1e-9)
    assert back[1] == pytest.approx(p[1], abs=1e-9)
    assert back[2] == pytest.approx(p[2], abs=1e-9)


def test_rigid_transform_rejects_malformed_shapes() -> None:
    short_rotation: tuple[float, ...] = (1, 0, 0, 0, 1, 0, 0, 0)
    short_translation: tuple[float, ...] = (0, 0)
    with pytest.raises(ValueError):
        RigidTransform3D(
            rotation=short_rotation, translation=(0, 0, 0)  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        RigidTransform3D(
            rotation=(1, 0, 0, 0, 1, 0, 0, 0, 1),
            translation=short_translation,  # type: ignore[arg-type]
        )


def test_rigid_transform_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError):
        RigidTransform3D(
            rotation=(1, 0, 0, 0, 1, 0, 0, 0, float("nan")), translation=(0, 0, 0)
        )
