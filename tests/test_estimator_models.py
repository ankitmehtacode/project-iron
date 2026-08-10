"""Day 20, Objective 1 — motion and measurement models.

Determinism, positive-definiteness of Q/R, and the envelope-derived R
growing monotonically with distance are the three properties the objective
names explicitly; each gets its own test rather than being folded into a
single "it works" assertion, so a regression in one property does not hide
behind a passing assertion of another.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.estimator.measurement_model import (
    MeasurementModel,
    MeasurementModelError,
    illustrative_position_envelope,
    measurement_model_for,
)
from src.estimator.motion_model import (
    MOTION_ENTITY_KINDS,
    STATE_DIM,
    ConstantVelocityMotionModel,
    MotionModelError,
    motion_model_for,
)
from src.model.envelope import Envelope, EnvelopeCurve
from src.model.measurement import WorldPositionMeasurement

SEED = 20260731


def _is_positive_definite(matrix: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return False
    return True


# ---------------------------------------------------------------------------
# Motion models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", MOTION_ENTITY_KINDS)
def test_motion_model_for_every_kind_builds(kind: str) -> None:
    model = motion_model_for(kind)  # type: ignore[arg-type]
    assert model.kind == kind


def test_f_and_F_agree_for_every_kind() -> None:
    rng = np.random.default_rng(SEED)
    state = rng.normal(size=STATE_DIM)
    for kind in MOTION_ENTITY_KINDS:
        model = motion_model_for(kind)  # type: ignore[arg-type]
        np.testing.assert_array_equal(model.f(state, 0.1), model.F(0.1) @ state)


def test_constant_velocity_predicts_straight_line() -> None:
    model = motion_model_for("person")
    state = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 0.0])
    predicted = model.f(state, dt_s=2.0)
    np.testing.assert_allclose(predicted, [2.0, 4.0, 0.0, 1.0, 2.0, 0.0])


def test_fixture_transition_is_identity() -> None:
    model = motion_model_for("fixture")
    rng = np.random.default_rng(SEED)
    state = rng.normal(size=STATE_DIM)
    np.testing.assert_array_equal(model.f(state, dt_s=5.0), state)


@pytest.mark.parametrize("kind", MOTION_ENTITY_KINDS)
def test_motion_model_deterministic_under_pinned_seed(kind: str) -> None:
    """Two independent runs from the same seed produce bit-identical output."""

    def run() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(SEED)
        state = rng.normal(size=STATE_DIM)
        dt = float(rng.uniform(0.05, 0.5))
        model = motion_model_for(kind)  # type: ignore[arg-type]
        return model.f(state, dt), model.F(dt), model.Q(dt)

    f1, F1, Q1 = run()
    f2, F2, Q2 = run()
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_array_equal(F1, F2)
    np.testing.assert_array_equal(Q1, Q2)


@pytest.mark.parametrize("kind", MOTION_ENTITY_KINDS)
def test_Q_is_positive_definite(kind: str) -> None:
    model = motion_model_for(kind)  # type: ignore[arg-type]
    for dt in (1e-3, 0.1, 1.0, 5.0):
        assert _is_positive_definite(model.Q(dt)), f"{kind} Q not PD at dt={dt}"


def test_asset_carried_Q_larger_than_person_Q() -> None:
    """The 'inflated Q' requirement, checked directly rather than by eye."""
    person = motion_model_for("person")
    carried = motion_model_for("asset_carried")
    dt = 0.2
    # Velocity-block variance is where the inflation is easiest to compare
    # directly: larger sigma_a scales every entry of Q, so the (3,3) entry
    # (vx variance) alone is a faithful proxy for "more process noise".
    assert carried.Q(dt)[3, 3] > person.Q(dt)[3, 3]


def test_asset_static_Q_smaller_than_person_Q() -> None:
    static = motion_model_for("asset_static")
    person = motion_model_for("person")
    dt = 0.2
    assert static.Q(dt)[3, 3] < person.Q(dt)[3, 3]


def test_carrier_entity_id_rejected_for_non_carried_kinds() -> None:
    with pytest.raises(MotionModelError):
        ConstantVelocityMotionModel(
            kind="person", sigma_a_mps2=1.0, carrier_entity_id="carrier-1"
        )


def test_carrier_entity_id_accepted_for_asset_carried() -> None:
    model = motion_model_for("asset_carried", carrier_entity_id="carrier-1")
    assert isinstance(model, ConstantVelocityMotionModel)
    assert model.carrier_entity_id == "carrier-1"
    # Interface only, today: the coupling has no numerical effect yet.
    uncoupled = motion_model_for("asset_carried")
    np.testing.assert_array_equal(model.Q(0.3), uncoupled.Q(0.3))


def test_negative_dt_raises() -> None:
    for kind in MOTION_ENTITY_KINDS:
        model = motion_model_for(kind)  # type: ignore[arg-type]
        with pytest.raises(MotionModelError):
            model.F(-0.1)
        with pytest.raises(MotionModelError):
            model.Q(-0.1)


def test_sha_differs_by_kind_and_params() -> None:
    shas = {}
    for kind in MOTION_ENTITY_KINDS:
        shas[kind] = motion_model_for(kind).sha  # type: ignore[arg-type]
    assert len(set(shas.values())) == len(shas), f"sha collision across kinds: {shas}"


def test_sha_stable_for_identical_params() -> None:
    a = motion_model_for("person")
    b = motion_model_for("person")
    assert a.sha == b.sha


def test_unknown_entity_kind_raises() -> None:
    with pytest.raises(MotionModelError):
        motion_model_for("spaceship")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Measurement models
# ---------------------------------------------------------------------------


def test_measurement_model_h_extracts_position() -> None:
    model = measurement_model_for("cam-1")
    state = np.array([1.0, 2.0, 3.0, 9.0, 9.0, 9.0])
    np.testing.assert_array_equal(model.h(state), [1.0, 2.0, 3.0])


def test_R_positive_definite_across_distances() -> None:
    model = measurement_model_for("cam-1")
    for distance in (0.0, 1.0, 3.0, 8.0, 15.0, 25.0, 100.0):
        assert _is_positive_definite(model.R(distance)), f"R not PD at {distance}m"


def test_R_grows_monotonically_with_distance_within_envelope() -> None:
    model = measurement_model_for("cam-1")
    distances = [1.0, 3.0, 8.0, 15.0, 25.0]
    sigmas = [model.sigma_m(d) for d in distances]
    assert sigmas == sorted(sigmas)
    assert len(set(sigmas)) == len(
        sigmas
    ), "sigma must be STRICTLY increasing, not flat"


def test_observation_outside_envelope_gets_inflated_R_not_discarded() -> None:
    """The explicit test the objective asks for: outside-envelope enters as
    high-R, and R() still RETURNS a value -- there is no code path where an
    out-of-range distance raises or is silently dropped."""
    model = measurement_model_for("cam-1")
    boundary_distance = model.envelope.curve.points[-1][0]
    far_outside = boundary_distance + 50.0

    assert model.is_within_envelope(boundary_distance)
    assert not model.is_within_envelope(far_outside)

    boundary_R = model.R(boundary_distance)
    outside_R = model.R(far_outside)  # must not raise, must not return None

    assert outside_R is not None
    assert np.diag(outside_R)[0] > np.diag(boundary_R)[0], (
        "an out-of-envelope reading must be trusted LESS (higher R) than a "
        "reading at the boundary, not equally or more"
    )
    # exact inflation factor, not just "bigger"
    np.testing.assert_allclose(
        np.diag(outside_R)[0],
        np.diag(boundary_R)[0] * model.outside_envelope_inflation**2,
    )


def test_measurement_model_rejects_speed_indexed_curve() -> None:
    speed_curve = EnvelopeCurve(
        independent_variable="speed_mps", points=((0.2, 1.0), (4.9, 5.0))
    )
    bad_envelope = Envelope(
        capability="state_estimation",
        camera_id="cam-1",
        twin_rev=0,
        curve=speed_curve,
        sample_count=4,
        manifest_sha="test",
    )
    with pytest.raises(MeasurementModelError):
        MeasurementModel(envelope=bad_envelope)


def test_measurement_model_rejects_inflation_below_one() -> None:
    envelope = illustrative_position_envelope("cam-1")
    with pytest.raises(MeasurementModelError):
        MeasurementModel(envelope=envelope, outside_envelope_inflation=0.5)


def test_illustrative_envelope_is_marked_unmeasured() -> None:
    envelope = illustrative_position_envelope("cam-1")
    assert "illustrative" in envelope.manifest_sha
    assert "unmeasured" in envelope.manifest_sha


def test_measurement_model_sha_differs_with_envelope_camera() -> None:
    a = measurement_model_for("cam-1")
    b = measurement_model_for("cam-2")
    assert a.sha != b.sha


def test_negative_distance_raises() -> None:
    model = measurement_model_for("cam-1")
    with pytest.raises(MeasurementModelError):
        model.R(-1.0)


# ---------------------------------------------------------------------------
# WorldPositionMeasurement
# ---------------------------------------------------------------------------


def test_world_position_measurement_constructs() -> None:
    m = WorldPositionMeasurement(x_m=1.0, y_m=2.0, z_m=3.0)
    assert (m.x_m, m.y_m, m.z_m) == (1.0, 2.0, 3.0)


def test_world_position_measurement_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        WorldPositionMeasurement(x_m=float("nan"), y_m=0.0, z_m=0.0)
    with pytest.raises(ValueError):
        WorldPositionMeasurement(x_m=float("inf"), y_m=0.0, z_m=0.0)
