"""Day 20, Objective 3 — the consistency stage: NIS/NEES and the routed stubs.

Live NIS computation and the STRUCTURAL "no estimate without residuals"
rule are already exercised indirectly by tests/test_estimator_filter.py
(every update in a real filter run carries one). This file isolates the
consistency-stage functions themselves and the StateEstimate/
ConsistencyResidual construction rules, so a regression here is diagnosed
at the unit that owns it rather than surfacing as a filter test failure.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.estimator import consistency
from src.estimator.state import ConsistencyResidual, StateEstimate, StateEstimateError

_VALID_SHAS = dict(
    motion_model_sha="mm-1", measurement_model_sha="mo-1", update_rule_sha="ur-1"
)


def _estimate(
    residuals: tuple[ConsistencyResidual, ...], **overrides: object
) -> StateEstimate:
    kwargs: dict[str, object] = dict(
        ts_ns=1,
        mean=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        cov=tuple(tuple(row) for row in np.eye(6).tolist()),
        observed=True,
        graph_rev=1,
        residuals=residuals,
        **_VALID_SHAS,
    )
    kwargs.update(overrides)
    return StateEstimate(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# chi2_upper_bound
# ---------------------------------------------------------------------------


def test_chi2_upper_bound_matches_known_values() -> None:
    # Standard textbook values (Bar-Shalom et al.), 95% confidence.
    assert consistency.chi2_upper_bound(1, 0.95) == pytest.approx(3.841, abs=1e-2)
    assert consistency.chi2_upper_bound(3, 0.95) == pytest.approx(7.815, abs=1e-2)
    assert consistency.chi2_upper_bound(6, 0.95) == pytest.approx(12.592, abs=1e-2)


def test_chi2_upper_bound_grows_with_dof() -> None:
    bounds = [consistency.chi2_upper_bound(d) for d in (1, 2, 3, 6)]
    assert bounds == sorted(bounds)


def test_chi2_upper_bound_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        consistency.chi2_upper_bound(0)
    with pytest.raises(ValueError):
        consistency.chi2_upper_bound(3, confidence=1.5)


# ---------------------------------------------------------------------------
# compute_nis / compute_nees
# ---------------------------------------------------------------------------


def test_compute_nis_zero_innovation_is_well_within_bound() -> None:
    innovation = np.zeros(3)
    S = np.eye(3)
    residual = consistency.compute_nis(innovation, S)
    assert residual.kind == "nis"
    assert residual.value == 0.0
    assert residual.within_bound is True
    assert residual.dof == 3


def test_compute_nis_large_innovation_exceeds_bound() -> None:
    innovation = np.array([100.0, 100.0, 100.0])
    S = np.eye(3)
    residual = consistency.compute_nis(innovation, S)
    assert residual.within_bound is False
    assert residual.value > residual.chi2_bound  # type: ignore[operator]


def test_compute_nees_matches_compute_nis_mechanics() -> None:
    error = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cov = np.eye(6)
    residual = consistency.compute_nees(error, cov)
    assert residual.kind == "nees"
    assert residual.value == pytest.approx(1.0)
    assert residual.dof == 6
    assert residual.chi2_bound == pytest.approx(consistency.chi2_upper_bound(6))


# ---------------------------------------------------------------------------
# Stub residuals — routing and consumers
# ---------------------------------------------------------------------------


def test_stub_residuals_carry_no_value() -> None:
    for stub_fn in (
        consistency.no_observation_nis_stub,
        consistency.constraint_stub,
        consistency.calibration_stub,
        consistency.coverage_stub,
    ):
        residual = stub_fn()
        assert residual.value is None
        assert residual.dof is None
        assert residual.chi2_bound is None
        assert residual.within_bound is None
        assert residual.note


def test_stub_residuals_name_the_right_consumer() -> None:
    assert "twin-revision hypothesis" in consistency.constraint_stub().consumer
    assert "recalibration event" in consistency.calibration_stub().consumer
    assert "envelope drift" in consistency.coverage_stub().consumer


def test_stub_residuals_helper_returns_constraint_calibration_coverage() -> None:
    stubs = consistency.stub_residuals()
    assert [r.kind for r in stubs] == ["constraint", "calibration", "coverage"]


# ---------------------------------------------------------------------------
# fraction_outside_bound
# ---------------------------------------------------------------------------


def test_fraction_outside_bound_counts_correctly() -> None:
    residuals = [
        consistency.compute_nis(np.zeros(3), np.eye(3)),  # within
        consistency.compute_nis(np.array([50.0, 50.0, 50.0]), np.eye(3)),  # outside
        consistency.compute_nis(np.zeros(3), np.eye(3)),  # within
        consistency.constraint_stub(),  # not nis, excluded
    ]
    assert consistency.fraction_outside_bound(residuals, "nis") == pytest.approx(1 / 3)


def test_fraction_outside_bound_nan_when_nothing_scored() -> None:
    residuals = [consistency.no_observation_nis_stub(), consistency.constraint_stub()]
    assert np.isnan(consistency.fraction_outside_bound(residuals, "nis"))
    assert np.isnan(consistency.fraction_outside_bound(residuals, "nees"))


# ---------------------------------------------------------------------------
# ConsistencyResidual construction rules
# ---------------------------------------------------------------------------


def test_consistency_residual_rejects_unknown_kind() -> None:
    with pytest.raises(StateEstimateError):
        ConsistencyResidual(kind="bogus", consumer="x")  # type: ignore[arg-type]


def test_consistency_residual_rejects_empty_consumer() -> None:
    with pytest.raises(StateEstimateError):
        ConsistencyResidual(kind="nis", consumer="")


def test_consistency_residual_rejects_partial_value_set() -> None:
    with pytest.raises(StateEstimateError):
        ConsistencyResidual(
            kind="nis", consumer="x", value=1.0
        )  # dof/bound/within missing


# ---------------------------------------------------------------------------
# StateEstimate STRUCTURAL — no estimate without residuals
# ---------------------------------------------------------------------------


def test_state_estimate_rejects_empty_residuals() -> None:
    with pytest.raises(StateEstimateError, match="residuals must not be empty"):
        _estimate(residuals=())


def test_state_estimate_rejects_residuals_with_no_nis_entry() -> None:
    with pytest.raises(StateEstimateError, match="nis"):
        _estimate(
            residuals=(consistency.constraint_stub(), consistency.calibration_stub())
        )


def test_state_estimate_accepts_stub_nis_plus_other_stubs() -> None:
    estimate = _estimate(
        residuals=(consistency.no_observation_nis_stub(), *consistency.stub_residuals())
    )
    assert len(estimate.residuals) == 4


def test_state_estimate_rejects_wrong_length_mean() -> None:
    with pytest.raises(StateEstimateError):
        _estimate(
            residuals=(consistency.no_observation_nis_stub(),),
            mean=(0.0, 0.0, 0.0),
        )


def test_state_estimate_rejects_wrong_shape_cov() -> None:
    with pytest.raises(StateEstimateError):
        _estimate(
            residuals=(consistency.no_observation_nis_stub(),),
            cov=((1.0, 0.0), (0.0, 1.0)),
        )


@pytest.mark.parametrize(
    "missing_field", ["motion_model_sha", "measurement_model_sha", "update_rule_sha"]
)
def test_state_estimate_rejects_empty_sha(missing_field: str) -> None:
    kwargs = dict(_VALID_SHAS)
    kwargs[missing_field] = ""
    with pytest.raises(StateEstimateError):
        _estimate(residuals=(consistency.no_observation_nis_stub(),), **kwargs)


# ---------------------------------------------------------------------------
# StateEstimate.require_comparable — §15's three-way key
# ---------------------------------------------------------------------------


def test_require_comparable_passes_when_all_three_shas_match() -> None:
    a = _estimate(residuals=(consistency.no_observation_nis_stub(),))
    b = _estimate(residuals=(consistency.no_observation_nis_stub(),), ts_ns=2)
    a.require_comparable(b)  # must not raise


@pytest.mark.parametrize(
    "differing_field", ["motion_model_sha", "measurement_model_sha", "update_rule_sha"]
)
def test_require_comparable_raises_when_one_sha_differs(differing_field: str) -> None:
    a = _estimate(residuals=(consistency.no_observation_nis_stub(),))
    kwargs = dict(_VALID_SHAS)
    kwargs[differing_field] = "different"
    b = _estimate(residuals=(consistency.no_observation_nis_stub(),), **kwargs)
    with pytest.raises(StateEstimateError, match=differing_field):
        a.require_comparable(b)
