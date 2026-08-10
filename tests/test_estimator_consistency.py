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
# Day 22, Objective 1 -- posterior-family guard on compute_nees
# ---------------------------------------------------------------------------


def test_compute_nees_default_family_is_gaussian_and_unchanged() -> None:
    error = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cov = np.eye(6)
    explicit = consistency.compute_nees(error, cov, posterior_family="gaussian")
    implicit = consistency.compute_nees(error, cov)
    assert explicit == implicit


def test_compute_nees_raises_on_mixture_posterior_family() -> None:
    error = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cov = np.eye(6)
    with pytest.raises(consistency.PosteriorFamilyError, match="gaussian_mixture"):
        consistency.compute_nees(error, cov, posterior_family="gaussian_mixture")


def test_compute_nees_raises_on_any_non_gaussian_family() -> None:
    with pytest.raises(consistency.PosteriorFamilyError):
        consistency.compute_nees(
            np.zeros(3), np.eye(3), posterior_family="bogus"  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Day 22, Objective 1 -- the collapsed-Gaussian diagnostic (kept, labeled)
# ---------------------------------------------------------------------------


def test_collapsed_diagnostic_matches_compute_nees_numerically() -> None:
    """Same math as compute_nees -- only the labeling differs, deliberately:
    Objective 1 asks for the old number "alongside" the new ones, not a
    different number."""
    error = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    cov = np.eye(6) * 2.0
    plain = consistency.compute_nees(error, cov)
    diagnostic = consistency.compute_collapsed_gaussian_nees_diagnostic(error, cov)
    assert diagnostic.value == pytest.approx(plain.value)
    assert diagnostic.within_bound == plain.within_bound


def test_collapsed_diagnostic_consumer_names_itself_invalid() -> None:
    residual = consistency.compute_collapsed_gaussian_nees_diagnostic(
        np.zeros(3), np.eye(3)
    )
    assert "NOT a valid consistency check" in residual.consumer
    assert "diagnostic" in residual.note


# ---------------------------------------------------------------------------
# Day 22, Objective 1 -- compute_mixture_nees
# ---------------------------------------------------------------------------


def test_compute_mixture_nees_single_mode_matches_compute_nees() -> None:
    """A one-mode 'mixture' must reduce exactly to the plain single-Gaussian
    computation -- the weighted sum with one weight=1.0 term is the same
    quadratic form."""
    gt = np.zeros(6)
    mean = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cov = np.eye(6)
    mixture = consistency.compute_mixture_nees(
        gt, {"only": 1.0}, {"only": mean}, {"only": cov}
    )
    plain = consistency.compute_nees(mean - gt, cov)
    assert mixture.value == pytest.approx(plain.value)


def test_compute_mixture_nees_is_the_weighted_sum() -> None:
    gt = np.zeros(6)
    mean_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mean_b = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    cov = np.eye(6)
    weights = {"a": 0.25, "b": 0.75}
    means = {"a": mean_a, "b": mean_b}
    covs = {"a": cov, "b": cov}
    residual = consistency.compute_mixture_nees(gt, weights, means, covs)
    nees_a = float((mean_a - gt) @ np.linalg.solve(cov, mean_a - gt))
    nees_b = float((mean_b - gt) @ np.linalg.solve(cov, mean_b - gt))
    expected = 0.25 * nees_a + 0.75 * nees_b
    assert residual.value == pytest.approx(expected)
    assert residual.kind == "nees"
    assert "mixture-aware" in residual.note


def test_compute_mixture_nees_rejects_empty_mixture() -> None:
    with pytest.raises(consistency.PosteriorFamilyError):
        consistency.compute_mixture_nees(np.zeros(3), {}, {}, {})


# ---------------------------------------------------------------------------
# Day 22, Objective 1 -- mixture_density and empirical_coverage_by_sampling
# ---------------------------------------------------------------------------


def test_mixture_density_single_mode_matches_scipy_pdf() -> None:
    from scipy.stats import multivariate_normal as mvn

    mean = np.array([1.0, 2.0])
    cov = np.eye(2) * 0.5
    point = np.array([1.1, 1.9])
    density = consistency.mixture_density(
        point, {"only": 1.0}, {"only": mean}, {"only": cov}
    )
    expected = mvn.pdf(point, mean=mean, cov=cov)
    assert density[0] == pytest.approx(expected)


def test_mixture_density_is_the_weighted_sum_of_components() -> None:
    from scipy.stats import multivariate_normal as mvn

    mean_a, mean_b = np.zeros(2), np.array([3.0, 3.0])
    cov = np.eye(2)
    point = np.array([0.5, 0.5])
    weights = {"a": 0.3, "b": 0.7}
    density = consistency.mixture_density(
        point, weights, {"a": mean_a, "b": mean_b}, {"a": cov, "b": cov}
    )
    expected = 0.3 * mvn.pdf(point, mean=mean_a, cov=cov) + 0.7 * mvn.pdf(
        point, mean=mean_b, cov=cov
    )
    assert density[0] == pytest.approx(expected)


def test_empirical_coverage_requires_explicit_rng() -> None:
    with pytest.raises(consistency.PosteriorFamilyError, match="rng"):
        consistency.empirical_coverage_by_sampling(
            np.zeros(2), {"only": 1.0}, {"only": np.zeros(2)}, {"only": np.eye(2)}
        )


def test_empirical_coverage_true_at_the_mode_mean() -> None:
    rng = np.random.default_rng(0)
    mean = np.zeros(6)
    cov = np.eye(6)
    covered = consistency.empirical_coverage_by_sampling(
        mean, {"only": 1.0}, {"only": mean}, {"only": cov}, rng=rng
    )
    assert covered is True


def test_empirical_coverage_false_far_from_every_mode() -> None:
    rng = np.random.default_rng(0)
    mean = np.zeros(6)
    cov = np.eye(6) * 0.01  # tight -- a far point should clearly fall outside
    far = np.full(6, 50.0)
    covered = consistency.empirical_coverage_by_sampling(
        far, {"only": 1.0}, {"only": mean}, {"only": cov}, rng=rng
    )
    assert covered is False


def test_empirical_coverage_is_calibrated_for_a_single_gaussian_mode() -> None:
    """The sampling method's own self-check: for a mixture that is really
    just one Gaussian mode, GT drawn from that SAME Gaussian should be
    'covered' close to `confidence` of the time -- the same interpretation
    NEES pass rate has for a single-Gaussian posterior, reproduced here by
    a completely different (nonparametric) mechanism as a cross-check."""
    mean = np.zeros(4)
    cov = np.eye(4) * 0.5
    confidence = 0.95
    rng = np.random.default_rng(42)
    gt_draws = rng.multivariate_normal(mean, cov, size=300)
    covered_count = 0
    for gt in gt_draws:
        sample_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        if consistency.empirical_coverage_by_sampling(
            gt,
            {"only": 1.0},
            {"only": mean},
            {"only": cov},
            confidence=confidence,
            n_samples=1000,
            rng=sample_rng,
        ):
            covered_count += 1
    coverage = covered_count / len(gt_draws)
    assert coverage == pytest.approx(confidence, abs=0.06)


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


# ---------------------------------------------------------------------------
# Day 22, Objective 1 -- StateEstimate.mode_states / mode_components
# ---------------------------------------------------------------------------

_MODE_MEAN = tuple(np.zeros(6).tolist())
_MODE_COV = tuple(tuple(row) for row in np.eye(6).tolist())


def test_mode_states_without_mode_probabilities_raises() -> None:
    with pytest.raises(StateEstimateError, match="mode_probabilities"):
        _estimate(
            residuals=(consistency.no_observation_nis_stub(),),
            mode_states=(("only", _MODE_MEAN, _MODE_COV),),
        )


def test_mode_probabilities_without_mode_states_raises() -> None:
    with pytest.raises(StateEstimateError, match="mode_states"):
        _estimate(
            residuals=(consistency.no_observation_nis_stub(),),
            mode_probabilities=(("only", 1.0),),
            imm_config_sha="imm-1",
        )


def test_mode_states_name_mismatch_raises() -> None:
    with pytest.raises(StateEstimateError, match="must match"):
        _estimate(
            residuals=(consistency.no_observation_nis_stub(),),
            mode_probabilities=(("a", 1.0),),
            mode_states=(("b", _MODE_MEAN, _MODE_COV),),
            imm_config_sha="imm-1",
        )


def test_mode_states_wrong_mean_length_raises() -> None:
    with pytest.raises(StateEstimateError, match="mean must have"):
        _estimate(
            residuals=(consistency.no_observation_nis_stub(),),
            mode_probabilities=(("a", 1.0),),
            mode_states=(("a", (0.0, 0.0), _MODE_COV),),
            imm_config_sha="imm-1",
        )


def test_mode_states_wrong_cov_shape_raises() -> None:
    with pytest.raises(StateEstimateError, match="cov must be"):
        _estimate(
            residuals=(consistency.no_observation_nis_stub(),),
            mode_probabilities=(("a", 1.0),),
            mode_states=(("a", _MODE_MEAN, ((1.0, 0.0), (0.0, 1.0))),),
            imm_config_sha="imm-1",
        )


def test_mode_components_returns_weight_mean_cov_by_name() -> None:
    estimate = _estimate(
        residuals=(consistency.no_observation_nis_stub(),),
        mode_probabilities=(("a", 0.4), ("b", 0.6)),
        mode_states=(("a", _MODE_MEAN, _MODE_COV), ("b", _MODE_MEAN, _MODE_COV)),
        imm_config_sha="imm-1",
    )
    components = estimate.mode_components()
    assert components is not None
    assert set(components) == {"a", "b"}
    weight, mean, cov = components["a"]
    assert weight == pytest.approx(0.4)
    np.testing.assert_array_equal(mean, np.zeros(6))
    np.testing.assert_array_equal(cov, np.eye(6))


def test_mode_components_none_for_single_model_estimate() -> None:
    estimate = _estimate(residuals=(consistency.no_observation_nis_stub(),))
    assert estimate.mode_components() is None
