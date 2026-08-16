"""Day 31, Objective 1 — informativeness, and the co-emission rule.

The structural claim under test: a calibration figure cannot be emitted
without its informativeness margin. The behavioural claim: a
constant-variance predictor scores a margin of zero against itself, and a
genuinely informative estimator scores a positive one — so the metric
distinguishes the two cases Day 30's calibration criterion could not.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.estimator.consistency import chi2_upper_bound
from src.estimator.informativeness import (
    STATE_DOF,
    UNINFORMATIVE_MARGIN_NATS,
    CalibrationAndInformativeness,
    ConstantCovarianceBaseline,
    InformativenessError,
    constant_baseline_coverage,
    fit_constant_covariance,
    gaussian_log_density,
    mixture_log_density,
    require_informativeness,
)


def _errors(rng: np.random.Generator, n: int, sigma_p: float, sigma_v: float):
    position = rng.normal(scale=sigma_p, size=(n, 3))
    velocity = rng.normal(scale=sigma_v, size=(n, 3))
    return (
        [float(e @ e) for e in position],
        [float(e @ e) for e in velocity],
    )


# --------------------------------------------------------------------------
# The baseline
# --------------------------------------------------------------------------


def test_fit_recovers_the_variance_it_was_generated_with() -> None:
    """The MLE for an isotropic Gaussian is mean(|e|^2)/k. If this did not
    recover a known sigma, every margin below would be measured against
    the wrong reference."""
    rng = np.random.default_rng(31)
    position_sq, velocity_sq = _errors(rng, 40_000, sigma_p=0.3, sigma_v=1.1)
    baseline = fit_constant_covariance(position_sq, velocity_sq, "test")
    assert baseline.position_variance_m2 == pytest.approx(0.09, rel=0.03)
    assert baseline.velocity_variance_m2s2 == pytest.approx(1.21, rel=0.03)
    assert baseline.n == 40_000
    assert baseline.fitted_on == "test"


def test_fit_rejects_empty_and_mismatched_input() -> None:
    with pytest.raises(InformativenessError, match="non-empty"):
        fit_constant_covariance([], [], "test")
    with pytest.raises(InformativenessError, match="non-empty"):
        fit_constant_covariance([1.0, 2.0], [1.0], "test")


def test_fit_rejects_a_degenerate_zero_error_fit() -> None:
    """Zero error everywhere has no uncertainty question to answer, and a
    zero variance would make every log density infinite."""
    with pytest.raises(InformativenessError, match="degenerate"):
        fit_constant_covariance([0.0, 0.0], [0.0, 0.0], "test")


def test_baseline_rejects_non_positive_variance() -> None:
    with pytest.raises(InformativenessError, match="positive"):
        ConstantCovarianceBaseline(0.0, 1.0, 5, "test")


def test_the_fitted_baseline_is_itself_calibrated() -> None:
    """The baseline has to be a cheapest-way-to-PASS predictor, not a
    straw man. On Gaussian errors the MLE constant should land near
    nominal coverage — which is what makes it the same predictor the
    objective describes as 'tuned to be calibrated'."""
    rng = np.random.default_rng(131)
    position_sq, velocity_sq = _errors(rng, 20_000, sigma_p=0.3, sigma_v=1.1)
    baseline = fit_constant_covariance(position_sq, velocity_sq, "test")
    coverage = constant_baseline_coverage(
        baseline, position_sq, velocity_sq, chi2_upper_bound(STATE_DOF, 0.95)
    )
    assert coverage == pytest.approx(0.95, abs=0.02)


# --------------------------------------------------------------------------
# The margin: does it separate an estimator from a constant?
# --------------------------------------------------------------------------


def test_a_constant_variance_predictor_scores_a_zero_margin_against_itself() -> None:
    """The defining case. An estimator whose covariance is exactly the
    fitted constant carries no information the constant does not, and its
    margin must be zero — not small, zero."""
    rng = np.random.default_rng(231)
    position_sq, velocity_sq = _errors(rng, 5_000, sigma_p=0.4, sigma_v=0.9)
    baseline = fit_constant_covariance(position_sq, velocity_sq, "test")

    estimator = [
        gaussian_log_density(baseline.nees(p, v), baseline.log_det)
        for p, v in zip(position_sq, velocity_sq)
    ]
    constant = [baseline.log_density(p, v) for p, v in zip(position_sq, velocity_sq)]
    assert float(np.mean(estimator) - np.mean(constant)) == pytest.approx(
        0.0, abs=1e-12
    )


def test_an_estimator_that_knows_the_per_frame_variance_scores_positive() -> None:
    """The other direction, and the falsifiability half: if a genuinely
    informative estimator did NOT score positive, the metric would be
    measuring nothing."""
    rng = np.random.default_rng(331)
    n = 20_000
    # Half the frames are ten times noisier. A constant cannot express
    # that; a per-frame covariance can.
    scales = np.where(rng.random(n) < 0.5, 0.1, 1.0)
    position = rng.normal(scale=scales[:, None], size=(n, 3))
    velocity = rng.normal(scale=scales[:, None], size=(n, 3))
    position_sq = [float(e @ e) for e in position]
    velocity_sq = [float(e @ e) for e in velocity]

    baseline = fit_constant_covariance(position_sq, velocity_sq, "test")
    informed = [
        gaussian_log_density((p + v) / scale**2, STATE_DOF * math.log(scale**2))
        for p, v, scale in zip(position_sq, velocity_sq, scales)
    ]
    constant = [baseline.log_density(p, v) for p, v in zip(position_sq, velocity_sq)]
    margin = float(np.mean(informed) - np.mean(constant))
    assert margin > UNINFORMATIVE_MARGIN_NATS
    assert margin > 1.0, "a 10x variance split should be worth well over a nat"


def test_an_over_hedged_estimator_scores_a_negative_margin() -> None:
    """Config B's actual measured shape (Day 31: -0.94 to -1.11 nats on
    v6-motion). Reporting a variance far larger than the errors warrant
    passes calibration and is WORSE than the best constant — the log
    score is proper, so it penalizes that rather than rewarding it the
    way a variance ratio would."""
    rng = np.random.default_rng(431)
    position_sq, velocity_sq = _errors(rng, 5_000, sigma_p=0.3, sigma_v=0.3)
    baseline = fit_constant_covariance(position_sq, velocity_sq, "test")

    over_hedged = ConstantCovarianceBaseline(
        baseline.position_variance_m2 * 25.0,
        baseline.velocity_variance_m2s2 * 25.0,
        baseline.n,
        "over-hedged",
    )
    margin = float(
        np.mean(
            [
                over_hedged.log_density(p, v) - baseline.log_density(p, v)
                for p, v in zip(position_sq, velocity_sq)
            ]
        )
    )
    assert margin < 0.0
    # ...while its coverage is better than nominal, i.e. it passes
    # calibration exactly as config B did.
    coverage = constant_baseline_coverage(
        over_hedged, position_sq, velocity_sq, chi2_upper_bound(STATE_DOF, 0.95)
    )
    assert coverage > 0.95


# --------------------------------------------------------------------------
# Mixtures
# --------------------------------------------------------------------------


def test_a_one_component_mixture_equals_the_gaussian_density() -> None:
    """Sanity anchor between the two density paths: with one mode, the
    mixture formula must reduce exactly to the Gaussian one, or IMM's
    numbers are not comparable with the single-model ones."""
    truth = np.array([1.0, -2.0, 0.5, 0.1, 0.0, -0.3])
    mean = np.array([1.2, -1.7, 0.4, 0.0, 0.2, -0.1])
    cov = np.diag([0.04, 0.04, 0.04, 0.25, 0.25, 0.25])
    error = mean - truth
    nees = float(error @ np.linalg.solve(cov, error))
    _, log_det = np.linalg.slogdet(cov)

    assert mixture_log_density(truth, [1.0], [mean], [cov]) == pytest.approx(
        gaussian_log_density(nees, float(log_det))
    )


def test_mixture_weights_need_not_be_normalised() -> None:
    truth = np.zeros(6)
    means = [np.zeros(6), np.ones(6) * 0.1]
    covs = [np.eye(6), np.eye(6) * 2.0]
    assert mixture_log_density(truth, [0.5, 0.5], means, covs) == pytest.approx(
        mixture_log_density(truth, [5.0, 5.0], means, covs)
    )


def test_mixture_rejects_malformed_input() -> None:
    truth = np.zeros(6)
    with pytest.raises(InformativenessError, match="equal-length"):
        mixture_log_density(truth, [], [], [])
    with pytest.raises(InformativenessError, match="equal-length"):
        mixture_log_density(truth, [1.0], [np.zeros(6), np.zeros(6)], [np.eye(6)])
    with pytest.raises(InformativenessError, match="positive finite"):
        mixture_log_density(truth, [0.0], [np.zeros(6)], [np.eye(6)])


def test_mixture_rejects_a_singular_component() -> None:
    with pytest.raises(InformativenessError, match="positive-definite"):
        mixture_log_density(np.zeros(6), [1.0], [np.zeros(6)], [np.zeros((6, 6))])


def test_gaussian_log_density_rejects_non_finite_input() -> None:
    with pytest.raises(InformativenessError, match="finite"):
        gaussian_log_density(float("nan"), 0.0)


# --------------------------------------------------------------------------
# STRUCTURAL: co-emission
# --------------------------------------------------------------------------


def test_a_calibration_figure_cannot_be_constructed_without_a_margin() -> None:
    """The structural half. Every field is required and undefaulted, so
    there is no representation of 'coverage, and no informativeness' —
    the same rule scorecard.py enforces between gate.wake_fraction and
    gate.recall_retained, for the same reason."""
    with pytest.raises(TypeError):
        CalibrationAndInformativeness(coverage_95=0.99, n=100)  # type: ignore[call-arg]


def test_emitting_coverage_without_a_margin_raises() -> None:
    """The runtime half, for the dict-shaped report blocks that cross
    into JSON where a dataclass's required fields cannot reach."""
    with pytest.raises(InformativenessError, match="no\n?\\s*informativeness margin"):
        require_informativeness({"coverage_95": 0.99}, "by_regime.cessation")
    with pytest.raises(InformativenessError):
        require_informativeness({"empirical_coverage_95": 0.99}, "pooled")


def test_emitting_both_is_fine_and_emitting_neither_is_fine() -> None:
    """A block with no calibration figure at all has nothing to pair —
    the rule is 'both or neither', not 'always both'."""
    require_informativeness({"coverage_95": 0.99, "elpd_margin_nats": 0.4}, "ok")
    require_informativeness({"n": 0}, "empty regime")


def test_the_margin_and_the_informative_flag_agree_with_the_threshold() -> None:
    # Exactly AT the threshold is not informative -- the docstring says
    # "at or below", so the comparison is strict. Built by subtracting
    # from zero so the float arithmetic is exact and the boundary is
    # actually tested rather than approached.
    at_threshold = CalibrationAndInformativeness(
        coverage_95=0.95,
        baseline_coverage_95=0.95,
        elpd_nats=UNINFORMATIVE_MARGIN_NATS,
        baseline_elpd_nats=0.0,
        n=10,
    )
    assert at_threshold.elpd_margin_nats == UNINFORMATIVE_MARGIN_NATS
    assert not at_threshold.informative

    below = CalibrationAndInformativeness(
        coverage_95=0.95,
        baseline_coverage_95=0.95,
        elpd_nats=-0.9,
        baseline_elpd_nats=0.0,
        n=10,
    )
    assert not below.informative

    clearly_over = CalibrationAndInformativeness(
        coverage_95=0.95,
        baseline_coverage_95=0.95,
        elpd_nats=2.0,
        baseline_elpd_nats=1.0,
        n=10,
    )
    assert clearly_over.informative
    assert clearly_over.as_dict()["elpd_margin_nats"] == pytest.approx(1.0)
