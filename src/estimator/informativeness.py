"""Is a calibrated estimator's uncertainty actually INFORMATIVE?
(Day 31, Objective 1.)

The blind spot this closes
--------------------------
:mod:`src.estimator.consistency` asks whether reported uncertainty is
HONEST: are the standardized errors consistent with the covariance the
filter claims? That is a real question and this module does not replace
it. But it is only half of one, because **a constant-variance predictor
passes it by declining to estimate.**

Day 30 measured that exact failure. Config B's posterior velocity
uncertainty was pinned at its floor on 100.00% of scored frames of
v6-motion — a constant 1.5 m/s, on every frame of every regime — while
config A's own natural sigma_v converges to 0.25-0.60 m/s. Config B's
cessation coverage moved 0.5013 -> 0.9946 and the directional criterion
recorded a PASS. A filter that is never confident cannot be caught being
overconfident.

This module supplies the second half, mirroring the Day-12 baseline rule
(a number without a trivial baseline is not a result) applied to the
SECOND moment rather than the first.

The baseline: the best constant-variance predictor there is
--------------------------------------------------------------
:func:`fit_constant_covariance` fits a block-isotropic constant
covariance — one variance for the three position axes, one for the three
velocity axes — to the estimator's own errors by maximum likelihood. The
baseline then predicts with **the estimator's own means** and that
constant covariance, so the comparison isolates the second moment and
nothing else.

Two deliberate choices, both stated because both make the baseline
STRONGER than it has any right to be:

1. **It is fitted on the data it is scored against.** No deployable
   predictor gets that. The baseline is an oracle.
2. **It is fitted by maximum likelihood, not "tuned until calibrated."**
   Among all constant-covariance predictors the MLE one maximizes the log
   score, so it dominates any calibration-tuned constant. A margin
   measured against it is therefore a LOWER BOUND on the margin against
   the cheapest-way-to-pass predictor the objective actually names.
   :func:`constant_baseline_coverage` reports the fitted baseline's own
   coverage so a reader can confirm it does in fact pass the calibration
   test — i.e. that it really is a cheapest-way-to-pass predictor.

Consequence: a small positive margin is not impressive, and a margin at
or near zero is damning.

Why block-isotropic and not one scalar
----------------------------------------
The state is ``[x, y, z, vx, vy, vz]`` — three metres and three
metres-per-second. A single scalar variance across all six is
dimensionally incoherent; it would be comparing a position error in m^2
against a velocity error in (m/s)^2 through one number. NEES gets away
with a scalar output only because it divides by the estimator's OWN
covariance, which carries the block structure. A constant baseline has to
carry it too, so it has two parameters instead of one. Two is still
trivial.

Why expected log predictive density, and not a variance ratio
----------------------------------------------------------------
The obvious alternative is a mean predictive-variance ratio — "how much
tighter is the estimator than the constant." It is rejected here for one
reason: **it is not a proper scoring rule.** A filter that reports
absurdly tiny covariance everywhere would score arbitrarily well on it
while being wildly overconfident. That failure is caught by the
calibration test, so a variance ratio is only safe when read TOGETHER
with calibration — and the whole lesson of Day 30 is that two numbers
which must be read together get separated.

Expected log predictive density (ELPD) is strictly proper: in
expectation it is maximized by, and only by, the true predictive
distribution. Over- and under-confidence are both penalized, so the
single number cannot be gamed in either direction. It is also almost
free here — for a Gaussian,

    log N(x; mean, P) = -0.5 * (d*log(2*pi) + log det P + NEES)

and NEES is already computed for every scored frame. Only ``log det P``
is new.

Units are nats per frame. The margin is a difference of log densities, so
it reads directly as a log likelihood ratio per frame: +0.7 nats means
the estimator's uncertainty makes the observed truth about 2x more likely
per frame than the best constant does.

Mixture posteriors are scored as mixtures
--------------------------------------------
IMM's predictive distribution is a Gaussian MIXTURE, and Day 22
established that collapsing it to one (mean, cov) and testing Gaussianity
tests an assumption the estimator explicitly violates. ELPD has no such
problem — the log density of a mixture is exact and cheap
(:func:`mixture_log_density`, via log-sum-exp) — so IMM is scored on the
density it actually reports, not a collapsed diagnostic. This is the first
consistency-adjacent number in this project that is valid for both
posterior families without a separate mixture-only variant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

POSITION_DOF = 3
VELOCITY_DOF = 3
STATE_DOF = POSITION_DOF + VELOCITY_DOF

UNINFORMATIVE_MARGIN_NATS = 0.05
"""At or below this ELPD margin, an estimator's per-frame uncertainty is
not distinguishable from the best constant — it is passing calibration by
refusing to estimate.

Declared here, before any configuration is scored, so the threshold is
not chosen after seeing which side of it config B lands on. 0.05 nats per
frame corresponds to a likelihood ratio of about 1.05x, i.e. the
estimator's whole per-frame uncertainty model buys ~5% more likelihood
than a single fitted constant does. Anything that small is noise on a
few-hundred-frame regime, not a capability.

This is a REPORTING threshold, not a bound on anything physical — the
margin itself is the measurement and is always reported as a number."""


class InformativenessError(ValueError):
    """A calibration figure was emitted without its informativeness
    margin, or an informativeness margin was computed from inputs that
    cannot support one."""


@dataclass(frozen=True)
class ConstantCovarianceBaseline:
    """The best constant-variance predictor for one set of errors.

    Attributes:
        position_variance_m2: MLE isotropic position variance, m^2.
        velocity_variance_m2s2: MLE isotropic velocity variance, (m/s)^2.
        n: Frames it was fitted on.
        fitted_on: Human-readable scope of the fit, carried so a reader
            can tell a globally-fitted baseline from a per-regime one
            without tracing the call site. This matters more than it
            looks: a baseline fitted PER REGIME would be given the
            ground-truth regime label, which no deployable predictor has,
            and would understate the estimator's informativeness by
            handing the trivial predictor a real capability.
    """

    position_variance_m2: float
    velocity_variance_m2s2: float
    n: int
    fitted_on: str

    def __post_init__(self) -> None:
        for name in ("position_variance_m2", "velocity_variance_m2s2"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise InformativenessError(
                    f"ConstantCovarianceBaseline.{name} must be finite and "
                    f"positive, got {value}"
                )
        if self.n <= 0:
            raise InformativenessError(
                "a constant-covariance baseline cannot be fitted on zero frames"
            )

    @property
    def log_det(self) -> float:
        """``log det`` of the block-isotropic covariance."""
        return POSITION_DOF * math.log(self.position_variance_m2) + (
            VELOCITY_DOF * math.log(self.velocity_variance_m2s2)
        )

    def nees(self, position_sq_error: float, velocity_sq_error: float) -> float:
        """This baseline's NEES for one frame — the same quantity
        :func:`src.estimator.consistency.compute_nees` computes, for the
        constant covariance instead of the estimator's own."""
        return (
            position_sq_error / self.position_variance_m2
            + velocity_sq_error / self.velocity_variance_m2s2
        )

    def log_density(self, position_sq_error: float, velocity_sq_error: float) -> float:
        """Log predictive density at the true state, nats."""
        return -0.5 * (
            STATE_DOF * math.log(2.0 * math.pi)
            + self.log_det
            + self.nees(position_sq_error, velocity_sq_error)
        )


def fit_constant_covariance(
    position_sq_errors: Sequence[float],
    velocity_sq_errors: Sequence[float],
    fitted_on: str,
) -> ConstantCovarianceBaseline:
    """Maximum-likelihood block-isotropic constant covariance.

    For an isotropic Gaussian over ``k`` axes, the MLE variance is the
    mean squared error per axis — ``mean(|e|^2) / k``. Closed form, no
    optimizer, nothing to converge.

    Args:
        fitted_on: Recorded verbatim on the result. Pass the scope the fit
            actually covers ("v6-motion, all scored frames"), never the
            scope it will be USED on — see
            :class:`ConstantCovarianceBaseline` for why a per-regime fit
            would be a different, much stronger baseline.

    Raises:
        InformativenessError: on empty input, or if every error is
            exactly zero (a degenerate fit with no positive variance to
            report — real on a synthetic set only if something upstream
            has collapsed).
    """
    position = np.asarray(position_sq_errors, dtype=np.float64)
    velocity = np.asarray(velocity_sq_errors, dtype=np.float64)
    if position.size == 0 or position.size != velocity.size:
        raise InformativenessError(
            "fit_constant_covariance needs a non-empty, equal-length pair of "
            f"squared-error sequences, got {position.size} and {velocity.size}"
        )
    position_variance = float(np.mean(position)) / POSITION_DOF
    velocity_variance = float(np.mean(velocity)) / VELOCITY_DOF
    if position_variance <= 0.0 or velocity_variance <= 0.0:
        raise InformativenessError(
            "constant-covariance fit is degenerate: mean squared error is "
            f"zero (position {position_variance}, velocity {velocity_variance}). "
            "An estimator with exactly zero error everywhere has no "
            "uncertainty question to answer."
        )
    return ConstantCovarianceBaseline(
        position_variance_m2=position_variance,
        velocity_variance_m2s2=velocity_variance,
        n=int(position.size),
        fitted_on=fitted_on,
    )


def gaussian_log_density(nees: float, log_det_covariance: float) -> float:
    """Log density of a ``STATE_DOF``-dimensional Gaussian at the truth.

    Takes NEES rather than the error vector because NEES is exactly the
    Mahalanobis term and is already computed for every scored frame — so
    this adds one quantity (``log det``) to the evaluation rather than a
    parallel computation that could disagree with the consistency stage.
    """
    if not math.isfinite(nees) or not math.isfinite(log_det_covariance):
        raise InformativenessError(
            f"gaussian_log_density needs finite inputs, got nees={nees}, "
            f"log_det={log_det_covariance}"
        )
    return -0.5 * (STATE_DOF * math.log(2.0 * math.pi) + log_det_covariance + nees)


def mixture_log_density(
    truth: FloatArray,
    weights: Sequence[float],
    means: Sequence[FloatArray],
    covariances: Sequence[FloatArray],
) -> float:
    """Log density of a Gaussian MIXTURE evaluated at ``truth``, via
    log-sum-exp.

    Args:
        truth: The true state, ``[STATE_DOF]``. Not an error vector —
            each component measures its own offset from the truth, and
            passing a residual relative to the COMBINED mean would silently
            score every mode against the wrong point.
        weights: Mode probabilities. Normalised here, so they need not sum
            to one already.
        means: Per-mode means, each ``[STATE_DOF]``.
        covariances: Per-mode covariances, each ``[STATE_DOF, STATE_DOF]``.

    Exact, not a diagnostic: unlike NEES, a log density has no
    single-Gaussian assumption to violate, so IMM is scored here on the
    posterior it actually reports. That is the point of choosing a log
    score over a variance ratio.

    Raises:
        InformativenessError: if the mixture is empty, the three
            sequences disagree in length, the weights do not form a usable
            distribution, or a component covariance is not
            positive-definite.
    """
    if not weights or len(weights) != len(means) or len(weights) != len(covariances):
        raise InformativenessError(
            "mixture_log_density needs equal-length, non-empty weights/means/"
            f"covariances, got {len(weights)}/{len(means)}/{len(covariances)}"
        )
    total_weight = float(np.sum(weights))
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise InformativenessError(
            f"mixture weights must sum to a positive finite value, got {total_weight}"
        )
    truth_array = np.asarray(truth, dtype=np.float64)
    terms: list[float] = []
    for weight, mean, covariance in zip(weights, means, covariances):
        if weight <= 0.0:
            continue
        offset = np.asarray(mean, dtype=np.float64) - truth_array
        sign, log_det = np.linalg.slogdet(np.asarray(covariance, dtype=np.float64))
        if sign <= 0:
            raise InformativenessError(
                "a mixture component has non-positive-definite covariance; "
                "its log density is undefined"
            )
        quadratic = float(offset @ np.linalg.solve(covariance, offset))
        terms.append(
            math.log(weight / total_weight)
            - 0.5 * (STATE_DOF * math.log(2.0 * math.pi) + float(log_det) + quadratic)
        )
    if not terms:
        raise InformativenessError("every mixture component had zero weight")
    return float(np.logaddexp.reduce(np.array(terms, dtype=np.float64)))


def constant_baseline_coverage(
    baseline: ConstantCovarianceBaseline,
    position_sq_errors: Sequence[float],
    velocity_sq_errors: Sequence[float],
    chi2_bound: float,
) -> float:
    """The fitted constant baseline's OWN empirical coverage.

    Reported alongside every margin so a reader can confirm the baseline
    really is a cheapest-way-to-pass predictor rather than a straw man. If
    this number sits near nominal, the MLE-fitted constant and the
    "tuned until calibrated" constant the objective names are the same
    predictor, and the margin measured against it answers the objective's
    question directly.

    Args:
        chi2_bound: The same 95% chi-square bound the estimator's own
            coverage is judged against, for ``STATE_DOF`` degrees of
            freedom — passed in rather than recomputed so the two coverage
            figures cannot end up judged against different bounds.
    """
    position = np.asarray(position_sq_errors, dtype=np.float64)
    velocity = np.asarray(velocity_sq_errors, dtype=np.float64)
    if position.size == 0:
        return float("nan")
    nees = (
        position / baseline.position_variance_m2
        + velocity / baseline.velocity_variance_m2s2
    )
    return float(np.mean(nees <= chi2_bound))


@dataclass(frozen=True)
class CalibrationAndInformativeness:
    """STRUCTURAL: a calibration figure and its informativeness margin,
    or neither.

    Every field is required and undefaulted, so there is no way to
    construct this carrying a coverage number and no margin — the same
    co-emission rule ``src/data/scorecard.py`` already enforces between
    ``gate.wake_fraction`` and ``gate.recall_retained``, and for the same
    reason: two numbers that are only meaningful together will be
    separated by someone quoting the flattering one.

    Day 30 is the worked example. Config B's cessation coverage of 0.9946
    was quoted, adopted, and defended across three ADR revisions. Its
    velocity uncertainty was a constant the whole time, and nothing in the
    report format made that visible next to the number it explained.

    Attributes:
        coverage_95: Empirical 95% coverage of the estimator.
        baseline_coverage_95: The fitted constant baseline's own coverage
            — reported so a reader can confirm the baseline really does
            pass the calibration test, i.e. that it is a
            cheapest-way-to-pass predictor and not a straw man.
        elpd_nats: Estimator's mean log predictive density, nats/frame.
        baseline_elpd_nats: The constant baseline's, same units.
        n: Frames both figures are computed over.
    """

    coverage_95: float
    baseline_coverage_95: float
    elpd_nats: float
    baseline_elpd_nats: float
    n: int

    @property
    def elpd_margin_nats(self) -> float:
        """Estimator minus baseline. Positive = the per-frame uncertainty
        carries information a constant cannot."""
        return self.elpd_nats - self.baseline_elpd_nats

    @property
    def informative(self) -> bool:
        """Whether the margin clears :data:`UNINFORMATIVE_MARGIN_NATS`."""
        margin = self.elpd_margin_nats
        return bool(math.isfinite(margin) and margin > UNINFORMATIVE_MARGIN_NATS)

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "n": self.n,
            "coverage_95": self.coverage_95,
            "baseline_coverage_95": self.baseline_coverage_95,
            "elpd_nats": self.elpd_nats,
            "baseline_elpd_nats": self.baseline_elpd_nats,
            "elpd_margin_nats": self.elpd_margin_nats,
            "informative": self.informative,
        }


def require_informativeness(block: dict[str, object], where: str) -> None:
    """Raise unless ``block`` carries an informativeness margin alongside
    its coverage figure.

    The runtime half of the co-emission rule, for the dict-shaped report
    blocks that cross into JSON and back — where the dataclass's own
    required fields cannot reach. Mirrors
    ``src/data/scorecard.py``'s ``gate.wake_fraction`` pairing check.

    Raises:
        InformativenessError: if a coverage figure is present without a
            margin beside it.
    """
    has_coverage = "coverage_95" in block or "empirical_coverage_95" in block
    if not has_coverage:
        return
    if "elpd_margin_nats" not in block:
        raise InformativenessError(
            f"{where}: a calibration figure was emitted with no "
            "informativeness margin beside it. A constant-variance "
            "predictor passes calibration by declining to estimate (Day 30, "
            "config B), so coverage alone cannot say whether an estimator "
            "is doing anything. Emit both or neither."
        )
