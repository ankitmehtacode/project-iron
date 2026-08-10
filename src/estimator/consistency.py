"""NIS/NEES and the routed stub residuals — §15 stage 4, always runs.

NIS (normalized innovation squared) is the filter's own self-check: it
compares what the filter predicted a measurement would be against what it
actually got, scaled by how uncertain the filter thought that prediction
was. It needs no ground truth and runs live, on every update.

NEES (normalized estimation error squared) compares the filter's estimate
against ground truth directly. It needs GT and therefore cannot run live in
production — no deployed filter has access to it — so it is an
*evaluation-time* computation (:mod:`scripts.eval_estimator`), not something
attached to a live :class:`~src.estimator.state.StateEstimate`. Confusing
the two would be circular: a filter that graded itself against the answer it
already produced could never appear overconfident.

Both are chi-square distributed under a correctly-calibrated filter (Bar-
Shalom et al.), so :func:`chi2_upper_bound` gives the value a healthy filter
should stay under some fraction of the time. **An overconfident filter shows
NIS/NEES ABOVE this bound too often** — it is claiming tighter uncertainty
than its actual errors support, which is the continuous-state analogue of
"reports 92% confidence, right 70% of the time" (§10).

Posterior families, and why this module now raises instead of guessing
(Day 22, Objective 1)
------------------------------------------------------------------------
The chi-square identity above is a fact about a **single Gaussian**
posterior. Day 21's IMM output is a **Gaussian mixture** — several modes,
each with its own ``(mean, cov)``, blended by mode probability. Collapsing
that mixture to one first-two-moments ``(mean, cov)`` (as
:func:`~src.estimator.imm._combine` does for the reported estimate) and
then running :func:`compute_nees` on the result tests a Gaussianity
assumption the estimator explicitly violates — a genuinely bimodal
mixture, collapsed this way, can produce a combined covariance that
matches neither mode's actual spread, so "pooled NEES worse" may be
measuring the collapse, not the filter. This is the ninth instance in
this project of an instrument, not the code under test, being the defect
(see ``FOUNDATION_REPORT.md``, Day 22) — and the last one this module
lets happen silently: :func:`compute_nees` now requires its caller to
name the posterior family it is being applied to, and raises
:class:`PosteriorFamilyError` for anything other than ``"gaussian"``.
:func:`compute_mixture_nees` and :func:`empirical_coverage_by_sampling`
are the two mixture-valid alternatives (Day 22 Objective 1(b) and 1(a)
respectively); :func:`compute_collapsed_gaussian_nees_diagnostic` keeps
the old (invalid-for-a-mixture) number available, explicitly labeled, for
side-by-side comparison rather than silently deleting it.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2, multivariate_normal

from src.estimator.state import ConsistencyResidual

FloatArray = npt.NDArray[np.float64]

PosteriorFamily = Literal["gaussian", "gaussian_mixture"]
"""Which shape of posterior a consistency check is being applied to.
``compute_nees``/``compute_nis`` are valid only for ``"gaussian"`` --
see the module docstring."""


class PosteriorFamilyError(ValueError):
    """Raised when a consistency check is applied to a posterior family it
    was not derived for -- e.g. a single-Gaussian NEES run against a
    Gaussian-mixture (IMM) posterior collapsed to one (mean, cov)."""


DEFAULT_CONFIDENCE = 0.95
"""The fraction of updates a correctly-calibrated filter should fall within
bound. Not a tunable per-run knob — changing it changes what "in bound"
means for every consumer of a within_bound flag, so it is a module
constant, not a function parameter with a different default per call site."""

_NIS_CONSUMER = (
    "filter self-check (this stage) and the accuracy scorecard's pass-rate summary"
)
_NEES_CONSUMER = "accuracy evaluation against exact GT (scripts/eval_estimator.py)"
_CONSTRAINT_CONSUMER = "twin-revision hypothesis (not yet implemented)"
_CALIBRATION_CONSUMER = "recalibration event (not yet implemented)"
_COVERAGE_CONSUMER = "envelope drift (not yet implemented)"


def chi2_upper_bound(dof: int, confidence: float = DEFAULT_CONFIDENCE) -> float:
    """The value below which ``confidence`` fraction of a chi-square(dof)
    variable falls, e.g. the 95th percentile."""
    if dof < 1:
        raise ValueError(f"dof must be >= 1, got {dof}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return float(chi2.ppf(confidence, dof))


def compute_nis(
    innovation: FloatArray,
    innovation_cov: FloatArray,
    confidence: float = DEFAULT_CONFIDENCE,
) -> ConsistencyResidual:
    """NIS = innovation^T @ inv(S) @ innovation, for one update step."""
    dof = int(innovation.shape[0])
    value = float(innovation @ np.linalg.solve(innovation_cov, innovation))
    bound = chi2_upper_bound(dof, confidence)
    return ConsistencyResidual(
        kind="nis",
        consumer=_NIS_CONSUMER,
        value=value,
        dof=dof,
        chi2_bound=bound,
        within_bound=value <= bound,
    )


def compute_nees(
    error: FloatArray,
    cov: FloatArray,
    confidence: float = DEFAULT_CONFIDENCE,
    *,
    posterior_family: PosteriorFamily = "gaussian",
) -> ConsistencyResidual:
    """NEES = error^T @ inv(cov) @ error, ``error`` = estimate - ground truth.

    Valid only for ``posterior_family="gaussian"`` (the default, and the
    only value every pre-Day-22 caller ever passed implicitly). Raises
    :class:`PosteriorFamilyError` otherwise: see the module docstring for
    why a Gaussian-mixture posterior (IMM's combined estimate) cannot be
    scored this way. Use :func:`compute_mixture_nees` or
    :func:`empirical_coverage_by_sampling` for a ``gaussian_mixture``
    posterior, or :func:`compute_collapsed_gaussian_nees_diagnostic` if the
    old collapsed number is wanted anyway, explicitly labeled as invalid.
    """
    if posterior_family != "gaussian":
        raise PosteriorFamilyError(
            f"compute_nees is valid only for posterior_family='gaussian'; "
            f"got {posterior_family!r}. A gaussian_mixture posterior (e.g. "
            "IMM's combined estimate) must use compute_mixture_nees or "
            "empirical_coverage_by_sampling instead -- collapsing a mixture "
            "to one (mean, cov) and testing Gaussianity tests an assumption "
            "the estimator explicitly violates. See this module's docstring."
        )
    dof = int(error.shape[0])
    value = float(error @ np.linalg.solve(cov, error))
    bound = chi2_upper_bound(dof, confidence)
    return ConsistencyResidual(
        kind="nees",
        consumer=_NEES_CONSUMER,
        value=value,
        dof=dof,
        chi2_bound=bound,
        within_bound=value <= bound,
    )


_COLLAPSED_NEES_CONSUMER = (
    "Day-22 Objective-1 diagnostic ONLY -- NOT a valid consistency check for "
    "a gaussian_mixture posterior (see compute_nees's PosteriorFamilyError "
    "and this project's mixture-valid alternatives: compute_mixture_nees, "
    "empirical_coverage_by_sampling)"
)


def compute_collapsed_gaussian_nees_diagnostic(
    error: FloatArray, cov: FloatArray, confidence: float = DEFAULT_CONFIDENCE
) -> ConsistencyResidual:
    """The number Day 21 silently reported as IMM's "pooled NEES": a
    mixture collapsed to its first two moments, scored as if Gaussian.

    Kept, not deleted — Day 22 Objective 1 asks for it "alongside" the
    mixture-valid metrics precisely so a reader can see all three side by
    side and judge whether the collapse itself explains Day 21's result.
    Its own ``consumer`` field names itself invalid for judging the
    filter's consistency, so nothing downstream can mistake it for a
    validated check the way the unlabeled Day-21 number could.
    """
    residual = compute_nees(error, cov, confidence, posterior_family="gaussian")
    return ConsistencyResidual(
        kind=residual.kind,
        consumer=_COLLAPSED_NEES_CONSUMER,
        value=residual.value,
        dof=residual.dof,
        chi2_bound=residual.chi2_bound,
        within_bound=residual.within_bound,
        note=(
            "diagnostic only: mixture collapsed to (mean, cov), then "
            "tested as Gaussian"
        ),
    )


_MIXTURE_NEES_CONSUMER = (
    "accuracy evaluation against exact GT, mixture-aware "
    "(scripts/eval_estimator.py, Day 22 Objective 1)"
)


def compute_mixture_nees(
    gt: FloatArray,
    mode_weights: dict[str, float],
    mode_means: dict[str, FloatArray],
    mode_covs: dict[str, FloatArray],
    confidence: float = DEFAULT_CONFIDENCE,
) -> ConsistencyResidual:
    """Probability-weighted sum of each mode's OWN NEES against its OWN
    ``(mean, cov)`` — valid for a ``gaussian_mixture`` posterior in a way
    pooling the mixture into one ``(mean, cov)`` and calling
    :func:`compute_nees` is not.

    ``value = sum_i w_i * (mean_i - gt)^T @ inv(cov_i) @ (mean_i - gt)``.
    Under a correctly-calibrated mode ``i``, ``E[NEES_i] = dof`` (Bar-Shalom
    et al.), and mode weights sum to 1, so this weighted sum's expectation
    is still ``dof`` under a correctly-calibrated mixture —
    :func:`chi2_upper_bound` is used as the same reference bound, but this
    is declared **approximate, not exact**: a probability-weighted sum of
    (generally non-identical) chi-square variables is not itself
    chi-square distributed. Reported as one of two independent lines of
    mixture-valid evidence, alongside :func:`empirical_coverage_by_sampling`
    (which assumes no parametric form at all) — not as a replacement for it.

    Args:
        gt: Ground-truth state vector, same dimensionality as each mode's mean.
        mode_weights, mode_means, mode_covs: Keyed by mode name, same keys
            in all three — typically built from
            :meth:`~src.estimator.state.StateEstimate.mode_components`.
    """
    names = list(mode_weights)
    if not names:
        raise PosteriorFamilyError("compute_mixture_nees needs at least one mode")
    dof = int(gt.shape[0])
    value = 0.0
    for name in names:
        error = mode_means[name] - gt
        value += mode_weights[name] * float(
            error @ np.linalg.solve(mode_covs[name], error)
        )
    bound = chi2_upper_bound(dof, confidence)
    return ConsistencyResidual(
        kind="nees",
        consumer=_MIXTURE_NEES_CONSUMER,
        value=value,
        dof=dof,
        chi2_bound=bound,
        within_bound=value <= bound,
        note=(
            "mixture-aware: probability-weighted per-mode NEES; bound is "
            "approximate, not exact (see docstring)"
        ),
    )


def mixture_density(
    points: FloatArray,
    mode_weights: dict[str, float],
    mode_means: dict[str, FloatArray],
    mode_covs: dict[str, FloatArray],
) -> FloatArray:
    """Gaussian-mixture probability density at each row of ``points``
    (shape ``[N, dim]`` or ``[dim]``), evaluated exactly (no sampling) —
    the building block :func:`empirical_coverage_by_sampling` thresholds."""
    pts = np.atleast_2d(points)
    density = np.zeros(pts.shape[0], dtype=np.float64)
    for name, weight in mode_weights.items():
        density += weight * multivariate_normal.pdf(
            pts, mean=mode_means[name], cov=mode_covs[name], allow_singular=True
        )
    return density


MIXTURE_COVERAGE_DEFAULT_N_SAMPLES = 2000
"""Monte-Carlo sample count per frame for :func:`empirical_coverage_by_sampling`.
Large enough that the sampled 95th-density-percentile threshold is not
itself dominated by sampling noise; small enough that scoring several
thousand frames (v3-indoor's sustained regime alone: 2414) finishes in a
practical development-loop time. Not tuned to move any specific result —
see this function's docstring for what would happen if it were."""


def empirical_coverage_by_sampling(
    gt: FloatArray,
    mode_weights: dict[str, float],
    mode_means: dict[str, FloatArray],
    mode_covs: dict[str, FloatArray],
    confidence: float = DEFAULT_CONFIDENCE,
    n_samples: int = MIXTURE_COVERAGE_DEFAULT_N_SAMPLES,
    rng: np.random.Generator | None = None,
) -> bool:
    """Whether ``gt`` falls inside the mixture's empirical (nonparametric)
    ``confidence``-level credible region — Objective 1(a): "assumes no
    Gaussianity at all" about the REGION'S SHAPE (each component is still
    modeled as Gaussian; the region formed by mixing them is not assumed
    to be an ellipse the way a single Gaussian's would be).

    Method: draw ``n_samples`` from the mixture (component chosen by
    ``mode_weights``, then a draw from that component's own Gaussian).
    Compute each sample's mixture density (:func:`mixture_density`); the
    density value at the ``(1 - confidence)`` sample quantile is a
    Monte-Carlo estimate of the density threshold whose super-level set
    has probability mass ``confidence`` under the mixture — the standard
    nonparametric minimum-volume / highest-posterior-density (HPD) region
    construction. ``gt`` is "covered" iff its own mixture density is at or
    above that threshold.

    Aggregating this boolean's mean over many frames (the same way NEES
    pass rate is aggregated) gives an empirical coverage fraction directly
    comparable to :func:`fraction_outside_bound`'s NEES-based one, without
    inheriting NEES's single-Gaussian assumption.

    Args:
        rng: Caller-supplied generator for reproducibility — this function
            takes no default seed of its own, the same convention
            :mod:`scripts.eval_estimator` uses for observation noise
            (``EVAL_SEED``); an unseeded default would make two runs on
            the same data disagree on coverage by sampling noise alone.
    """
    if rng is None:
        raise PosteriorFamilyError(
            "empirical_coverage_by_sampling needs an explicit rng for "
            "reproducibility -- see this function's docstring"
        )
    names = list(mode_weights)
    if not names:
        raise PosteriorFamilyError(
            "empirical_coverage_by_sampling needs at least one mode"
        )
    probs = np.array([mode_weights[n] for n in names], dtype=np.float64)
    probs = probs / probs.sum()
    counts = rng.multinomial(n_samples, probs)
    draws = [
        rng.multivariate_normal(mode_means[name], mode_covs[name], size=int(count))
        for name, count in zip(names, counts)
        if count > 0
    ]
    samples = np.concatenate(draws, axis=0)
    sample_density = mixture_density(samples, mode_weights, mode_means, mode_covs)
    gt_density = float(mixture_density(gt, mode_weights, mode_means, mode_covs)[0])
    threshold = float(np.quantile(sample_density, 1.0 - confidence))
    return bool(gt_density >= threshold)


def no_observation_nis_stub() -> ConsistencyResidual:
    """NIS for a predict-only step: no observation, so no innovation exists
    to score. Still an explicit residual (never an absent one) — see
    StateEstimate's STRUCTURAL requirement that a 'nis' entry is always
    present, computed or not."""
    return ConsistencyResidual(
        kind="nis",
        consumer=_NIS_CONSUMER,
        note=(
            "not applicable: no observation at this step (predicted forward, "
            "e.g. an occlusion gap or bootstrap)"
        ),
    )


def constraint_stub() -> ConsistencyResidual:
    return ConsistencyResidual(
        kind="constraint",
        consumer=_CONSTRAINT_CONSUMER,
        note="stub: not computed today",
    )


def calibration_stub() -> ConsistencyResidual:
    return ConsistencyResidual(
        kind="calibration",
        consumer=_CALIBRATION_CONSUMER,
        note="stub: not computed today",
    )


def coverage_stub() -> ConsistencyResidual:
    return ConsistencyResidual(
        kind="coverage", consumer=_COVERAGE_CONSUMER, note="stub: not computed today"
    )


def stub_residuals() -> tuple[ConsistencyResidual, ...]:
    """The three always-routed stubs, in the order §15 names them."""
    return (constraint_stub(), calibration_stub(), coverage_stub())


def fraction_outside_bound(residuals: "list[ConsistencyResidual]", kind: str) -> float:
    """Share of ``kind`` residuals (nis or nees) that fell outside their bound.

    Only residuals with an actual computed value count — stubs and
    no-observation entries are excluded, since "outside bound" is
    undefined for a residual that was never computed. NaN when nothing of
    that kind was computed, same convention as
    :func:`src.eval.baselines.margin`.
    """
    scored = [r for r in residuals if r.kind == kind and r.within_bound is not None]
    if not scored:
        return float("nan")
    outside = sum(1 for r in scored if not r.within_bound)
    return outside / len(scored)
