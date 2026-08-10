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
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

from src.estimator.state import ConsistencyResidual

FloatArray = npt.NDArray[np.float64]

DEFAULT_CONFIDENCE = 0.95
"""The fraction of updates a correctly-calibrated filter should fall within
bound. Not a tunable per-run knob — changing it changes what "in bound"
means for every consumer of a within_bound flag, so it is a module
constant, not a function parameter with a different default per call site."""

_NIS_CONSUMER = "filter self-check (this stage) and the accuracy scorecard's pass-rate summary"
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
    innovation: FloatArray, innovation_cov: FloatArray, confidence: float = DEFAULT_CONFIDENCE
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
    error: FloatArray, cov: FloatArray, confidence: float = DEFAULT_CONFIDENCE
) -> ConsistencyResidual:
    """NEES = error^T @ inv(cov) @ error, ``error`` = estimate - ground truth."""
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


def no_observation_nis_stub() -> ConsistencyResidual:
    """NIS for a predict-only step: no observation, so no innovation exists
    to score. Still an explicit residual (never an absent one) — see
    StateEstimate's STRUCTURAL requirement that a 'nis' entry is always
    present, computed or not."""
    return ConsistencyResidual(
        kind="nis",
        consumer=_NIS_CONSUMER,
        note="not applicable: no observation at this step (predicted forward, e.g. an occlusion gap or bootstrap)",
    )


def constraint_stub() -> ConsistencyResidual:
    return ConsistencyResidual(kind="constraint", consumer=_CONSTRAINT_CONSUMER, note="stub: not computed today")


def calibration_stub() -> ConsistencyResidual:
    return ConsistencyResidual(kind="calibration", consumer=_CALIBRATION_CONSUMER, note="stub: not computed today")


def coverage_stub() -> ConsistencyResidual:
    return ConsistencyResidual(kind="coverage", consumer=_COVERAGE_CONSUMER, note="stub: not computed today")


def stub_residuals() -> tuple[ConsistencyResidual, ...]:
    """The three always-routed stubs, in the order §15 names them."""
    return (constraint_stub(), calibration_stub(), coverage_stub())


def fraction_outside_bound(
    residuals: "list[ConsistencyResidual]", kind: str
) -> float:
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
