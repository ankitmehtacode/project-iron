"""StateEstimate and ConsistencyResidual: the estimator's output shape (§15, §17).

Two rules, both structural
---------------------------
1. Two state estimates are comparable only when they agree on
   ``motion_model_sha``, ``measurement_model_sha``, and ``update_rule_sha``
   (§15) — :meth:`StateEstimate.require_comparable` raises otherwise, same
   pattern as ``MeasuredEnvelope.require_comparable`` and
   ``Scorecard.require_comparable`` elsewhere in this codebase.
2. A state estimate emitted without its consistency residuals is refused at
   construction (§10's calibration rule, extended to continuous state):
   ``residuals`` is a required field with no default, and it must contain at
   least one ``"nis"``-kind entry, because stage 4 of §15 always runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt

from src.estimator.motion_model import STATE_DIM

FloatArray = npt.NDArray[np.float64]

ConsistencyKind = Literal["nis", "nees", "constraint", "calibration", "coverage"]
"""Which family of self-check a residual belongs to.

``nis``/``nees`` are computed today (:mod:`src.estimator.consistency`). The
other three are routed as STUBS — present in the vocabulary, carrying no
numeric value, each naming the consumer that will read it once implemented:
``constraint`` -> twin-revision hypothesis, ``calibration`` -> recalibration
event, ``coverage`` -> envelope drift.
"""
CONSISTENCY_KINDS: tuple[ConsistencyKind, ...] = get_args(ConsistencyKind)


class StateEstimateError(ValueError):
    """Raised when a StateEstimate or ConsistencyResidual is malformed."""


@dataclass(frozen=True)
class ConsistencyResidual:
    """One typed residual from the always-on consistency stage.

    A "computed" residual (``value`` is not None) must set ``dof``,
    ``chi2_bound``, and ``within_bound`` together — a residual that reports
    a value with no bound to judge it against is not a consistency check,
    it is a number. A stub residual (nothing computed today) leaves all four
    as ``None`` and uses ``note``/``consumer`` to say why.
    """

    kind: ConsistencyKind
    consumer: str
    value: float | None = None
    dof: int | None = None
    chi2_bound: float | None = None
    within_bound: bool | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in CONSISTENCY_KINDS:
            raise StateEstimateError(
                f"unknown ConsistencyResidual.kind {self.kind!r}; expected one "
                f"of {CONSISTENCY_KINDS}"
            )
        if not self.consumer:
            raise StateEstimateError("ConsistencyResidual.consumer must not be empty")
        fields = (self.value, self.dof, self.chi2_bound, self.within_bound)
        provided = [f is not None for f in fields]
        if any(provided) and not all(provided):
            raise StateEstimateError(
                f"ConsistencyResidual(kind={self.kind!r}) must set value/dof/"
                "chi2_bound/within_bound all together or not at all -- a "
                "partial residual cannot be judged against its own bound"
            )
        if self.dof is not None and self.dof < 1:
            raise StateEstimateError(f"ConsistencyResidual.dof must be >= 1, got {self.dof}")


@dataclass(frozen=True)
class StateEstimate:
    """One resolved state — a marginal mean and covariance at ``ts_ns``.

    Attributes:
        ts_ns: When this estimate is valid for.
        mean: ``(x_m, y_m, z_m, vx_mps, vy_mps, vz_mps)``.
        cov: 6x6 covariance, row-major nested tuples (hashable/comparable by
            value, unlike a bare ndarray).
        observed: True if a real observation was incorporated at exactly
            ``ts_ns`` (a correction step); False if this is a prediction
            extrapolated forward through a gap with no observation.
        motion_model_sha, measurement_model_sha, update_rule_sha: §15's
            three-way comparability key. See :meth:`require_comparable`.
        graph_rev: Which :class:`~src.model.episode.StateGraph` revision
            produced this estimate — pins it to a specific, reproducible
            factor history.
        residuals: STRUCTURAL — required, non-empty, and must include at
            least one ``"nis"``-kind entry. See module docstring.
    """

    ts_ns: int
    mean: tuple[float, ...]
    """Length :data:`~src.estimator.motion_model.STATE_DIM`, checked at
    construction — kept variable-length in the type (rather than a fixed
    6-tuple annotation) because every numpy ``.tolist()`` conversion into
    this field would otherwise need an unchecked cast; the runtime check in
    ``__post_init__`` is the actual enforcement either way."""
    cov: tuple[tuple[float, ...], ...]
    """``STATE_DIM x STATE_DIM``, same reasoning as :attr:`mean`."""
    observed: bool
    motion_model_sha: str
    measurement_model_sha: str
    update_rule_sha: str
    graph_rev: int
    residuals: tuple[ConsistencyResidual, ...]

    def __post_init__(self) -> None:
        if len(self.mean) != STATE_DIM:
            raise StateEstimateError(
                f"StateEstimate.mean must have {STATE_DIM} entries, got {len(self.mean)}"
            )
        if len(self.cov) != STATE_DIM or any(len(row) != STATE_DIM for row in self.cov):
            raise StateEstimateError(
                f"StateEstimate.cov must be {STATE_DIM}x{STATE_DIM}, got shape "
                f"({len(self.cov)}, {[len(r) for r in self.cov]})"
            )
        for name, value in (
            ("motion_model_sha", self.motion_model_sha),
            ("measurement_model_sha", self.measurement_model_sha),
            ("update_rule_sha", self.update_rule_sha),
        ):
            if not value:
                raise StateEstimateError(f"StateEstimate.{name} must not be empty")
        if self.graph_rev < 0:
            raise StateEstimateError(
                f"StateEstimate.graph_rev must be >= 0, got {self.graph_rev}"
            )
        # STRUCTURAL: a state estimate emitted without its consistency
        # residuals raises. Stage 4 always runs (§15), so "no residuals at
        # all" and "residuals present but no NIS entry" are both refused —
        # NIS (or its no-observation stub) is the one residual every single
        # predict-or-update step produces, computed or not.
        if not self.residuals:
            raise StateEstimateError(
                "StateEstimate.residuals must not be empty. §15's stage 4 "
                "(consistency) always runs; a state estimate with no residuals "
                "attached was emitted without running it."
            )
        if not any(r.kind == "nis" for r in self.residuals):
            raise StateEstimateError(
                "StateEstimate.residuals must include an 'nis'-kind entry "
                "(computed, or a stub noting why it does not apply this step) "
                "-- every predict-or-update step produces one."
            )

    def require_comparable(self, other: "StateEstimate") -> None:
        """Raise unless both states were produced by the same three models/rule.

        Same pattern as ``MeasuredEnvelope.require_comparable`` and
        ``Scorecard.require_comparable`` elsewhere in this codebase (§15).
        """
        mismatches = [
            field
            for field, a, b in (
                ("motion_model_sha", self.motion_model_sha, other.motion_model_sha),
                (
                    "measurement_model_sha",
                    self.measurement_model_sha,
                    other.measurement_model_sha,
                ),
                ("update_rule_sha", self.update_rule_sha, other.update_rule_sha),
            )
            if a != b
        ]
        if mismatches:
            raise StateEstimateError(
                f"these state estimates are not comparable: {', '.join(mismatches)} "
                f"differ. Two states produced by different models/update rules "
                "are not the same measurement."
            )

    def mean_array(self) -> FloatArray:
        return np.array(self.mean, dtype=np.float64)

    def cov_array(self) -> FloatArray:
        return np.array(self.cov, dtype=np.float64)

    def position_m(self) -> FloatArray:
        return self.mean_array()[:3]

    def velocity_mps(self) -> FloatArray:
        return self.mean_array()[3:]
