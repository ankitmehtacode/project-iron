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
            raise StateEstimateError(
                f"ConsistencyResidual.dof must be >= 1, got {self.dof}"
            )


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
            least one ``"nis"``-kind entry. See module docstring. An IMM
            estimate (Day 21) includes one combined ``"nis"`` entry plus
            one per-mode ``"nis"`` entry (distinguished by ``consumer``/
            ``note``, naming the mode) — the structural rule ("at least
            one nis entry") is unchanged; IMM simply emits more of them.
        imm_config_sha: ``None`` for a single-model estimate (Day 20,
            unchanged). Set for an IMM estimate (Day 21) — §15's
            comparability key extends to four shas when this is set; see
            :meth:`require_comparable`.
        mode_probabilities: ``None`` for a single-model estimate.
            ``((mode_name, probability), ...)`` for an IMM estimate,
            summing to 1.0 — the IMM's own directly-useful output (Day 21
            Objective 5), not just an internal quantity used to compute
            the combined mean/cov and then discarded.
        mode_states: ``None`` for a single-model estimate. For an IMM
            estimate (Day 22), ``((mode_name, mean, cov), ...)`` — the
            same per-mode Gaussian components :attr:`mode_probabilities`
            weights, in the same mode order. Added alongside the
            probabilities themselves (Day 22 Objective 1) because a
            mixture-aware consistency check needs each component's own
            ``(mean, cov)``, not just how much weight it carries — without
            this, computing a mixture-valid NEES or coverage figure would
            require reaching into IMM's private per-mode state
            (``_ImmAppendedState``) from outside the module. ``mean`` has
            length :data:`~src.estimator.motion_model.STATE_DIM`; ``cov``
            is ``STATE_DIM x STATE_DIM``, same shape rules as :attr:`mean`/
            :attr:`cov`.

            **Documented, not yet wired to, downstream consumers** (Day 21
            Objective 5 — intentionally documentation only): a high
            probability on a ``static``-family mode over a sustained span
            is the raw signal a POSE verb ("sat", "stood") or dwell
            detection would key off; a shift away from a ``static``-family
            mode is the raw signal motion-onset event triggering would key
            off; and the same aggregate is exactly what the motion gate
            scorecard's existing ``low_activity`` classification
            (``src/data/golden.py``, Day 15/16) is already trying to
            characterize from the outside, without the estimator's own
            view of it. None of these are implemented today.

            **Uncalibrated.** A mode probability is the IMM's own internal
            belief, not a validated confidence — Day 21 measured that the
            *combined estimate's* covariance is itself measurably
            overconfident in some regimes (see the Day-21 report), and
            mode probability has not been checked against real motion
            labels at all. Treat it as a strong hint for triage, not as a
            number safe to threshold and alert on, until that validation
            — which is its own, separate objective — has happened.
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
    imm_config_sha: str | None = None
    mode_probabilities: tuple[tuple[str, float], ...] | None = None
    mode_states: (
        tuple[tuple[str, tuple[float, ...], tuple[tuple[float, ...], ...]], ...] | None
    ) = None

    def __post_init__(self) -> None:
        if len(self.mean) != STATE_DIM:
            raise StateEstimateError(
                f"StateEstimate.mean must have {STATE_DIM} entries, got "
                f"{len(self.mean)}"
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
        if self.mode_probabilities is not None:
            if self.imm_config_sha is None:
                raise StateEstimateError(
                    "StateEstimate.mode_probabilities is set but imm_config_sha "
                    "is not -- a mode-probability output with no recorded IMM "
                    "config cannot be traced back to what produced it"
                )
            if not self.mode_probabilities:
                raise StateEstimateError(
                    "StateEstimate.mode_probabilities must not be an empty tuple "
                    "when set; pass None instead if there are no modes"
                )
            total = sum(p for _, p in self.mode_probabilities)
            if abs(total - 1.0) > 1e-6:
                raise StateEstimateError(
                    f"StateEstimate.mode_probabilities must sum to 1.0, got {total}"
                )
            if any(p < 0.0 or p > 1.0 for _, p in self.mode_probabilities):
                raise StateEstimateError(
                    f"StateEstimate.mode_probabilities entries must lie in "
                    f"[0, 1], got {self.mode_probabilities}"
                )
        # STRUCTURAL: mode_states and mode_probabilities travel together --
        # a mixture-aware consistency check needs both a mode's weight and
        # its own (mean, cov); either alone is not enough to reconstruct
        # the posterior mixture-aware code depends on.
        if self.mode_states is not None and self.mode_probabilities is None:
            raise StateEstimateError(
                "StateEstimate.mode_states is set but mode_probabilities is "
                "not -- per-mode components with no weight to combine them "
                "cannot describe a mixture posterior"
            )
        if self.mode_probabilities is not None and self.mode_states is None:
            raise StateEstimateError(
                "StateEstimate.mode_probabilities is set but mode_states is "
                "not -- an IMM estimate must carry each mode's own (mean, "
                "cov), not just its weight, or mixture-aware consistency "
                "checks (Day 22) have nothing to compute against"
            )
        if self.mode_states is not None:
            assert self.mode_probabilities is not None  # checked just above
            prob_names = {name for name, _ in self.mode_probabilities}
            state_names = {name for name, _, _ in self.mode_states}
            if prob_names != state_names:
                raise StateEstimateError(
                    f"StateEstimate.mode_states names {sorted(state_names)} must "
                    f"match mode_probabilities names {sorted(prob_names)}"
                )
            for name, mean, cov in self.mode_states:
                if len(mean) != STATE_DIM:
                    raise StateEstimateError(
                        f"StateEstimate.mode_states[{name!r}] mean must have "
                        f"{STATE_DIM} entries, got {len(mean)}"
                    )
                if len(cov) != STATE_DIM or any(len(row) != STATE_DIM for row in cov):
                    raise StateEstimateError(
                        f"StateEstimate.mode_states[{name!r}] cov must be "
                        f"{STATE_DIM}x{STATE_DIM}, got shape "
                        f"({len(cov)}, {[len(r) for r in cov]})"
                    )

    def require_comparable(self, other: "StateEstimate") -> None:
        """Raise unless both states were produced by the same models/rule.

        Same pattern as ``MeasuredEnvelope.require_comparable`` and
        ``Scorecard.require_comparable`` elsewhere in this codebase (§15).
        Extends to ``imm_config_sha`` (Day 21): ``None`` compares equal to
        ``None`` (two single-model estimates), but a single-model estimate
        is never comparable to an IMM one, and two IMM estimates are only
        comparable under the same transition-matrix/mode configuration.
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
                ("imm_config_sha", self.imm_config_sha, other.imm_config_sha),
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

    def dominant_mode(self) -> tuple[str, float] | None:
        """The IMM mode with the highest probability, or ``None`` for a
        single-model estimate (``mode_probabilities is None``).

        A convenience for the documented-but-unwired consumers described
        on :attr:`mode_probabilities` — most of them want "which mode is
        this, right now", not the full distribution. Still uncalibrated;
        see that attribute's docstring before using this for anything
        alerting-adjacent.
        """
        if self.mode_probabilities is None:
            return None
        return max(self.mode_probabilities, key=lambda pair: pair[1])

    def mode_components(self) -> dict[str, tuple[float, FloatArray, FloatArray]] | None:
        """``{mode_name: (weight, mean_array, cov_array)}``, or ``None`` for
        a single-model estimate (``mode_probabilities is None``).

        The array form of :attr:`mode_probabilities` + :attr:`mode_states`
        combined -- what :mod:`src.estimator.consistency`'s mixture-aware
        functions (Day 22 Objective 1) actually consume, so callers do not
        each re-implement the same tuple-to-ndarray unpacking.
        """
        if self.mode_probabilities is None:
            return None
        assert self.mode_states is not None  # STRUCTURAL invariant, see __post_init__
        probs = dict(self.mode_probabilities)
        return {
            name: (
                probs[name],
                np.array(mean, dtype=np.float64),
                np.array(cov, dtype=np.float64),
            )
            for name, mean, cov in self.mode_states
        }
