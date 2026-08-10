"""Interacting Multiple Model (IMM) filtering over MotionModel (§15, Day 21).

Day 20 built one Kalman filter with one motion model per entity. Day 21's
diagnosis (see the Day-21 report section) found the specific failure a
single model cannot avoid: v3-indoor's steady walking and v4.1-gate's
motion-transient (cessation, specifically — see below) frames need
genuinely different process-noise regimes, and the innovation sequence is
measurably NOT white in both — direct evidence the problem is *which
model* is running, not how one model's Q is tuned (a mis-tuned but
correctly-specified filter still produces white innovations; a wrong
model does not). IMM runs several motion models in parallel, weighted by
how well each currently explains the observations, and blends their
outputs — the filter decides, frame by frame, whether "steady motion" or
"just changed" better describes what happened, instead of one Q trying to
cover both.

Three modes, sized to what Day 21 actually measured
-------------------------------------------------------
- ``static``: :class:`~src.estimator.motion_model.NearlyConstantPositionMotionModel`
  — position does not advance by velocity at all (no F coupling between
  them), not just a CV model with a tighter Q. That distinction turned out
  to matter: a first attempt reused ``asset_static``'s tight-Q *CV* model,
  which shares the same velocity-to-position F as every other mode, so a
  nonzero velocity mixed in from another mode kept propagating just as
  well as under ``constant_velocity`` — it only claimed a tighter
  covariance while doing so, which let it win the mode-probability
  contest on covariance width alone, including during genuine sustained
  walking. See ``tests/test_estimator_imm.py`` for the regression test.
- ``constant_velocity``: ``person``'s Day-20 model, unchanged. Wins during
  steady walking — the regime Day 20/21 already showed this model handles
  well alone (v3-indoor sustained: 99.4% NEES coverage).
- ``maneuvering``: a new, high-process-noise CV model
  (:class:`~src.estimator.motion_model.ConstantVelocityMotionModel`,
  ``kind="maneuvering"``). Wins during a transient — Day 21's measured
  failure was specifically **cessation** (a walker stopping), not onset
  (which was already well-calibrated under the single ``person`` model) —
  by tolerating far more unmodeled acceleration than the other two modes.

Not four, not five modes: Day 21's per-regime table found onset already
well-calibrated (v3-indoor: 96.5% coverage, white innovations) under the
single ``constant_velocity`` model. A fourth mode tuned for onset would
solve a problem the measurement did not show existed. Three modes covers
the actual measured failure without inventing capacity the evidence does
not justify.

The standard IMM cycle
-----------------------
Per update (Blackman & Popoli; Bar-Shalom, Li & Kirubarajan — the same
textbook :mod:`src.estimator.consistency`'s chi-square bounds come from):

1. **Mixing** — blend each mode's previous ``(mean, cov)`` using the
   transition probability matrix, producing one mixed initial condition
   per mode for this step.
2. **Mode-matched filtering** — one ordinary Kalman predict+update per
   mode, from its own mixed initial condition, using that mode's own
   motion model (H/R do not change per mode — the sensor did not change).
3. **Mode-probability update** — reweight each mode by the Gaussian
   likelihood of its own innovation, combined with the mixing-predicted
   probability.
4. **Combination** — the reported estimate is the probability-weighted
   mixture of every mode's ``(mean, cov)``.

STRUCTURAL — the prior firewall holds (§17)
----------------------------------------------
Mode probabilities are derived ONLY from observation likelihoods (step 3)
— never seeded from, or nudged by, any external belief about what the
target is "probably" doing. The initial mode distribution is uniform
(``1/n_modes`` each) at bootstrap: the least-informative distribution
available, not a behavioural assumption. :func:`run_imm_filter` takes no
parameter that could carry a prior — tested the same way Day 20's
single-model filter was
(``tests/test_estimator_imm.py::test_run_imm_filter_has_no_prior_parameter``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from src.estimator import consistency
from src.estimator.measurement_model import MeasurementModel
from src.estimator.motion_model import (
    STATE_DIM,
    ConstantVelocityMotionModel,
    MotionModel,
    NearlyConstantPositionMotionModel,
    apply_velocity_covariance_floor,
    motion_model_for,
)
from src.estimator.state import ConsistencyResidual, StateEstimate
from src.model.episode import Factor, StateGraph, StateQuery
from src.model.measurement import WorldPositionMeasurement
from src.model.observation import Observation
from src.model.ulid import generate_ulid

FloatArray = npt.NDArray[np.float64]

MANEUVERING_SIGMA_A_MPS2 = 8.0
"""Higher than every Day-20 kind (the previous maximum was
``asset_carried``'s 4.0 m/s^2) — deliberately: this mode exists to win the
mode-probability contest during a genuine transient, which needs a
process-noise budget wide enough that its predicted covariance actually
covers a cessation-magnitude residual. Day 21 measured NEES up to ~800
(against a chi-square(6) 95% bound of 12.6) in the frames immediately
after a walker stopped, under the single ``constant_velocity`` model —
this mode's Q is sized to make that residual plausible, not merely
"bigger", so the mode-probability update can actually favour it there."""

LARGE_VELOCITY_VARIANCE_MPS2 = 100.0
"""Same bootstrap convention as
:data:`src.estimator.filter.LARGE_VELOCITY_VARIANCE_MPS2` — no informative
prior on initial velocity, replicated identically across every mode."""

UPDATE_RULE_VERSION = "imm-kalman-joseph-form-v1"
UPDATE_RULE_SHA = hashlib.sha256(UPDATE_RULE_VERSION.encode("utf-8")).hexdigest()

_LIKELIHOOD_FLOOR = 1e-300
"""Numerical floor: when every mode finds an observation astronomically
unlikely, fall back to the mixing-predicted probabilities rather than
dividing by (effectively) zero."""


class ImmError(RuntimeError):
    """Raised when an IMM config or filter run is malformed."""


def _default_modes(
    velocity_covariance_floor: bool = False,
) -> tuple[tuple[str, MotionModel], ...]:
    """Day 22, Objective 2: ``velocity_covariance_floor`` applies ONLY to
    the ``constant_velocity`` mode (:func:`motion_model_for`'s ``person``
    build) — that is the mode representing a pedestrian's own kinematics.
    ``static`` and ``maneuvering`` are IMM kinematic-regime labels, not
    entity kinds, and Objective 2's per-entity-kind directive does not
    name either of them; both remain unfloored regardless of this flag."""
    return (
        ("static", NearlyConstantPositionMotionModel()),
        (
            "constant_velocity",
            motion_model_for(
                "person", velocity_covariance_floor=velocity_covariance_floor
            ),
        ),
        (
            "maneuvering",
            ConstantVelocityMotionModel(
                kind="maneuvering", sigma_a_mps2=MANEUVERING_SIGMA_A_MPS2
            ),
        ),
    )


@dataclass(frozen=True)
class ImmConfig:
    """Mode set + transition probability matrix, versioned as one unit.

    Attributes:
        modes: Ordered ``(name, MotionModel)`` pairs. Order defines
            ``transition_matrix``'s row/column order.
        transition_matrix: ``transition_matrix[i][j]`` = P(mode j at k |
            mode i at k-1). Config-driven and versioned via :attr:`sha` —
            not hardcoded inside the filter, so it can be tuned (or
            measured) later without touching filter logic.
    """

    modes: tuple[tuple[str, MotionModel], ...]
    transition_matrix: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        n = len(self.modes)
        if n < 2:
            raise ImmError(f"ImmConfig needs at least 2 modes, got {n}")
        names = [name for name, _ in self.modes]
        if len(set(names)) != len(names):
            raise ImmError(f"ImmConfig mode names must be unique, got {names}")
        if len(self.transition_matrix) != n or any(
            len(row) != n for row in self.transition_matrix
        ):
            raise ImmError(
                f"ImmConfig.transition_matrix must be {n}x{n} (one row/col "
                f"per mode), got shape ({len(self.transition_matrix)}, "
                f"{[len(r) for r in self.transition_matrix]})"
            )
        for i, row in enumerate(self.transition_matrix):
            if any(p < 0.0 or p > 1.0 for p in row):
                raise ImmError(
                    f"transition_matrix row {i} has an entry outside [0, 1]: {row}"
                )
            total = sum(row)
            if abs(total - 1.0) > 1e-6:
                raise ImmError(
                    f"transition_matrix row {i} must sum to 1.0, got {total}"
                )

    @property
    def mode_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.modes)

    @property
    def sha(self) -> str:
        payload = {
            "modes": [(name, model.sha) for name, model in self.modes],
            "transition_matrix": [list(row) for row in self.transition_matrix],
        }
        canonical = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def default_imm_config(
    persistence_probability: float = 0.95,
    velocity_covariance_floor: bool = False,
) -> ImmConfig:
    """The 3-mode config Day 21 ships, with a symmetric "stay put, switch
    rarely" transition matrix.

    Args:
        persistence_probability: diagonal entries — the probability a mode
            persists from one step to the next. Off-diagonal mass is split
            evenly across the remaining modes. 0.95 at 12fps implies a
            ~20-frame (~1.7s) expected dwell time before switching, which
            is the right ORDER of magnitude for the transient length Day
            21 actually measured (the cessation NEES decay took roughly
            10-14 frames to resolve under the single-model filter) — not
            fitted to that specific number, chosen to be plausible at the
            same scale.
        velocity_covariance_floor: Day 22, Objective 2 -- forwarded to
            :func:`_default_modes`; see its docstring for which mode this
            actually affects.
    """
    if not 0.0 < persistence_probability < 1.0:
        raise ImmError(
            f"persistence_probability must be in (0, 1), got {persistence_probability}"
        )
    modes = _default_modes(velocity_covariance_floor)
    n = len(modes)
    off_diagonal = (1.0 - persistence_probability) / (n - 1)
    matrix = tuple(
        tuple(persistence_probability if i == j else off_diagonal for j in range(n))
        for i in range(n)
    )
    return ImmConfig(modes=modes, transition_matrix=matrix)


@dataclass(frozen=True)
class _ModeState:
    mean: FloatArray
    cov: FloatArray


@dataclass(frozen=True)
class _ImmAppendedState:
    """StateGraph payload for an IMM-produced factor — mirrors
    :class:`src.estimator.filter._AppendedState`'s role for the
    single-model case, extended with everything IMM needs to continue
    mixing at the next step."""

    estimate: StateEstimate
    imm_config: ImmConfig
    measurement_model: MeasurementModel
    mode_states: dict[str, _ModeState]
    mode_probabilities: dict[str, float]


def _gaussian_likelihood(innovation: FloatArray, innovation_cov: FloatArray) -> float:
    """N(innovation; 0, innovation_cov) — the standard IMM mode-likelihood term."""
    dim = innovation.shape[0]
    sign, logdet = np.linalg.slogdet(innovation_cov)
    if sign <= 0:
        raise ImmError("innovation covariance is not positive definite")
    mahalanobis = float(innovation @ np.linalg.solve(innovation_cov, innovation))
    log_likelihood = -0.5 * (dim * np.log(2 * np.pi) + logdet + mahalanobis)
    return float(np.exp(log_likelihood))


def _combine(
    weights: dict[str, float], means: dict[str, FloatArray], covs: dict[str, FloatArray]
) -> tuple[FloatArray, FloatArray]:
    """The IMM combination formula, reused for mixing and for the final
    output combination — same weighted-mean-plus-spread structure both
    times."""
    names = list(weights)
    combined_mean = np.zeros(STATE_DIM)
    for name in names:
        combined_mean += weights[name] * means[name]
    combined_cov = np.zeros((STATE_DIM, STATE_DIM))
    for name in names:
        diff = means[name] - combined_mean
        combined_cov += weights[name] * (covs[name] + np.outer(diff, diff))
    return combined_mean, combined_cov


def run_imm_filter(
    graph: StateGraph,
    observations: Sequence[Observation],
    imm_config: ImmConfig,
    measurement_model: MeasurementModel,
    manifest_sha: str,
    sensor_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> StateGraph:
    """IMM predict-mix-filter-combine over ``observations``.

    Same shape as :func:`src.estimator.filter.run_single_entity_filter`:
    strictly increasing ``ts_ns``, ``WorldPositionMeasurement`` only, one
    Factor appended per observation, no prior parameter anywhere. See that
    function's docstring for the reasoning; not repeated here.

    Raises:
        ImmError: on empty input, out-of-order timestamps, a non-position
            measurement, or a non-positive ``dt`` between observations.
    """
    if not observations:
        raise ImmError("run_imm_filter needs at least one observation")
    timestamps = [obs.ts_ns for obs in observations]
    if timestamps != sorted(timestamps):
        raise ImmError(
            "observations must be strictly increasing in ts_ns; out-of-order "
            "input is refused rather than silently re-sorted"
        )

    origin = np.array(sensor_origin_m, dtype=np.float64)
    mode_names = imm_config.mode_names
    n_modes = len(mode_names)
    H = measurement_model.H()

    previous: _ImmAppendedState | None = None
    previous_factor_id: str | None = None

    for observation in observations:
        if not isinstance(observation.measurement, WorldPositionMeasurement):
            raise ImmError(
                f"observation {observation.observation_id} carries a "
                f"{type(observation.measurement).__name__}, not "
                "WorldPositionMeasurement — the IMM filter only consumes "
                "unprojected 3D position readings"
            )
        z = np.array(
            [
                observation.measurement.x_m,
                observation.measurement.y_m,
                observation.measurement.z_m,
            ],
            dtype=np.float64,
        )
        distance_m = float(np.linalg.norm(z - origin))
        R = measurement_model.R(distance_m)

        inputs: tuple[str, ...]
        if previous is None:
            # Bootstrap: identical to Day 20's single-model bootstrap,
            # replicated across every mode. Mode probabilities start
            # uniform — the least-informative distribution, never seeded
            # from any external belief (§17).
            bootstrap_mean = np.concatenate([z, np.zeros(3, dtype=np.float64)])
            bootstrap_cov = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
            bootstrap_cov[:3, :3] = R
            bootstrap_cov[3:, 3:] = LARGE_VELOCITY_VARIANCE_MPS2 * np.eye(
                3, dtype=np.float64
            )
            mode_states = {
                name: _ModeState(mean=bootstrap_mean.copy(), cov=bootstrap_cov.copy())
                for name in mode_names
            }
            mode_probs = {name: 1.0 / n_modes for name in mode_names}
            combined_mean, combined_cov = bootstrap_mean, bootstrap_cov
            per_mode_residuals = tuple(
                ConsistencyResidual(
                    kind="nis",
                    consumer=f"IMM mode-matched filter self-check (mode={name})",
                    note=f"mode={name}; bootstrap, no innovation",
                )
                for name in mode_names
            )
            combined_nis = consistency.no_observation_nis_stub()
            factor_kind = "bootstrap"
            inputs = (str(observation.observation_id),)
        else:
            dt_s = (observation.ts_ns - previous.estimate.ts_ns) / 1e9
            if dt_s <= 0:
                raise ImmError(
                    f"non-positive dt ({dt_s}s) between observations at "
                    f"{previous.estimate.ts_ns} and {observation.ts_ns}"
                )

            # 1. Mixing.
            prior_states = previous.mode_states
            prior_probs = previous.mode_probabilities
            predicted_mode_probs = {
                mode_names[j]: sum(
                    imm_config.transition_matrix[i][j] * prior_probs[mode_names[i]]
                    for i in range(n_modes)
                )
                for j in range(n_modes)
            }
            mixed_states: dict[str, _ModeState] = {}
            for j, name_j in enumerate(mode_names):
                c_j = predicted_mode_probs[name_j]
                weights = {}
                for i, name_i in enumerate(mode_names):
                    weights[name_i] = (
                        imm_config.transition_matrix[i][j] * prior_probs[name_i] / c_j
                        if c_j > _LIKELIHOOD_FLOOR
                        else 1.0 / n_modes
                    )
                mean, cov = _combine(
                    weights,
                    {n: prior_states[n].mean for n in mode_names},
                    {n: prior_states[n].cov for n in mode_names},
                )
                mixed_states[name_j] = _ModeState(mean=mean, cov=cov)

            # 2. Mode-matched filtering.
            new_mode_states: dict[str, _ModeState] = {}
            likelihoods: dict[str, float] = {}
            per_mode_residuals_list: list[ConsistencyResidual] = []
            predicted_means: dict[str, FloatArray] = {}
            predicted_covs: dict[str, FloatArray] = {}
            for name_j, model_j in imm_config.modes:
                F = model_j.F(dt_s)
                Q = model_j.Q(dt_s)
                prior = mixed_states[name_j]
                predicted_mean = F @ prior.mean
                predicted_cov = F @ prior.cov @ F.T + Q
                predicted_means[name_j] = predicted_mean
                predicted_covs[name_j] = predicted_cov

                innovation = z - H @ predicted_mean
                innovation_cov = H @ predicted_cov @ H.T + R
                kalman_gain = predicted_cov @ H.T @ np.linalg.inv(innovation_cov)
                new_mean = predicted_mean + kalman_gain @ innovation
                i_kh = np.eye(STATE_DIM, dtype=np.float64) - kalman_gain @ H
                new_cov = (
                    i_kh @ predicted_cov @ i_kh.T + kalman_gain @ R @ kalman_gain.T
                )
                # Day 22, Objective 2: a no-op for every mode except
                # constant_velocity when built with velocity_covariance_floor=True.
                new_cov = apply_velocity_covariance_floor(
                    new_cov, model_j.velocity_covariance_floor_mps2(dt_s)
                )
                new_mode_states[name_j] = _ModeState(mean=new_mean, cov=new_cov)
                likelihoods[name_j] = _gaussian_likelihood(innovation, innovation_cov)

                mode_nis = consistency.compute_nis(innovation, innovation_cov)
                per_mode_residuals_list.append(
                    ConsistencyResidual(
                        kind="nis",
                        consumer=f"IMM mode-matched filter self-check (mode={name_j})",
                        value=mode_nis.value,
                        dof=mode_nis.dof,
                        chi2_bound=mode_nis.chi2_bound,
                        within_bound=mode_nis.within_bound,
                        note=f"mode={name_j}",
                    )
                )
            per_mode_residuals = tuple(per_mode_residuals_list)

            # 3. Mode-probability update.
            normalizer = sum(
                predicted_mode_probs[name] * likelihoods[name] for name in mode_names
            )
            if normalizer <= 0.0:
                new_mode_probs = predicted_mode_probs
            else:
                new_mode_probs = {
                    name: predicted_mode_probs[name] * likelihoods[name] / normalizer
                    for name in mode_names
                }

            # 4. Combination -- the reported estimate.
            combined_mean, combined_cov = _combine(
                new_mode_probs,
                {n: new_mode_states[n].mean for n in mode_names},
                {n: new_mode_states[n].cov for n in mode_names},
            )

            # A combined-level NIS for the STRUCTURAL "at least one nis
            # entry" requirement and for parity with the single-model
            # report: the mixture's own pre-update predicted mean/cov
            # (mode-probability-weighted), the same combination formula
            # used for the output, applied one step earlier.
            combined_predicted_mean, combined_predicted_cov = _combine(
                predicted_mode_probs, predicted_means, predicted_covs
            )
            combined_innovation = z - H @ combined_predicted_mean
            combined_innovation_cov = H @ combined_predicted_cov @ H.T + R
            combined_nis = consistency.compute_nis(
                combined_innovation, combined_innovation_cov
            )

            mode_states = new_mode_states
            mode_probs = new_mode_probs
            factor_kind = "measurement_update"
            assert previous_factor_id is not None
            inputs = (previous_factor_id, str(observation.observation_id))

        estimate = StateEstimate(
            ts_ns=observation.ts_ns,
            mean=tuple(combined_mean.tolist()),
            cov=tuple(tuple(row) for row in combined_cov.tolist()),
            observed=True,
            motion_model_sha="+".join(sorted(m.sha for _, m in imm_config.modes)),
            measurement_model_sha=measurement_model.sha,
            update_rule_sha=UPDATE_RULE_SHA,
            graph_rev=graph.graph_rev + 1,
            residuals=(
                combined_nis,
                *per_mode_residuals,
                *consistency.stub_residuals(),
            ),
            imm_config_sha=imm_config.sha,
            mode_probabilities=tuple(sorted(mode_probs.items())),
            mode_states=tuple(
                (
                    name,
                    tuple(state.mean.tolist()),
                    tuple(tuple(r) for r in state.cov.tolist()),
                )
                for name, state in sorted(mode_states.items())
            ),
        )
        payload = _ImmAppendedState(
            estimate=estimate,
            imm_config=imm_config,
            measurement_model=measurement_model,
            mode_states=mode_states,
            mode_probabilities=mode_probs,
        )
        factor = graph.append_factor(
            str(generate_ulid()), factor_kind, inputs, manifest_sha, payload=payload
        )
        previous = payload
        previous_factor_id = factor.factor_id

    return graph


def _payload_factors(
    graph: StateGraph, graph_rev: int
) -> list[tuple[Factor, _ImmAppendedState]]:
    out: list[tuple[Factor, _ImmAppendedState]] = []
    for f in graph.factors_as_of(graph_rev):
        payload = graph.payload_for(f.factor_id)
        if isinstance(payload, _ImmAppendedState):
            out.append((f, payload))
    return out


def resolve_imm_state(query: StateQuery, graph: StateGraph) -> StateEstimate:
    """The IMM counterpart of :func:`src.estimator.filter.resolve_state`.

    Raises:
        NotImplementedError: if ``query.horizon_kind == "smoothed"``.
        ImmError: if the graph has no resolvable IMM state before
            ``query.at_ts_ns``, or resolving it needs more extrapolation
            than ``query.horizon_ns`` allows.
    """
    if query.horizon_kind == "smoothed":
        raise NotImplementedError(
            "smoothed-horizon resolution is out of scope for both the "
            "single-model and IMM filters today — see "
            "src.estimator.filter.resolve_state's docstring"
        )

    candidates = _payload_factors(graph, query.graph_rev)
    if not candidates:
        raise ImmError(
            f"no resolvable IMM state in this graph at graph_rev="
            f"{query.graph_rev} — the graph is empty, or carries no "
            "IMM-produced factors at or before this revision"
        )

    at_or_before = [(f, p) for f, p in candidates if p.estimate.ts_ns <= query.at_ts_ns]
    if not at_or_before:
        earliest = candidates[0][1].estimate.ts_ns
        raise ImmError(
            f"query.at_ts_ns={query.at_ts_ns} predates this graph's earliest "
            f"resolvable IMM state (ts_ns={earliest})"
        )

    _, latest = at_or_before[-1]
    dt_ns = query.at_ts_ns - latest.estimate.ts_ns
    if dt_ns == 0:
        return latest.estimate
    if dt_ns > query.horizon_ns:
        raise ImmError(
            f"resolving at_ts_ns={query.at_ts_ns} needs {dt_ns}ns of "
            f"extrapolation beyond the last resolvable factor "
            f"(ts_ns={latest.estimate.ts_ns}), exceeding "
            f"horizon_ns={query.horizon_ns}"
        )

    # Extrapolation with no observation: propagate every mode forward
    # under its own F/Q, keep mode probabilities frozen (nothing has been
    # observed to update them), then combine -- same structure as a
    # normal step minus the mixing/update/reweight stages.
    dt_s = dt_ns / 1e9
    predicted_means: dict[str, FloatArray] = {}
    predicted_covs: dict[str, FloatArray] = {}
    for name, model in latest.imm_config.modes:
        prior = latest.mode_states[name]
        F = model.F(dt_s)
        predicted_means[name] = F @ prior.mean
        predicted_covs[name] = apply_velocity_covariance_floor(
            F @ prior.cov @ F.T + model.Q(dt_s),
            model.velocity_covariance_floor_mps2(dt_s),
        )
    combined_mean, combined_cov = _combine(
        latest.mode_probabilities, predicted_means, predicted_covs
    )
    per_mode_residuals = tuple(
        ConsistencyResidual(
            kind="nis",
            consumer=f"IMM mode-matched filter self-check (mode={name})",
            note=f"mode={name}; predicted, no observation at this timestamp",
        )
        for name in latest.imm_config.mode_names
    )
    return StateEstimate(
        ts_ns=query.at_ts_ns,
        mean=tuple(combined_mean.tolist()),
        cov=tuple(tuple(row) for row in combined_cov.tolist()),
        observed=False,
        motion_model_sha=latest.estimate.motion_model_sha,
        measurement_model_sha=latest.estimate.measurement_model_sha,
        update_rule_sha=latest.estimate.update_rule_sha,
        graph_rev=query.graph_rev,
        residuals=(
            consistency.no_observation_nis_stub(),
            *per_mode_residuals,
            *consistency.stub_residuals(),
        ),
        imm_config_sha=latest.imm_config.sha,
        mode_probabilities=tuple(sorted(latest.mode_probabilities.items())),
        mode_states=tuple(
            (
                name,
                tuple(predicted_means[name].tolist()),
                tuple(tuple(r) for r in predicted_covs[name].tolist()),
            )
            for name in sorted(predicted_means)
        ),
    )
