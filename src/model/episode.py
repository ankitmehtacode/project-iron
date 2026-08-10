"""Episode, ActivityMode, and the StateGraph skeleton.

Episode, not Episode-plus-Interaction
--------------------------------------
An earlier draft of this model had a separate ``Interaction`` type for
"an episode where two people did something to each other". It added a
second primitive for a property (at least two non-object participants)
that :class:`Episode` already had every field needed to express — see
ADR 0006 for the primitive-proliferation reasoning that killed it.
:attr:`Episode.is_interaction` is a derived property, not a class.

ActivityMode is inadmissible by construction
----------------------------------------------
:class:`ActivityMode` is a statistical behaviour-pattern label — "this
trajectory looks like browsing" — never a fact. It is scoped to an
episode or a trajectory, never to an enrolled identity: ``scope_kind`` is
a closed two-value vocabulary that has no "identity" option, so there is
no field to populate even if someone wanted to attach a behaviour label
to a named person by default. Its score is typed as
:class:`~src.model.evidence.UncalibratedScore`, not a bare float, so it
cannot silently read as a probability. ``admissible`` is always ``False``,
checked at construction, and :func:`attach_to_alert` /
:func:`attach_to_evidence` unconditionally raise — there is no code path
that lets a behaviour-pattern label trigger a page or back a claim.

StateGraph is a skeleton, on purpose — filled Day 20, single-entity only
--------------------------------------------------------------------------
The factor-graph solver was explicitly out of scope for Day 13 (see the
Day-13 prompt's scope discipline). What was in scope then is the
append-only factor store and the monotonically increasing ``graph_rev`` —
because :class:`~src.model.evidence.Evidence` needs something to reference,
even before anything could resolve a query against it.

Day 20 fills :func:`solve_state` for the **single-entity** case only —
:mod:`src.estimator`'s motion/measurement models and Kalman filter — via a
deferred import inside the function body (not a module-level import: that
would make ``src.model.episode`` depend on ``src.estimator``, which itself
imports ``src.model.episode`` for :class:`StateGraph`/:class:`StateQuery`,
a genuine import cycle that a call-time import breaks the same way
``observability_partition`` in :mod:`src.data.scorecard` defers its
``src.cascade.envelope`` import). The multi-entity factor graph, smoothing
across the full graph, and hypothesis management are still out of scope —
see :data:`StateQuery.horizon_kind` for how "smoothed" stays a documented
``NotImplementedError`` rather than a silent wrong answer.

``StateGraph``'s own shape is unchanged from Day 13 in every way a Day-13
caller depends on: :class:`Factor` is still frozen with the same four
fields, ``append_factor``'s first four parameters and
``factors_as_of`` are untouched. The one addition is an *optional* payload
slot on ``append_factor`` (default ``None``) so a filter can attach the
actual numeric :class:`~src.estimator.state.StateEstimate` a factor
represents, retrievable via the new :meth:`StateGraph.payload_for` —
additive, not breaking, and exactly what lets :func:`solve_state` replay a
graph's history deterministically without recomputing it from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, NoReturn, get_args

from src.model.evidence import Evidence, UncalibratedScore

ParticipantRole = Literal["primary", "counterpart", "object", "bystander", "container"]
PARTICIPANT_ROLES: tuple[ParticipantRole, ...] = get_args(ParticipantRole)


class EpisodeError(ValueError):
    """Raised when an Episode, ActivityMode, or StateGraph record is malformed."""


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Participant:
    """One entity's role within an Episode."""

    entity_id: str
    role: ParticipantRole

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise EpisodeError("Participant.entity_id must not be empty")
        if self.role not in PARTICIPANT_ROLES:
            raise EpisodeError(
                f"unknown participant role {self.role!r}; expected one of "
                f"{PARTICIPANT_ROLES}"
            )


@dataclass(frozen=True)
class Episode:
    """A segmented span of activity with roled participants.

    Attributes:
        episode_id: Unique id.
        site_id: Which installation.
        start_ns: Episode start, nanoseconds since the epoch.
        end_ns: Episode end. Must be after ``start_ns``.
        participants: At least one. STRUCTURAL-adjacent: an episode with
            no participants describes nothing.
        boundary_confidence: In [0, 1]. Segmentation — where an episode
            starts and ends — is a hypothesis about the data, not a fact
            read off it; this is required precisely so nothing downstream
            can treat an episode's boundaries as certain.
        evidence_refs: Opaque references into the evidence layer.
        manifest_sha: The run that produced this record.
    """

    episode_id: str
    site_id: str
    start_ns: int
    end_ns: int
    participants: tuple[Participant, ...]
    boundary_confidence: float
    evidence_refs: tuple[str, ...]
    manifest_sha: str

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise EpisodeError("Episode.episode_id must not be empty")
        if not self.site_id:
            raise EpisodeError("Episode.site_id must not be empty")
        if self.end_ns <= self.start_ns:
            raise EpisodeError(
                f"Episode must span a positive interval: start_ns="
                f"{self.start_ns} end_ns={self.end_ns}"
            )
        if not self.participants:
            raise EpisodeError("Episode.participants must not be empty")
        if not 0.0 <= self.boundary_confidence <= 1.0:
            raise EpisodeError(
                f"Episode.boundary_confidence must lie in [0, 1], got "
                f"{self.boundary_confidence}"
            )
        if not self.manifest_sha:
            raise EpisodeError("Episode.manifest_sha must not be empty")

    @property
    def is_interaction(self) -> bool:
        """Whether this episode qualifies as an interaction.

        An interaction is an episode with at least two non-``object``
        participants — derived here, not a separate type. See the module
        docstring and ADR 0006.
        """
        non_object = [p for p in self.participants if p.role != "object"]
        return len(non_object) >= 2


# ---------------------------------------------------------------------------
# ActivityMode — inadmissible statistical labels.
# ---------------------------------------------------------------------------

ActivityLabel = Literal[
    "browsing",
    "working_at_station",
    "idle",
    "transiting",
    "congregating",
    "servicing_equipment",
]
ACTIVITY_LABELS: tuple[ActivityLabel, ...] = get_args(ActivityLabel)

ActivityScopeKind = Literal["episode", "trajectory"]
"""Deliberately excludes any identity-scoped option — see the module docstring."""
ACTIVITY_SCOPE_KINDS: tuple[ActivityScopeKind, ...] = get_args(ActivityScopeKind)


@dataclass(frozen=True)
class ActivityMode:
    """A statistical behaviour-pattern label, never a fact.

    STRUCTURAL: ``admissible`` is always ``False``, checked here even
    though its type already pins it — a caller passing ``True`` at
    runtime (bypassing the type checker) is still rejected. Use
    :func:`attach_to_alert` / :func:`attach_to_evidence` to see the
    unconditional raise this buys.
    """

    scope_id: str
    scope_kind: ActivityScopeKind
    label: ActivityLabel
    score: UncalibratedScore
    admissible: Literal[False] = False

    def __post_init__(self) -> None:
        if not self.scope_id:
            raise EpisodeError("ActivityMode.scope_id must not be empty")
        if self.scope_kind not in ACTIVITY_SCOPE_KINDS:
            raise EpisodeError(
                f"unknown activity scope kind {self.scope_kind!r}; expected "
                f"one of {ACTIVITY_SCOPE_KINDS}"
            )
        if self.label not in ACTIVITY_LABELS:
            raise EpisodeError(
                f"unknown activity label {self.label!r}; expected one of "
                f"{ACTIVITY_LABELS}"
            )
        if self.admissible is not False:
            raise EpisodeError(
                "ActivityMode.admissible must always be False: a behaviour "
                "pattern is a statistical label, never a fact strong enough "
                "to alert on or admit as evidence"
            )


def attach_to_alert(mode: ActivityMode) -> NoReturn:
    """Always raises: an ActivityMode can never trigger or back an alert."""
    raise EpisodeError(
        f"ActivityMode {mode.label!r} (scope={mode.scope_kind}:{mode.scope_id}) "
        "cannot trigger an alert. Behaviour-pattern labels are statistical "
        "correlations over an episode or trajectory, not observed facts — "
        "paging someone for a correlation is exactly what admissible=False "
        "exists to prevent."
    )


def attach_to_evidence(mode: ActivityMode, evidence: Evidence) -> NoReturn:
    """Always raises: an ActivityMode can never be admitted as evidence."""
    raise EpisodeError(
        f"ActivityMode {mode.label!r} (scope={mode.scope_kind}:{mode.scope_id}) "
        f"cannot be admitted into evidence chain {evidence.evidence_id!r}. "
        "See attach_to_alert's docstring — the same rule applies to evidence."
    )


# ---------------------------------------------------------------------------
# StateGraph skeleton — append-only, monotonic graph_rev, no solver.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Factor:
    """One append-only entry in the state graph.

    Frozen and never replaced in place: correcting a factor means
    appending a new one, exactly like every other record in this model.
    """

    factor_id: str
    factor_kind: str
    inputs: tuple[str, ...]
    graph_rev: int
    manifest_sha: str

    def __post_init__(self) -> None:
        if not self.factor_id:
            raise EpisodeError("Factor.factor_id must not be empty")
        if not self.factor_kind:
            raise EpisodeError("Factor.factor_kind must not be empty")
        if not self.inputs:
            raise EpisodeError("Factor.inputs must not be empty")
        if self.graph_rev < 1:
            raise EpisodeError(f"Factor.graph_rev must be >= 1, got {self.graph_rev}")
        if not self.manifest_sha:
            raise EpisodeError("Factor.manifest_sha must not be empty")


HorizonKind = Literal["filtered", "smoothed"]
"""Which solve MODE answers this query — orthogonal to ``horizon_ns``
(which asks *how far*, not *which algorithm*).

``filtered``: causal, online — the marginal at ``at_ts_ns`` uses only
factors at or before it (a Kalman-style forward pass). This is what Day 20
implements.

``smoothed``: uses factors *after* ``at_ts_ns`` too (an RTS-style backward
pass), which is strictly more accurate at any point that has later evidence
but is not a single-entity-scope concern — it needs the same factor-replay
machinery plus a second pass, and Day 20's scope is single-entity filtering
only. ``solve_state`` raises ``NotImplementedError`` for this value, not
silently falling back to ``filtered`` — a caller who asked for smoothing
and got a filtered answer with no error would not know to distrust it.
"""
HORIZON_KINDS: tuple[HorizonKind, ...] = get_args(HorizonKind)


@dataclass(frozen=True)
class StateQuery:
    """A request to resolve state at a point in time, against a graph revision.

    Attributes:
        at_ts_ns: The instant to resolve state for.
        horizon_ns: How far the query is willing to extrapolate beyond
            observed factors. Zero means "interpolate only".
        graph_rev: Which revision of the factor graph to answer against —
            pinned explicitly so a query's answer is reproducible even as
            the graph keeps growing.
        horizon_kind: ``"filtered"`` (default, Day 13's original implicit
            meaning — every pre-Day-20 ``StateQuery(...)`` call site keeps
            working unchanged) or ``"smoothed"``. See :data:`HorizonKind`.
    """

    at_ts_ns: int
    horizon_ns: int
    graph_rev: int
    horizon_kind: HorizonKind = "filtered"

    def __post_init__(self) -> None:
        if self.horizon_ns < 0:
            raise EpisodeError(
                f"StateQuery.horizon_ns must be >= 0, got {self.horizon_ns}"
            )
        if self.graph_rev < 0:
            raise EpisodeError(
                f"StateQuery.graph_rev must be >= 0, got {self.graph_rev}"
            )
        if self.horizon_kind not in HORIZON_KINDS:
            raise EpisodeError(
                f"unknown StateQuery.horizon_kind {self.horizon_kind!r}; "
                f"expected one of {HORIZON_KINDS}"
            )


class StateGraph:
    """An append-only factor store with a monotonically increasing revision.

    There is no method that mutates or removes an appended
    :class:`Factor` — only :meth:`append_factor`. ``graph_rev`` increases
    by exactly one per append and is never reused, so
    ``Evidence.state_refs`` entries stay meaningful even as the graph
    keeps growing underneath them.

    Day 20: ``append_factor`` accepts an optional ``payload`` — the actual
    numeric result (a :class:`~src.estimator.state.StateEstimate`) a
    predict/update step produced, retrievable via :meth:`payload_for`.
    ``Factor`` itself stays untouched: the payload lives in a side table
    keyed by ``factor_id``, not on the frozen dataclass, so every Day-13
    caller that never passes one gets exactly the old behaviour (``None``
    back from a lookup nothing ever performs). This is what lets
    :func:`solve_state` replay a graph's history instead of needing a
    parallel, separately-passed-in numeric store.
    """

    def __init__(self) -> None:
        self._factors: list[Factor] = []
        self._graph_rev = 0
        self._payloads: dict[str, Any] = {}

    @property
    def graph_rev(self) -> int:
        return self._graph_rev

    def append_factor(
        self,
        factor_id: str,
        factor_kind: str,
        inputs: tuple[str, ...],
        manifest_sha: str,
        payload: Any = None,
    ) -> Factor:
        """Append a new factor, advancing ``graph_rev`` by one.

        Args:
            payload: Optional numeric result this factor represents (Day
                20). Stored by ``factor_id``, retrievable via
                :meth:`payload_for`. ``None`` (the default) stores nothing,
                matching every Day-13 call site exactly.

        Raises:
            EpisodeError: if ``factor_id`` was already used in this graph —
                payloads are keyed by ``factor_id``, so a reused id would
                silently overwrite a prior factor's numeric result.
        """
        if factor_id in self._payloads or any(
            f.factor_id == factor_id for f in self._factors
        ):
            raise EpisodeError(
                f"factor_id {factor_id!r} was already appended to this graph; "
                "factor ids must be unique within a graph"
            )
        next_rev = self._graph_rev + 1
        factor = Factor(
            factor_id=factor_id,
            factor_kind=factor_kind,
            inputs=inputs,
            graph_rev=next_rev,
            manifest_sha=manifest_sha,
        )
        self._factors.append(factor)
        self._graph_rev = next_rev
        if payload is not None:
            self._payloads[factor_id] = payload
        return factor

    def factors_as_of(self, graph_rev: int) -> tuple[Factor, ...]:
        """Every factor appended at or before ``graph_rev``, in append order."""
        return tuple(f for f in self._factors if f.graph_rev <= graph_rev)

    def payload_for(self, factor_id: str) -> Any | None:
        """The numeric payload attached to ``factor_id``, or ``None``.

        ``None`` covers two cases identically: no such factor exists, and
        the factor exists but was appended with no payload. Callers that
        need to distinguish those should check ``factors_as_of`` /
        ``factor_id`` membership separately.
        """
        return self._payloads.get(factor_id)


def solve_state(query: StateQuery, graph: StateGraph) -> Any:
    """Resolve ``query`` against ``graph``'s factors: the single-entity case.

    Day 20 fills this for ``query.horizon_kind == "filtered"`` — a causal
    replay of ``graph.factors_as_of(query.graph_rev)`` via
    :mod:`src.estimator`'s single-entity Kalman filter, returning a
    :class:`~src.estimator.state.StateEstimate`. Imported inside the
    function body, not at module level, to avoid a cycle: ``src.estimator``
    imports :class:`StateGraph`/:class:`StateQuery` from this module, so
    this module cannot import ``src.estimator`` back at import time (see
    the module docstring).

    Still not implemented, and still explicitly so rather than silently
    wrong: the multi-entity factor graph, smoothing across the full graph
    (``query.horizon_kind == "smoothed"``), and hypothesis management. Any
    of those landing without a measured accuracy figure attached would
    violate the same rule this whole package's Objective 0 exists to
    enforce — no metric without a baseline, no claim without a number —
    which is why Day 20 also shipped ``scripts/eval_estimator.py`` in the
    same change as this function, not after it.

    Raises:
        NotImplementedError: if ``query.horizon_kind == "smoothed"``.
        EpisodeError: if the graph has no resolvable state before
            ``query.at_ts_ns``, or if resolving it needs more extrapolation
            than ``query.horizon_ns`` allows.
    """
    from src.estimator.filter import FilterError, resolve_state

    try:
        return resolve_state(query, graph)
    except FilterError as exc:
        raise EpisodeError(str(exc)) from exc
