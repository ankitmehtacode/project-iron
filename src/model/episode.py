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

StateGraph is a skeleton, on purpose
--------------------------------------
The factor-graph solver is explicitly out of scope for Day 13 (see the
Day-13 prompt's scope discipline). What is in scope is the append-only
factor store and the monotonically increasing ``graph_rev`` — because
:class:`~src.model.evidence.Evidence` needs something to reference
*today*, even though nothing can resolve a query against it yet.
:func:`solve_state` raises ``NotImplementedError`` naming exactly what
will fill it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NoReturn, get_args

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
    """

    at_ts_ns: int
    horizon_ns: int
    graph_rev: int

    def __post_init__(self) -> None:
        if self.horizon_ns < 0:
            raise EpisodeError(
                f"StateQuery.horizon_ns must be >= 0, got {self.horizon_ns}"
            )
        if self.graph_rev < 0:
            raise EpisodeError(
                f"StateQuery.graph_rev must be >= 0, got {self.graph_rev}"
            )


class StateGraph:
    """An append-only factor store with a monotonically increasing revision.

    There is no method that mutates or removes an appended
    :class:`Factor` — only :meth:`append_factor`. ``graph_rev`` increases
    by exactly one per append and is never reused, so
    ``Evidence.state_refs`` entries stay meaningful even as the graph
    keeps growing underneath them.
    """

    def __init__(self) -> None:
        self._factors: list[Factor] = []
        self._graph_rev = 0

    @property
    def graph_rev(self) -> int:
        return self._graph_rev

    def append_factor(
        self,
        factor_id: str,
        factor_kind: str,
        inputs: tuple[str, ...],
        manifest_sha: str,
    ) -> Factor:
        """Append a new factor, advancing ``graph_rev`` by one."""
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
        return factor

    def factors_as_of(self, graph_rev: int) -> tuple[Factor, ...]:
        """Every factor appended at or before ``graph_rev``, in append order."""
        return tuple(f for f in self._factors if f.graph_rev <= graph_rev)


def solve_state(query: StateQuery, graph: StateGraph) -> NoReturn:
    """Resolve ``query`` against ``graph``'s factors.

    Not implemented. This is the factor-graph solver / estimator, and Day
    13's scope is primitives and rules — types, validation, serialization,
    migration, structural guards — not inference; see the Day-13 prompt's
    scope discipline. What belongs here, in a later measured phase: an
    inference pass (e.g. belief propagation or an equivalent factor-graph
    solver) over ``graph.factors_as_of(query.graph_rev)``, producing a
    state estimate at ``query.at_ts_ns`` with uncertainty that honestly
    reflects ``query.horizon_ns`` — wide when extrapolating past the last
    observed factor, narrow when interpolating between two nearby ones.
    Landing that estimator without a measured accuracy figure attached
    would violate the same rule this whole package's Objective 0 exists
    to enforce: no metric without a baseline, no claim without a number.

    Raises:
        NotImplementedError: always.
    """
    raise NotImplementedError(
        "the factor-graph solver is out of scope for Day 13 — see "
        "solve_state's docstring for what will fill it and why it is not "
        "here yet"
    )
