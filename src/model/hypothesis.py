"""The hypothesis store, and why a hypothesis cannot die without a typed cause.

Closes ``docs/data_model/v0.3.md``'s "The hypothesis store and its typed
death causes, including ``PRUNED_BY_BUDGET``" finding: ``PRUNED_BY_BUDGET``,
``HypothesisStore``, ``DeathCause``, and ``death_cause`` appeared ZERO
times anywhere in this repository before today (Day 28), despite
"hypothesis management" being carried on the punch list since Day 5 and
``PRUNED_BY_BUDGET`` specifically being reasoned about in argument as if
it already existed.

Why ``PRUNED_BY_BUDGET`` is not a rejection
--------------------------------------------
"We ruled it out" and "we never evaluated it" are different answers to
an investigator asking what else was considered. A hypothesis refuted by
a hard constraint, an observation, or a dominant competitor was
EVALUATED and found wanting. A hypothesis pruned by budget was never
evaluated at all — it lost a resource competition (too many live
hypotheses, not enough compute to carry it further) before evidence ever
spoke to whether it was right. A store that could not distinguish these
would, when asked "was X considered and rejected," have no way to avoid
answering the wrong one of the two.

Because that distinction matters, a budget-pruned hypothesis is not
discarded: it is retained as a record — its proposition and its support
score AT THE MOMENT of death, deliberately WITHOUT the full state that a
live hypothesis carries (retaining full state for every hypothesis a
budget ever forced out would reintroduce the exact resource problem
budget pruning exists to relieve). Every forensic query over "what else
was considered" (:meth:`HypothesisStore.considered_alternatives`)
surfaces budget-pruned records alongside refuted ones, by construction —
there is no separate "rejected only" query for a caller to reach for by
mistake and no filter to apply after the fact.

STRUCTURAL: the only way a hypothesis leaves the alive set is
:meth:`HypothesisStore.kill`, which requires a typed :data:`DeathCause`
as a non-optional argument and unconditionally records a
:class:`DeadHypothesis` before returning. There is no ``discard``,
``remove``, or ``pop`` on this store — a hypothesis cannot die without a
typed cause because there is no code path by which it can die at all
except the one that assigns one.

Full multi-hypothesis management stays skeleton today: nothing in this
module runs multiple competing track hypotheses against each other or
decides when to spawn one. This module establishes the store, the
lifecycle, and the death-cause discipline so that whenever hypothesis
management does arrive, it cannot be built without them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

DeathCauseKind = Literal[
    "refuted_by_hard_constraint",
    "refuted_by_observation",
    "dominated_by_likelihood",
    "merged_into",
    "expired_horizon",
    "pruned_by_budget",
]


@dataclass(frozen=True)
class RefutedByHardConstraint:
    """Evaluated and found physically impossible — see
    ``src/model/constraint.py::HardConstraintViolation``, the only type
    that can construct one of these (via
    ``src/estimator/constraints.py::prune_for_hard_violation``)."""

    constraint_name: str
    kind: Literal["refuted_by_hard_constraint"] = field(
        default="refuted_by_hard_constraint", init=False
    )


@dataclass(frozen=True)
class RefutedByObservation:
    """Evaluated and found inconsistent with an observation (e.g. NIS/NEES
    far outside bound)."""

    detail: str
    kind: Literal["refuted_by_observation"] = field(
        default="refuted_by_observation", init=False
    )


@dataclass(frozen=True)
class DominatedByLikelihood:
    """Evaluated and found strictly less likely than a surviving
    competitor — not physically impossible, just worse-supported."""

    dominant_hypothesis_id: str
    kind: Literal["dominated_by_likelihood"] = field(
        default="dominated_by_likelihood", init=False
    )


@dataclass(frozen=True)
class MergedInto:
    """Absorbed into another hypothesis that now represents both —
    evaluated, not rejected."""

    hypothesis_id: str
    kind: Literal["merged_into"] = field(default="merged_into", init=False)


@dataclass(frozen=True)
class ExpiredHorizon:
    """Never refuted; simply outlived the horizon this store tracks
    alternatives within."""

    horizon_ns: int
    kind: Literal["expired_horizon"] = field(default="expired_horizon", init=False)


@dataclass(frozen=True)
class PrunedByBudget:
    """NOT a rejection — see module docstring. This hypothesis lost a
    resource competition before evidence ever spoke to whether it was
    right."""

    budget: int
    kind: Literal["pruned_by_budget"] = field(default="pruned_by_budget", init=False)


DeathCause = Union[
    RefutedByHardConstraint,
    RefutedByObservation,
    DominatedByLikelihood,
    MergedInto,
    ExpiredHorizon,
    PrunedByBudget,
]


@dataclass(frozen=True)
class Hypothesis:
    """A live track hypothesis. ``proposition`` is a human-readable claim
    (e.g. "entity-7 is the same physical body as entity-3 after a
    12-frame occlusion"); ``support`` is whatever score the caller's
    evaluation logic currently assigns it — this store does not interpret
    or compare support values itself, only stores and retires them."""

    id: str
    proposition: str
    support: float


@dataclass(frozen=True)
class DeadHypothesis:
    """What survives after :meth:`HypothesisStore.kill`: the proposition
    and the support score at death, deliberately WITHOUT full state (see
    module docstring) — plus the typed cause, always present, never
    optional."""

    id: str
    proposition: str
    support_at_death: float
    cause: DeathCause


class HypothesisStoreError(KeyError):
    """Raised for an operation on a hypothesis id the store does not have
    in the state the operation requires (e.g. killing an id that is
    already dead, or that was never proposed)."""


class HypothesisStore:
    """Tracks live hypotheses and retains typed records of dead ones.

    Skeleton by design (see module docstring): this store does not
    decide when to spawn, merge, or budget-prune a hypothesis — it only
    enforces that whatever caller does decide records a typed cause and
    never silently loses a record.
    """

    def __init__(self) -> None:
        self._alive: dict[str, Hypothesis] = {}
        self._dead: dict[str, DeadHypothesis] = {}

    def propose(
        self, hypothesis_id: str, proposition: str, support: float
    ) -> Hypothesis:
        """Add a new live hypothesis. Raises if ``hypothesis_id`` was ever
        used before, alive or dead — ids are not recycled, so a dead
        hypothesis's record is never at risk of being overwritten by a
        new proposal that reuses its id."""
        if hypothesis_id in self._alive or hypothesis_id in self._dead:
            raise HypothesisStoreError(
                f"hypothesis id {hypothesis_id!r} already exists (alive or dead)"
            )
        hypothesis = Hypothesis(
            id=hypothesis_id, proposition=proposition, support=support
        )
        self._alive[hypothesis_id] = hypothesis
        return hypothesis

    def kill(self, hypothesis_id: str, cause: DeathCause) -> DeadHypothesis:
        """The only way a hypothesis leaves the alive set. ``cause`` is
        required and typed — there is no default and no way to call this
        without one, so a hypothesis cannot die without a typed cause.
        Always records a :class:`DeadHypothesis` before returning; never
        silently drops the record regardless of ``cause``, including
        :class:`PrunedByBudget`."""
        if hypothesis_id not in self._alive:
            raise HypothesisStoreError(f"no alive hypothesis {hypothesis_id!r} to kill")
        hypothesis = self._alive.pop(hypothesis_id)
        record = DeadHypothesis(
            id=hypothesis.id,
            proposition=hypothesis.proposition,
            support_at_death=hypothesis.support,
            cause=cause,
        )
        self._dead[hypothesis_id] = record
        return record

    def alive(self) -> tuple[Hypothesis, ...]:
        return tuple(self._alive.values())

    def considered_alternatives(self) -> tuple[DeadHypothesis, ...]:
        """Forensic query: every hypothesis that ever died, every cause
        included. Budget-pruned records are NOT filtered out — see the
        module docstring for why a query that dropped them would answer
        "we ruled it out" when the truth is "we never evaluated it."
        """
        return tuple(self._dead.values())
