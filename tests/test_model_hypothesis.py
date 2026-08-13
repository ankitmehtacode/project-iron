"""Day 28, Objective 3 — the hypothesis store's typed death causes.

STRUCTURAL rules under test:
  - A hypothesis cannot die without a typed DeathCause (kill() requires
    one; there is no discard/remove/pop that skips it).
  - PRUNED_BY_BUDGET hypotheses are retained as records (proposition +
    support at death, no full state) and are NOT filtered out of the
    forensic considered_alternatives() query.
"""

from __future__ import annotations

import pytest

from src.model.hypothesis import (
    DeadHypothesis,
    DominatedByLikelihood,
    ExpiredHorizon,
    HypothesisStore,
    HypothesisStoreError,
    MergedInto,
    PrunedByBudget,
    RefutedByHardConstraint,
    RefutedByObservation,
)


def _store_with_one_hypothesis(support: float = 0.5) -> tuple[HypothesisStore, str]:
    store = HypothesisStore()
    store.propose("h1", "entity-7 continues entity-3 after occlusion", support)
    return store, "h1"


def test_propose_adds_a_live_hypothesis() -> None:
    store, hid = _store_with_one_hypothesis()
    assert [h.id for h in store.alive()] == [hid]


def test_propose_rejects_a_reused_id_even_after_death() -> None:
    store, hid = _store_with_one_hypothesis()
    store.kill(hid, PrunedByBudget(budget=4))
    with pytest.raises(HypothesisStoreError):
        store.propose(hid, "a different claim", 0.1)


def test_kill_requires_a_death_cause_argument() -> None:
    store, hid = _store_with_one_hypothesis()
    with pytest.raises(TypeError):
        store.kill(hid)  # type: ignore[call-arg]


def test_kill_removes_from_alive() -> None:
    store, hid = _store_with_one_hypothesis()
    store.kill(hid, ExpiredHorizon(horizon_ns=1_000_000_000))
    assert store.alive() == ()


def test_kill_unknown_id_raises() -> None:
    store = HypothesisStore()
    with pytest.raises(HypothesisStoreError):
        store.kill("ghost", PrunedByBudget(budget=4))


@pytest.mark.parametrize(
    "cause",
    [
        RefutedByHardConstraint(constraint_name="one_body_one_place"),
        RefutedByObservation(detail="NIS 40.2 vs bound 7.8"),
        DominatedByLikelihood(dominant_hypothesis_id="h2"),
        MergedInto(hypothesis_id="h2"),
        ExpiredHorizon(horizon_ns=2_000_000_000),
        PrunedByBudget(budget=6),
    ],
)
def test_every_death_cause_kind_is_retained_as_a_record(cause: object) -> None:
    store, hid = _store_with_one_hypothesis(support=0.73)
    record = store.kill(hid, cause)  # type: ignore[arg-type]
    assert isinstance(record, DeadHypothesis)
    assert record.id == hid
    assert record.support_at_death == 0.73
    assert record.cause == cause
    assert record in store.considered_alternatives()


def test_dead_hypothesis_record_has_no_full_state_field() -> None:
    store, hid = _store_with_one_hypothesis()
    record = store.kill(hid, PrunedByBudget(budget=4))
    assert set(vars(record)) == {"id", "proposition", "support_at_death", "cause"}


def test_budget_pruned_hypotheses_are_not_filtered_out_of_considered_alternatives() -> (
    None
):
    """The specific rule the objective names: 'we ruled it out' and 'we
    never evaluated it' are different answers, and a forensic query that
    silently dropped PRUNED_BY_BUDGET records would collapse them."""
    store = HypothesisStore()
    store.propose("refuted", "hypothesis that violated a hard constraint", 0.9)
    store.propose("budget-pruned", "hypothesis that lost the budget competition", 0.4)
    store.kill("refuted", RefutedByHardConstraint(constraint_name="one_body_one_place"))
    store.kill("budget-pruned", PrunedByBudget(budget=6))

    alternatives = store.considered_alternatives()
    causes_by_id = {record.id: record.cause for record in alternatives}

    assert "budget-pruned" in causes_by_id
    assert isinstance(causes_by_id["budget-pruned"], PrunedByBudget)
    assert "refuted" in causes_by_id


def test_merged_into_carries_the_target_hypothesis_id() -> None:
    store, hid = _store_with_one_hypothesis()
    record = store.kill(hid, MergedInto(hypothesis_id="survivor"))
    assert isinstance(record.cause, MergedInto)
    assert record.cause.hypothesis_id == "survivor"
