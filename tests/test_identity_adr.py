"""Guards on ADR 0001 and the consent promise it is paired with.

These are tests on documents, which is unusual and deliberate. The §7
limitation in the consent template is a promise made to a named person who
signed it, and ADR 0001 is the architectural reason that promise can one day
be strengthened. The failure mode worth catching is someone softening the
limitation because the ADR *describes* a better position, before the adapter
architecture exists and a withdrawal drill has actually been run — which would
turn a documented plan into an unearned assurance.

They assert on short distinctive phrases rather than whole paragraphs, so the
prose can be edited freely and only the load-bearing claims are pinned.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ADR = REPO_ROOT / "docs" / "adr" / "0001-identity-adapter-architecture.md"
CONSENT = REPO_ROOT / "docs" / "site_zero_consent_TEMPLATE.md"


@pytest.fixture(scope="module")
def adr() -> str:
    return ADR.read_text()


@pytest.fixture(scope="module")
def consent() -> str:
    return CONSENT.read_text()


def test_adr_exists_and_is_accepted(adr: str) -> None:
    assert "**Status:** Accepted" in adr


def test_adr_states_the_frozen_backbone_constraint(adr: str) -> None:
    """The single sentence the whole document exists to fix in place."""
    assert "frozen" in adr
    assert "never lives in fine-tuned backbone weights" in adr


def test_adr_keeps_the_expensive_deletion_tier_empty(adr: str) -> None:
    """Tier 3 must be recorded as not-applicable, not as a fallback.

    If backbone retraining ever becomes a real tier, the erasure argument
    collapses back to "weeks, so in practice never" and the decision has been
    silently reversed.
    """
    assert "Not applicable by design" in adr


def test_adr_requires_the_gap_to_be_measured_not_assumed(adr: str) -> None:
    """An accuracy cost accepted without measurement is a guess."""
    assert "must be measured" in adr


def test_adr_requires_a_withdrawal_drill_before_enrolment(adr: str) -> None:
    """An erasure path that has never been run is a claim, not a capability."""
    assert "withdrawal drill" in adr.lower()


def test_consent_still_states_the_honest_limitation(consent: str) -> None:
    """The promise must not be strengthened ahead of the architecture.

    This is the assertion the whole file exists for. Today no adapter exists,
    so deleting footage genuinely does not undo training, and the participant
    must be told that in plain words.
    """
    assert "We cannot reverse training." in consent


def test_consent_points_at_the_adr(consent: str) -> None:
    """The two documents must be revised together, so each names the other."""
    assert "0001-identity-adapter-architecture.md" in consent


def test_consent_records_the_precondition_for_softening_it(consent: str) -> None:
    assert "Do not soften this paragraph" in consent


def test_counsel_question_is_still_open_in_both(adr: str, consent: str) -> None:
    """Neither document may quietly answer a question only counsel can."""
    assert "Counsel:" in consent
    assert "For counsel:" in adr
