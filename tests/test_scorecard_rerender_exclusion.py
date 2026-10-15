"""Scorecards that include Cosmos re-render frames record their post-cut
exclusion window, and two different windows are not comparable (Day 42,
Objective A2)."""

from __future__ import annotations

from typing import Any

import pytest

from src.data.scorecard import OBSERVABILITY_BOX_AREA, Scorecard, ScorecardError


def _card(**kwargs: Any) -> Scorecard:
    return Scorecard(
        golden_set_version="vtest",
        golden_set_sha="a" * 64,
        domain="indoor",
        clips_scored=1,
        observability_proxy=OBSERVABILITY_BOX_AREA,
        envelope={"envelope_sha": "e" * 64},
        **kwargs,
    )


def _window(frames: int) -> dict[str, Any]:
    return {"frames_after_cut": frames, "source": "day42_rerender_transient.json"}


def test_rerender_frames_without_a_window_are_refused() -> None:
    with pytest.raises(ScorecardError, match="no post-cut exclusion window"):
        _card(includes_rerender_frames=True)


def test_window_without_a_source_is_refused() -> None:
    with pytest.raises(ScorecardError, match="source"):
        _card(
            includes_rerender_frames=True, rerender_exclusion={"frames_after_cut": 30}
        )


def test_negative_window_is_refused() -> None:
    with pytest.raises(ScorecardError, match="negative"):
        _card(includes_rerender_frames=True, rerender_exclusion=_window(-1))


def test_window_on_a_card_without_rerender_frames_is_refused() -> None:
    with pytest.raises(ScorecardError, match="declares no"):
        _card(rerender_exclusion=_window(0))


def test_zero_is_an_explicit_statement_and_is_serialised() -> None:
    card = _card(includes_rerender_frames=True, rerender_exclusion=_window(0))
    payload = card.as_dict()
    assert payload["includes_rerender_frames"] is True
    assert payload["rerender_exclusion"]["frames_after_cut"] == 0


def test_different_windows_refuse_comparison() -> None:
    a = _card(includes_rerender_frames=True, rerender_exclusion=_window(0))
    b = _card(includes_rerender_frames=True, rerender_exclusion=_window(240))
    with pytest.raises(ScorecardError, match="re-render frames differently"):
        a.require_comparable(b)
    with pytest.raises(ScorecardError, match="re-render frames differently"):
        b.require_comparable(a)


def test_rerender_card_and_source_only_card_refuse_comparison() -> None:
    with pytest.raises(ScorecardError, match="re-render frames differently"):
        _card(
            includes_rerender_frames=True, rerender_exclusion=_window(240)
        ).require_comparable(_card())


def test_same_window_is_comparable() -> None:
    _card(
        includes_rerender_frames=True, rerender_exclusion=_window(240)
    ).require_comparable(
        _card(includes_rerender_frames=True, rerender_exclusion=_window(240))
    )
    _card().require_comparable(_card())
