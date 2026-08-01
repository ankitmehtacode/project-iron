"""A golden set whose ground truth is not observable must not be mintable.

v2-indoor scored ``observable_fraction`` 0.4444 — agents spent most of their
frames behind a furniture slab or off the bottom frame edge. Its recall of
1.0 rode on 41 moving frames and was a statement about the set rather than
about the gate, and nobody could see that until the observability partition
existed.

The floor moves that discovery before minting, because after minting the set's
sha is quoted and every metric computed against it inherits the problem.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.golden import (
    MIN_OBSERVABLE_FRACTION,
    Domain,
    GoldenClip,
    GoldenSet,
    GoldenSetError,
    Status,
    load_golden_set,
    mint_golden_set,
)


def _clip(name: str, hard: bool = False) -> GoldenClip:
    return GoldenClip(clip_id=name, content_sha="a" * 64, hard_coverage=hard)


def _set(*clips: GoldenClip, version: str = "v3-test") -> GoldenSet:
    return GoldenSet(
        version=version,
        domain=Domain.INDOOR,
        status=Status.ACTIVE,
        clips=clips,
    )


def test_a_set_below_the_floor_is_refused(tmp_path: Path) -> None:
    """The deliberately bad set. This is v2-indoor's actual number."""
    golden = _set(_clip("a"), _clip("b"))
    with pytest.raises(GoldenSetError, match="below the 0.80 floor"):
        mint_golden_set(tmp_path, golden, {"a": 0.44, "b": 0.44})

    assert not list(tmp_path.glob("*")), (
        "a refused set must leave nothing behind; a half-written manifest is "
        "one someone can still cite"
    )


def test_a_set_above_the_floor_mints(tmp_path: Path) -> None:
    golden = _set(_clip("a"), _clip("b"))
    path = mint_golden_set(tmp_path, golden, {"a": 0.9, "b": 0.85})
    assert path.exists()
    assert load_golden_set(tmp_path, "v3-test").version == "v3-test"


def test_hard_coverage_clips_are_excluded_from_the_floor(tmp_path: Path) -> None:
    """The blind-spot scenario must stay representable.

    Without the exclusion, keeping one genuine coverage gap would drag the
    aggregate down and the only way to mint would be to delete the hard case —
    authoring away the exact scenario the product has to handle.
    """
    golden = _set(_clip("good1"), _clip("good2"), _clip("blindspot", hard=True))
    path = mint_golden_set(
        tmp_path, golden, {"good1": 0.9, "good2": 0.9, "blindspot": 0.05}
    )
    assert path.exists(), (
        "a set with two strong clips and one tagged blind spot must mint; the "
        "0.05 belongs on its own line, not in the aggregate"
    )


def test_hard_coverage_tag_cannot_rescue_a_bad_set(tmp_path: Path) -> None:
    """Tagging the weak clips is not a way under the floor.

    The tag is for scenarios that are deliberately hard, not for ones that came
    out badly. Here the untagged remainder is still below the floor, so it is
    still refused.
    """
    golden = _set(_clip("weak1", hard=True), _clip("weak2"), _clip("weak3"))
    with pytest.raises(GoldenSetError, match="below the 0.80 floor"):
        mint_golden_set(tmp_path, golden, {"weak1": 0.1, "weak2": 0.4, "weak3": 0.5})


def test_a_set_of_nothing_but_blind_spots_is_refused(tmp_path: Path) -> None:
    """Tagging everything would leave the floor with nothing to check."""
    golden = _set(_clip("a", hard=True), _clip("b", hard=True))
    with pytest.raises(GoldenSetError, match="nothing but blind spots"):
        mint_golden_set(tmp_path, golden, {"a": 0.1, "b": 0.1})


def test_an_unmeasured_clip_is_refused(tmp_path: Path) -> None:
    """Acceptance runs on measurements, never on assumptions."""
    golden = _set(_clip("a"), _clip("b"))
    with pytest.raises(GoldenSetError, match="no measured observable fraction"):
        mint_golden_set(tmp_path, golden, {"a": 0.95})


def test_the_floor_is_exactly_where_it_says(tmp_path: Path) -> None:
    """Boundary: at the floor mints, a hair below refuses."""
    golden = _set(_clip("a"))
    assert mint_golden_set(
        tmp_path / "at", golden, {"a": MIN_OBSERVABLE_FRACTION}
    ).exists()
    with pytest.raises(GoldenSetError):
        mint_golden_set(
            tmp_path / "below", golden, {"a": MIN_OBSERVABLE_FRACTION - 0.001}
        )


def test_hard_coverage_survives_a_manifest_round_trip(tmp_path: Path) -> None:
    """The tag must persist, or the exclusion silently stops applying."""
    golden = _set(_clip("plain"), _clip("blindspot", hard=True))
    mint_golden_set(tmp_path, golden, {"plain": 0.9, "blindspot": 0.05})

    reloaded = load_golden_set(tmp_path, "v3-test")
    tagged = {c.clip_id: c.hard_coverage for c in reloaded.clips}
    assert tagged == {"plain": False, "blindspot": True}
