"""Tests for golden-set versioning and immutability.

A golden set is a measuring instrument, so these target the ways an instrument
silently stops being the one your old numbers were measured on: editing it in
place, swapping a clip underneath the manifest, and citing a legacy set as a
product metric.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import IronConfig
from src.data.golden import (
    Condition,
    Domain,
    GoldenClip,
    GoldenSet,
    GoldenSetError,
    SiteZeroPlan,
    Status,
    available_versions,
    load_golden_set,
    write_golden_set,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SEEDED = REPO_ROOT / "configs" / "golden"


def clip(name: str, sha: str = "a" * 64, **kw: object) -> GoldenClip:
    return GoldenClip(clip_id=name, content_sha=sha, **kw)  # type: ignore[arg-type]


def a_set(*clips: GoldenClip, version: str = "vtest") -> GoldenSet:
    return GoldenSet(
        version=version, domain=Domain.INDOOR, status=Status.ACTIVE, clips=clips
    )


# ---------------------------------------------------------------------------
# The seeded manifests
# ---------------------------------------------------------------------------


def test_both_versions_are_seeded() -> None:
    assert set(available_versions(SEEDED)) == {"v1-driving", "v2-indoor"}


def test_driving_set_is_legacy_and_refuses_product_use() -> None:
    """The finding that forced this module, pinned.

    A driving-domain number reported as a product metric is not a small
    error — it is a real, reproducible measurement of something the product
    does not do.
    """
    legacy = load_golden_set(SEEDED, "v1-driving")
    assert legacy.domain is Domain.DRIVING
    assert legacy.status is Status.LEGACY
    assert legacy.usable_for_product_metrics is False
    with pytest.raises(GoldenSetError, match="must not back a product metric"):
        legacy.require_product_usable()


def test_indoor_set_is_active_and_supersedes_driving() -> None:
    indoor = load_golden_set(SEEDED, "v2-indoor")
    assert indoor.domain is Domain.INDOOR
    assert indoor.status is Status.ACTIVE
    assert indoor.supersedes == "v1-driving"
    indoor.require_product_usable()


def test_config_selects_the_indoor_set() -> None:
    config = IronConfig.load()
    assert config.eval.golden_set_version == "v2-indoor"
    assert config.eval.allow_legacy_golden_set is False


def test_indoor_set_is_empty_and_that_is_honest() -> None:
    """Empty until Site Zero lands. An empty set measuring nothing is the
    correct state; a populated-looking one would be worse."""
    assert load_golden_set(SEEDED, "v2-indoor").clips == ()


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_writing_over_an_existing_manifest_is_refused(tmp_path: Path) -> None:
    write_golden_set(tmp_path, a_set(clip("c1")))
    with pytest.raises(GoldenSetError, match="immutable"):
        write_golden_set(tmp_path, a_set(clip("c1"), clip("c2")))


def test_adding_clips_mints_a_new_version() -> None:
    original = a_set(clip("c1"), version="v2-indoor")
    grown = original.with_clips((clip("c2", sha="b" * 64),), version="v3-indoor")

    assert len(original.clips) == 1, "the original must be untouched"
    assert len(grown.clips) == 2
    assert grown.supersedes == "v2-indoor"
    assert grown.set_sha != original.set_sha


def test_adding_under_the_same_version_is_an_error() -> None:
    original = a_set(clip("c1"), version="v2-indoor")
    with pytest.raises(GoldenSetError, match="editing is an error"):
        original.with_clips((clip("c2"),), version="v2-indoor")


def test_a_swapped_clip_changes_the_set_sha() -> None:
    """Content-addressing through members: the instrument's identity is its
    contents, not its name."""
    before = a_set(clip("c1", sha="a" * 64))
    after = a_set(clip("c1", sha="f" * 64))
    assert before.set_sha != after.set_sha


def test_editing_a_manifest_on_disk_is_detected(tmp_path: Path) -> None:
    """The set_sha check catches a hand-edited manifest at load time.

    Without it, an edit would silently invalidate every number previously
    reported against the set, undetectably.
    """
    path = write_golden_set(tmp_path, a_set(clip("c1"), clip("c2")))
    payload = json.loads(path.read_text())
    payload["clips"].pop()  # remove a clip, leave the recorded set_sha
    path.write_text(json.dumps(payload))

    with pytest.raises(GoldenSetError, match="does not match its recorded"):
        load_golden_set(tmp_path, "vtest")


def test_duplicate_clip_ids_are_rejected() -> None:
    """A clip counted twice silently weights the metric toward it."""
    with pytest.raises(GoldenSetError, match="duplicate clip_id"):
        a_set(clip("c1"), clip("c1", sha="b" * 64))


def test_a_clip_without_a_content_sha_is_rejected() -> None:
    with pytest.raises(GoldenSetError, match="content_sha"):
        GoldenClip(clip_id="c1", content_sha="")


# ---------------------------------------------------------------------------
# Condition coverage
# ---------------------------------------------------------------------------


def test_coverage_reports_zeroes_not_just_presences() -> None:
    """An absent condition is exactly what you want to see."""
    golden = a_set(clip("c1", conditions=(Condition.DAYLIGHT,)))
    coverage = golden.condition_coverage()
    assert coverage[Condition.DAYLIGHT] == 1
    assert coverage[Condition.FAR_FIELD] == 0
    assert set(coverage) == set(Condition)


def test_missing_conditions_lists_the_gap() -> None:
    golden = a_set(clip("c1", conditions=(Condition.DAYLIGHT,)))
    missing = golden.missing_conditions()
    assert Condition.DAYLIGHT not in missing
    assert Condition.LENS_SMUDGE in missing


def test_site_zero_plan_reports_the_capture_gap() -> None:
    """The plan is data so the gap is computed, not re-estimated from prose."""
    empty = load_golden_set(SEEDED, "v2-indoor")
    gap = SiteZeroPlan().gap(empty)
    assert gap["complete"] is False
    assert gap["clips_short"] == 30
    assert len(gap["conditions_short"]) == len(Condition)


def test_a_complete_set_reports_complete() -> None:
    plan = SiteZeroPlan(target_clips=2, min_clips_per_condition=1)
    golden = a_set(
        clip("c1", conditions=tuple(Condition)),
        clip("c2", sha="b" * 64, conditions=tuple(Condition)),
    )
    assert plan.gap(golden)["complete"] is True


# ---------------------------------------------------------------------------
# The eval entrypoint
# ---------------------------------------------------------------------------


def test_eval_report_refuses_an_empty_set() -> None:
    """An empty set reporting no failures is not a passing grade."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import eval_report

    assert eval_report.main([]) == 1


def test_eval_report_names_the_legacy_domain() -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import eval_report

    assert eval_report.main(["--version", "v1-driving"]) == 1
