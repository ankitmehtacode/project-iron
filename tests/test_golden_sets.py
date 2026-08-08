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


def test_every_minted_version_is_retained() -> None:
    """Superseded sets are kept, never deleted.

    v2 is retained even though v3 replaced it: it is the record of what was
    measured before, and deleting it would erase the instrument that produced
    every number reported against it. v1 is retained on the same grounds.
    """
    assert set(available_versions(SEEDED)) == {
        "v1-driving",
        "v2-indoor",
        "v3-indoor",
        "v4-gate",
    }


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


def test_config_selects_the_active_indoor_set() -> None:
    """The configured set must be indoor, active, and the current one.

    Pinned to v3-indoor deliberately. v2 is still on disk and still loadable,
    so a config left pointing at it would keep scoring against a set whose
    ground truth is only 44% observable while looking entirely normal.
    """
    config = IronConfig.load()
    assert config.eval.golden_set_version == "v3-indoor"
    assert config.eval.allow_legacy_golden_set is False

    active = load_golden_set(SEEDED, config.eval.golden_set_version)
    assert active.domain is Domain.INDOOR
    assert active.status is Status.ACTIVE
    assert active.usable_for_product_metrics is True
    assert active.supersedes == "v2-indoor"


def test_indoor_set_is_populated_from_synthetic_clips_only() -> None:
    """Day 5 asserted this set was empty, and that was honest then.

    It is now populated by ``scripts/gen_synthetic_indoor.py`` so that ``make
    eval`` produces a scorecard at all. The honesty requirement did not go
    away, it moved: every clip must be traceable to the synthetic dataset, so
    that no captured footage can arrive here without the source label
    changing. Site Zero clips still supersede these.
    """
    indoor = load_golden_set(SEEDED, "v2-indoor")
    assert indoor.clips, "set is empty; run gen_synthetic_indoor.py --write-golden"
    assert all(c.source_dataset == "synthetic-indoor-v1" for c in indoor.clips)
    assert "SYNTHETIC-ONLY" in indoor.description


def test_indoor_set_still_declares_its_capture_gap() -> None:
    """Populating with synthetic clips must not read as "done".

    The conditions that need real capture are precisely the appearance-driven
    ones an analytic renderer cannot produce, so they must still be reported
    missing rather than being quietly satisfied by a synthetic stand-in.
    """
    missing = load_golden_set(SEEDED, "v2-indoor").missing_conditions()
    for appearance_only in (
        Condition.LENS_SMUDGE,
        Condition.CLOTHING_CHANGE,
        Condition.SIMILAR_CLOTHING,
    ):
        assert appearance_only in missing


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
    """The plan is data so the gap is computed, not re-estimated from prose.

    Synthetic clips shrink the gap without closing it. The assertion is on the
    arithmetic rather than on a frozen constant, so populating the set further
    updates the expectation instead of breaking the test.
    """
    golden = load_golden_set(SEEDED, "v2-indoor")
    plan = SiteZeroPlan()
    gap = plan.gap(golden)

    assert gap["complete"] is False
    assert gap["clips_short"] == plan.target_clips - len(golden.clips)
    assert 0 < gap["clips_short"] < plan.target_clips, "synthetic clips must count"
    assert gap["conditions_short"], "no renderer covers every condition"


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


def test_eval_report_refuses_an_empty_set(tmp_path: Path) -> None:
    """An empty set reporting no failures is not a passing grade.

    Until Objective 5 this ran against v2-indoor, which was empty at the time.
    Now that the set is populated, the invariant has to be tested against a set
    that is genuinely empty — otherwise the test would have been "fixed" by
    deleting the very refusal it exists to guard.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import eval_report

    write_golden_set(tmp_path, a_set(version="v9-empty"))
    assert eval_report.main(["--version", "v9-empty", "--root", str(tmp_path)]) == 1


@pytest.mark.slow
def test_eval_report_scores_the_populated_indoor_set(
    synthetic_indoor_clips: Path,
) -> None:
    """The other half: a populated set must actually produce a scorecard.

    Guards the opposite failure — a refusal path broad enough to swallow a
    real run would make ``make eval`` permanently green-by-abstention.

    Takes ``synthetic_indoor_clips`` because it used to depend on those clips
    happening to be on disk from a previous ``make eval``. That made it pass on
    a development machine and fail in a fresh clone, so it was testing the
    developer's working directory rather than the repository. The fixture
    renders them when absent.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import eval_report

    assert sorted(synthetic_indoor_clips.glob("*.npz")), (
        "the fixture must have materialised clips; scoring an empty directory "
        "would make this test pass by abstention, which is what it guards"
    )
    assert eval_report.main([]) == 0


def test_eval_report_names_the_legacy_domain() -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import eval_report

    assert eval_report.main(["--version", "v1-driving"]) == 1
