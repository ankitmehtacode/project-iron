"""Tests for the synthetic indoor generator and the scorecard.

The generator is deterministic and its GT is analytic, so these assert the
properties a downstream metric depends on: exact depth, exact occlusion, and
scene coverage that actually contains the hard cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gen_synthetic_indoor as gen  # noqa: E402

from src.data.golden import Condition  # noqa: E402


def test_asset_clearance_covers_every_asset() -> None:
    """Synthetic is not a license. Every asset needs a recorded origin."""
    assert gen.ASSET_CLEARANCE
    for asset in gen.ASSET_CLEARANCE:
        assert {"asset", "origin", "license", "smpl_derived"} <= set(asset)
        assert asset["origin"], f"{asset['asset']} has no recorded origin"


def test_no_asset_is_smpl_derived() -> None:
    """The SMPL trap, asserted rather than assumed.

    SMPL-derived human assets carry a commercial-use requirement that survives
    being rendered. Everything here is generated from primitives.
    """
    for asset in gen.ASSET_CLEARANCE:
        assert asset["smpl_derived"].lower().startswith("no")


def test_scene_set_covers_the_hard_cases() -> None:
    """Overlap and a coverage gap cannot be added by annotation later."""
    scenes = gen.build_scenes(frames=8, fps=12.0)
    names = {s.name for s in scenes}
    assert "overlap_pair_2agents" in names
    assert "coverage_gap_3agents" in names

    overlap = next(s for s in scenes if s.name == "overlap_pair_2agents")
    assert len(overlap.cameras) == 2, "an overlap pair needs two cameras"

    conditions = {c for s in scenes for c in s.conditions}
    for required in (
        Condition.CROWDED,
        Condition.FAR_FIELD,
        Condition.LIGHTS_TRANSIENT,
        Condition.GLARE,
        Condition.BLIND_SPOT_TRAVERSAL,
        Condition.OCCLUSION_FURNITURE,
    ):
        assert required in conditions, f"{required.value} not covered"


def test_both_resolutions_are_present() -> None:
    """The product claims 720p and 1080p, so both must be measurable."""
    resolutions = {
        (c.width, c.height) for s in gen.build_scenes(8, 12.0) for c in s.cameras
    }
    assert (1280, 720) in resolutions
    assert (1920, 1080) in resolutions


def test_agent_count_spans_the_declared_range() -> None:
    counts = {len(s.agents) for s in gen.build_scenes(8, 12.0)}
    assert min(counts) <= 2 and max(counts) >= 6


def test_generation_is_deterministic(tmp_path: Path) -> None:
    """Same seed, same bytes — a fixture that drifts is not a fixture."""
    first = gen.generate(tmp_path / "a", frames=4, fps=12.0, seed=7)
    second = gen.generate(tmp_path / "b", frames=4, fps=12.0, seed=7)
    shas_a = {c["clip_id"]: c["content_sha"] for c in first["clips"]}
    shas_b = {c["clip_id"]: c["content_sha"] for c in second["clips"]}
    assert shas_a == shas_b


def test_depth_is_metric_and_positive(tmp_path: Path) -> None:
    """Analytic depth in metres, so it can back a metric claim."""
    gen.generate(tmp_path, frames=4, fps=12.0, seed=3)
    clip = next(tmp_path.glob("*.npz"))
    with np.load(clip) as data:
        depth = np.asarray(data["depth_m"])
    assert np.all(depth > 0), "non-positive depth is behind the camera"
    assert np.all(np.isfinite(depth))
    assert 0.1 < float(np.median(depth)) < 100.0


def test_occlusion_flags_exist_and_vary(tmp_path: Path) -> None:
    """An occlusion flag that is constant tests nothing."""
    gen.generate(tmp_path, frames=8, fps=12.0, seed=5)
    any_variation = False
    for clip in tmp_path.glob("*.npz"):
        with np.load(clip) as data:
            occ = np.asarray(data["track_occluded"])
        if occ.size and 0 < occ.mean() < 1:
            any_variation = True
    assert any_variation, "no clip has partial occlusion; the set is too easy"


def test_intrinsics_are_exact_not_guessed(tmp_path: Path) -> None:
    """Unlike compute_intrinsics, these are known rather than invented."""
    gen.generate(tmp_path, frames=4, fps=12.0, seed=9)
    clip = next(tmp_path.glob("*.npz"))
    with np.load(clip) as data:
        fx, fy, cx, cy = np.asarray(data["intrinsics"])
        extrinsics = np.asarray(data["extrinsics"])
    assert fx > 0 and fy > 0
    assert extrinsics.shape == (4, 4)
    np.testing.assert_allclose(extrinsics[3], [0, 0, 0, 1])


def test_manifest_declares_the_synthetic_limitation(tmp_path: Path) -> None:
    """A synthetic scorecard quoted externally is the failure mode."""
    manifest = gen.generate(tmp_path, frames=4, fps=12.0, seed=11)
    assert "MUST NOT be quoted externally" in manifest["supersession"]
    assert manifest["lane"] == "S"
    assert "why_not_kubric" in manifest["renderer"]


def test_scorecard_reports_false_negatives_prominently(tmp_path: Path) -> None:
    """The metric that matters most must be present and correctly directed.

    A missed wake loses the event outright, so false negatives are reported as
    a raw count with higher_is_better=False rather than folded into an
    aggregate that can look healthy.
    """
    from src.config import IronConfig
    from src.data.golden import Domain, GoldenClip, GoldenSet, Status
    from src.data.scorecard import compute

    manifest = gen.generate(tmp_path, frames=8, fps=12.0, seed=13)
    clips = tuple(
        GoldenClip(clip_id=c["clip_id"], content_sha=c["content_sha"])
        for c in manifest["clips"][:2]
    )
    golden = GoldenSet(
        version="vtest", domain=Domain.INDOOR, status=Status.ACTIVE, clips=clips
    )
    card = compute(golden, tmp_path, IronConfig.load().cascade.motion_gate_config())

    names = {m.name for m in card.metrics}
    assert "motion_gate.false_negatives" in names
    assert "motion_gate.recall" in names

    fn = next(m for m in card.metrics if m.name == "motion_gate.false_negatives")
    assert fn.higher_is_better is False
    assert card.clips_scored == 2


def test_scorecard_reports_set_difficulty_not_just_score(tmp_path: Path) -> None:
    """A perfect score on easy clips says nothing; difficulty must be visible."""
    from src.config import IronConfig
    from src.data.golden import Domain, GoldenClip, GoldenSet, Status
    from src.data.scorecard import compute

    manifest = gen.generate(tmp_path, frames=8, fps=12.0, seed=17)
    golden = GoldenSet(
        version="vtest",
        domain=Domain.INDOOR,
        status=Status.ACTIVE,
        clips=(
            GoldenClip(
                clip_id=manifest["clips"][0]["clip_id"],
                content_sha=manifest["clips"][0]["content_sha"],
            ),
        ),
    )
    card = compute(golden, tmp_path, IronConfig.load().cascade.motion_gate_config())
    assert any(m.name == "gt.occluded_track_fraction" for m in card.metrics)


def test_static_background_is_bit_identical_across_frames(tmp_path: Path) -> None:
    """A wall does not shimmer.

    Regression for a defect found while reading the first scorecard: the
    surface texture was drawn inside the frame loop, so every frame resampled
    it. That put sigma=3 grain on 83% of the pixels of a static scene and
    turned a declared surface property into undeclared sensor noise — which
    the motion gate was then scored against. Pixels no agent ever touches must
    be identical in every frame.
    """
    gen.generate(tmp_path, frames=6, fps=12.0, seed=23)
    clip = tmp_path / "near_static__cam_a.npz"
    with np.load(clip) as data:
        rgb = np.asarray(data["rgb"])
        instances = np.asarray(data["instances"])

    never_agent = (instances == 0).all(axis=0)
    assert never_agent.sum() > 1000, "not enough background to test"
    background = rgb[:, never_agent, :]
    np.testing.assert_array_equal(
        background,
        np.broadcast_to(background[0], background.shape),
        err_msg="background pixels changed between frames; texture is not static",
    )


def test_content_sha_reproduces_across_processes(tmp_path: Path) -> None:
    """The end-to-end version of the above, run in real subprocesses.

    Checks the property rather than the implementation, so it still holds if
    the seeding is rewritten. Two interpreters with different hash salts must
    agree on the bytes.
    """
    import json
    import subprocess
    import sys

    script = (
        "import json,sys;"
        f"sys.path.insert(0,{str(REPO_ROOT)!r});"
        f"sys.path.insert(0,{str(REPO_ROOT / 'scripts')!r});"
        "import gen_synthetic_indoor as g;"
        "m=g.generate(__import__('pathlib').Path(sys.argv[1]),4,12.0,29);"
        "print(json.dumps({c['clip_id']:c['content_sha'] for c in m['clips']}))"
    )
    results = []
    for index, salt in enumerate(("0", "1")):
        env = {**__import__("os").environ, "PYTHONHASHSEED": salt}
        out = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / f"run{index}")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        results.append(json.loads(out.stdout.strip().splitlines()[-1]))
    assert results[0] == results[1], "content_sha depends on the hash salt"
