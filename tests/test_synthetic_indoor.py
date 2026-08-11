"""Tests for the synthetic indoor generator and the scorecard.

The generator is deterministic and its GT is analytic, so these assert the
properties a downstream metric depends on: exact depth, exact occlusion, and
scene coverage that actually contains the hard cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

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


@pytest.mark.slow
def test_generation_is_deterministic(tmp_path: Path) -> None:
    """Same seed, same bytes — a fixture that drifts is not a fixture."""
    first = gen.generate(tmp_path / "a", frames=4, fps=12.0, seed=7)
    second = gen.generate(tmp_path / "b", frames=4, fps=12.0, seed=7)
    shas_a = {c["clip_id"]: c["content_sha"] for c in first["clips"]}
    shas_b = {c["clip_id"]: c["content_sha"] for c in second["clips"]}
    assert shas_a == shas_b


@pytest.mark.slow
def test_depth_is_metric_and_positive(tmp_path: Path) -> None:
    """Analytic depth in metres, so it can back a metric claim."""
    gen.generate(tmp_path, frames=4, fps=12.0, seed=3)
    clip = next(tmp_path.glob("*.npz"))
    with np.load(clip) as data:
        depth = np.asarray(data["depth_m"])
    assert np.all(depth > 0), "non-positive depth is behind the camera"
    assert np.all(np.isfinite(depth))
    assert 0.1 < float(np.median(depth)) < 100.0


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_manifest_declares_the_synthetic_limitation(tmp_path: Path) -> None:
    """A synthetic scorecard quoted externally is the failure mode."""
    manifest = gen.generate(tmp_path, frames=4, fps=12.0, seed=11)
    assert "MUST NOT be quoted externally" in manifest["supersession"]
    assert manifest["lane"] == "S"
    assert "why_not_kubric" in manifest["renderer"]


@pytest.mark.slow
def test_scorecard_reports_false_negatives_prominently(tmp_path: Path) -> None:
    """The metric that matters most must be present and correctly directed.

    A missed wake loses the event outright, so the miss count is reported
    as a raw count with higher_is_better=False rather than folded into an
    aggregate that can look healthy. Day 14 renamed the emitted metrics to
    gate.miss_cost / gate.recall_retained (see FOUNDATION_REPORT.md's
    Day-14 section, Objective 3) — same underlying counts, reframed
    against wake_fraction/compute_saved instead of precision/F1, which
    Day 12 showed do not discriminate a real gate from always-wake on
    this fixture.
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
    assert "gate.miss_cost" in names
    assert "gate.recall_retained" in names
    assert "gate.wake_fraction" in names
    assert "gate.compute_saved" in names

    fn = next(m for m in card.metrics if m.name == "gate.miss_cost")
    assert fn.higher_is_better is False
    assert card.clips_scored == 2


@pytest.mark.slow
def test_scorecard_carries_a_live_measurement_environment(tmp_path: Path) -> None:
    """Day 17: the environment that scored THIS run, not the envelope's.

    ``card.envelope["envelope_measured_stack"]`` is whatever stack calibrated
    the loaded envelope file, potentially days earlier. A scorecard produced
    by a different (drifted) interpreter would inherit that string
    unchanged and report it as if it described itself. This field is
    captured fresh, every call, from the running process.
    """
    import sys

    from src.config import IronConfig
    from src.data.golden import Domain, GoldenClip, GoldenSet, Status
    from src.data.scorecard import compute

    manifest = gen.generate(tmp_path, frames=8, fps=12.0, seed=13)
    clips = tuple(
        GoldenClip(clip_id=c["clip_id"], content_sha=c["content_sha"])
        for c in manifest["clips"][:1]
    )
    golden = GoldenSet(
        version="vtest", domain=Domain.INDOOR, status=Status.ACTIVE, clips=clips
    )
    card = compute(golden, tmp_path, IronConfig.load().cascade.motion_gate_config())

    env = card.measurement_environment
    assert env["sys_prefix"] == sys.prefix
    assert env["sys_executable"] == sys.executable
    assert env["library_versions"]["numpy"] not in ("", "unknown", "not installed")
    assert "measurement_environment" in card.as_dict()


@pytest.mark.slow
def test_scorecard_reports_per_condition_wake_fraction(tmp_path: Path) -> None:
    """Objective 3 (Day 15): a single aggregate wake_fraction across mixed
    conditions must not be the headline -- card.per_condition always
    carries all four buckets, even ones with zero clips.
    """
    from src.config import IronConfig
    from src.data.golden import Domain, GoldenClip, GoldenSet, Status
    from src.data.scorecard import _CONDITION_BUCKETS, compute

    manifest = gen.generate(tmp_path, frames=8, fps=12.0, seed=19)
    clips = tuple(
        GoldenClip(
            clip_id=c["clip_id"],
            content_sha=c["content_sha"],
            conditions=(Condition.DAYLIGHT, Condition.SINGLE_PERSON),
        )
        for c in manifest["clips"][:2]
    )
    golden = GoldenSet(
        version="vtest", domain=Domain.INDOOR, status=Status.ACTIVE, clips=clips
    )
    card = compute(golden, tmp_path, IronConfig.load().cascade.motion_gate_config())

    assert set(card.per_condition) == set(_CONDITION_BUCKETS)
    # Every clip tagged daylight/single_person lands in "occupied", and every
    # other bucket is present but empty -- not silently omitted.
    assert card.per_condition["occupied"]["clips"] == 2.0
    for bucket in ("empty", "night", "degenerate"):
        assert card.per_condition[bucket]["clips"] == 0.0
        assert np.isnan(card.per_condition[bucket]["wake_fraction"])


@pytest.mark.slow
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


@pytest.mark.slow
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


def test_stationary_agent_silhouette_is_bit_identical_across_frames(
    tmp_path: Path,
) -> None:
    """A person standing still does not shimmer either — not just the wall.

    Day 17 STRUCTURAL test. The regression this guards: gait phase was
    ``sin(2*pi*(t*4 + agent_id))`` -- a function of ELAPSED TIME, so a
    zero-velocity agent's legs kept swinging every frame even though
    ``position_at(t)`` never moved it. ``test_static_background_is_bit_
    identical_across_frames`` above only ever checked pixels no agent
    touches, so this passed undetected: the one clip v4-gate (Day 16)
    authored specifically to be motionless
    (``long_static_occupant``, speed_scale=0.0) rendered a silhouette that
    changed on ~92% of its frames, and the Day-9 silhouette-diff ground
    truth (``gt_moved_from_render``) scored it as very nearly always
    moving. Fixed: gait phase now drives off ``Agent.progress_at(t)``
    (distance along the path, clamped at arrival) rather than off ``t``
    directly, so a speed_scale=0.0 agent's phase is the same constant every
    frame. This asserts the frames touching the agent, not just the
    background — the region the previous test structurally could not see.
    """
    scenes = gen.build_scenes_v4_gate(frames=12, fps=12.0)
    scene = next(s for s in scenes if s.name == "long_static_occupant")
    assert (
        scene.agents and scene.agents[0].speed_scale == 0.0
    ), "long_static_occupant must stay the zero-velocity fixture this test targets"
    camera = scene.cameras[0]
    rng = __import__("numpy").random.default_rng(23)
    texture = rng.normal(0.0, 3.0, size=(camera.height, camera.width))

    frames = [
        gen.render_frame(scene, camera, frame, texture) for frame in range(scene.frames)
    ]

    for key in ("rgb", "depth_m", "instances"):
        first = frames[0][key]
        for index, later in enumerate(frames[1:], start=1):
            np.testing.assert_array_equal(
                later[key],
                first,
                err_msg=f"{key} differs at frame {index} for a speed_scale=0.0 "
                "agent — its silhouette must be bit-identical to frame 0, "
                "not just the background's",
            )


# ---------------------------------------------------------------------------
# Day 23, Objective 2 -- MultiSegmentAgent / PathSegment
# ---------------------------------------------------------------------------


def test_multi_segment_agent_single_leg_matches_plain_agent() -> None:
    """A one-segment MultiSegmentAgent must reduce to the same trajectory
    a plain Agent produces -- the generalisation adds no new behaviour for
    the case it subsumes."""
    plain = gen.Agent(0, (0.0, 0.0), (4.0, 8.0), speed_scale=1.0)
    multi = gen.MultiSegmentAgent(
        0, (0.0, 0.0), [gen.PathSegment(end=(4.0, 8.0), duration=1.0)]
    )
    for t in (0.0, 0.1, 0.37, 0.5, 0.9, 1.0):
        np.testing.assert_allclose(multi.position_at(t), plain.position_at(t))
        assert multi.progress_at(t) == pytest.approx(plain.progress_at(t))


def test_multi_segment_agent_pause_holds_position_and_progress() -> None:
    """A zero-length (pause) leg: position AND progress_at (gait phase)
    must stay exactly constant for its whole duration -- the Day-17
    invariant, generalised to a multi-leg path (Day 23)."""
    agent = gen.MultiSegmentAgent(
        0,
        (0.0, 0.0),
        [
            gen.PathSegment(end=(5.0, 0.0), duration=0.5),
            gen.PathSegment(end=(5.0, 0.0), duration=0.5),  # pause
        ],
    )
    held_position = agent.position_at(0.5)
    held_progress = agent.progress_at(0.5)
    for t in (0.6, 0.7, 0.85, 1.0):
        np.testing.assert_allclose(agent.position_at(t), held_position)
        assert agent.progress_at(t) == held_progress


def test_multi_segment_agent_progress_is_monotonic_and_reaches_one() -> None:
    agent = gen.MultiSegmentAgent(
        0,
        (0.0, 0.0),
        [
            gen.PathSegment(end=(3.0, 0.0), duration=0.3),
            gen.PathSegment(end=(3.0, 0.0), duration=0.2),  # pause
            gen.PathSegment(end=(3.0, 5.0), duration=0.5),
        ],
    )
    ts = np.linspace(0.0, 1.0, 50)
    progress = [agent.progress_at(float(t)) for t in ts]
    assert progress == sorted(progress)
    assert progress[0] == 0.0
    assert progress[-1] == 1.0


def test_multi_segment_agent_ease_out_slows_before_stopping() -> None:
    """Gradual deceleration: displacement over the FINAL fraction of an
    ease_out leg must be smaller than over an equal-sized earlier
    fraction -- speed genuinely ramps down, not just reaches the same
    endpoint on the same schedule as a linear (abrupt) leg."""
    eased = gen.MultiSegmentAgent(
        0, (0.0, 0.0), [gen.PathSegment(end=(10.0, 0.0), duration=1.0, ease_out=True)]
    )
    early_disp = np.linalg.norm(eased.position_at(0.55) - eased.position_at(0.5))
    late_disp = np.linalg.norm(eased.position_at(1.0) - eased.position_at(0.95))
    assert late_disp < early_disp


def test_multi_segment_agent_rejects_empty_segments() -> None:
    try:
        gen.MultiSegmentAgent(0, (0.0, 0.0), [])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Day 23, Objective 2 -- build_scenes_v5_cessation and its mint-time gate
# ---------------------------------------------------------------------------


def test_v5_cessation_scenes_stay_in_frame_and_unoccluded() -> None:
    """Every stop event must actually be observable -- geometry that pushes
    the agent out of frame or behind furniture would make cessation volume
    that mint_golden_set's observability floor then refuses anyway."""
    scenes = gen.build_scenes_v5_cessation(24, 12.0)
    assert len(scenes) >= 15
    for scene in scenes:
        camera = scene.cameras[0]
        agent = scene.agents[0]
        for frame in range(scene.frames):
            t = frame / max(1, scene.frames - 1)
            world = agent.position_at(t)
            u, v, _z = gen.project(world, camera)
            in_frame = (
                np.isfinite(u) and 0 <= u < camera.width and 0 <= v < camera.height
            )
            occluded = any(
                f.occludes(world, np.array(camera.position)) for f in scene.furniture
            )
            assert (
                in_frame and not occluded
            ), f"{scene.name} frame {frame}: out of frame or occluded"


def test_v5_cessation_clears_its_own_volume_requirement() -> None:
    """The acceptance criterion Objective 2 states explicitly: >=200
    cessation frames, every non-exempt regime >=30, measured directly from
    classify_track over every scene's raw track -- not asserted, computed."""
    from src.estimator.regime import MOTION_REGIMES, classify_track

    scenes = gen.build_scenes_v5_cessation(24, 12.0)
    counts = {r: 0 for r in MOTION_REGIMES}
    for scene in scenes:
        dt_s = 1.0 / scene.fps
        agent = scene.agents[0]
        track = np.array(
            [
                agent.position_at(f / max(1, scene.frames - 1))
                for f in range(scene.frames)
            ]
        )
        for regime in classify_track(track, dt_s):
            counts[regime] += 1

    assert counts["cessation"] >= 200
    for regime in MOTION_REGIMES:
        if regime in gen.V5_CESSATION_EXEMPT_REGIMES:
            continue
        assert counts[regime] >= 30, f"{regime}: only {counts[regime]} frames"


def test_v5_cessation_includes_stop_then_restart_scenes() -> None:
    scenes = gen.build_scenes_v5_cessation(24, 12.0)
    names = {s.name for s in scenes}
    assert "stop_then_restart_radial" in names
    assert "stop_then_restart_lateral" in names
    restart_scene = next(s for s in scenes if s.name == "stop_then_restart_radial")
    assert len(restart_scene.agents[0].segments) == 4  # walk, pause, walk, pause


def test_v5_cessation_varies_approach_speed_deceleration_and_distance() -> None:
    """The day's own required axes of variation, checked directly rather
    than trusted from the scene names."""
    scenes = gen.build_scenes_v5_cessation(24, 12.0)
    by_name = {s.name: s for s in scenes}

    durations = {
        round(s.agents[0].segments[0].duration * s.frames / s.fps, 1) for s in scenes
    }
    assert len(durations) >= 3, "approach durations do not actually vary"

    ease_flags = {s.agents[0].segments[0].ease_out for s in scenes}
    assert ease_flags == {True, False}, "both deceleration profiles must appear"

    radial_far = by_name["radial_slow_far_gradual"].agents[0].segments[0].end[1]
    radial_near = by_name["radial_slow_near_abrupt"].agents[0].segments[0].end[1]
    assert radial_far != radial_near, "radial distance-from-camera does not vary"


@pytest.mark.slow
def test_enforce_regime_volume_passes_on_v5_cessation(tmp_path: Path) -> None:
    manifest = gen.generate(
        tmp_path, frames=24, fps=12.0, seed=20260811, scene_set="v5-cessation"
    )
    counts = gen.enforce_regime_volume(
        tmp_path,
        manifest,
        min_cessation_frames=200,
        min_other_regime_frames=30,
        exempt_regimes=gen.V5_CESSATION_EXEMPT_REGIMES,
    )
    assert counts["cessation"] >= 200


@pytest.mark.slow
def test_enforce_regime_volume_refuses_a_set_with_no_cessation(tmp_path: Path) -> None:
    """v3's own scenes never stop (Day 21/22: constant velocity throughout)
    -- exactly the set this gate exists to refuse if someone tried to mint
    it as a cessation source."""
    manifest = gen.generate(tmp_path, frames=8, fps=12.0, seed=31, scene_set="v3")
    try:
        gen.enforce_regime_volume(
            tmp_path, manifest, min_cessation_frames=200, min_other_regime_frames=30
        )
        raise AssertionError("expected RegimeVolumeError")
    except gen.RegimeVolumeError as exc:
        assert "cessation" in str(exc)


def test_stationary_v5_cessation_agent_silhouette_is_bit_identical() -> None:
    """The Day-17 stopped-agent bit-identity check, reproduced for
    v5-cessation's own stop segments (Day 23) -- a paused
    MultiSegmentAgent must not shimmer any more than a speed_scale=0.0
    plain Agent does."""
    scenes = gen.build_scenes_v5_cessation(24, 12.0)
    scene = next(s for s in scenes if s.name == "radial_long_stop_near")
    camera = scene.cameras[0]
    rng = np.random.default_rng(23)
    texture = rng.normal(0.0, 3.0, size=(camera.height, camera.width))

    # The pause leg is segments[1]; find frames whose normalised time falls
    # inside it, away from its edges (so easing/finite-difference boundary
    # effects at the leg transition itself do not confound the assertion).
    agent = scene.agents[0]
    cumulative = agent._cumulative_duration_fractions()
    pause_start, pause_end = cumulative[1], cumulative[2]
    frames_in_pause = [
        f
        for f in range(scene.frames)
        if pause_start + 0.02 < f / max(1, scene.frames - 1) < pause_end - 0.02
    ]
    assert len(frames_in_pause) >= 3, "not enough held frames to test"

    rendered = [
        gen.render_frame(scene, camera, frame, texture) for frame in frames_in_pause
    ]
    for key in ("rgb", "depth_m", "instances"):
        first = rendered[0][key]
        for index, later in enumerate(rendered[1:], start=1):
            np.testing.assert_array_equal(
                later[key],
                first,
                err_msg=f"{key} differs between held frames "
                f"{frames_in_pause[0]} and {frames_in_pause[index]} -- a "
                "paused MultiSegmentAgent must not shimmer",
            )


@pytest.mark.slow
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
