"""The motion gate may only be scored against motion its camera can see.

The defect these tests encode: ``motion_gate_metrics`` derived ground truth
from ``agent_xyz`` — world-space displacement, maximised over *every* agent in
the scene — and compared it against one camera's wake decisions. A camera was
therefore charged a false negative for sleeping through motion that happened
outside its frustum, which is not a defect but the definition of a frustum.

Correcting that alone would produce a second false verdict, so GT motion is
partitioned into three buckets per camera, not two:

* **not observable** — no pixels of the mover reach this sensor (outside the
  frustum, or occluded by static geometry). Excluded from every denominator.
* **observable, below the capability envelope** — the mover is visible and
  unoccluded but subtends fewer gate pixels than ``min_foreground_fraction``
  can resolve. Excluded from the recall denominator and reported on its own
  line as ``envelope.limited_misses``. These are not gate defects; they are the
  camera's capability envelope becoming measurable for the first time.
* **observable, above the envelope** — the only honest recall denominator.

Suppressing the middle bucket into recall would hide the envelope. Counting it
as failure would send someone tuning ``min_foreground_fraction`` down until the
gate wakes on sensor noise. It gets its own number so it can do neither.

Bucket logic is tested on hand-built arrays, where each bucket can be placed
exactly. One test renders the real blind-spot camera, because the regression
being guarded is that the renderer's coverage-gap scene actually produces
unobservable frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import gen_synthetic_indoor as gen  # noqa: E402

from src.config import IronConfig  # noqa: E402
from src.data.scorecard import (  # noqa: E402
    FIRST_AGENT_INSTANCE_ID,
    Observability,
    motion_gate_metrics,
    observability_partition,
)

# The golden set is rendered at 24 frames and the gate warms up for 10. A
# shorter clip leaves nothing after warmup, so the partition would be measured
# over an empty slice and every assertion would pass vacuously.
FRAMES = 24
SEED = 23


@pytest.fixture(scope="module")
def gate_config():  # type: ignore[no-untyped-def]
    return IronConfig.load().cascade.motion_gate_config()


def _write_clip(
    path: Path,
    *,
    agent_areas_px: list[list[int]],
    moves: list[list[bool]],
    size: tuple[int, int] = (720, 1280),
) -> Path:
    """Build a clip whose observability is exactly what the test intends.

    ``agent_areas_px`` is per frame, per agent: how many native pixels that
    agent occupies. ``moves`` is per frame, per agent: whether it displaced
    since the previous frame. Everything else is the minimum the scorer reads.
    """
    height, width = size
    frames = len(agent_areas_px)
    agents = len(agent_areas_px[0]) if frames else 0

    instances = np.zeros((frames, height, width), dtype=np.int32)
    for frame, areas in enumerate(agent_areas_px):
        cursor = 0
        for agent, area in enumerate(areas):
            flat = instances[frame].reshape(-1)
            flat[cursor : cursor + area] = FIRST_AGENT_INSTANCE_ID + agent
            cursor += area

    # Positions are only read through their frame-to-frame delta, so a moving
    # agent is given a 1 m step and a still one none.
    xyz = np.zeros((frames, agents, 3), dtype=np.float32)
    for agent in range(agents):
        travelled = 0.0
        for frame in range(frames):
            if frame and moves[frame][agent]:
                travelled += 1.0
            xyz[frame, agent, 0] = travelled

    np.savez_compressed(
        path,
        rgb=np.zeros((frames, height, width, 3), dtype=np.uint8),
        depth_m=np.ones((frames, height, width), dtype=np.float32),
        instances=instances,
        agent_xyz=xyz,
        track_uv=np.zeros((frames, agents, 2), dtype=np.float32),
        track_occluded=np.zeros((frames, agents), dtype=bool),
        intrinsics=np.array([600.0, 600.0, width / 2, height / 2]),
        extrinsics=np.eye(4),
    )
    return path


def _envelope_native_px(gate_config, size: tuple[int, int] = (720, 1280)) -> float:
    """Native-pixel area sitting exactly on the envelope boundary."""
    gate_px = gate_config.gate_width * gate_config.gate_height
    native_px = size[0] * size[1]
    return gate_config.min_foreground_fraction * gate_px * (native_px / gate_px)


def test_motion_outside_the_frustum_is_excluded_not_charged(
    tmp_path: Path, gate_config
) -> None:
    """Bucket 1: a mover with no pixels on the sensor leaves the denominator."""
    clip = _write_clip(
        tmp_path / "blind.npz",
        agent_areas_px=[[0]] * 12,
        moves=[[False]] + [[True]] * 11,
    )
    partition = observability_partition(clip, gate_config)
    after = partition.after_warmup()

    assert np.all(after == Observability.NOT_OBSERVABLE)

    counts, rates = motion_gate_metrics(clip, gate_config)
    assert counts["fn"] == 0, "a camera cannot be charged for motion it never saw"
    assert counts["tp"] + counts["fn"] == 0, "nothing here is scoreable"
    assert counts["unobservable_frames"] == len(after)
    assert np.isnan(rates["recall"]), "no observable motion means no recall to report"


def test_below_envelope_motion_is_reported_not_counted_as_failure(
    tmp_path: Path, gate_config
) -> None:
    """Bucket 2: visible but unresolvable motion is its own number."""
    boundary = _envelope_native_px(gate_config)
    clip = _write_clip(
        tmp_path / "far.npz",
        agent_areas_px=[[int(boundary * 0.5)]] * 12,
        moves=[[False]] + [[True]] * 11,
    )
    partition = observability_partition(clip, gate_config)
    after = partition.after_warmup()

    assert np.all(after == Observability.BELOW_ENVELOPE)

    counts, _ = motion_gate_metrics(clip, gate_config)
    assert counts["fn"] == 0, (
        "below-envelope motion is the capability envelope, not a gate defect; "
        "counting it as a failure is what sends someone tuning the threshold "
        "down onto sensor noise"
    )
    assert counts["envelope_limited_misses"] == len(after), (
        "and it must still appear somewhere — a number that leaves the "
        "denominator without surfacing has been hidden, not excluded"
    )


def test_above_envelope_motion_is_the_recall_denominator(
    tmp_path: Path, gate_config
) -> None:
    """Bucket 3: resolvable motion is scored, and a miss is a real miss."""
    boundary = _envelope_native_px(gate_config)
    clip = _write_clip(
        tmp_path / "near.npz",
        agent_areas_px=[[int(boundary * 4)]] * 12,
        moves=[[False]] + [[True]] * 11,
    )
    partition = observability_partition(clip, gate_config)
    after = partition.after_warmup()

    assert np.all(after == Observability.ABOVE_ENVELOPE)

    counts, _ = motion_gate_metrics(clip, gate_config)
    assert counts["tp"] + counts["fn"] == len(after)
    assert counts["envelope_limited_misses"] == 0


def test_envelope_boundary_tracks_the_gate_config(tmp_path: Path, gate_config) -> None:
    """The boundary is the gate's resolving power, not a constant.

    A hardcoded pixel count would drift the moment the gate's resolution or
    foreground threshold changed, and the scorecard would silently start
    excluding the wrong frames.
    """
    boundary = _envelope_native_px(gate_config)
    clip = _write_clip(
        tmp_path / "marginal.npz",
        agent_areas_px=[[int(boundary * 0.5)]] * 12,
        moves=[[False]] + [[True]] * 11,
    )

    assert np.all(
        observability_partition(clip, gate_config).after_warmup()
        == Observability.BELOW_ENVELOPE
    )

    widened = gate_config.__class__(
        min_foreground_fraction=gate_config.min_foreground_fraction / 4,
        stay_awake_frames=gate_config.stay_awake_frames,
        diff_threshold=gate_config.diff_threshold,
        warmup_frames=gate_config.warmup_frames,
        gate_width=gate_config.gate_width,
        gate_height=gate_config.gate_height,
    )
    assert np.all(
        observability_partition(clip, widened).after_warmup()
        == Observability.ABOVE_ENVELOPE
    ), "lowering the foreground threshold must widen the envelope"


def test_envelope_is_resolution_independent(tmp_path: Path, gate_config) -> None:
    """The same scene at 1080p and 720p must land on the same side.

    The gate's threshold is a fraction of the frame it actually processes, so
    silhouette area only compares meaningfully in gate space. Comparing native
    pixel counts would put the 1080p rendition on the far side of the envelope
    from its own 720p twin.
    """
    small = (720, 1280)
    large = (1080, 1920)
    fraction_of_frame = 0.5 * gate_config.min_foreground_fraction

    clips = []
    for name, size in (("720.npz", small), ("1080.npz", large)):
        area = int(fraction_of_frame * size[0] * size[1])
        clips.append(
            _write_clip(
                tmp_path / name,
                agent_areas_px=[[area]] * 12,
                moves=[[False]] + [[True]] * 11,
                size=size,
            )
        )

    labels = [
        set(observability_partition(c, gate_config).after_warmup()) for c in clips
    ]
    assert labels[0] == labels[1], (
        "the same fraction of the frame must bucket identically at both " "renditions"
    )


def test_the_buckets_are_a_partition(tmp_path: Path, gate_config) -> None:
    """Every frame with GT motion falls in exactly one bucket."""
    boundary = _envelope_native_px(gate_config)
    clip = _write_clip(
        tmp_path / "mixed.npz",
        agent_areas_px=[[0, int(boundary * 0.4), int(boundary * 3)] for _ in range(12)],
        moves=[[False] * 3] + [[True, True, False]] * 11,
    )
    partition = observability_partition(clip, gate_config)

    moving = partition.labels != Observability.NO_MOTION
    bucketed = sum(
        int(np.sum(partition.labels == label))
        for label in (
            Observability.NOT_OBSERVABLE,
            Observability.BELOW_ENVELOPE,
            Observability.ABOVE_ENVELOPE,
        )
    )
    assert bucketed == int(np.sum(moving))


def test_the_easiest_visible_mover_sets_the_frames_bucket(
    tmp_path: Path, gate_config
) -> None:
    """One resolvable mover makes the frame scoreable regardless of the rest.

    If a frame containing both an out-of-frustum agent and a large near one
    were labelled by the worst case, real misses would be excused by the
    presence of an agent the camera cannot see.
    """
    boundary = _envelope_native_px(gate_config)
    clip = _write_clip(
        tmp_path / "both.npz",
        agent_areas_px=[[0, int(boundary * 3)]] * 12,
        moves=[[False, False]] + [[True, True]] * 11,
    )
    partition = observability_partition(clip, gate_config)
    assert np.all(partition.after_warmup() == Observability.ABOVE_ENVELOPE)


def test_wakes_on_excluded_frames_stay_visible(tmp_path: Path, gate_config) -> None:
    """Excluded frames must not become a hiding place for wakes.

    A wake with nothing resolvable to wake on is neither a TP nor an FP, but it
    is still compute, so it is counted on its own line.
    """
    clip = _write_clip(
        tmp_path / "blind2.npz",
        agent_areas_px=[[0]] * 12,
        moves=[[False]] + [[True]] * 11,
    )
    counts, _ = motion_gate_metrics(clip, gate_config)
    assert "wakes_outside_envelope" in counts


@pytest.mark.slow
def test_blind_spot_camera_really_does_produce_unobservable_frames(
    tmp_path: Path, gate_config
) -> None:
    """The regression itself, against the real renderer.

    Six of the seven false negatives on the first scorecard came from this one
    camera. If the coverage-gap scene ever stops pointing a camera away from
    its agents, the partition stops being exercised by the golden set and this
    test says so.
    """
    gen.generate(tmp_path, frames=FRAMES, fps=12.0, seed=SEED)
    clip = tmp_path / "coverage_gap_3agents__cam_c.npz"
    assert clip.exists(), "the coverage-gap scene must keep its blind-spot camera"

    partition = observability_partition(clip, gate_config)
    after = partition.after_warmup()
    assert np.any(after == Observability.NOT_OBSERVABLE), (
        "the blind-spot camera should have frames whose GT motion never "
        "reaches its sensor"
    )

    counts, _ = motion_gate_metrics(clip, gate_config)
    assert counts["fn"] == 0, (
        "every miss on this camera was a frustum or envelope artefact, so "
        "none of them may be charged as a gate defect"
    )
