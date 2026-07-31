"""Tests for the token→track mapping, pinning the temporal off-by-one shut.

These need no weights and no CoTracker. That is the point: the arithmetic used
to live on a method of a class that imports a tracking checkpoint at module
scope, so the only way to execute it was to run the whole pipeline — which is
how a one-line index bug survived.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.contracts import FrameGeometry, PatchTokens, TemporalSpan
from src.semantics.patch_mapping import (
    map_tracks_to_embeddings,
    patch_index_bilinear_weights,
    tokens_from_encoder_output,
)

GEOMETRY = FrameGeometry(width=224, height=224)
GRID = (14, 14)
N_SPATIAL = GRID[0] * GRID[1]
DIM = 8
ENCODER = "e" * 16


def span_for(frames: int, tubelet: int = 2) -> TemporalSpan:
    return TemporalSpan(
        start_ts_ns=0, end_ts_ns=max(1, frames), frames_covered=frames, tubelet=tubelet
    )


def tokens_with_slot_values(values: list[float], frames: int = 4) -> PatchTokens:
    """Tokens whose every patch in slot k carries the constant ``values[k]``.

    A constant per slot makes the mapping observable directly: whatever value
    comes back for a frame names the slot that frame was read from.
    """
    data = np.zeros((len(values), N_SPATIAL, DIM), dtype=np.float64)
    for slot, value in enumerate(values):
        data[slot, :, :] = value
    return PatchTokens(
        data=data,
        span=span_for(frames),
        grid=GRID,
        geometry=GEOMETRY,
        encoder_sha=ENCODER,
    )


def centre_tracks(frames: int, points: int = 5) -> np.ndarray:
    """Tracks parked mid-frame, away from any patch boundary."""
    return np.full((frames, points, 2), 112.0, dtype=np.float64)


# ---------------------------------------------------------------------------
# The defect, pinned exactly
# ---------------------------------------------------------------------------


def test_frame_to_slot_is_integer_division_not_a_clamp() -> None:
    """The whole bug, in one assertion.

    The old rule was ``min(t, T_out - 1)``. With 4 frames and 2 slots it
    misassigns frame 1 alone — which is why a 4-frame clip made this look like
    an edge case.
    """
    tokens = tokens_with_slot_values([10.0, 20.0])
    assert [tokens.slot_for_frame(t) for t in range(4)] == [0, 0, 1, 1]

    # What the old rule would have produced, for contrast.
    t_out = tokens.n_temporal
    assert [min(t, t_out - 1) for t in range(4)] == [0, 1, 1, 1]


def test_the_damage_grows_with_clip_length() -> None:
    """A clamp saturates; integer division keeps advancing.

    At 8 frames the old rule misassigns 5 of 8 — worse than the 4-frame case
    in both count and proportion. The checkpoint this pipeline uses is natively
    64-frame, so the short clip was hiding the severity.
    """
    tokens = PatchTokens(
        data=np.zeros((4, N_SPATIAL, DIM)),
        span=span_for(8),
        grid=GRID,
        geometry=GEOMETRY,
        encoder_sha=ENCODER,
    )
    correct = [tokens.slot_for_frame(t) for t in range(8)]
    old_rule = [min(t, tokens.n_temporal - 1) for t in range(8)]

    assert correct == [0, 0, 1, 1, 2, 2, 3, 3]
    assert old_rule == [0, 1, 2, 3, 3, 3, 3, 3]
    assert sum(a != b for a, b in zip(correct, old_rule)) == 5


# ---------------------------------------------------------------------------
# The Day-2 black/white probe, now green
# ---------------------------------------------------------------------------


def test_black_white_halves_group_by_tubelet() -> None:
    """The Day-2 temporal-alignment probe.

    Frames 0-1 read slot 0, frames 2-3 read slot 1. Under the old clamp, frame
    1 read slot 1 and this failed on its first assertion.
    """
    tokens = tokens_with_slot_values([0.0, 1.0])
    result = map_tracks_to_embeddings(tokens, centre_tracks(4), bilinear=False)

    # Slot values are constant, so after L2 normalisation compare direction:
    # slot 0 is the zero vector, slot 1 is not.
    zero_frames = [t for t in range(4) if np.allclose(result[t], 0.0)]
    assert zero_frames == [0, 1], (
        "frames sharing a tubelet must read the same slot; frame 1 reading "
        "slot 1 is the original defect"
    )


# ---------------------------------------------------------------------------
# Finer-grained: four distinct patterns, exact per-frame assignment
# ---------------------------------------------------------------------------


def test_four_distinct_patterns_assign_in_exact_order() -> None:
    """Per-frame assignment order, asserted exactly rather than in aggregate.

    Four frames, two slots carrying orthogonal unit directions. Every frame's
    embedding must equal its own slot's direction — not merely differ from the
    other, which a half-right mapping could also achieve.
    """
    data = np.zeros((2, N_SPATIAL, DIM), dtype=np.float64)
    data[0, :, 0] = 1.0  # slot 0 points along axis 0
    data[1, :, 1] = 1.0  # slot 1 points along axis 1
    tokens = PatchTokens(
        data=data,
        span=span_for(4),
        grid=GRID,
        geometry=GEOMETRY,
        encoder_sha=ENCODER,
    )

    result = map_tracks_to_embeddings(tokens, centre_tracks(4), bilinear=False)
    expected_axis = [0, 0, 1, 1]

    for frame, axis in enumerate(expected_axis):
        vector = result[frame, 0]
        assert vector[axis] == pytest.approx(
            1.0
        ), f"frame {frame} should read slot {axis} (axis {axis}), got {vector}"
        other = 1 - axis
        assert vector[other] == pytest.approx(
            0.0
        ), f"frame {frame} leaked slot {other} content"


def test_eight_frames_four_slots_exact_order() -> None:
    """The same assertion where the old rule fails on half the frames."""
    slots = 4
    data = np.zeros((slots, N_SPATIAL, DIM), dtype=np.float64)
    for slot in range(slots):
        data[slot, :, slot] = 1.0
    tokens = PatchTokens(
        data=data,
        span=span_for(8),
        grid=GRID,
        geometry=GEOMETRY,
        encoder_sha=ENCODER,
    )

    result = map_tracks_to_embeddings(tokens, centre_tracks(8), bilinear=False)
    for frame in range(8):
        expected_slot = frame // 2
        assert result[frame, 0, expected_slot] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The contract guards this path
# ---------------------------------------------------------------------------


def test_wrong_token_count_is_refused_at_the_boundary() -> None:
    """784 tokens for a 4-frame clip must raise, not be reshaped into place."""
    features = np.zeros((1, 784, DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="assumed tubelet=1"):
        tokens_from_encoder_output(
            features,
            batch_index=0,
            span=span_for(4),
            grid=GRID,
            geometry=GEOMETRY,
            encoder_sha=ENCODER,
        )


def test_correct_token_count_is_accepted() -> None:
    features = np.zeros((1, 392, DIM), dtype=np.float32)
    tokens = tokens_from_encoder_output(
        features,
        batch_index=0,
        span=span_for(4),
        grid=GRID,
        geometry=GEOMETRY,
        encoder_sha=ENCODER,
    )
    assert tokens.n_temporal == 2
    assert tokens.n_spatial == N_SPATIAL


def test_token_count_not_divisible_by_the_grid_is_refused() -> None:
    features = np.zeros((1, 400, DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="not a multiple"):
        tokens_from_encoder_output(
            features,
            batch_index=0,
            span=span_for(4),
            grid=GRID,
            geometry=GEOMETRY,
            encoder_sha=ENCODER,
        )


def test_track_frame_count_mismatch_is_refused_not_clamped() -> None:
    """Clamping one to the other is the original defect, generalised."""
    tokens = tokens_with_slot_values([0.0, 1.0], frames=4)
    with pytest.raises(ValueError, match="Refusing to guess"):
        map_tracks_to_embeddings(tokens, centre_tracks(6))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_bilinear_weights_sum_to_one() -> None:
    x = np.array([0.0, 50.0, 112.0, 223.0])
    y = np.array([0.0, 90.0, 112.0, 223.0])
    _, weights = patch_index_bilinear_weights(x, y, GRID, GEOMETRY)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


def test_bilinear_is_continuous_across_a_patch_boundary() -> None:
    """Nearest-patch indexing snaps; bilinear does not.

    A track drifting across a 16-pixel boundary would otherwise jump to a
    different embedding while the scene changed continuously.
    """
    data = np.zeros((2, N_SPATIAL, DIM), dtype=np.float64)
    data[0, :, 0] = np.arange(N_SPATIAL)  # distinct value per patch
    tokens = PatchTokens(
        data=data,
        span=span_for(4),
        grid=GRID,
        geometry=GEOMETRY,
        encoder_sha=ENCODER,
    )

    just_before = np.full((4, 1, 2), 111.9, dtype=np.float64)
    just_after = np.full((4, 1, 2), 112.1, dtype=np.float64)

    a = map_tracks_to_embeddings(tokens, just_before, bilinear=True)[0, 0]
    b = map_tracks_to_embeddings(tokens, just_after, bilinear=True)[0, 0]
    assert np.abs(a - b).max() < 0.05, "bilinear sampling jumped at a boundary"


def test_output_is_l2_normalised() -> None:
    """Un-normalised vectors make cosine similarity depend on magnitude, which
    varies with scene content rather than semantics."""
    tokens = tokens_with_slot_values([3.0, 7.0])
    result = map_tracks_to_embeddings(tokens, centre_tracks(4))
    norms = np.linalg.norm(result, axis=2)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-9)


def test_zero_vectors_stay_zero_rather_than_nan() -> None:
    """A NaN vector corrupts an entire index; division by a zero norm must not
    produce one."""
    tokens = tokens_with_slot_values([0.0, 0.0])
    result = map_tracks_to_embeddings(tokens, centre_tracks(4))
    assert np.all(np.isfinite(result))
    assert np.allclose(result, 0.0)
