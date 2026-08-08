"""Map tracked pixel coordinates onto encoder patch tokens.

Why this is its own module
--------------------------
It used to be a private method on ``SemanticExtractor``, which imports
CoTracker at module scope. Token indexing has nothing to do with point
tracking, but the coupling meant the indexing arithmetic could not be executed
— let alone tested — without a tracking checkpoint installed. That is how a
one-line index bug survived: the only way to run it was to run the whole
pipeline.

The bug this replaces
---------------------
The previous rule was::

    T_out = features.shape[1] // NUM_PATCHES   # correct: 392 // 196 = 2
    t_out = min(t, T_out - 1)                  # WRONG

The first line is right — the mapper read the real token count and never
assumed the repo's documented 784. The second is a *clamp* where the encoder's
actual layout calls for integer division by the tubelet. With 4 frames and 2
temporal slots it misassigns frame 1 alone:

======  ===================  ==================
frame   ``min(t, T_out-1)``  ``t // tubelet``
======  ===================  ==================
0       0                    0
1       **1**                **0**
2       1                    1
3       1                    1
======  ===================  ==================

The damage grows with clip length, because ``min`` saturates at the last slot
while the correct rule keeps advancing: at 8 frames it misassigns 5 of 8, and
the checkpoint this pipeline uses is natively 64-frame. A short clip made the
bug look like an edge case.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.contracts import FrameGeometry, PatchTokens, TemporalSpan

FloatArray = npt.NDArray[np.float64]


def patch_index_bilinear_weights(
    x: FloatArray, y: FloatArray, grid: tuple[int, int], geometry: FrameGeometry
) -> tuple[npt.NDArray[np.intp], FloatArray]:
    """Bilinear weights over the patch grid for subpixel track coordinates.

    Returns:
        ``(indices, weights)``, each ``[N, 4]``: the four neighbouring patch
        indices and their bilinear weights.

    Nearest-patch indexing (``floor(x / patch_size)``) quantises every track to
    a 16-pixel cell, so a point drifting across a patch boundary snaps
    discontinuously to a different embedding while the underlying scene changed
    continuously. Bilinear interpolation on the patch grid keeps the feature
    continuous in the coordinate, which is what a tracker's subpixel output
    deserves.
    """
    rows, cols = grid
    # Patch centres sit at half-integer grid coordinates, matching the
    # canonical pixel-centre convention in src.contracts.frames.
    gx = np.clip(x / geometry.width * cols - 0.5, 0, cols - 1)
    gy = np.clip(y / geometry.height * rows - 0.5, 0, rows - 1)

    x0 = np.floor(gx).astype(np.intp)
    y0 = np.floor(gy).astype(np.intp)
    x1 = np.minimum(x0 + 1, cols - 1)
    y1 = np.minimum(y0 + 1, rows - 1)

    wx = gx - x0
    wy = gy - y0

    indices = np.stack(
        [y0 * cols + x0, y0 * cols + x1, y1 * cols + x0, y1 * cols + x1], axis=1
    )
    weights = np.stack(
        [(1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy], axis=1
    )
    return indices, weights


def tokens_from_encoder_output(
    features: FloatArray,
    *,
    batch_index: int,
    span: TemporalSpan,
    grid: tuple[int, int],
    geometry: FrameGeometry,
    encoder_sha: str,
) -> PatchTokens:
    """Wrap one batch item of raw encoder output as :class:`PatchTokens`.

    This is the boundary where the temporal off-by-one becomes structurally
    impossible: ``PatchTokens.__post_init__`` asserts
    ``n_temporal == frames_covered // tubelet``, so an encoder emitting an
    unexpected token count, or a caller declaring the wrong tubelet, raises
    here rather than producing quietly misattributed embeddings.
    """
    tokens = np.asarray(features[batch_index], dtype=np.float64)
    if tokens.ndim != 2:
        raise ValueError(
            f"expected [tokens, dim] for one batch item, got {tokens.shape}"
        )

    n_spatial = grid[0] * grid[1]
    total, dim = tokens.shape
    if total % n_spatial != 0:
        raise ValueError(
            f"encoder emitted {total} tokens, which is not a multiple of the "
            f"{n_spatial}-patch spatial grid {grid}. The grid or the input "
            "resolution disagrees with the export."
        )

    return PatchTokens(
        data=tokens.reshape(total // n_spatial, n_spatial, dim),
        span=span,
        grid=grid,
        geometry=geometry,
        encoder_sha=encoder_sha,
    )


def map_tracks_to_embeddings(
    tokens: PatchTokens,
    tracks: FloatArray,
    *,
    bilinear: bool = True,
) -> FloatArray:
    """Look up an embedding for every tracked point at every frame.

    Args:
        tokens: Encoder output for one clip, carrying its own temporal layout.
        tracks: ``[T, N, 2]`` pixel ``(x, y)`` per tracked point per frame.
            ``T`` must equal ``tokens.span.frames_covered``.
        bilinear: Interpolate over the patch grid. See
            :func:`patch_index_bilinear_weights`.

    Returns:
        ``[T, N, dim]`` embeddings, L2-normalised.

    Raises:
        ValueError: if the track frame count disagrees with the span. That
            disagreement is precisely how a frame ends up reading another
            frame's semantics, so it is refused rather than clamped.
    """
    frames, points, _ = tracks.shape
    if frames != tokens.span.frames_covered:
        raise ValueError(
            f"tracks cover {frames} frames but the token span covers "
            f"{tokens.span.frames_covered}. Refusing to guess an alignment: "
            "silently clamping one to the other is the original defect."
        )

    out = np.zeros((frames, points, tokens.dim), dtype=np.float64)

    for frame in range(frames):
        # The whole fix, in one call: the span knows its own tubelet, so this
        # is integer division rather than a clamp, and it is validated by
        # PatchTokens rather than recomputed here.
        slot = tokens.slot_for_frame(frame)
        slot_tokens = tokens.data[slot]

        x = tracks[frame, :, 0].astype(np.float64)
        y = tracks[frame, :, 1].astype(np.float64)

        if bilinear:
            indices, weights = patch_index_bilinear_weights(
                x, y, tokens.grid, tokens.geometry
            )
            gathered = slot_tokens[indices]  # [N, 4, dim]
            out[frame] = np.einsum("nkd,nk->nd", gathered, weights)
        else:
            rows, cols = tokens.grid
            col = np.clip(
                (x / tokens.geometry.width * cols).astype(np.intp), 0, cols - 1
            )
            row = np.clip(
                (y / tokens.geometry.height * rows).astype(np.intp), 0, rows - 1
            )
            out[frame] = slot_tokens[row * cols + col]

    # L2-normalise before anything indexes these. An un-normalised vector makes
    # cosine similarity depend on magnitude, and magnitude varies with scene
    # content rather than with semantics.
    norms = np.linalg.norm(out, axis=2, keepdims=True)
    return np.divide(out, norms, out=np.zeros_like(out), where=norms > 0)
