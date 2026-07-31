"""Patch-token envelopes and the temporal span they cover.

The bug this kills
------------------
V-JEPA2 is a *tubelet* encoder: it consumes ``tubelet`` consecutive frames per
temporal token. With the standard tubelet of 2, a 4-frame clip produces **2**
temporal token slots, not 4.

This repository's docstrings say otherwise. ``semantic_extractor.py`` documents
its V-JEPA output as ``[B, T*196, 1024]`` and draws a data-flow diagram showing
one temporal slot per input frame. The mapping code then computes
``t_out = min(t, T_out - 1)``, which means input frames 2 and 3 both read
temporal slot 1, and frame 1 reads slot 1 as well. Every semantic vector after
the first is attributed to the wrong moment in time. Nothing raises; the output
array has exactly the shape the caller expected.

:class:`PatchTokens` refuses to exist in that state. The shape assertion in
``__post_init__`` is the permanent kill for this bug class: any code path that
produces or consumes the wrong temporal count fails loudly at the boundary
rather than quietly downstream.

Why a raise and not an ``assert``
---------------------------------
``assert`` statements are stripped by ``python -O``. A contract that evaporates
under an optimisation flag is a comment, not a contract, so these validations
raise ``ValueError`` unconditionally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.contracts.frames import FrameGeometry

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TemporalSpan:
    """The wall-clock interval and frame count a token block covers.

    Time is carried in nanoseconds since the epoch, not frame indices. Frame
    indices are presentation-only: they are not comparable across cameras, they
    reset on reconnect, and they do not survive a dropped frame. The timeline
    key for anything that must be joined is ``(site_id, ts_ns)``.
    """

    start_ts_ns: int
    end_ts_ns: int
    frames_covered: int
    tubelet: int

    def __post_init__(self) -> None:
        if self.end_ts_ns <= self.start_ts_ns:
            raise ValueError(
                f"TemporalSpan must advance in time: start_ts_ns="
                f"{self.start_ts_ns} end_ts_ns={self.end_ts_ns}"
            )
        if self.frames_covered <= 0:
            raise ValueError(
                f"TemporalSpan.frames_covered must be positive, "
                f"got {self.frames_covered}"
            )
        if self.tubelet <= 0:
            raise ValueError(
                f"TemporalSpan.tubelet must be positive, got {self.tubelet}"
            )
        if self.tubelet > self.frames_covered:
            raise ValueError(
                f"TemporalSpan.tubelet ({self.tubelet}) exceeds frames_covered "
                f"({self.frames_covered}); the encoder cannot emit a fractional "
                "temporal token"
            )

    @property
    def n_temporal(self) -> int:
        """Temporal token slots this span yields: ``frames_covered // tubelet``.

        Floor division. When the frame count is not a multiple of the tubelet
        the trailing frames are not represented by any token, which
        :attr:`frames_unrepresented` reports rather than hides.
        """
        return self.frames_covered // self.tubelet

    @property
    def frames_unrepresented(self) -> int:
        """Trailing frames that fall outside any temporal token.

        Nonzero means the encoder silently dropped the tail of the clip. That
        is legal but almost never intended, and a caller that cares should
        check rather than discover it as missing coverage later.
        """
        return self.frames_covered % self.tubelet

    @property
    def duration_ns(self) -> int:
        return self.end_ts_ns - self.start_ts_ns


@dataclass(frozen=True, eq=False)
class PatchTokens:
    """Encoder patch embeddings with their temporal and spatial layout attached.

    Equality is disabled for the same reason as
    :class:`~src.contracts.fields.DepthField`: ``==`` on a wrapped numpy array
    returns an array, not a bool.

    Attributes:
        data: ``[n_temporal, n_spatial, dim]`` embeddings.
        span: The interval covered, carrying the tubelet.
        grid: Spatial patch grid as ``(rows, cols)``.
        geometry: The pixel frame the grid was computed over. Needed to map a
            pixel coordinate to a patch without assuming a resolution.
        encoder_sha: Content hash of the encoder that produced these vectors.
            Embeddings from two different encoders are not comparable, and an
            index built from one and queried with the other returns confident
            nonsense. This field is what lets the query path refuse.
    """

    data: FloatArray
    span: TemporalSpan
    grid: tuple[int, int]
    geometry: FrameGeometry
    encoder_sha: str

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(
                "PatchTokens.data must be [n_temporal, n_spatial, dim], got "
                f"shape {self.data.shape}"
            )
        rows, cols = self.grid
        if rows <= 0 or cols <= 0:
            raise ValueError(f"PatchTokens.grid must be positive, got {self.grid}")
        if not self.encoder_sha:
            raise ValueError(
                "PatchTokens.encoder_sha is required: embeddings from different "
                "encoders are not comparable, and an unlabelled block cannot be "
                "checked against the index that stores it"
            )

        n_temporal, n_spatial, _dim = self.data.shape

        expected_temporal = self.span.n_temporal
        if n_temporal != expected_temporal:
            raise ValueError(
                f"PatchTokens temporal mismatch: data has {n_temporal} temporal "
                f"slots but the span covers {self.span.frames_covered} frames at "
                f"tubelet {self.span.tubelet}, which is {expected_temporal} slots. "
                "A tubelet encoder emits frames_covered // tubelet slots — if "
                f"you expected {self.span.frames_covered}, you have assumed "
                "tubelet=1."
            )

        expected_spatial = rows * cols
        if n_spatial != expected_spatial:
            raise ValueError(
                f"PatchTokens spatial mismatch: data has {n_spatial} spatial "
                f"tokens but grid {self.grid} implies {expected_spatial}"
            )

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return int(self.data.shape[2])

    @property
    def n_temporal(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_spatial(self) -> int:
        return int(self.data.shape[1])

    def frames_for_slot(self, slot: int) -> range:
        """Frame indices, relative to the span, that temporal ``slot`` covers.

        The inverse of the mapping that is currently wrong in
        ``semantic_extractor.py``: slot ``k`` covers frames
        ``[k * tubelet, (k + 1) * tubelet)``, so with tubelet 2 slot 0 covers
        frames 0 and 1. Provided so consumers stop rederiving it, each with
        their own off-by-one.

        Raises:
            IndexError: if ``slot`` is outside the available temporal slots.
        """
        if not 0 <= slot < self.n_temporal:
            raise IndexError(
                f"temporal slot {slot} out of range for {self.n_temporal} slots"
            )
        start = slot * self.span.tubelet
        return range(start, start + self.span.tubelet)

    def slot_for_frame(self, frame_index: int) -> int:
        """Temporal slot covering a frame index relative to the span.

        Raises:
            IndexError: if the frame is outside the span, or falls in the tail
                that no token represents.
        """
        if not 0 <= frame_index < self.span.frames_covered:
            raise IndexError(
                f"frame {frame_index} outside span of "
                f"{self.span.frames_covered} frames"
            )
        slot = frame_index // self.span.tubelet
        if slot >= self.n_temporal:
            raise IndexError(
                f"frame {frame_index} falls in the {self.span.frames_unrepresented} "
                "trailing frames that no temporal token covers"
            )
        return slot
