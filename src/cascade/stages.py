"""Stages 1-3: detector, tracker, semantics.

Day 42: the detector stage calls the :mod:`src.detect` contract when a
backend is bound, and raises when none is. Tracker and semantics remain
deliberately unimplemented. Each raises ``NotImplementedError`` from
:meth:`process` with a docstring naming exactly what will plug in and what it
must satisfy. They exist now so that :class:`~src.cascade.runner.CascadeRunner`
can be built and benchmarked against the real shape of the pipeline — the wake
accounting, short-circuit logic, and stats reporting are all exercisable today,
and are what determine whether the Tier-1 budget is achievable at all.

Filling these in with something plausible would be worse than leaving them
empty: a runner benchmarked against a fake detector reports a cost that has
nothing to do with the real one, and the number would be quoted long after
anyone remembered it was invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.cascade.stage import FrameBatch, StageContext, StageOutput

if TYPE_CHECKING:
    from src.detect.contract import Detector


@dataclass(frozen=True)
class DetectionWakePolicy:
    """When a detection wakes the tracker: classes and a score floor.

    The floor applies to an UNCALIBRATED score (Day 42): it is a ranking
    cut, not a probability, until a CalibrationRecord exists for the
    producer. Compiled from customer config, never a module constant.
    """

    classes: frozenset[str]
    min_score: float

    def __post_init__(self) -> None:
        if not self.classes:
            raise ValueError("a wake policy names at least one class")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score is a score in [0, 1]")


class DetectorStage:
    """Stage 1: detection. Wakes on motion; a thin caller of the Day-42
    :class:`~src.detect.contract.Detector` protocol.

    It owns no model. A backend (oracle, motion blob, signed pretrained) is
    bound at construction and reports canonical-frame boxes in a
    :class:`~src.detect.contract.DetectionSet`, which this stage puts in
    ``ctx.payload["detections"]``. With no backend bound it still raises
    ``NotImplementedError``: a cascade benchmarked against an invented
    detector reports a cost that has nothing to do with the real one.

    Still owed by the product path (see iron-cascade-runtime): an INT8
    OpenVINO export compiled once with ``INFERENCE_NUM_THREADS``.
    """

    def __init__(
        self,
        detector: "Detector | None" = None,
        policy: DetectionWakePolicy | None = None,
    ) -> None:
        if (detector is None) != (policy is None):
            raise ValueError("bind a detector and its wake policy together")
        self._detector = detector
        self._policy = policy

    @property
    def name(self) -> str:
        return "detector"

    def process(self, frame_batch: FrameBatch, ctx: StageContext) -> StageOutput:
        if self._detector is None or self._policy is None:
            raise NotImplementedError(
                "DetectorStage is a stub until a Detector backend is bound; "
                "see src/detect/ for the contract and its backends."
            )
        from src.contracts.frames import FrameGeometry
        from src.detect.contract import FrameInput

        image = frame_batch[-1]
        detections = self._detector.detect(
            FrameInput(
                image_bgr=image,
                ts_ns=ctx.ts_ns,
                frame_index=ctx.frame_index,
                geometry=FrameGeometry(width=image.shape[1], height=image.shape[0]),
            )
        )
        ctx.payload["detections"] = detections
        scores = detections.score_array()
        wanted = np.array(
            [label in self._policy.classes for label in detections.labels()], bool
        )
        hits = int(np.sum(wanted & (scores >= self._policy.min_score)))
        return StageOutput(
            woke=True,
            wake_next=hits > 0,
            detail={
                "backend": self._detector.name,
                "detections": len(detections),
                "policy_hits": hits,
            },
        )

    def should_wake_next(self, output: StageOutput) -> bool:
        return output.wake_next


class TrackerStage:
    """Stage 2: point tracking and metric geometry. Wakes on detection.

    To plug in:
        The existing CoTracker3 path in ``src/geometry/enhanced_cotracker.py``,
        fused with depth through the contract layer rather than the current
        direct-array path.

    Must satisfy:
        - Depth arrives as a :class:`~src.contracts.DepthField` with explicit
          units. The current projector treats Depth-Anything-V2's relative
          disparity as metres; see
          ``tests/test_known_bugs.py::test_depth_output_declares_units``.
          Wiring this stage to raw arrays would reproduce that defect in new
          code.
        - Intrinsics carried as :class:`~src.contracts.Intrinsics` valid for
          the frame geometry actually used, rescaled at the point of any
          resize.
        - Track identity keyed on ``(site_id, ts_ns)``, never on frame index.
          Frame indices reset on reconnect and do not survive a dropped frame.
    """

    @property
    def name(self) -> str:
        return "tracker"

    def process(self, frame_batch: FrameBatch, ctx: StageContext) -> StageOutput:
        raise NotImplementedError(
            "TrackerStage is a stub. Plug in the CoTracker3 path via the "
            "contract layer; see this class's docstring."
        )

    def should_wake_next(self, output: StageOutput) -> bool:
        return output.wake_next


class SemanticsStage:
    """Stage 3: embeddings, pose, human-object interaction. Wakes per track.

    To plug in:
        The existing V-JEPA2 embedding path in
        ``src/semantics/semantic_extractor.py``, once two confirmed defects are
        fixed — it currently feeds the encoder un-standardised pixels, and maps
        temporal tokens to frames with an off-by-one that misattributes every
        frame after the first.

    Must satisfy:
        - Output wrapped as :class:`~src.contracts.PatchTokens`, whose shape
          assertion makes the temporal misattribution structurally
          impossible.
        - Patch features sampled at track points by bilinear interpolation on
          the patch grid, not ``floor(x / patch_size)`` as the current mapper
          does. L2-normalise before any index insert.
        - The heaviest stage by far, so it wakes per active track under policy
          rather than per frame. This is where the wake hierarchy earns its
          cost.
    """

    @property
    def name(self) -> str:
        return "semantics"

    def process(self, frame_batch: FrameBatch, ctx: StageContext) -> StageOutput:
        raise NotImplementedError(
            "SemanticsStage is a stub. Plug in the V-JEPA2 embedding path via "
            "PatchTokens; see this class's docstring."
        )

    def should_wake_next(self, output: StageOutput) -> bool:
        return output.wake_next
