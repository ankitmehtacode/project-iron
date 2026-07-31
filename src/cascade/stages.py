"""Stages 1-3: detector, tracker, semantics.

Deliberately unimplemented. Each raises ``NotImplementedError`` from
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

from src.cascade.stage import FrameBatch, StageContext, StageOutput


class DetectorStage:
    """Stage 1: INT8 person and object detection. Wakes on motion.

    To plug in:
        An INT8 OpenVINO person/object detector, compiled once at startup and
        held for the process lifetime. Per-clip load/unload would let model
        loading dominate wall time — see iron-cascade-runtime on why the
        concurrency model is stage-parallel processes over bounded queues
        rather than sequential load per clip.

    Must satisfy:
        - Compiled with ``INFERENCE_NUM_THREADS`` from
          :class:`~src.config.RuntimeConfig`, never left at the OpenVINO
          default. Torch and OpenVINO each grab every core by default and
          together oversubscribe, roughly halving throughput.
        - Detections reported in the frame's own pixel space, or carrying an
          :class:`~src.contracts.AffineTransform` back to it if the stage
          resizes. It will resize — detectors have fixed input sizes — so the
          transform is mandatory, not optional.
        - ``should_wake_next`` gated on detection confidence and class policy
          compiled from customer config, not on hardcoded thresholds.
    """

    @property
    def name(self) -> str:
        return "detector"

    def process(self, frame_batch: FrameBatch, ctx: StageContext) -> StageOutput:
        raise NotImplementedError(
            "DetectorStage is a stub. Plug in the INT8 person/object detector; "
            "see this class's docstring for the constraints it must satisfy."
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
