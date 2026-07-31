"""The wake hierarchy: motion gate → detector → tracker → semantics.

Most frames of most cameras contain nothing moving. The cascade exists so that
those frames cost almost nothing, which is the only way the Tier-1 idle budget
(under 3% of one core per camera) is reachable on NUC-class hardware. Stage 0
is always on and nearly free; every later stage runs only when the one before
it says something is worth looking at.

Stages 1-3 are stubs today. The runner, the wake accounting, and the stats
reporting are real and benchmarkable now — see ``scripts/cascade_bench.py``.
"""

from src.cascade.motion import (
    FrameDifferenceBackend,
    Mog2Backend,
    MotionGate,
    MotionGateConfig,
)
from src.cascade.runner import CascadeRunner, CascadeStats, JsonlStatsSink
from src.cascade.stage import (
    FrameBatch,
    Stage,
    StageAccounting,
    StageContext,
    StageOutput,
    StageStats,
)
from src.cascade.stages import DetectorStage, SemanticsStage, TrackerStage

__all__ = [
    "CascadeRunner",
    "CascadeStats",
    "DetectorStage",
    "FrameBatch",
    "FrameDifferenceBackend",
    "JsonlStatsSink",
    "Mog2Backend",
    "MotionGate",
    "MotionGateConfig",
    "SemanticsStage",
    "Stage",
    "StageAccounting",
    "StageContext",
    "StageOutput",
    "StageStats",
    "TrackerStage",
]
