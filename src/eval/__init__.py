"""Evaluation harness support: baselines, comparability, structural checks.

This package is deliberately thin. The measurement-and-scoring code proper
lives in :mod:`src.data.scorecard`, :mod:`src.data.validity`,
:mod:`src.data.depth_eval` and the ``scripts/eval_*.py`` entry points; the
job here is the surrounding discipline that prevents a number from being
reported before the reader can interpret it.
"""

from src.eval.baselines import (
    Baseline,
    BaselineMissing,
    baseline_registry,
    compute_baselines,
    margin,
    register_baseline,
    require_baseline,
)

__all__ = [
    "Baseline",
    "BaselineMissing",
    "baseline_registry",
    "compute_baselines",
    "margin",
    "register_baseline",
    "require_baseline",
]
