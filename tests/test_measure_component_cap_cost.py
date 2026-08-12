"""Day 27, Objective 2 -- scripts/measure_component_cap_cost.py's own logic.

Not a re-test of the cap mechanism itself (that's tests/test_estimator_joint.py);
this covers the script's own aggregation arithmetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import measure_component_cap_cost as mccc  # noqa: E402

from src.estimator.consistency import compute_nees  # noqa: E402


def test_aggregate_empty_is_nan_not_a_crash() -> None:
    result = mccc._aggregate([])
    assert result["n"] == 0
    assert np.isnan(result["rmse_m"])
    assert np.isnan(result["coverage_95"])


def test_aggregate_known_values() -> None:
    nees_ok = compute_nees(np.zeros(6), np.eye(6))
    records = [
        {"sq_error": 4.0, "nees": nees_ok},
        {"sq_error": 16.0, "nees": nees_ok},
    ]
    result = mccc._aggregate(records)
    assert result["n"] == 2
    assert result["rmse_m"] == np.sqrt((4.0 + 16.0) / 2)


def test_clip_id_matches_the_six_agent_scene() -> None:
    assert "6agents" in mccc.CLIP_ID


def test_capped_config_is_tighter_than_uncapped() -> None:
    assert mccc.CAPPED.max_component_size < mccc.UNCAPPED.max_component_size
    assert mccc.CAPPED.max_component_size < 6 <= mccc.UNCAPPED.max_component_size
