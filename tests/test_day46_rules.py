"""Day 46, Objective 2: the rule outcomes recorded in artifacts/day46/rules.json
are what the product does. A decision written in a report and not enforced
by anything decays; these tests are the enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.day46_rules as rules
from src.pipeline.run import REPORTED_POSITION

RULES = Path(__file__).resolve().parent.parent / "artifacts" / "day46" / "rules.json"


def test_recorded_outcomes_reproduce_from_the_day44_artifacts() -> None:
    recorded = json.loads(RULES.read_text())
    fresh = {
        "R1": rules.r1_schedule(),
        "R2": rules.r2_tracker(),
        "R3": rules.r3_range_interval(),
        "R4": rules.r4_uncertainty(),
        "R5": rules.r5_position(),
        "R6": rules.r6_fusion(),
    }
    for key, result in fresh.items():
        assert result["outcome"] == recorded[key]["outcome"], key


def test_the_product_reports_the_position_r5_chose() -> None:
    outcome = json.loads(RULES.read_text())["R5"]["outcome"]
    assert REPORTED_POSITION == outcome


def test_iron_run_defaults_to_the_tracker_r2_chose() -> None:
    import argparse

    import src.main

    outcome = json.loads(RULES.read_text())["R2"]["outcome"]
    seen: dict[str, str] = {}

    def capture(args: argparse.Namespace) -> int:
        seen["tracker"] = args.tracker
        return 0

    original = src.main._cmd_run
    src.main._cmd_run = capture
    try:
        src.main.main(["run", "--video", "v.mp4", "--calibration", "c", "--out", "o"])
    finally:
        src.main._cmd_run = original
    assert seen["tracker"] == outcome
