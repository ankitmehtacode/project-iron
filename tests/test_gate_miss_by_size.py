"""scripts/smartspaces_gate_miss_by_size.py: the binning and the read-off.

The curve is read by people deciding where the 320x180 gate stops being
trustworthy, so its two pieces of arithmetic are pinned here on hand-built
inputs: every event lands in exactly one bin, and a recall level is read
from the first non-thin bin (walking down from the tallest) under it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import smartspaces_gate_miss_by_size as m  # noqa: E402


def test_every_event_lands_in_exactly_one_bin() -> None:
    heights = np.array([0.0, 19.9, 20.0, 1080.0, 500.0])
    missed = np.array([True, False, True, False, True])
    rows = m.curve(heights, missed, m.HEIGHT_EDGES_NATIVE)
    assert sum(r["events"] for r in rows) == heights.size
    assert sum(r["misses"] for r in rows) == int(missed.sum())
    first = rows[0]
    assert (first["events"], first["misses"]) == (2, 1)


def test_empty_bin_has_no_rate_and_is_thin() -> None:
    rows = m.curve(np.array([500.0]), np.array([False]), [0, 10, 1081])
    assert rows[0]["miss_rate"] is None and rows[0]["too_thin"]


def test_recall_crossing_skips_thin_bins_and_reads_from_the_top() -> None:
    edges = [0, 10, 20, 30]
    heights = np.concatenate([np.full(40, 5.0), np.full(5, 15.0), np.full(40, 25.0)])
    missed = np.concatenate(
        [
            np.ones(40, bool),
            np.ones(5, bool),
            np.r_[np.ones(6, bool), np.zeros(34, bool)],
        ]
    )
    rows = m.curve(heights, missed, edges)
    crossings = m.recall_crossings(rows)
    # top bin: recall 34/40 = 0.85 -> under 0.9, not under 0.8
    assert crossings["recall_below_0.9"]["bin_native_px"] == [20, 30]
    # middle bin is thin (5 < 30) and skipped; bottom bin recall 0
    assert crossings["recall_below_0.8"]["bin_native_px"] == [0, 10]
    assert crossings["recall_below_0.5"]["bin_native_px"] == [0, 10]
