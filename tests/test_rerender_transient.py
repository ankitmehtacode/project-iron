"""scripts/smartspaces_rerender_transient.py: cut detection, the exclusion
mask and the transient-end rule, on hand-built inputs (Day 42, A2)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import smartspaces_rerender_transient as t  # noqa: E402


def test_recorded_mog2_history_matches_the_gate_code() -> None:
    source = (REPO_ROOT / "src" / "cascade" / "motion.py").read_text()
    match = re.search(r"createBackgroundSubtractorMOG2\(\s*history=(\d+)", source)
    assert match, "MOG2 construction not found in the gate"
    assert int(match.group(1)) == t.MOG2_HISTORY


def test_cuts_are_found_where_they_are_not_where_they_are_expected() -> None:
    means = np.zeros((3000, 3))
    means[900:] += 50.0  # on a boundary
    means[1234:] += 30.0  # off any multiple of 900
    means += np.random.default_rng(0).normal(0, 0.3, means.shape)
    assert t.detect_cuts(means).tolist() == [900, 1234]


def test_segments_start_after_frame_zero_and_end_at_the_next_cut() -> None:
    assert t.cut_segments(np.array([0, 900, 1800]), 2700) == [
        (900, 1800),
        (1800, 2700),
    ]


def test_exclusion_mask_drops_window_after_each_cut_only() -> None:
    keep = t.exclusion_mask(np.array([900, 1800]), 2700, 30)
    assert not keep[900:930].any() and not keep[1800:1830].any()
    assert keep[:900].all() and keep[930:1800].all() and keep[1830:].all()


def _pair(decay: int, offset: float, cuts: list[int], seed: int, frames: int = 2700):
    """Re-render wakes at 1.0 for ``decay`` frames after each cut, then at
    source + offset; the source sits at 0.3 throughout. Each pair gets its
    own seed: identical units would give zero-width intervals."""
    rng = np.random.default_rng(seed)
    src = (rng.random(frames) < 0.3).astype(float)
    rer = (rng.random(frames) < 0.3 + offset).astype(float)
    for c in cuts:
        rer[c : c + decay] = 1.0
    return ({"wake": rer}, {"wake": src}, np.array(cuts))


def test_transient_end_is_found_when_the_curve_decays_to_the_source() -> None:
    pairs = [_pair(120, 0.0, [900, 1800], seed) for seed in range(30)]
    curve = t.transient_curve(pairs, "wake")
    assert curve["reaches_steady_state_k"] is not None
    assert 90 <= curve["reaches_steady_state_k"] <= 150


def test_a_persistent_offset_is_not_a_transient() -> None:
    pairs = [_pair(0, 0.4, [900, 1800], seed) for seed in range(30)]
    curve = t.transient_curve(pairs, "wake")
    assert curve["returns_to_source_k"] is None
    assert curve["late_chunk_difference"][0] > 0.3


def test_a_flat_curve_is_stable_from_the_first_bin_most_of_the_time() -> None:
    """The multiple-comparisons guard: on pairs with no transient at all,
    the steady-state reading should land at k=0 in nearly every replicate
    (pointwise intervals made this fail about 70% of the time)."""
    hits = 0
    for rep in range(20):
        pairs = [_pair(0, 0.0, [900, 1800], 1000 * rep + s) for s in range(30)]
        hits += t.transient_curve(pairs, "wake")["reaches_steady_state_k"] == 0
    assert hits >= 16
