"""Shared fixtures.

The synthetic dataset lives under ``data/``, which is correctly gitignored:
the clips are large and reproducible from ``scripts/gen_synthetic_indoor.py``
plus a seed. That is the right call for the repository and it created a test
that only passed by accident.

``test_eval_report_scores_the_populated_indoor_set`` scores the configured
golden set and asserts a scorecard comes out. On a development machine the
clips are already on disk from the last ``make eval``, so it passed. In a fresh
clone they do not exist, the run refuses, and the test fails — which is how the
day-7 fresh-clone verification found it.

Skipping when the clips are absent would be worse than failing. That test
exists to catch ``make eval`` going permanently green-by-abstention, so a
version of it that abstains in CI is the very defect it guards against. The
fixture therefore *materialises* the dataset instead: generated once per
session if absent, reused if present.
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

SLOW_THRESHOLD_S = 15.0
"""Day 24, Objective 4: test_synthetic_indoor.py had 13 tests genuinely
costing 16-421s with no @pytest.mark.slow, discovered only by someone
running --durations=0 and reading the output -- exactly the failure mode
this threshold exists to stop. 15s sits above every currently-unmarked
test's measured cost (6.68s is the current repo-wide maximum outside
slow/requires_weights/decoder_dependent, per a clean, uncontended run;
see FOUNDATION_REPORT.md's Day-24 section for why "clean" needed stating)
and comfortably below the 16s+ tier that should have been marked
already. A test that legitimately needs more than this either earns one
of the exempt markers below or gets sped up -- it does not get
discovered by a slow terminal."""

_SLOW_THRESHOLD_EXEMPT_MARKERS = ("slow", "requires_weights", "decoder_dependent")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo
) -> Generator[None, None, None]:
    """Fail a test that exceeds SLOW_THRESHOLD_S without an exempting
    marker, so the next expensive test marks itself instead of being
    found by someone watching a terminal (Day 24, Objective 4)."""
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or report.outcome != "passed":
        return
    if call.duration <= SLOW_THRESHOLD_S:
        return
    if any(item.get_closest_marker(name) for name in _SLOW_THRESHOLD_EXEMPT_MARKERS):
        return
    report.outcome = "failed"
    report.longrepr = (
        f"{item.nodeid} took {call.duration:.1f}s, over the {SLOW_THRESHOLD_S:.0f}s "
        "threshold, with none of slow/requires_weights/decoder_dependent set. "
        "If this genuinely needs the time, mark it @pytest.mark.slow (or the "
        "appropriate exempting marker) rather than letting it join the "
        "quick local loop unmarked."
    )


@pytest.fixture(scope="session")
def synthetic_indoor_clips() -> Path:
    """Path to the rendered synthetic indoor clips, generating them if needed.

    Generation is deterministic in the seed, so a run that generates and a run
    that reuses measure the same instrument. The defaults must match the ones
    ``configs/golden/v2-indoor.golden.json`` was minted from, or the clip
    content hashes will not match the golden set and every score becomes a
    score against a different set.
    """
    import gen_synthetic_indoor as gen

    from src.config import IronConfig

    config = IronConfig.load()
    root = config.paths.resolved_data_dir / "synthetic" / gen.DATASET_NAME

    if not sorted(root.glob("*.npz")):
        # Defaults taken from gen_synthetic_indoor.main's argparse: the golden
        # set was minted with them, so anything else would produce a set that
        # does not match its own manifest.
        gen.generate(root, frames=24, fps=12.0, seed=20260801)

    return root
