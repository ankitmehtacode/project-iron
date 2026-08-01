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
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


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
