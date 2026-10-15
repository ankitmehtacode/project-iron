"""scripts/check_lockfile.py: the lockfile is exact, and drift is reported."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_lockfile as c  # noqa: E402


def test_the_repository_lockfile_has_only_exact_pins() -> None:
    """Day 42: nine entries used >= and four had no version at all."""
    pins = c.parse_requirements(c.LOCKFILE.read_text())
    assert "torch" in pins and "transformers" in pins
    assert pins["huggingface-hub"] == "0.36.2"


def test_a_range_is_refused() -> None:
    with pytest.raises(ValueError, match="not an exact pin"):
        c.parse_requirements("numpy>=1.26\n")
    with pytest.raises(ValueError, match="not an exact pin"):
        c.parse_requirements("types-psutil\n")


def test_index_url_options_and_comments_are_ignored() -> None:
    text = (
        "# heading\n"
        "torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu\n"
        "cotracker @ git+https://github.com/x/y.git@abc123\n"
    )
    assert c.parse_requirements(text) == {
        "torch": "2.2.0",
        "cotracker": "git+https://github.com/x/y.git@abc123",
    }


def test_drift_missing_and_name_normalisation() -> None:
    locked = {"pyyaml": "6.0.3", "torch": "2.2.0", "scipy": "1.15.3"}
    installed = c.parse_freeze("PyYAML==6.0.2\ntorch==2.2.0+cpu\n")
    problems = c.differences(locked, installed)
    assert problems == [
        "pyyaml: locked 6.0.3, installed 6.0.2",
        "scipy: locked 1.15.3, NOT INSTALLED",
    ]
