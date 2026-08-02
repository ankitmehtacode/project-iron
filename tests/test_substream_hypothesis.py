"""The sub-stream hypothesis harness: refuses without matched captures, works with them.

The harness itself is scaffolding for a comparison that needs real camera
footage this project does not have yet (Day 12). What is tested here is
that the scaffolding is correct: it refuses to fabricate a comparison from
mismatched or missing inputs, and produces the right numbers when given
genuinely matched ones — proven with synthetic arrays rather than a camera.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "measure_substream_hypothesis.py"


def test_refuses_when_no_captures_given(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(tmp_path / "out.json")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "cannot run" in result.stderr
    assert not (tmp_path / "out.json").exists()


def test_refuses_on_frame_count_mismatch(tmp_path: Path) -> None:
    main_path = tmp_path / "main.npz"
    sub_path = tmp_path / "sub.npz"
    np.savez(main_path, rgb=np.zeros((10, 180, 320, 3), dtype=np.uint8))
    np.savez(sub_path, rgb=np.zeros((7, 180, 320, 3), dtype=np.uint8))

    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--main-capture", str(main_path),
            "--sub-capture", str(sub_path),
            "--out", str(tmp_path / "out.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "frame count mismatch" in result.stderr


def test_identical_captures_produce_perfect_parity(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (15, 180, 320, 3)).astype(np.uint8)
    main_path = tmp_path / "main.npz"
    sub_path = tmp_path / "sub.npz"
    np.savez(main_path, rgb=frames)
    np.savez(sub_path, rgb=frames.copy())
    out_path = tmp_path / "out.json"

    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--main-capture", str(main_path),
            "--sub-capture", str(sub_path),
            "--out", str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()

    import json

    payload = json.loads(out_path.read_text())
    assert payload["wake_decision_parity"] == 1.0
    assert payload["mog2_variance_delta_sub_minus_main"] == 0.0
    assert payload["n_frames"] == 15
