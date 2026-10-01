"""Validate the harness's CHIRLA CMC/mAP against CHIRLA's own eval code (Day 40).

Runs bdager/CHIRLA's ``benchmark/reid/evaluate_reid.py`` (``run_evaluation``,
``per_subset=True``) on the SAME embeddings scripts/eval_chirla_baselines.py
scored -- one similarity matrix, two metric implementations -- and compares
every CMC@k and mAP against artifacts/chirla/day40_untrained_baselines.json.
Any disagreement beyond float tolerance exits non-zero: the harness, not
the reference, is then wrong.

CHIRLA's repository has no LICENSE file (checked at fcb6f53: 404), so its
code is NOT vendored here. This script downloads that one file at the pinned
commit into data/derived/ (git-ignored) at run time and imports it from there.

It needs h5py, scikit-learn, tqdm and matplotlib (CHIRLA's imports), none
of which the project depends on, and imports nothing from this repository,
so it runs in a throwaway venv:

    python3.10 -m venv /tmp/chirla-ref && /tmp/chirla-ref/bin/pip install \\
        numpy h5py scikit-learn tqdm matplotlib pillow
    /tmp/chirla-ref/bin/python scripts/chirla_reference_check.py

Run scripts/eval_chirla_baselines.py first; it writes the embeddings this
reads (data/derived/chirla_eval/<method>/<scenario>_{query,gallery}.npz).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CHIRLA_COMMIT = "fcb6f53359d5888b6e8fb745b65a411697dcc22c"
REFERENCE_URL = (
    f"https://raw.githubusercontent.com/bdager/CHIRLA/{CHIRLA_COMMIT}/"
    "benchmark/reid/evaluate_reid.py"
)
DERIVED = REPO_ROOT / "data" / "derived"
EMBEDDINGS = DERIVED / "chirla_eval"
RESULTS = REPO_ROOT / "artifacts" / "chirla" / "day40_untrained_baselines.json"
METHODS = ("hsv_full", "hsv_stripes", "chance_seed0")
TOPK = [1, 5, 10]
TOLERANCE = 1e-6


def _load_reference() -> object:
    path = DERIVED / "chirla_reference" / CHIRLA_COMMIT / "evaluate_reid.py"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(REFERENCE_URL, timeout=60) as response:
            path.write_bytes(response.read())
    spec = importlib.util.spec_from_file_location("chirla_evaluate_reid", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_h5(npz_path: Path) -> Path:
    data = np.load(npz_path)
    h5_path = npz_path.with_suffix(".h5")
    with h5py.File(h5_path, "w") as f:
        f["embeddings"] = data["embeddings"].astype(np.float32)
        f["ids"] = data["ids"].astype(np.int32)
        f["paths"] = data["paths"].astype("S")
    return h5_path


def main() -> int:
    reference = _load_reference()
    harness = json.loads(RESULTS.read_text())["results"]
    worst = 0.0
    failures = []
    for scenario, scenario_out in harness.items():
        for method in METHODS:
            ours = scenario_out["methods"][method]
            ref = reference.run_evaluation(  # type: ignore[attr-defined]
                str(_to_h5(EMBEDDINGS / method / f"{scenario}_gallery.npz")),
                str(_to_h5(EMBEDDINGS / method / f"{scenario}_query.npz")),
                topk=TOPK,
                per_subset=True,
                show_logs=False,
            )
            pairs = [
                (f"CMC@{k}", ours["cmc"][str(k)], ref["cmc_scores"][i])
                for i, k in enumerate(TOPK)
            ]
            pairs.append(("mAP", ours["mAP"], ref["mAP"]))
            for name, a, b in pairs:
                diff = abs(float(a) - float(b))
                worst = max(worst, diff)
                if diff > TOLERANCE:
                    failures.append((scenario, method, name, a, b))
                print(
                    f"{'OK ' if diff <= TOLERANCE else 'BAD'} {scenario:18s} "
                    f"{method:13s} {name:7s} harness {float(a):.9f}  "
                    f"chirla {float(b):.9f}  |diff| {diff:.2e}"
                )
    print(
        f"max |diff| = {worst:.3e} (tolerance {TOLERANCE:g}); "
        f"{len(failures)} disagreement(s)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
