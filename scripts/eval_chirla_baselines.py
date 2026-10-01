"""CHIRLA query -> gallery re-ID with two weight-free baselines (Day 40).

The project's first measurement of an appearance capability on real data.
No learned backbone: a backbone needs its own licence snapshot (model slate
rule R2), so the two methods here need no weights at all:

- ``chance``      random embeddings -> random rankings, mean over SEEDS.
- ``hsv_full``    HSV colour histogram of the whole crop.
- ``hsv_stripes`` the same, upper and lower half separately ("shirt and
                  trousers") -- the clothing-colour shortcut a re-ID model
                  has to beat to be worth anything.

Histograms are compared with the Bhattacharyya coefficient: each histogram
is L1-normalised and square-rooted, so the cosine of two such vectors IS
their Bhattacharyya coefficient. That lets the exact same embeddings be
scored by this harness and by CHIRLA's own ``evaluate_reid.py`` (which
takes embeddings and computes cosine itself) -- one similarity matrix, two
metric implementations (see scripts/chirla_reference_check.py).

Protocol, read from CHIRLA's eval code (bdager/CHIRLA@fcb6f53,
benchmark/reid/evaluate_reid.py, ``--per-subset``), not assumed:
  - query subset ``test_k`` is scored against gallery subset ``train_k``;
  - closed set: queries with a negative id (distractor) or an id absent
    from that gallery subset are excluded;
  - cosine similarity; no same-camera exclusion;
  - a subset whose CMC and mAP are all zero is DROPPED from the average
    (``valid_results``) -- reproduced, and the drop count reported;
  - final number = unweighted mean over the remaining subsets.
Only the ``gallery`` and ``query`` splits are read. ``train`` and ``val``
are never opened (lane R: evaluation only).

Predictions, stated before the first run (Day-40 prompt):
  P1 hsv beats chance clearly on reid_reappearance (clothing persists).
  P2 hsv is near chance on reid_long_term (clothing changes over months).
  P3 if hsv does well on reid_long_term, clothing-colour cues leak across
     the split -- and the published 18.81% CMC@1 must be read against it.

Data handling: crops are decoded in memory and never written anywhere.
The only files written are numeric -- embeddings, ids and image *paths*
under data/derived/ (git-ignored) for the reference check, and the scores
JSON under artifacts/. No timing, throughput, CPU or latency is measured.

Usage:
    .venv-pinned/bin/python scripts/eval_chirla_baselines.py
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import IronConfig  # noqa: E402
from src.data.registry import (  # noqa: E402
    DatasetRegistry,
    DeploymentScopeUnresolved,
)
from src.identity.bakeoff import (  # noqa: E402
    open_bakeoff_eval_set,
    query_gallery_retrieval,
)

REGISTRY_PATH = REPO_ROOT / "configs" / "datasets.yaml"
OUT_PATH = REPO_ROOT / "artifacts" / "chirla" / "day40_untrained_baselines.json"
SCENARIOS = ("reid_reappearance", "reid_long_term")
TOPK = (1, 5, 10)
SEEDS = tuple(range(10))
CHANCE_DIM = 64
HSV_BINS = (16, 4, 4)  # hue, saturation, value
LABEL = (
    "CHIRLA — lane R, eval-only, internal benchmarking. "
    "deployment_scope_restriction: UNRESOLVED. Not a product claim."
)


def subset_of(path: str) -> str:
    """CHIRLA's ``extract_subset_from_path`` (hierarchical branch): the
    ``train_k``/``test_k`` component following ``/train/`` or ``/test/``."""
    parts = path.split("/")
    for i, part in enumerate(parts[:-1]):
        if part in ("train", "test") and parts[i + 1].startswith(("train_", "test_")):
            return parts[i + 1]
    raise ValueError(f"no train_k/test_k subset in {path!r}")


def load_split(root: Path, scenario: str, split: str) -> dict[str, Any]:
    if split not in ("gallery", "query"):
        raise ValueError(f"lane R: only gallery/query are read, not {split!r}")
    files = sorted(
        p
        for p in (root / scenario / "data").glob(f"*_{split}.parquet/{split}-*.parquet")
        if p.is_file()
    )
    if not files:
        raise FileNotFoundError(f"no {split} parquet for {scenario} under {root}")
    table = pa.concat_tables(
        [pq.read_table(f, columns=["image", "image_path", "id"]) for f in files]
    )
    return {
        "images": [cell["bytes"] for cell in table.column("image").to_pylist()],
        "paths": np.array(table.column("image_path").to_pylist()),
        "ids": np.array(table.column("id").to_pylist(), dtype=np.int64),
    }


def _sqrt_hist(hsv: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, list(HSV_BINS), [0, 180, 0, 256, 0, 256]
    )
    hist = hist.flatten().astype(np.float64)
    total = hist.sum()
    if total == 0:
        raise ValueError("empty crop")
    return np.sqrt(hist / total)


def hsv_features(images: list[bytes], stripes: bool) -> np.ndarray:
    """One unit-norm vector per crop; cosine == Bhattacharyya coefficient
    (for ``stripes``, the mean of the upper- and lower-half coefficients)."""
    out = []
    for raw in images:
        rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        if stripes:
            half = hsv.shape[0] // 2
            vec = np.concatenate([_sqrt_hist(hsv[:half]), _sqrt_hist(hsv[half:])])
            vec /= np.sqrt(2.0)
        else:
            vec = _sqrt_hist(hsv)
        out.append(vec)
    return np.asarray(out, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Identical arithmetic to CHIRLA's ``cosine_similarity``."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


def chirla_protocol(
    q_emb: np.ndarray,
    q_ids: np.ndarray,
    q_paths: np.ndarray,
    g_emb: np.ndarray,
    g_ids: np.ndarray,
    g_paths: np.ndarray,
) -> dict[str, Any]:
    """Per-subset scoring and averaging exactly as CHIRLA's ``--per-subset``."""
    q_sub = np.array([subset_of(p) for p in q_paths])
    g_sub = np.array([subset_of(p) for p in g_paths])
    per_subset = []
    for subset in sorted(set(q_sub)):
        qm = q_sub == subset
        gm = g_sub == subset.replace("test_", "train_")
        if not gm.any():  # CHIRLA falls back to the whole gallery
            gm = np.ones_like(g_sub, dtype=bool)
        sim = cosine(q_emb[qm], g_emb[gm])
        try:
            s = query_gallery_retrieval(sim, q_ids[qm], g_ids[gm], TOPK)
            cmc, mean_ap, n = s.cmc, s.mean_ap, s.n_queries
        except ValueError:  # no known query; CHIRLA's open-set fallback is all-zero
            cmc, mean_ap, n = {k: 0.0 for k in TOPK}, 0.0, 0
        per_subset.append(
            {"subset": subset, "cmc": cmc, "mAP": mean_ap, "n_scored_queries": n}
        )
    valid = [
        r for r in per_subset if r["mAP"] > 0 or any(v > 0 for v in r["cmc"].values())
    ]
    if not valid:
        raise ValueError("every subset scored zero")
    return {
        "cmc": {k: float(np.mean([r["cmc"][k] for r in valid])) for k in TOPK},
        "mAP": float(np.mean([r["mAP"] for r in valid])),
        "subsets_total": len(per_subset),
        "subsets_dropped_all_zero": len(per_subset) - len(valid),
        "per_subset": per_subset,
    }


def _export(
    out_dir: Path,
    method: str,
    scenario: str,
    split: str,
    emb: np.ndarray,
    data: dict[str, Any],
) -> None:
    path = out_dir / method / f"{scenario}_{split}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, embeddings=emb, ids=data["ids"].astype(np.int32), paths=data["paths"]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    registry = DatasetRegistry.load(REGISTRY_PATH)
    entry = open_bakeoff_eval_set(registry, "CHIRLA")  # lane-R eval gate
    try:
        registry.require_product_claim_clearance("CHIRLA")
    except DeploymentScopeUnresolved as exc:
        clearance = {"cleared": False, "refusal": str(exc)}
    else:
        clearance = {"cleared": True, "refusal": None}
    if clearance["cleared"]:
        # The label below asserts UNRESOLVED; refuse rather than mislabel.
        raise SystemExit("deployment scope resolved -- update LABEL before reporting")

    assert entry.license_snapshot is not None
    commit = entry.license_snapshot.resolved_commit_sha
    data_dir = IronConfig.load().paths.resolved_data_dir
    root = data_dir / "raw" / "CHIRLA" / str(commit)
    derived = data_dir / "derived" / "chirla_eval"

    results: dict[str, Any] = {}
    for scenario in SCENARIOS:
        manifest = json.loads((root / scenario / "_manifest.json").read_text())
        gallery = load_split(root, scenario, "gallery")
        query = load_split(root, scenario, "query")
        print(f"{scenario}: {len(query['ids'])} queries, {len(gallery['ids'])} gallery")

        embeddings = {
            "hsv_full": (
                hsv_features(query["images"], stripes=False),
                hsv_features(gallery["images"], stripes=False),
            ),
            "hsv_stripes": (
                hsv_features(query["images"], stripes=True),
                hsv_features(gallery["images"], stripes=True),
            ),
        }
        chance_runs = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            q = rng.standard_normal((len(query["ids"]), CHANCE_DIM)).astype(np.float32)
            g = rng.standard_normal((len(gallery["ids"]), CHANCE_DIM)).astype(
                np.float32
            )
            if seed == SEEDS[0]:
                embeddings["chance_seed0"] = (q, g)
            chance_runs.append(
                chirla_protocol(
                    q, query["ids"], query["paths"], g, gallery["ids"], gallery["paths"]
                )
            )

        scenario_out: dict[str, Any] = {
            "manifest_commit_sha": manifest["commit_sha"],
            "n_query_rows": int(len(query["ids"])),
            "n_gallery_rows": int(len(gallery["ids"])),
            "methods": {},
        }
        chance = {
            "cmc": {
                k: float(np.mean([r["cmc"][k] for r in chance_runs])) for k in TOPK
            },
            "mAP": float(np.mean([r["mAP"] for r in chance_runs])),
            "seeds": list(SEEDS),
            "cmc1_per_seed": [r["cmc"][1] for r in chance_runs],
            "mAP_per_seed": [r["mAP"] for r in chance_runs],
            "subsets_dropped_all_zero_per_seed": [
                r["subsets_dropped_all_zero"] for r in chance_runs
            ],
        }
        scenario_out["methods"]["chance"] = chance
        for method, (q_emb, g_emb) in embeddings.items():
            _export(derived, method, scenario, "query", q_emb, query)
            _export(derived, method, scenario, "gallery", g_emb, gallery)
            if method == "chance_seed0":
                scenario_out["methods"][method] = chirla_protocol(
                    q_emb,
                    query["ids"],
                    query["paths"],
                    g_emb,
                    gallery["ids"],
                    gallery["paths"],
                )
                continue
            scored = chirla_protocol(
                q_emb,
                query["ids"],
                query["paths"],
                g_emb,
                gallery["ids"],
                gallery["paths"],
            )
            scored["margin_over_chance"] = {
                "cmc1": scored["cmc"][1] - chance["cmc"][1],
                "mAP": scored["mAP"] - chance["mAP"],
            }
            scenario_out["methods"][method] = scored
        results[scenario] = scenario_out
        for method, s in scenario_out["methods"].items():
            print(
                f"  {method:13s} CMC@1 {s['cmc'][1]:.4f}  CMC@5 {s['cmc'][5]:.4f}  "
                f"CMC@10 {s['cmc'][10]:.4f}  mAP {s['mAP']:.4f}"
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "label": LABEL,
                "product_claim_clearance": clearance,
                "dataset": {"name": "CHIRLA", "commit_sha": commit, "lane": entry.lane},
                "splits_read": ["gallery", "query"],
                "protocol": "CHIRLA evaluate_reid.py --per-subset, closed set "
                "(bdager/CHIRLA@fcb6f53)",
                "hsv_bins": list(HSV_BINS),
                "chance_dim": CHANCE_DIM,
                "results": results,
                "no_timing_claim": "No timing, throughput, CPU or latency measured.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
