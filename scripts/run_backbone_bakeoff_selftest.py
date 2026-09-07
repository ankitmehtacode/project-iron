"""Run the backbone bake-off harness end to end — SELF_TEST ONLY, against
the synthetic stand-in backbone. Built to run the moment a real
FrozenBackbone and a real lane-R eval set (e.g. CHIRLA, once its checklist
in docs/chirla_verification_checklist.md is cleared) exist; not run against
either today.

============================================================================
SELF_TEST — not a trained model, not to be quoted as re-ID performance.
This script does NOT select a backbone. The decision ledger's "Dense-
semantics encoder: open" line is unchanged by anything printed below.
============================================================================

    python scripts/run_backbone_bakeoff_selftest.py
    python scripts/run_backbone_bakeoff_selftest.py \
        --json outputs/identity/bakeoff_selftest_day36.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.identity.bakeoff import (  # noqa: E402
    clothing_change_robustness,
    patch_boundary_discontinuity,
    same_object_retrieval_map,
    temporal_embedding_stability,
)
from src.identity.selftest import (  # noqa: E402
    SELF_TEST_LABEL,
    SyntheticStandInBackbone,
    apply_synthetic_clothing_change,
    make_synthetic_identity_gallery,
    make_synthetic_patch_positions,
)


def run(args: argparse.Namespace) -> dict:
    print(SELF_TEST_LABEL)
    print("This script does NOT select a backbone.")
    print("=" * len(SELF_TEST_LABEL))

    backbone = SyntheticStandInBackbone(
        input_dim=args.input_dim, feature_dim=args.feature_dim, seed=args.seed
    )
    clips, labels = make_synthetic_identity_gallery(
        n_identities=args.n_identities,
        samples_per_identity=args.samples_per_identity,
        input_dim=args.input_dim,
        seed=args.seed,
    )

    results = {}

    pos0, posk = make_synthetic_patch_positions(len(labels), seed=args.seed)
    retrieval = same_object_retrieval_map(
        backbone=backbone,
        clips=clips,
        identity_labels=labels,
        pos0=pos0,
        posk=posk,
        self_test=True,
    )
    print(retrieval.render())
    results["retrieval_map"] = retrieval.as_evidence()

    base = clips[0:1]
    sequence = base + 0.05 * torch.randn_like(base).repeat(10, 1)
    stability = temporal_embedding_stability(
        backbone=backbone, clip_sequence=sequence, self_test=True
    )
    print(stability.render())
    results["temporal_stability"] = stability.as_evidence()

    boundary = patch_boundary_discontinuity(
        backbone=backbone, clips_a=clips, clips_b=clips + 0.1, self_test=True
    )
    print(boundary.render())
    results["patch_boundary_l2"] = boundary.as_evidence()

    perturbed = apply_synthetic_clothing_change(clips, labels, seed=args.seed)
    clothing = clothing_change_robustness(
        backbone=backbone,
        clean_clips=clips,
        perturbed_clips=perturbed,
        identity_labels=labels,
        self_test=True,
    )
    print(clothing.render())
    results["clothing_change_map"] = clothing.as_evidence()

    print()
    print(f"[{SELF_TEST_LABEL}] All four probes ran end to end against the")
    print("synthetic stand-in only. No backbone was selected or recommended.")

    payload = {"self_test_label": SELF_TEST_LABEL, "probes": results}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"Written to {args.json}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-identities", type=int, default=6)
    parser.add_argument("--samples-per-identity", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
