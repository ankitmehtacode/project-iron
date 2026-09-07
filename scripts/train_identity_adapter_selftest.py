"""Run the adapter/trainer/promotion-gate pipeline end to end — SELF_TEST
ONLY, against a synthetic in-memory identity gallery.

============================================================================
SELF_TEST — not a trained model, not to be quoted as re-ID performance.
============================================================================

This proves the plumbing (backbone -> adapter -> triplet loss -> gradient
step -> checkpoint manifest -> promotion gate) moves correctly. It proves
NOTHING about re-identification capability: the "backbone" is a fixed
random projection that has never seen an image, and the "identities" are
gaussian clusters in a 32-dimensional space, not people. See
docs/adr/0001-identity-adapter-architecture.md and
FOUNDATION_REPORT.md's Day-36 section for why nothing in this repository
can be trained on real data today (lane C has zero clips; lane S fails the
appearance-learned validity gate).

    python scripts/train_identity_adapter_selftest.py
    python scripts/train_identity_adapter_selftest.py --json outputs/identity/selftest_day36.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.identity.adapter import Adapter, AdapterConfig  # noqa: E402
from src.identity.checkpoint import write_selftest_checkpoint_manifest  # noqa: E402
from src.identity.promotion import evaluate_promotion  # noqa: E402
from src.identity.selftest import (  # noqa: E402
    SELF_TEST_LABEL,
    SyntheticStandInBackbone,
    make_synthetic_identity_gallery,
    make_triplet_sampler,
)
from src.identity.trainer import TrainingConfig, train_adapter  # noqa: E402


def run(args: argparse.Namespace) -> dict:
    print(SELF_TEST_LABEL)
    print("=" * len(SELF_TEST_LABEL))

    backbone = SyntheticStandInBackbone(
        input_dim=args.input_dim, feature_dim=args.feature_dim, seed=args.seed
    )
    adapter_config = AdapterConfig(
        input_dim=args.feature_dim, hidden_dim=64, output_dim=32, num_layers=2
    )
    adapter = Adapter(adapter_config)

    clips, labels = make_synthetic_identity_gallery(
        n_identities=args.n_identities,
        samples_per_identity=args.samples_per_identity,
        input_dim=args.input_dim,
        seed=args.seed,
        cluster_scale=args.cluster_scale,
        noise_scale=args.noise_scale,
    )
    sampler = make_triplet_sampler(
        clips, labels, batch_size=args.batch_size, seed=args.seed
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        triplet_margin=args.triplet_margin,
        seed=args.seed,
    )

    print(
        f"training on {len(labels)} synthetic samples across "
        f"{args.n_identities} synthetic clusters ({SELF_TEST_LABEL})"
    )
    losses = train_adapter(
        backbone=backbone, adapter=adapter, triplets=sampler, config=training_config
    )
    print(f"triplet loss: {losses[0]:.4f} -> {losses[-1]:.4f} over {len(losses)} steps")

    result = evaluate_promotion(
        backbone=backbone, adapter=adapter, clips=clips, identity_labels=labels
    )
    print(f"[{SELF_TEST_LABEL}] {result.render()}")

    manifest = write_selftest_checkpoint_manifest(
        Path(args.output),
        adapter=adapter,
        backbone_sha=backbone.backbone_sha,
        synthetic_dataset_label="synthetic-identity-gallery-day36-selftest",
        training_config_sha=training_config.config_sha(),
        promotion_evidence=result.as_evidence(),
    )
    print(f"checkpoint manifest -> {args.output}")
    print(f"  self_test_label : {manifest.self_test_label}")
    print(f"  promoted        : {manifest.promoted}  (always False for a self-test)")

    payload = {
        "self_test_label": SELF_TEST_LABEL,
        "losses": losses,
        "promotion": result.as_evidence(),
        "checkpoint_manifest_path": args.output,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"Written to {args.json}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-identities", type=int, default=8)
    parser.add_argument("--samples-per-identity", type=int, default=12)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--triplet-margin", type=float, default=0.3)
    parser.add_argument(
        "--cluster-scale",
        type=float,
        default=1.2,
        help=(
            "identity-cluster separation relative to noise_scale=1.0; kept "
            "close to 1.0 deliberately so the initial (random-init) "
            "adapter starts with real triplet violations — a wider "
            "separation makes the problem trivially separable before any "
            "gradient step, which would run the pipeline without actually "
            "demonstrating it moves gradients"
        ),
    )
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument(
        "--output",
        default="outputs/identity/selftest_checkpoint_manifest.json",
        help="path for the SELF_TEST checkpoint manifest",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
