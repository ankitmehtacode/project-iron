"""Run ADR 0001's withdrawal drill — evict, retrain, confirm — on synthetic
identities. Day 36, Objective 5: the first real verification of this
project's central erasure claim.

    python scripts/run_erasure_drill_selftest.py
    python scripts/run_erasure_drill_selftest.py --json outputs/identity/erasure_drill_day36.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.identity.erasure_drill import run_erasure_drill  # noqa: E402
from src.identity.selftest import SELF_TEST_LABEL  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-identities", type=int, default=8)
    parser.add_argument("--samples-per-identity", type=int, default=12)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument(
        "--backbone-path",
        default="outputs/identity/erasure_drill_backbone.bin",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print(SELF_TEST_LABEL)
    print("Synthetic identities, not real people. Structural result only —")
    print("no timing claim.")
    print("=" * 78)

    result = run_erasure_drill(
        backbone_weights_path=Path(args.backbone_path),
        n_identities=args.n_identities,
        samples_per_identity=args.samples_per_identity,
        input_dim=args.input_dim,
        feature_dim=args.feature_dim,
        seed=args.seed,
        epochs=args.epochs,
    )
    print(result.render())

    if not result.backbone_untouched:
        print(
            "\n!! FINDING: the backbone artifact was NOT byte-identical "
            "throughout. This invalidates ADR 0001's erasure architecture "
            "as currently implemented and must be reported as the day's "
            "primary finding, not a footnote.",
            file=sys.stderr,
        )

    payload = {
        "self_test_label": SELF_TEST_LABEL,
        "backbone_untouched": result.backbone_untouched,
        "adapter_shas_distinct": result.adapter_shas_distinct,
        "backbone_file_hash_before": result.backbone_file_hash_before,
        "backbone_file_hash_after_first_train": result.backbone_file_hash_after_first_train,
        "backbone_file_hash_after_retrain": result.backbone_file_hash_after_retrain,
        "first_checkpoint_adapter_sha": result.first_manifest.adapter_sha,
        "retrained_checkpoint_adapter_sha": result.retrained_manifest.adapter_sha,
        "withdrawn_identity": result.withdrawn_identity,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"Written to {args.json}")

    return 0 if (result.backbone_untouched and result.adapter_shas_distinct) else 1


if __name__ == "__main__":
    raise SystemExit(main())
