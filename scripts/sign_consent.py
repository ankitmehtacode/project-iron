"""Sign a sole-subject consent record for a lane-C capture (Day 45).

The smallest command that attaches a consent record, modelled on the licence
signature (``scripts/fetch_dataset.py --verify-license ... --i-have-read-it``):
a human runs it, the flag is the signature, the tool records and does not
judge. It is for ONE case only: the person recording is the only person in
frame and consents for themselves. Any capture with another person needs the
site consent template, which is not legally reviewed
(``consent-template-counsel-review`` on the Blocker Ledger) and is not what
this command signs.

    python scripts/sign_consent.py site-zero-solo --signed-by "<name>" \\
        --captured-on 2026-11-12 --i-am-the-sole-subject

Writes ``docs/consent/<dataset>.json`` (the statement, the signer, the date)
and attaches ``consent_record`` (path, sha256, subjects=1, captured_on,
recorded_by, purpose_note) to the dataset's entry in configs/datasets.yaml,
editing only that field. The registry then opens the dataset; editing the
record afterwards makes it refuse again (sha mismatch).

Exit codes: 0 signed; 2 refused (no signature flag, unknown dataset, not
lane C, or already signed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.fetch_dataset import REGISTRY_PATH, _update_registry_yaml  # noqa: E402
from src.data.registry import DatasetRegistry  # noqa: E402

PURPOSE = (
    "Development and evaluation of the project-iron video pipeline on footage "
    "of the signer alone, recorded by the signer for this purpose. Not for "
    "identification, not shared outside this repository's owner, frames of "
    "the signer never committed to git, docs or reports. Withdrawable by "
    "deleting the footage and this record."
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dataset")
    p.add_argument("--signed-by", required=True)
    p.add_argument("--captured-on", type=date.fromisoformat, required=True)
    p.add_argument(
        "--i-am-the-sole-subject",
        action="store_true",
        help="the signature: you recorded this, and you are the only person in it",
    )
    p.add_argument("--consent-dir", type=Path, default=ROOT / "docs" / "consent")
    args = p.parse_args(argv)

    if not args.i_am_the_sole_subject:
        print(
            "Refusing: pass --i-am-the-sole-subject. This flag is a recorded "
            "signature that you made the recording and are the only person in "
            "it. Footage with anyone else in it is not covered by this command.",
            file=sys.stderr,
        )
        return 2
    registry = DatasetRegistry.load(REGISTRY_PATH)
    entry = registry.get(args.dataset)
    if entry.lane != "C":
        print(f"Refusing: {entry.name!r} is lane {entry.lane}, not C.", file=sys.stderr)
        return 2
    if entry.consent_record is not None:
        print(
            f"Refusing: {entry.name!r} already has a consent record "
            f"({entry.consent_record.path}); a signature is not overwritten.",
            file=sys.stderr,
        )
        return 2

    record = {
        "dataset": entry.name,
        "subjects": 1,
        "subject": "the signer, sole person in frame",
        "captured_on": args.captured_on.isoformat(),
        "signed_by": args.signed_by,
        "signed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "statement": (
            "I recorded this footage, I am the only person in it, and I consent "
            "to its use for the purpose below."
        ),
        "purpose": PURPOSE,
    }
    args.consent_dir.mkdir(parents=True, exist_ok=True)
    path = args.consent_dir / f"{entry.name}.json"
    if path.exists():
        print(
            f"Refusing: {path} exists; a signature is not overwritten.", file=sys.stderr
        )
        return 2
    raw = (json.dumps(record, indent=2) + "\n").encode()
    path.write_bytes(raw)
    _update_registry_yaml(
        entry.name,
        {
            "consent_record": {
                "path": str(path.resolve().relative_to(ROOT)),
                "sha": hashlib.sha256(raw).hexdigest(),
                "subjects": 1,
                "captured_on": args.captured_on.isoformat(),
                "recorded_by": args.signed_by,
                "purpose_note": "sole subject, self-consent: " + PURPOSE,
            }
        },
    )
    DatasetRegistry.load(REGISTRY_PATH).open_for_eval(entry.name)
    print(f"signed {path} and attached it to {entry.name!r}; the registry opens it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
