"""Emit a CVAT label configuration generated FROM the event schema.

One source, not two. If the annotation spec were hand-written, it would drift
from the `Verb` enum the moment either changed, and the drift would surface as
GT that fails validation months later — or worse, as GT that validates while
meaning something different from what production means by the same word.

So the label config is generated. Adding a verb to the enum changes what
annotators see the next time this runs; deleting one makes the old label
invalid immediately. The generator also encodes verb/object coherence into the
CVAT form itself: INTERACTION labels get a required object attribute, POSE
labels get none, so an annotator physically cannot record "sat a laptop".

    python scripts/cvat_project.py                    # print the config
    python scripts/cvat_project.py -o labels.json     # write it
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.events import (
    ENTITY_KINDS,
    GEOMETRIC_VERBS,
    INTERACTION_VERBS,
    POSE_VERBS,
    SCHEMA_VERSION,
    Verb,
)

# CVAT attribute names. Kept as constants because the ingest side reads them
# back; a typo on one side only would silently produce empty attributes.
ATTR_OBJECT = "object_id"
ATTR_OBJECT_KIND = "object_kind"
ATTR_ZONE = "zone_id"
ATTR_SUBJECT = "subject_id"
ATTR_SUBJECT_KIND = "subject_kind"
ATTR_OBSERVED = "observed"
ATTR_CONFIDENCE = "confidence"


def _select(values: list[str], default: str = "") -> dict[str, Any]:
    return {
        "input_type": "select",
        "values": values,
        "default_value": default or values[0],
        "mutable": False,
    }


def _text(default: str = "") -> dict[str, Any]:
    return {
        "input_type": "text",
        "values": [default],
        "default_value": default,
        "mutable": False,
    }


def label_for(verb: Verb) -> dict[str, Any]:
    """Build the CVAT label for one verb, with coherence baked into the form.

    The attribute set is decided by the verb's group, which is the same rule
    ``Event.__post_init__`` enforces. Encoding it here means an incoherent
    annotation is not merely rejected at ingest — it cannot be entered.
    """
    attributes: list[dict[str, Any]] = [
        {"name": ATTR_SUBJECT_KIND, **_select(list(ENTITY_KINDS), "session")},
        {"name": ATTR_SUBJECT, **_text()},
        # observed=False marks an inferred fact. Annotators need it for the
        # blind-spot traversals in the capture protocol, where the subject is
        # known to have crossed but was not seen doing it.
        {"name": ATTR_OBSERVED, **_select(["true", "false"], "true")},
        {"name": ATTR_CONFIDENCE, **_text("1.0")},
        {"name": ATTR_ZONE, **_text()},
    ]

    if verb in INTERACTION_VERBS:
        # Required: "picked_up" is not a fact until it says what was picked up.
        attributes.append({"name": ATTR_OBJECT_KIND, **_select(["asset"], "asset")})
        attributes.append({"name": ATTR_OBJECT, **_text()})

    group = (
        "INTERACTION"
        if verb in INTERACTION_VERBS
        else "POSE"
        if verb in POSE_VERBS
        else "GEOMETRIC"
    )
    return {
        "name": verb.value,
        "type": "any",
        "attributes": attributes,
        "_group": group,
    }


def build_label_config() -> list[dict[str, Any]]:
    """The full CVAT label list, one label per verb in the closed vocabulary."""
    return [label_for(verb) for verb in Verb]


def build_project_spec() -> dict[str, Any]:
    """Label config plus the provenance needed to check it later.

    ``schema_version`` travels with the spec so an export produced under one
    vocabulary cannot be silently ingested under another.
    """
    return {
        "generated_from": "src/events/schema.py Verb enum",
        "schema_version": SCHEMA_VERSION,
        "verb_counts": {
            "geometric": len(GEOMETRIC_VERBS),
            "pose": len(POSE_VERBS),
            "interaction": len(INTERACTION_VERBS),
            "total": len(list(Verb)),
        },
        "labels": build_label_config(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="emit only the labels array, which is what CVAT's import expects",
    )
    args = parser.parse_args(argv)

    spec = build_project_spec()
    payload = spec["labels"] if args.labels_only else spec
    text = json.dumps(payload, indent=2) + "\n"

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        counts = spec["verb_counts"]
        print(
            f"Wrote {counts['total']} labels to {args.output} "
            f"({counts['geometric']} geometric, {counts['pose']} pose, "
            f"{counts['interaction']} interaction)."
        )
        print(
            "Generated from the Verb enum: re-run after any schema change so "
            "the annotation spec cannot drift from production."
        )
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
