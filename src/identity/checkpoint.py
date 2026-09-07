"""Checkpoint provenance: every trained adapter's manifest, written before
it can be mistaken for something it is not.

Two entrypoints, deliberately not one
--------------------------------------
:func:`write_checkpoint_manifest` is the ONLY way to record a real
checkpoint, and it refuses to write if any contributing dataset is not
lane C — the same "no manifest, no load" / "wrong provenance, refuse at
write time" discipline :func:`src.provenance.require_export_manifest`
already applies to exported model artifacts, applied here to training
instead of export.

:func:`write_selftest_checkpoint_manifest` is the ONLY way to record a
SELF_TEST checkpoint (Objective 3's synthetic stand-in run). It never
touches the dataset registry — there is no real dataset to check — and it
refuses outright to be marked ``promoted=True``: promotion means "eligible
for deployment", and nothing trained on synthetic in-memory tensors is
eligible for that, regardless of what its own promotion-gate arithmetic
says. Keeping this as a second function, not a flag on the first, is
deliberate: a boolean ``self_test: bool = False`` parameter on one function
is exactly the kind of one-character mistake that turns a self-test
manifest into something that reads as real. Two names cannot be typo'd into
each other the same way.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.data.registry import DatasetEntry
from src.identity.adapter import Adapter
from src.identity.selftest import SELF_TEST_LABEL
from src.provenance import ManifestError

SCHEMA_VERSION = "1.0"


class CheckpointManifest(BaseModel):
    """Immutable record of everything that produced one adapter checkpoint."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    created_at_utc: str
    backbone_sha: str
    adapter_config_sha: str
    training_config_sha: str
    dataset_shas_and_lanes: dict[str, str]
    adapter_sha: str
    promoted: bool = False
    promotion_evidence: dict[str, Any] | None = None
    self_test_label: str | None = None
    """None for a real checkpoint. Equal to SELF_TEST_LABEL, verbatim, for
    every checkpoint produced by write_selftest_checkpoint_manifest — never
    any other value, so a reader can grep for the exact string rather than
    infer self-test status from context."""

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path


def _require_lane_c(dataset_entries: dict[str, DatasetEntry]) -> dict[str, str]:
    """Every contributing dataset must be lane C, or the manifest write
    itself refuses — see module docstring.

    Raises:
        ManifestError: naming every non-lane-C dataset, if any.
    """
    non_c = {
        name: entry.lane for name, entry in dataset_entries.items() if entry.lane != "C"
    }
    if non_c:
        raise ManifestError(
            "checkpoint manifest refuses to record training data outside "
            f"lane C: {non_c}. A checkpoint manifest is what later proves a "
            "shipped adapter's identity capability came only from "
            "consented data (ADR 0001) — recording a non-lane-C dataset "
            "here would misrepresent that even if a training run somehow "
            "reached it despite src.identity.lane_gate."
        )
    return {
        name: (entry.content_sha or "NO_CONTENT_SHA_RECORDED")
        for name, entry in dataset_entries.items()
    }


def write_checkpoint_manifest(
    path: Path,
    *,
    adapter: Adapter,
    backbone_sha: str,
    dataset_entries: dict[str, DatasetEntry],
    training_config_sha: str,
    promoted: bool = False,
    promotion_evidence: dict[str, Any] | None = None,
) -> CheckpointManifest:
    """Write a REAL checkpoint's manifest. Refuses on non-lane-C data.

    Raises:
        ManifestError: via :func:`_require_lane_c`, if ``dataset_entries``
            contains anything outside lane C.
    """
    dataset_shas_and_lanes = {
        name: f"C:{sha}" for name, sha in _require_lane_c(dataset_entries).items()
    }
    manifest = CheckpointManifest(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        backbone_sha=backbone_sha,
        adapter_config_sha=adapter.config.config_sha(),
        training_config_sha=training_config_sha,
        dataset_shas_and_lanes=dataset_shas_and_lanes,
        adapter_sha=adapter.adapter_sha,
        promoted=promoted,
        promotion_evidence=promotion_evidence,
        self_test_label=None,
    )
    manifest.write(path)
    return manifest


def write_selftest_checkpoint_manifest(
    path: Path,
    *,
    adapter: Adapter,
    backbone_sha: str,
    synthetic_dataset_label: str,
    training_config_sha: str,
    promotion_evidence: dict[str, Any] | None = None,
) -> CheckpointManifest:
    """Write a SELF_TEST checkpoint's manifest. Never touches the dataset
    registry (there is no real dataset) and is never promotable.

    Raises:
        ManifestError: unconditionally, if the caller passes anything that
            would try to mark this ``promoted`` — see module docstring.
    """
    manifest = CheckpointManifest(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        backbone_sha=backbone_sha,
        adapter_config_sha=adapter.config.config_sha(),
        training_config_sha=training_config_sha,
        dataset_shas_and_lanes={
            synthetic_dataset_label: (
                "SELF_TEST — synthetic in-memory tensors, not a registered "
                "dataset, not lane C, not any lane at all"
            )
        },
        adapter_sha=adapter.adapter_sha,
        promoted=False,
        promotion_evidence=promotion_evidence,
        self_test_label=SELF_TEST_LABEL,
    )
    manifest.write(path)
    return manifest
