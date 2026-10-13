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

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.data.registry import DatasetEntry
from src.identity.adapter import Adapter
from src.identity.selftest import SELF_TEST_LABEL
from src.provenance import ManifestError

SCHEMA_VERSION = "1.1"
"""1.1 (2026-10-14): adds ``unresolved_restrictions``."""


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
    unresolved_restrictions: dict[str, dict[str, str | None]] = Field(
        default_factory=dict
    )
    """Every contributing dataset's UNRESOLVED
    :class:`~src.data.registry.DeploymentScopeRestriction`, keyed by dataset
    name, verbatim ``text`` and ``source`` (2026-10-14). Carried into the
    checkpoint because a restriction on the data travels with the weights:
    resolving it later in the registry does not retroactively clear a
    checkpoint whose manifest recorded it open. Empty means none were open
    at write time — not that the datasets were audited for one."""

    @model_validator(mode="after")
    def _not_promoted_with_open_restrictions(self) -> "CheckpointManifest":
        self._require_promotable()
        return self

    def _require_promotable(self) -> None:
        """The invariant lives on the model, not only on its one writer:
        direct construction and ``model_validate`` hit the validator, and
        ``write`` re-checks because ``model_copy(update=...)`` skips it."""
        if self.promoted and self.unresolved_restrictions:
            raise ManifestError(
                "checkpoint manifest refuses promoted=True while contributing "
                "datasets carry unresolved restrictions: "
                f"{sorted(self.unresolved_restrictions)}. Resolving one is a "
                "human (counsel) decision recorded in configs/datasets.yaml, "
                "never inferred here — see DeploymentScopeRestriction."
            )

    def write(self, path: Path) -> Path:
        self._require_promotable()
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


def _unresolved_restrictions(
    dataset_entries: dict[str, DatasetEntry],
) -> dict[str, dict[str, str | None]]:
    return {
        name: {
            "text": entry.deployment_scope_restriction.text,
            "source": entry.deployment_scope_restriction.source,
        }
        for name, entry in dataset_entries.items()
        if entry.deployment_scope_restriction is not None
        and not entry.deployment_scope_restriction.resolved
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
    """Write a REAL checkpoint's manifest. Refuses on non-lane-C data, and
    refuses ``promoted=True`` while any contributing dataset carries an
    unresolved restriction.

    This is the promotion gate's durable half: :func:`src.identity.promotion.
    evaluate_promotion` decides whether the METRIC clears its baseline, and
    a checkpoint only becomes promoted when that decision is recorded here
    — so this is where an open legal question about the training data
    blocks it, regardless of how good the number is.

    Raises:
        ManifestError: via :func:`_require_lane_c`, if ``dataset_entries``
            contains anything outside lane C; or if ``promoted`` is True and
            any entry's ``deployment_scope_restriction`` is unresolved.
    """
    dataset_shas_and_lanes = {
        name: f"C:{sha}" for name, sha in _require_lane_c(dataset_entries).items()
    }
    unresolved = _unresolved_restrictions(dataset_entries)
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
        unresolved_restrictions=unresolved,
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
