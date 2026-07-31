"""Derived-artifact metadata and the compatibility guard that reads it.

The disaster this prevents
--------------------------
A FAISS index is only meaningful against the exact encoder and preprocessing
that produced its vectors. Re-export the model, change a normalisation
constant, or re-quantize, and every vector in the index becomes a set of
coordinates in a space the query vectors no longer live in. Nothing crashes.
Search keeps returning results, ranked confidently, and they are wrong.

There is no way to detect this after the fact from the vectors themselves, so
it has to be recorded at write time and checked at read time. Every derived
artifact — index, embedding parquet, PCA matrix — gets a ``<name>.meta.json``
sidecar naming the shas that produced it, and the read path refuses on
mismatch.

Missing metadata is a refusal, not a pass
-----------------------------------------
An artifact with no sidecar was written before this existed, which means it was
written by the pipeline that applied no channel standardisation. Those vectors
are void. Treating an absent sidecar as "probably fine" would silently
readmit exactly the data this guard exists to exclude, so it raises like any
other mismatch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

METADATA_SUFFIX = ".meta.json"
SCHEMA_VERSION: Literal["1.0"] = "1.0"

REBUILD_SCRIPT = "scripts/rebuild_index.py"

ArtifactKind = Literal["faiss_index", "embedding_parquet", "pca_matrix"]


class ArtifactMismatch(RuntimeError):
    """Raised when a derived artifact does not match the current producers."""


@dataclass(frozen=True)
class ArtifactMetadata:
    """What produced a derived artifact.

    Attributes:
        kind: What sort of artifact this describes.
        encoder_sha: Content hash of the encoder that produced the vectors.
        preprocess_sha: Hash of the :class:`~src.models.preprocess.PreprocessSpec`
            in force when they were produced. Two vectors are comparable only
            if both shas match.
        pca_sha: Hash of the PCA matrix applied, when one was.
        manifest_sha: The run that wrote this artifact.
        created_at_utc: When.
        vector_count: How many vectors, for a cheap sanity check on load.
        notes: Free text for humans.
    """

    kind: ArtifactKind
    encoder_sha: str
    preprocess_sha: str
    manifest_sha: str
    pca_sha: str | None = None
    created_at_utc: str = ""
    vector_count: int | None = None
    notes: str = ""
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("encoder_sha", "preprocess_sha", "manifest_sha"):
            if not getattr(self, name):
                raise ValueError(
                    f"ArtifactMetadata.{name} is required: an artifact that "
                    "cannot name its producers cannot be validated against them"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "encoder_sha": self.encoder_sha,
            "preprocess_sha": self.preprocess_sha,
            "pca_sha": self.pca_sha,
            "manifest_sha": self.manifest_sha,
            "created_at_utc": self.created_at_utc
            or datetime.now(timezone.utc).isoformat(),
            "vector_count": self.vector_count,
            "notes": self.notes,
            "schema_version": self.schema_version,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArtifactMetadata":
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ArtifactMismatch(
                f"artifact metadata schema_version {version!r} is not "
                f"{SCHEMA_VERSION!r}; migrate the sidecar rather than "
                "reinterpreting its fields"
            )
        try:
            return cls(
                kind=payload["kind"],
                encoder_sha=payload["encoder_sha"],
                preprocess_sha=payload["preprocess_sha"],
                manifest_sha=payload["manifest_sha"],
                pca_sha=payload.get("pca_sha"),
                created_at_utc=payload.get("created_at_utc", ""),
                vector_count=payload.get("vector_count"),
                notes=payload.get("notes", ""),
                extra=payload.get("extra", {}),
            )
        except KeyError as exc:
            raise ArtifactMismatch(
                f"artifact metadata is missing required field {exc}"
            ) from exc


def sidecar_path_for(artifact_path: Path) -> Path:
    """Metadata path for an artifact: ``index.faiss`` → ``index.faiss.meta.json``."""
    return artifact_path.with_name(artifact_path.name + METADATA_SUFFIX)


def write_metadata(artifact_path: Path, metadata: ArtifactMetadata) -> Path:
    """Write the sidecar beside an artifact.

    Call this in the same operation that writes the artifact. A sidecar written
    later can describe a different state than the one that produced the file.
    """
    path = sidecar_path_for(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def read_metadata(artifact_path: Path) -> ArtifactMetadata:
    """Read an artifact's sidecar.

    Raises:
        ArtifactMismatch: if the sidecar is absent or unreadable. See the
            module docstring: absence means the artifact predates this
            mechanism, and those vectors are void.
    """
    path = sidecar_path_for(artifact_path)
    if not path.exists():
        raise ArtifactMismatch(
            f"{artifact_path} has no metadata sidecar at {path}. Artifacts "
            "written before provenance coupling existed were produced by the "
            "pipeline that applied no channel standardisation, so their "
            "vectors are void and cannot be validated. Rebuild with "
            f"{REBUILD_SCRIPT}."
        )
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactMismatch(f"cannot read {path}: {exc}") from exc
    return ArtifactMetadata.from_dict(payload)


def require_compatible(
    artifact_path: Path,
    *,
    encoder_sha: str,
    preprocess_sha: str,
    pca_sha: str | None = None,
) -> ArtifactMetadata:
    """Refuse to use an artifact whose producers differ from the current ones.

    Call at load time, before the first query. Checking lazily on first
    mismatch-sensitive operation means some queries have already returned
    confident nonsense by the time anyone notices.

    Args:
        artifact_path: The index or parquet about to be read.
        encoder_sha: Hash of the encoder now in use.
        preprocess_sha: Hash of the preprocessing spec now in use.
        pca_sha: Hash of the PCA matrix now in use, if any.

    Returns:
        The artifact's metadata, when everything matches.

    Raises:
        ArtifactMismatch: naming every field that differs and the rebuild
            script.
    """
    metadata = read_metadata(artifact_path)

    differences: list[str] = []
    if metadata.encoder_sha != encoder_sha:
        differences.append(
            f"encoder_sha: artifact {metadata.encoder_sha[:12]}... vs current "
            f"{encoder_sha[:12]}..."
        )
    if metadata.preprocess_sha != preprocess_sha:
        differences.append(
            f"preprocess_sha: artifact {metadata.preprocess_sha[:12]}... vs "
            f"current {preprocess_sha[:12]}..."
        )
    if (metadata.pca_sha or None) != (pca_sha or None):
        differences.append(f"pca_sha: artifact {metadata.pca_sha} vs current {pca_sha}")

    if differences:
        raise ArtifactMismatch(
            f"{artifact_path} was built by different producers than are now in "
            "use, so its vectors are not comparable with query vectors:\n  "
            + "\n  ".join(differences)
            + f"\nRebuild it with {REBUILD_SCRIPT}. Querying across this "
            "mismatch returns confidently-ranked results computed in the wrong "
            "space."
        )
    return metadata
