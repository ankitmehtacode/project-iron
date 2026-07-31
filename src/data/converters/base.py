"""The canonical form every dataset converts into, and the converter protocol.

Why a canonical form
--------------------
Every public dataset ships its own annotation dialect: NTU encodes the action
in the filename, CVAT exports XML tracks, MEVA has its own activity YAML. If
the eval harness understood each dialect, every metric would be implemented
N times and disagree in N ways. Instead, converters translate once, at the
edge, into the pipeline's own vocabulary — clips plus GT records typed as
:class:`src.events.Event` — and everything downstream speaks exactly one
language. The annotation spec and the production schema cannot drift, because
they are the same types.

GT events carry provenance like any other event: the ``manifest_sha`` of the
conversion run that produced them. "Which converter version made this GT" is a
query, not an archaeology project — conversions have bugs too, and a GT file
that cannot be traced to the code that wrote it cannot be re-examined when a
metric looks wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from src.events import Event, write_events_parquet


@dataclass(frozen=True)
class CanonicalClip:
    """One clip in canonical form.

    Attributes:
        clip_id: Stable identifier, unique within the source dataset.
        source_dataset: Registry name of the dataset this came from, so the
            lane travels with the clip.
        frames: Frame count.
        fps: Frame rate the timestamps were synthesized at, when the source
            provides frame indices rather than wall-clock time.
        media_path: Decoded frames or an ffmpeg-pinned mp4, when media exists.
            ``None`` for annotation-only sources (skeleton files carry no
            pixels).
    """

    clip_id: str
    source_dataset: str
    frames: int
    fps: float
    media_path: Path | None = None


@dataclass
class CanonicalClips:
    """A converter's output: clips, their GT events, and what got skipped.

    ``skipped`` is part of the contract, not a log line. A converter that
    silently drops what it cannot map produces a GT set whose gaps are
    invisible, and a verb detector evaluated against it gets credit for
    "correctly" not firing on activities the GT simply never mentions.
    """

    dataset: str
    clips: list[CanonicalClip] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    """clip_id -> reason it produced no GT."""

    def write_events(self, path: Path) -> Path:
        """Write the GT events through the production parquet path.

        The same writer production uses, deliberately: a GT file that only a
        special reader can load is a second dialect, which is the disease the
        canonical form exists to cure.
        """
        return write_events_parquet(self.events, path)


@runtime_checkable
class Converter(Protocol):
    """A dataset-specific translator into the canonical form."""

    @property
    def dataset(self) -> str:
        """Registry name of the dataset this converter understands."""
        ...

    def convert(self, raw_dir: Path) -> CanonicalClips:
        """Translate a raw dataset directory into canonical clips + GT."""
        ...
