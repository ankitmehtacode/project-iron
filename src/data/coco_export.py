"""Split rules for exporting SmartSpaces frames as COCO (Day 42, B4).

The rules are code, not convention, because the failure they prevent is
silent: a Cosmos Transfer re-render is its source scene's trajectories under
different RGB, so a re-render in train with its source in val (or the other
way round) is the val set's ground truth in the training data.

- :func:`assign_splits` places every scene. A re-render goes wherever its
  source goes, and a re-render whose source is in train has nowhere legal to
  go: it is EXCLUDED (and counted), never moved to val.
- :func:`validate_split_plan` refuses, at export time, any plan where a
  re-render is in train or a re-render and its source are split apart —
  whatever produced the plan.
- :class:`FrameRecord` carries ``frames_since_cut`` and ``in_transient`` so
  metrics can drop frames inside a re-render's measured post-cut transient
  (Objective A2) without re-deriving cuts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Split = Literal["train", "val"]


class ExportRefused(RuntimeError):
    """Raised when an export plan would break a split rule."""


@dataclass(frozen=True)
class ScenePlacement:
    scene: str
    dataset_split: Split
    """The split the dataset itself puts the scene in."""
    rerender_of: str | None


def assign_splits(
    placements: list[ScenePlacement],
) -> tuple[dict[str, Split], dict[str, str]]:
    """``(split by scene, excluded scene -> reason)``.

    A source scene keeps the dataset's split. A re-render follows its source;
    if that is train, the re-render is excluded, because train may not hold a
    re-render and moving it to val would leak its source's trajectories.
    """
    by_scene = {p.scene: p for p in placements}
    splits: dict[str, Split] = {}
    excluded: dict[str, str] = {}
    for p in placements:
        if p.rerender_of is None:
            splits[p.scene] = p.dataset_split
    for p in placements:
        if p.rerender_of is None:
            continue
        source = by_scene.get(p.rerender_of)
        if source is None:
            excluded[p.scene] = f"source {p.rerender_of} not fetched"
        elif splits[source.scene] == "train":
            excluded[p.scene] = (
                f"re-render of train scene {source.scene}: train may not hold a "
                "re-render, and val may not hold its source's trajectories"
            )
        else:
            splits[p.scene] = splits[source.scene]
    validate_split_plan(splits, {p.scene: p.rerender_of for p in placements})
    return splits, excluded


def validate_split_plan(
    splits: Mapping[str, str], rerender_of: Mapping[str, str | None]
) -> None:
    """Refuse a plan with a re-render in train, or a re-render on a different
    side from its source."""
    problems = []
    for scene, split in splits.items():
        source = rerender_of.get(scene)
        if source is None:
            continue
        if split == "train":
            problems.append(f"{scene} is a Cosmos re-render placed in train")
        if source in splits and splits[source] != split:
            problems.append(
                f"{scene} ({split}) and its source {source} ({splits[source]}) "
                "are on different sides of the split"
            )
    if problems:
        raise ExportRefused("; ".join(problems))


@dataclass(frozen=True)
class FrameRecord:
    """One exported frame."""

    image_id: int
    scene: str
    camera: str
    frame_index: int
    split: Split
    ct25_rerender: bool
    frames_since_cut: int | None
    """Frames since the last style cut (re-renders only; None for sources)."""
    in_transient: bool
    """True when ``frames_since_cut`` is inside the measured post-cut
    transient (A2). Such frames are kept in the export and excluded from
    metrics."""
    file_name: str

    def __post_init__(self) -> None:
        if self.ct25_rerender and self.split == "train":
            raise ExportRefused(f"{self.scene} is a re-render; it cannot be in train")
        if not self.ct25_rerender and (
            self.frames_since_cut is not None or self.in_transient
        ):
            raise ExportRefused("only re-render frames have cuts")


def height_bucket(height_px: float, edges: tuple[float, ...]) -> int:
    """Index of the bucket ``height_px`` falls in (``edges`` ascending)."""
    for i in range(len(edges) - 1):
        if edges[i] <= height_px < edges[i + 1]:
            return i
    return len(edges) - 2
