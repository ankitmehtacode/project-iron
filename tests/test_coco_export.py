"""SmartSpaces -> COCO split rules refuse structurally (Day 42, B4)."""

from __future__ import annotations

import pytest

from src.data.coco_export import (
    ExportRefused,
    FrameRecord,
    ScenePlacement,
    assign_splits,
    height_bucket,
    validate_split_plan,
)


def _p(scene: str, split: str, of: str | None = None) -> ScenePlacement:
    return ScenePlacement(
        scene=scene, dataset_split=split, rerender_of=of  # type: ignore[arg-type]
    )


def test_rerender_of_a_train_scene_is_excluded_not_moved() -> None:
    splits, excluded = assign_splits([_p("W000", "train"), _p("W001", "train", "W000")])
    assert splits == {"W000": "train"}
    assert "W001" in excluded


def test_rerender_follows_a_val_source() -> None:
    splits, excluded = assign_splits([_p("W020", "val"), _p("W021", "val", "W020")])
    assert splits == {"W020": "val", "W021": "val"} and not excluded


def test_rerender_whose_dataset_split_differs_still_follows_its_source() -> None:
    splits, _ = assign_splits([_p("S", "val"), _p("R", "train", "S")])
    assert splits["R"] == "val"


def test_plan_with_a_rerender_in_train_is_refused() -> None:
    with pytest.raises(ExportRefused, match="re-render placed in train"):
        validate_split_plan({"S": "train", "R": "train"}, {"R": "S"})


def test_plan_splitting_a_rerender_from_its_source_is_refused() -> None:
    with pytest.raises(ExportRefused, match="different sides"):
        validate_split_plan({"S": "train", "R": "val"}, {"R": "S"})


def test_a_rerender_frame_record_in_train_is_refused() -> None:
    with pytest.raises(ExportRefused, match="cannot be in train"):
        FrameRecord(1, "R", "c", 5, "train", True, 5, True, "f.png")


def test_source_frames_carry_no_cut() -> None:
    with pytest.raises(ExportRefused, match="only re-render"):
        FrameRecord(1, "S", "c", 5, "val", False, 5, False, "f.png")


def test_height_buckets_are_half_open_and_cover_everything() -> None:
    edges = (0.0, 32.0, 64.0, float("inf"))
    assert [height_bucket(h, edges) for h in (0, 31.9, 32, 63.9, 64, 1e6)] == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]
