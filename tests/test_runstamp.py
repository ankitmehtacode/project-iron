"""Run-stamped outputs never overwrite (Day 43)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.runstamp import latest, stamped, write_exclusive, write_run_stamped


def test_two_runs_write_two_files(tmp_path: Path) -> None:
    first = write_run_stamped(tmp_path, "eval", ".json", "one")
    second = write_run_stamped(tmp_path, "eval", ".json", "two")
    assert first != second
    assert first.read_text() == "one" and second.read_text() == "two"
    assert latest(tmp_path / "eval.json") in {first, second}


def test_existing_file_is_refused_not_truncated(tmp_path: Path) -> None:
    path = write_exclusive(tmp_path / "scorecard.json", "original")
    with pytest.raises(FileExistsError):
        write_exclusive(path, "replacement")
    assert path.read_text() == "original"


def test_scorecard_write_refuses_an_existing_path(tmp_path: Path) -> None:
    from src.data import scorecard

    path = tmp_path / "card.json"
    path.write_text("another run's card")

    class _Card:
        def as_dict(self) -> dict[str, str]:
            return {"x": "y"}

    with pytest.raises(FileExistsError):
        scorecard.write(_Card(), path)  # type: ignore[arg-type]
    assert path.read_text() == "another run's card"


def test_stamped_names_sort_by_time() -> None:
    a = stamped(Path("logs/run.log"), "20261111T000000Z-aaaaaa")
    b = stamped(Path("logs/run.log"), "20261111T000001Z-000000")
    assert a.name == "run.20261111T000000Z-aaaaaa.log"
    assert sorted([b, a]) == [a, b]
