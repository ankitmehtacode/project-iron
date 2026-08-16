"""Tests that depth values cannot be published as something they are not.

Audit finding 4 was that Depth-Anything-V2's relative inverse depth reached a
Parquet column named ``z`` documented as "Depth (meters)", via a projector using
a focal length nobody measured. Two independent lies, both silent, both
producing numbers that plot convincingly.

These tests pin both halves shut: the schema now says what it contains, and
unprojection refuses guessed intrinsics.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from src.contracts import DepthField, FrameGeometry, Intrinsics, unproject
from src.contracts.errors import UncalibratedIntrinsics
from src.contracts.frames import placeholder_intrinsics
from src.contracts.geometry import UNCALIBRATED_OVERRIDE_ENV
from src.utils.parquet_writer import (
    DEFAULT_DEPTH_UNITS,
    TRACKING_SCHEMA,
    ParquetWriter,
    read_parquet,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def write_sample(path: Path, **kwargs: object) -> Path:
    writer = ParquetWriter(str(path))
    writer.write_batch(
        track_ids=np.array([1, 2], dtype=np.int64),
        frame_indices=np.array([0, 0], dtype=np.int64),
        x_coords=np.array([10.0, 20.0], dtype=np.float32),
        y_coords=np.array([30.0, 40.0], dtype=np.float32),
        z_coords=np.array([0.5, 0.9], dtype=np.float32),
        ocr_texts=["", ""],
        confidences=np.array([0.9, 0.8], dtype=np.float32),
        **kwargs,  # type: ignore[arg-type]
    )
    writer.close()
    return path


# ---------------------------------------------------------------------------
# The schema says what it holds
# ---------------------------------------------------------------------------


def test_schema_has_no_column_named_z() -> None:
    """``z`` implied a camera-space metric coordinate. It never was one."""
    assert "z" not in TRACKING_SCHEMA.names


def test_schema_names_the_depth_column_honestly() -> None:
    assert "disparity_rel" in TRACKING_SCHEMA.names
    assert "depth_units" in TRACKING_SCHEMA.names


def test_parquet_written_today_carries_its_units(tmp_path: Path) -> None:
    """The red-turned-green case: units travel with the data.

    A column name can be misread by a consumer that never looks at the schema
    docs. A units column has to be selected past.
    """
    path = write_sample(tmp_path / "tracks.parquet")
    table = pq.read_table(path)

    assert "disparity_rel" in table.column_names
    assert "z" not in table.column_names

    units = set(table.column("depth_units").to_pylist())
    assert units == {DEFAULT_DEPTH_UNITS} == {"disparity_rel"}


def test_units_column_is_populated_on_every_row(tmp_path: Path) -> None:
    """A null units value is the same failure as no units column at all."""
    table = pq.read_table(write_sample(tmp_path / "t.parquet"))
    values = table.column("depth_units").to_pylist()
    assert len(values) == table.num_rows
    assert all(value for value in values)


def test_metric_units_must_be_declared_explicitly(tmp_path: Path) -> None:
    """Metric output is possible, but only if the writer says so.

    This is what an anchoring step would pass once it exists. The point is that
    "meters" has to be an explicit claim rather than the default reading of an
    unlabelled column.
    """
    path = write_sample(tmp_path / "metric.parquet", depth_units="meters")
    table = pq.read_table(path)
    assert set(table.column("depth_units").to_pylist()) == {"meters"}


def test_read_parquet_round_trips(tmp_path: Path) -> None:
    frame = read_parquet(str(write_sample(tmp_path / "t.parquet")))
    assert "disparity_rel" in frame.columns
    assert "depth_units" in frame.columns


def test_readme_no_longer_claims_metres_for_the_depth_column() -> None:
    """The specific line the audit cited: README.md's schema table.

    Scoped to the table itself: the surrounding prose deliberately explains
    that the column was once wrongly called metres, and that explanation must
    survive.
    """
    readme = (REPO_ROOT / "README.md").read_text()
    assert (
        "| z | float32 | Depth (meters) |" not in readme
    ), "the exact line the audit cited is still present"

    table_rows = [
        line
        for line in readme.splitlines()
        if line.startswith("|") and "float32" in line
    ]
    # A row mentioning metres only to deny them is the correction, not the
    # defect — matching those would make this test forbid its own fix.
    offenders = [
        row
        for row in table_rows
        if re.search(r"met(er|re)s", row, re.I)
        and not re.search(r"not\s+met(er|re)s", row, re.I)
    ]
    assert (
        not offenders
    ), "README schema table still claims metric depth:\n  " + "\n  ".join(offenders)


def test_projector_output_columns_are_not_named_as_metric() -> None:
    """The legacy projector's DataFrame must not present X/Y/Z as camera space."""
    import inspect

    from src.geometry import projector_vectorized

    source = inspect.getsource(projector_vectorized.project_to_3d)
    assert '"disparity_rel"' in source
    assert '"Z":' not in source


# ---------------------------------------------------------------------------
# Guessed intrinsics cannot silently produce 3D
# ---------------------------------------------------------------------------


def metric_depth(size: int = 8) -> DepthField:
    geometry = FrameGeometry(width=size, height=size)
    return DepthField(
        data=np.full(geometry.shape, 3.0, dtype=np.float64),
        units="meters",
        geometry=geometry,
        valid_mask=np.ones(geometry.shape, dtype=np.bool_),
    )


def test_placeholder_intrinsics_are_marked_uncalibrated() -> None:
    K = placeholder_intrinsics(FrameGeometry(width=1280, height=720))
    assert K.calibrated is False
    # The same guess compute_intrinsics was making silently.
    assert K.fx == 1280.0 and K.fy == 1280.0


def test_unproject_rejects_uncalibrated_intrinsics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A guessed focal length makes every output distance arbitrary."""
    monkeypatch.delenv(UNCALIBRATED_OVERRIDE_ENV, raising=False)
    depth = metric_depth()
    K = placeholder_intrinsics(depth.geometry)
    with pytest.raises(UncalibratedIntrinsics, match="placeholders"):
        unproject(depth, np.array([[4.0, 4.0]]), K)


def test_override_allows_uncalibrated_and_logs_loudly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Development escape hatch, but it must be impossible to miss in a log."""
    monkeypatch.setenv(UNCALIBRATED_OVERRIDE_ENV, "1")
    depth = metric_depth()
    K = placeholder_intrinsics(depth.geometry)

    with caplog.at_level("WARNING"):
        result = unproject(depth, np.array([[4.0, 4.0]]), K)

    assert result.shape == (1, 3)
    assert "UNCALIBRATED GEOMETRY" in caplog.text
    assert "ARBITRARY SCALE" in caplog.text


@pytest.mark.parametrize("value", ["", "0", "true", "yes", "TRUE"])
def test_override_requires_exactly_one(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Only ``1`` enables it, so a stray truthy-looking value cannot.

    An override that accepts "false" or "no" because they are non-empty strings
    is worse than no override.
    """
    monkeypatch.setenv(UNCALIBRATED_OVERRIDE_ENV, value)
    depth = metric_depth()
    with pytest.raises(UncalibratedIntrinsics):
        unproject(depth, np.array([[4.0, 4.0]]), placeholder_intrinsics(depth.geometry))


def test_calibrated_intrinsics_still_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard must not obstruct a real calibration."""
    monkeypatch.delenv(UNCALIBRATED_OVERRIDE_ENV, raising=False)
    geometry = FrameGeometry(width=100, height=100)
    depth = DepthField(
        data=np.full(geometry.shape, 2.0, dtype=np.float64),
        units="meters",
        geometry=geometry,
        valid_mask=np.ones(geometry.shape, dtype=np.bool_),
    )
    K = Intrinsics(
        fx=50.0,
        fy=50.0,
        cx=50.0,
        cy=50.0,
        distortion=(0.0,),
        valid_for=geometry,
        calibrated=True,
    )
    assert K.calibrated is True
    np.testing.assert_allclose(
        unproject(depth, np.array([[60.0, 70.0]]), K), [[0.4, 0.8, 2.0]], rtol=1e-12
    )


def test_units_are_checked_before_calibration() -> None:
    """Non-metric depth fails on units regardless of the intrinsics.

    Ordering matters for the error message: telling someone to calibrate a
    camera when the real problem is that their depth is disparity sends them
    down the wrong path.
    """
    geometry = FrameGeometry(width=8, height=8)
    depth = DepthField(
        data=np.full(geometry.shape, 3.0, dtype=np.float64),
        units="disparity_rel",
        geometry=geometry,
        valid_mask=np.ones(geometry.shape, dtype=np.bool_),
    )
    from src.contracts.errors import UnitsError

    with pytest.raises(UnitsError):
        unproject(depth, np.array([[4.0, 4.0]]), placeholder_intrinsics(geometry))


def test_rescaling_preserves_the_uncalibrated_flag() -> None:
    """A resize must not launder a guess into a calibration."""
    K = placeholder_intrinsics(FrameGeometry(width=1280, height=720))
    rescaled = K.rescaled_to(FrameGeometry(width=640, height=360))
    assert rescaled.calibrated is False
