"""Model weights through the dataset registry (Day 42, Objective B2).

One registry, not two: ``kind: model`` entries take the same snapshot and
signature gate as datasets, plus ``ship_blocked`` and a consumption manifest
that refuses to be promotable when anything it consumed cannot ship.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from src.data.registry import (
    DatasetEntry,
    DatasetRegistry,
    LicenseNotVerified,
    LicenseSnapshot,
    ShipBlocked,
    normalise,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import fetch_dataset  # noqa: E402

SHA = "5650961749fa93567c0d46fc7f43ea4f9e914107"


def _snapshot(verified_class: str = "Apache-2.0") -> LicenseSnapshot:
    return LicenseSnapshot(
        url=f"https://huggingface.co/PekingU/rtdetr_v2_r18vd/raw/{SHA}/README.md",
        verified_date=date(2026, 10, 15),
        text_sha256="0" * 64,
        verified_by="a human",
        verified_class=verified_class,
        resolved_commit_sha=SHA,
    )


def _model(**kwargs: Any) -> DatasetEntry:
    base: dict[str, Any] = {
        "name": "Test-Detector",
        "kind": "model",
        "lane": "R",
        "hosting": "huggingface",
        "hypothesis_class": "Apache-2.0",
        "hf_files": ["config.json"],
    }
    base.update(kwargs)
    return DatasetEntry.model_validate(base)


def _registry(*entries: DatasetEntry) -> DatasetRegistry:
    return DatasetRegistry({normalise(e.name): e for e in entries})


def test_repository_shortlist_is_registered_unsigned() -> None:
    registry = DatasetRegistry.load(REPO_ROOT / "configs" / "datasets.yaml")
    for name in ("RT-DETRv2-S-COCO", "YOLOX-S-COCO"):
        entry = registry.get(name)
        assert entry.kind == "model" and entry.base_training_data
        assert entry.license_snapshot is None, "only a human signs (B2)"
    assert registry.get("Ultralytics-YOLO").ship_blocked


@pytest.mark.parametrize(
    "licence", ["AGPL-3.0", "CC-BY-NC-4.0", "research only", "GPL-3.0"]
)
def test_a_ship_blocking_licence_forces_the_flag(licence: str) -> None:
    with pytest.raises(ValueError, match="ship_blocked"):
        _model(hypothesis_class=licence)
    assert _model(hypothesis_class=licence, ship_blocked=True).ship_blocked


def test_a_signed_class_that_blocks_shipping_also_forces_the_flag() -> None:
    with pytest.raises(ValueError, match="ship_blocked"):
        _model(license_snapshot=_snapshot("AGPL-3.0"))


def test_fetching_weights_without_a_snapshot_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = _registry(_model())
    monkeypatch.setattr(DatasetRegistry, "load", classmethod(lambda cls, p: registry))
    touched: list[str] = []
    monkeypatch.setattr(
        fetch_dataset, "_fetch_hf_file", lambda *a, **k: touched.append("download")
    )
    assert fetch_dataset.main(["Test-Detector"]) == 2
    assert "license snapshot" in capsys.readouterr().err
    assert touched == []
    with pytest.raises(LicenseNotVerified):
        registry.require_fetchable("Test-Detector")


def test_weights_are_only_fetched_at_the_signed_commit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = _registry(_model(license_snapshot=_snapshot()))
    monkeypatch.setattr(DatasetRegistry, "load", classmethod(lambda cls, p: registry))
    assert fetch_dataset.main(["Test-Detector", "--revision", "a" * 40]) == 2
    assert "not the signed commit" in capsys.readouterr().err


def test_ship_blocked_model_may_be_evaluated_offline() -> None:
    entry = _model(
        hypothesis_class="AGPL-3.0", ship_blocked=True, license_snapshot=_snapshot()
    )
    assert _registry(entry).open_for_eval("Test-Detector") is entry


def test_a_manifest_consuming_a_ship_blocked_model_cannot_be_promotable() -> None:
    blocked = _model(
        hypothesis_class="AGPL-3.0", ship_blocked=True, license_snapshot=_snapshot()
    )
    registry = _registry(blocked)
    offline = registry.consumption_manifest(["Test-Detector"])
    assert not offline.promotable and "Test-Detector" in offline.ship_blockers
    with pytest.raises(ShipBlocked, match="ship_blocked is true"):
        registry.consumption_manifest(["Test-Detector"], promotable=True)
    with pytest.raises(ShipBlocked):
        type(offline).model_validate({**offline.model_dump(), "promotable": True})


def test_an_unsigned_model_is_not_shippable_either() -> None:
    with pytest.raises(ShipBlocked, match="no licence snapshot"):
        _registry(_model()).require_shippable("Test-Detector")


def test_a_signed_permissive_model_is_promotable() -> None:
    registry = _registry(_model(license_snapshot=_snapshot()))
    assert registry.consumption_manifest(["Test-Detector"], promotable=True).promotable


def test_model_urls_parse_and_dataset_urls_do_not_pass_as_models() -> None:
    url = f"https://huggingface.co/PekingU/rtdetr_v2_r18vd/raw/{SHA}/README.md"
    assert fetch_dataset._parse_hf_dataset_url(url, "model") == (
        "PekingU/rtdetr_v2_r18vd",
        SHA,
        "README.md",
    )
    with pytest.raises(ValueError):
        fetch_dataset._parse_hf_dataset_url(
            "https://huggingface.co/datasets/a/b/raw/main/README.md", "model"
        )
