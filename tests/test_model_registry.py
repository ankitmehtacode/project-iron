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
        "base_training_data": "COCO 2017 train",
    }
    base.update(kwargs)
    return DatasetEntry.model_validate(base)


def _registry(*entries: DatasetEntry) -> DatasetRegistry:
    return DatasetRegistry({normalise(e.name): e for e in entries})


def test_repository_shortlist_is_signed_by_a_named_human_at_a_pinned_source() -> None:
    """Registered unsigned on Day 42 (B2); signed by the human at the B2
    checkpoint the same day. Each snapshot names its signer and a source that
    cannot move: a commit-pinned raw URL, recorded commit for HuggingFace."""
    registry = DatasetRegistry.load(REPO_ROOT / "configs" / "datasets.yaml")
    for name in ("RT-DETRv2-S-COCO", "YOLOX-S-COCO"):
        entry = registry.get(name)
        assert entry.kind == "model" and entry.base_training_data
        snap = entry.license_snapshot
        assert snap is not None and snap.verified_by == "Ankit Mehta"
        assert snap.verified_class == "Apache-2.0"
    rt = registry.get("RT-DETRv2-S-COCO").license_snapshot
    assert rt is not None and rt.resolved_commit_sha == SHA and SHA in rt.url
    yolox = registry.get("YOLOX-S-COCO").license_snapshot
    assert yolox is not None
    assert "6ddff4824372906469a7fae2dc3206c7aa4bbaee" in yolox.url
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


def test_signed_model_is_promotable_only_once_training_data_is_resolved() -> None:
    unresolved = _registry(_model(license_snapshot=_snapshot()))
    with pytest.raises(ShipBlocked, match="training-data licence unresolved"):
        unresolved.consumption_manifest(["Test-Detector"], promotable=True)
    resolved = _registry(
        _model(
            lane="S",
            license_snapshot=_snapshot(),
            base_training_data_resolution="counsel, 2026-11-01: memo ref",
        )
    )
    assert resolved.require_shippable("Test-Detector")
    lane_r = _registry(
        _model(
            license_snapshot=_snapshot(),
            base_training_data_resolution="counsel, 2026-11-01: memo ref",
        )
    )
    with pytest.raises(ShipBlocked, match="lane R"):
        lane_r.require_shippable("Test-Detector")


def test_a_model_must_state_its_training_data() -> None:
    with pytest.raises(ValueError, match="base_training_data"):
        _model(base_training_data="")


def test_an_empty_resolution_is_refused() -> None:
    with pytest.raises(ValueError, match="who resolved"):
        _model(base_training_data_resolution="  ")


def test_the_signed_shortlist_is_still_not_shippable() -> None:
    """Day 42: signing the weight licences must not make the models
    shippable while COCO's image-licence question is open."""
    registry = DatasetRegistry.load(REPO_ROOT / "configs" / "datasets.yaml")
    for name in ("RT-DETRv2-S-COCO", "YOLOX-S-COCO"):
        with pytest.raises(ShipBlocked, match="training-data"):
            registry.require_shippable(name)


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


def test_verify_license_records_a_pinned_snapshot_for_a_model_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact command printed at the B2 checkpoint, run against a sandbox
    registry with the network faked: a model URL (no /datasets/) resolves,
    the raw README at the pinned commit is hashed, and the snapshot written
    back loads with the commit and the signer's name."""
    import hashlib

    import yaml

    registry_path = tmp_path / "datasets.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {
                        "name": "RT-DETRv2-S-COCO",
                        "kind": "model",
                        "lane": "R",
                        "hosting": "huggingface",
                        "hypothesis_class": "Apache-2.0",
                        "hf_files": ["config.json"],
                        "base_training_data": "COCO 2017 train",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(fetch_dataset, "REGISTRY_PATH", registry_path)
    text = b"---\nlicense: apache-2.0\n---\n"
    kinds: list[str] = []

    def fake_raw(repo_id: str, sha: str, path: str, token: Any, kind: str) -> bytes:
        kinds.append(kind)
        assert (repo_id, sha, path) == ("PekingU/rtdetr_v2_r18vd", SHA, "README.md")
        return text

    monkeypatch.setattr(fetch_dataset, "_fetch_hf_raw_file", fake_raw)
    url = f"https://huggingface.co/PekingU/rtdetr_v2_r18vd/raw/{SHA}/README.md"
    code = fetch_dataset.main(
        [
            "RT-DETRv2-S-COCO",
            "--verify-license",
            "--license-url",
            url,
            "--verified-class",
            "Apache-2.0",
            "--verified-by",
            "Test Signer",
            "--i-have-read-it",
        ]
    )
    assert code == 0 and kinds == ["model"]
    entry = DatasetRegistry.load(registry_path).get("RT-DETRv2-S-COCO")
    snap = entry.license_snapshot
    assert snap is not None and snap.url == url
    assert snap.resolved_commit_sha == SHA and snap.verified_by == "Test Signer"
    assert snap.text_sha256 == hashlib.sha256(text).hexdigest()
    assert entry.kind == "model" and entry.hf_files == ["config.json"]


@pytest.mark.parametrize(
    "licence", ["CC BY-NC 4.0", "Research use only", "GNU GPL v3", "cc_by_nc_sa"]
)
def test_human_written_licence_spellings_also_force_the_flag(licence: str) -> None:
    """Review finding (Day 42): the markers matched only SPDX-like spellings."""
    with pytest.raises(ValueError, match="ship_blocked"):
        _model(hypothesis_class=licence)


def test_direct_construction_cannot_claim_a_clean_slate() -> None:
    """Review finding (Day 42): the validator trusted the self-reported
    ship_blockers field; it now re-derives them from the registry on disk."""
    from src.data.registry import ConsumptionManifest

    with pytest.raises(ShipBlocked, match="Ultralytics-YOLO"):
        ConsumptionManifest(consumes=["Ultralytics-YOLO"], promotable=True)
    with pytest.raises(ShipBlocked, match="RT-DETRv2-S-COCO"):
        ConsumptionManifest(consumes=["RT-DETRv2-S-COCO"], promotable=True)


def test_a_tampered_local_weight_is_refused_and_not_rebaselined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review finding (Day 42): a re-fetch trusted files already on disk and
    rewrote content_sha to whatever they hashed to."""
    import hashlib
    from types import SimpleNamespace

    import yaml

    good, bad = b"real weights", b"tampered weights"
    recorded = "f" * 64
    registry_path = tmp_path / "datasets.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {
                        "name": "Test-Detector",
                        "kind": "model",
                        "lane": "R",
                        "hosting": "huggingface",
                        "hypothesis_class": "Apache-2.0",
                        "hf_files": ["model.safetensors"],
                        "base_training_data": "COCO 2017 train",
                        "content_sha": recorded,
                        "license_snapshot": _snapshot().model_dump(mode="json"),
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(fetch_dataset, "REGISTRY_PATH", registry_path)
    monkeypatch.setenv("IRON_PATHS__MODELS_DIR", str(tmp_path / "models"))
    weights = tmp_path / "models" / "weights" / "Test-Detector" / SHA
    weights.mkdir(parents=True)
    (weights / "model.safetensors").write_bytes(bad)
    item = SimpleNamespace(
        path="model.safetensors",
        lfs=SimpleNamespace(sha256=hashlib.sha256(good).hexdigest()),
        blob_id=None,
    )
    monkeypatch.setattr(
        fetch_dataset.huggingface_hub.HfApi,
        "list_repo_tree",
        lambda self, *a, **k: [item],
    )
    assert fetch_dataset.main(["Test-Detector"]) == 1
    after = DatasetRegistry.load(registry_path).get("Test-Detector")
    assert after.content_sha == recorded
