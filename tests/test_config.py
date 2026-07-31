"""Tests for the typed configuration layer.

The properties under test are the ones the rest of the system depends on:
paths must not depend on the working directory, ``config_sha`` must be stable
enough to compare two runs, and ``IRON_*`` must actually win over the YAML file.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import (
    DEFAULT_CONFIG_PATH,
    IronConfig,
    PathsConfig,
    apply_runtime_settings,
)


def test_config_sha_is_stable_across_loads() -> None:
    """Two loads of an unchanged config must hash identically.

    This is the property the run manifest relies on to answer "did the
    configuration change between these two runs?".
    """
    assert IronConfig.load().config_sha() == IronConfig.load().config_sha()


def test_config_sha_changes_when_a_value_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = IronConfig.load().config_sha()
    monkeypatch.setenv("IRON_PIPELINE__CLIP_FRAMES", "8")
    assert IronConfig.load().config_sha() != baseline


def test_config_sha_ignores_machine_specific_project_root() -> None:
    """Relocating the checkout must not change the config hash.

    Otherwise the same configuration run on a developer laptop and in CI would
    be reported as two different configurations, and no cross-machine
    comparison would ever be valid.
    """
    default = IronConfig.load()
    relocated = default.model_copy(
        update={
            "paths": default.paths.model_copy(
                update={"project_root": Path("/somewhere/else")}
            )
        }
    )
    assert relocated.config_sha() == default.config_sha()


def test_env_var_overrides_yaml_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRON_RUNTIME__SEED", "4242")
    assert IronConfig.load().runtime.seed == 4242


def test_paths_do_not_depend_on_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model paths must resolve identically from any working directory.

    The pre-refactor code used ``"../models/..."``, so running the pipeline
    from the repository root — the documented location — failed to find files
    that were sitting right there.
    """
    from_here = IronConfig.load().paths.resolved_vjepa_xml
    monkeypatch.chdir(tmp_path)
    from_elsewhere = IronConfig.load().paths.resolved_vjepa_xml
    assert from_here == from_elsewhere
    assert from_here.is_absolute()


def test_model_paths_live_under_the_repository() -> None:
    """Weights resolve inside the checkout, not in a sibling directory."""
    paths = IronConfig.load().paths
    assert paths.project_root in paths.resolved_vjepa_xml.parents
    assert paths.project_root in paths.resolved_cotracker_checkpoint.parents


def test_absolute_path_overrides_are_left_alone() -> None:
    paths = PathsConfig(vjepa_xml=Path("/opt/models/custom.xml"))
    assert paths.resolved_vjepa_xml == Path("/opt/models/custom.xml")


def test_default_yaml_exists_and_parses() -> None:
    assert DEFAULT_CONFIG_PATH.exists(), f"{DEFAULT_CONFIG_PATH} is missing"
    assert IronConfig.load(DEFAULT_CONFIG_PATH).pipeline.clip_frames > 0


def test_explicitly_named_missing_config_is_an_error(tmp_path: Path) -> None:
    """A caller who names a file expects that file, not silent defaults."""
    with pytest.raises(FileNotFoundError):
        IronConfig.load(tmp_path / "nope.yaml")


def test_preserved_legacy_values() -> None:
    """Pin the values migrated out of endurance_run.py.

    This refactor moved constants; it must not have retuned them. If a later
    change to these numbers is intended and measured, update this test in the
    same commit as the measurement.
    """
    config = IronConfig.load()
    assert config.pipeline.grid_size == 10
    assert config.pipeline.clip_frames == 4
    assert config.pipeline.clip_h == 224
    assert config.pipeline.clip_w == 224
    assert config.pipeline.clip_channels == 3
    assert config.endurance.num_clips == 200
    assert config.runtime.device == "CPU"

    # leak_threshold_mb (150 MB of total growth) is deliberately gone. It was
    # half of a two-magic-number rule joined by AND, which let a real leak pass
    # on any run short enough. Replaced by a rate with units attached; see
    # src/endurance/gates.py.
    assert not hasattr(config.endurance, "leak_threshold_mb")
    assert config.endurance.leak_mb_per_hour_max == 50.0


def test_derived_pipeline_geometry() -> None:
    config = IronConfig.load()
    assert config.pipeline.clip_shape == (1, 4, 3, 224, 224)
    assert config.pipeline.spatial_patches == 196
    # 4 frames at tubelet 2 is 2 temporal slots, not 4. See iron-contracts.
    assert config.pipeline.temporal_slots == 2


def test_config_is_immutable() -> None:
    config = IronConfig.load()
    with pytest.raises(Exception):
        config.pipeline.clip_frames = 99  # type: ignore[misc]


def test_openvino_properties_empty_when_threads_unset() -> None:
    """Zero threads must emit no property, leaving OpenVINO's default intact."""
    config = IronConfig.load()
    assert config.runtime.openvino_properties() == {}


def test_openvino_properties_set_when_threads_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IRON_RUNTIME__OV_NUM_THREADS", "4")
    assert IronConfig.load().runtime.openvino_properties() == {
        "INFERENCE_NUM_THREADS": 4
    }


def test_apply_runtime_settings_reports_what_it_did() -> None:
    """Skipped settings must be reported, never silently dropped.

    A run that believes it is seeded but is not produces irreproducible results
    that look reproducible.
    """
    applied = apply_runtime_settings(IronConfig.load())
    assert applied.numpy_seeded is True
    assert applied.seed == IronConfig.load().runtime.seed
    try:
        import torch  # noqa: F401
    except ImportError:
        assert applied.torch_seeded is False
        assert any("torch" in reason for reason in applied.skipped)
    else:
        assert applied.torch_seeded is True
        assert applied.skipped == ()


def test_seeding_makes_synthetic_clips_reproducible() -> None:
    """The seed must actually determine the numbers the harness generates."""
    import numpy as np

    config = IronConfig.load()
    apply_runtime_settings(config)
    first = np.random.rand(*config.pipeline.clip_shape)
    apply_runtime_settings(config)
    second = np.random.rand(*config.pipeline.clip_shape)
    assert np.array_equal(first, second)


def test_env_prefix_is_namespaced() -> None:
    """Unprefixed variables must not leak into config."""
    assert os.environ.get("SEED") != "1"  # guard against a confusing local env
    assert IronConfig.model_config["env_prefix"] == "IRON_"
