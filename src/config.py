"""Typed, layered configuration for project-iron.

Why this module exists
----------------------
Configuration used to live as module-level constants inside entrypoints, most
visibly in ``endurance_run.py``. That had three concrete failure modes:

1. **CWD-dependent paths.** Model paths were written as ``"../models/..."``,
   which only resolve when the process happens to be started from one specific
   directory. Run the same script from the repository root and it exits with
   "model file not found" even though the file is right there.
2. **No way to vary a run without editing code.** Changing the clip count for a
   smoke test meant a source edit, which meant the thing you measured was not
   the thing in git.
3. **Nothing to hash.** Provenance needs a single value that answers "what
   configuration produced this result?". Scattered constants cannot be hashed.

Every path here resolves against :attr:`PathsConfig.project_root`, which is
derived from *this file's* location on disk, never from the current working
directory. Where you launch the process from can no longer change which files
it reads.

Layering and precedence
-----------------------
Highest priority wins::

    IRON_* environment variables   >   configs/default.yaml   >   field defaults

Environment variables use ``__`` to descend into sections, so
``IRON_RUNTIME__SEED=7`` sets :attr:`RuntimeConfig.seed` and
``IRON_PIPELINE__CLIP_FRAMES=8`` sets :attr:`PipelineConfig.clip_frames`.

Note that environment variables also override values passed to the constructor.
That is deliberate and is what makes ``IronConfig.load()`` layer correctly: the
YAML file is fed in through the constructor, so env vars must outrank it. The
practical rule is simply "``IRON_*`` always wins", which is what an operator
debugging a container expects.

Values mirror the previous hardcoded constants exactly
------------------------------------------------------
``grid_size=10``, ``clip_frames=4``, ``clip_h=clip_w=224``, ``num_clips=200``,
``leak_threshold_mb=150``. Several of these are known to be wrong for V-JEPA2
(whose published checkpoint is 256px / 64-frame), but correcting them changes
numerical output and therefore requires a measured, separately-gated change.
This module is a refactor: it moves the numbers, it does not tune them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# Resolved from this file's location: <project_root>/src/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

Device = Literal["CPU", "GPU", "AUTO"]


class PathsConfig(BaseModel):
    """Filesystem layout, expressed relative to the repository root.

    Fields hold *relative* paths on purpose. Absolute paths are produced on
    demand by the ``resolved_*`` properties, so the configuration itself stays
    machine-independent and two machines running the same config produce the
    same :meth:`IronConfig.config_sha`.
    """

    model_config = ConfigDict(frozen=True)

    project_root: Path = Field(
        default=PROJECT_ROOT,
        description=(
            "Absolute repository root, derived from src/config.py's location. "
            "Excluded from config_sha because it is machine-specific; the "
            "machine identity belongs in the run manifest instead."
        ),
    )
    models_dir: Path = Path("models")
    data_dir: Path = Path("data")
    cache_dir: Path = Path(".cache")
    output_dir: Path = Path("outputs")
    log_dir: Path = Path("logs")

    vjepa_xml: Path = Path("int8/vjepa2_vitl_int8.xml")
    """The PRODUCTION V-JEPA2 IR, relative to :attr:`models_dir`.

    This path is reserved. Only the artifact that production actually ran may
    ever occupy it, because it is the sole evidence of what the stored
    embeddings were computed with. A freshly exported model placed here would
    silently become "the production artifact" in every later forensic
    comparison, and the real one can never be reconstructed.

    Fresh exports go to :attr:`current_ir` under ``models/export/<utc-date>/``.
    """

    current_ir: Path | None = None
    """The IR this process should actually load, when it differs from production.

    ``None`` means "use the production artifact". Set it to point at a fresh
    export while the production one is unavailable, or to run a replacement
    candidate side by side with production. Keeping the two as separate keys is
    what stops an export from being mistaken for the thing it replaces.
    """

    cotracker_checkpoint: Path = Path("weights/cotracker3/scaled_offline.pth")
    """CoTracker3 checkpoint, relative to :attr:`models_dir`."""

    def resolve(self, path: Path) -> Path:
        """Resolve ``path`` against the project root, leaving absolutes alone."""
        return path if path.is_absolute() else (self.project_root / path)

    @property
    def resolved_models_dir(self) -> Path:
        return self.resolve(self.models_dir)

    @property
    def resolved_data_dir(self) -> Path:
        return self.resolve(self.data_dir)

    @property
    def resolved_cache_dir(self) -> Path:
        return self.resolve(self.cache_dir)

    @property
    def resolved_output_dir(self) -> Path:
        return self.resolve(self.output_dir)

    @property
    def resolved_log_dir(self) -> Path:
        return self.resolve(self.log_dir)

    @property
    def resolved_vjepa_xml(self) -> Path:
        """Absolute path to the V-JEPA2 IR.

        Replaces the old ``"../models/int8/vjepa2_vitl_int8.xml"`` literal,
        which pointed *outside* the repository and only worked from one
        undocumented working directory.
        """
        if self.vjepa_xml.is_absolute():
            return self.vjepa_xml
        return self.resolved_models_dir / self.vjepa_xml

    @property
    def resolved_current_ir(self) -> Path:
        """The IR to load: :attr:`current_ir` when set, else production."""
        if self.current_ir is None:
            return self.resolved_vjepa_xml
        if self.current_ir.is_absolute():
            return self.current_ir
        return self.resolved_models_dir / self.current_ir

    @property
    def using_production_ir(self) -> bool:
        """Whether the loaded IR is the production artifact.

        Callers that report a forensic verdict must check this. A verdict on a
        replacement export is not a verdict on production.
        """
        return self.current_ir is None

    @property
    def resolved_cotracker_checkpoint(self) -> Path:
        """Absolute path to the CoTracker3 checkpoint."""
        if self.cotracker_checkpoint.is_absolute():
            return self.cotracker_checkpoint
        return self.resolved_models_dir / self.cotracker_checkpoint


class RuntimeConfig(BaseModel):
    """Device selection, thread caps, and the determinism seed.

    Thread counts default to ``0``, meaning "leave the runtime's own default in
    place". That preserves the pre-existing behaviour, where nothing set thread
    counts at all. Pinning them to a nonzero value changes throughput, so it is
    an explicit opt-in rather than something this refactor imposes.
    """

    model_config = ConfigDict(frozen=True)

    device: Device = "CPU"
    ov_num_threads: int = Field(default=0, ge=0)
    torch_num_threads: int = Field(default=0, ge=0)
    seed: int = Field(default=0, ge=0)

    def openvino_properties(self) -> dict[str, int]:
        """OpenVINO compile-time properties implied by this config.

        Returned as a dict rather than applied globally because OpenVINO thread
        settings are a property of a compiled model, not of the process.
        """
        if self.ov_num_threads == 0:
            return {}
        return {"INFERENCE_NUM_THREADS": self.ov_num_threads}


class PipelineConfig(BaseModel):
    """Clip geometry and tracking density.

    These mirror the constants previously hardcoded in ``endurance_run.py`` and
    ``semantic_extractor.py``. They are almost certainly not the right values —
    see the module docstring — but changing them is a measured change, not a
    refactor.
    """

    model_config = ConfigDict(frozen=True)

    grid_size: int = Field(default=10, gt=0)
    """Points tracked per axis; total tracked points is ``grid_size ** 2``."""

    clip_frames: int = Field(default=4, gt=0)
    clip_channels: int = Field(default=3, gt=0)
    clip_h: int = Field(default=224, gt=0)
    clip_w: int = Field(default=224, gt=0)

    tubelet: int = Field(default=2, gt=0)
    """V-JEPA2 temporal patch depth.

    The encoder consumes ``tubelet`` frames per temporal token, so a clip of
    ``clip_frames`` frames yields ``clip_frames // tubelet`` temporal token
    slots — not ``clip_frames``. The repository's own docstrings claim
    ``T * 196`` tokens for ``T = 4``, which contradicts this. See
    ``tests/test_known_bugs.py::test_vjepa_token_count_matches_tubelet``.
    """

    patch_size: int = Field(default=16, gt=0)

    @property
    def clip_shape(self) -> tuple[int, int, int, int, int]:
        """Synthetic/expected clip shape ``[B, T, C, H, W]`` with ``B == 1``."""
        return (1, self.clip_frames, self.clip_channels, self.clip_h, self.clip_w)

    @property
    def spatial_patches(self) -> int:
        """Number of spatial patches per frame, ``(H/patch) * (W/patch)``."""
        return (self.clip_h // self.patch_size) * (self.clip_w // self.patch_size)

    @property
    def temporal_slots(self) -> int:
        """Number of temporal token slots the encoder should emit for a clip."""
        return self.clip_frames // self.tubelet


class CascadeConfig(BaseModel):
    """Wake-hierarchy tuning.

    Gate resolution is the lever that decides whether the Tier-1 idle budget
    (under 3% of one core per camera) is reachable. Motion gating asks "did
    anything move", not "what is it", so it does not need the resolution the
    detector needs.
    """

    model_config = ConfigDict(frozen=True)

    gate_width: int = Field(default=320, ge=0)
    gate_height: int = Field(default=180, ge=0)
    """Resolution the motion gate runs at; 0 disables downscaling."""

    min_foreground_fraction: float = Field(default=0.002, ge=0.0, le=1.0)
    stay_awake_frames: int = Field(default=12, ge=0)
    diff_threshold: int = Field(default=25, ge=0, le=255)
    warmup_frames: int = Field(default=10, ge=0)

    stats_interval_s: float = Field(default=10.0, gt=0)

    idle_core_budget_fraction: float = Field(default=0.03, gt=0, le=1.0)
    """PRODUCT BUDGET: share of one core stage 0 may consume per camera.

    The Tier-1 claim is exactly as true as this being met on reference
    hardware. It is **currently missed** — see ``regression_ceiling_fraction``.
    This number describes what the product needs, and is never adjusted to
    match what the code does.
    """

    regression_ceiling_fraction: float = Field(default=0.055, gt=0, le=1.0)
    """CI REGRESSION CEILING — deliberately NOT the product budget.

    Set from the worst measured cost on the pinned stack (4.51% on the
    upscaled real-footage scenario, 2026-08-01) plus headroom for shared-runner
    variance. Its only job is to catch a change that makes stage 0 *worse*
    than it is today.

    Keeping this separate from ``idle_core_budget_fraction`` is the whole
    point. Collapsing them would let CI go green by redefining the target,
    which is how a missed budget quietly becomes a met one.
    """

    measured_stack: str = "opencv 4.8.1.78 / numpy 1.26.2 / python 3.10.20"
    """Stack the ceiling was measured on.

    Recorded because the measurement does not transfer: the same code scored
    2.97% on OpenCV 5.0.0 and 3.24% on the pinned 4.8.1.78, whose resize and
    MOG2 implementations differ. A cost number without its stack is not a
    number.
    """

    def motion_gate_config(self) -> Any:
        """Build a :class:`~src.cascade.motion.MotionGateConfig` from this.

        Imported lazily so that ``src.config`` stays free of any dependency on
        the cascade package, keeping ``import src.config`` cheap.
        """
        from src.cascade.motion import MotionGateConfig

        return MotionGateConfig(
            min_foreground_fraction=self.min_foreground_fraction,
            stay_awake_frames=self.stay_awake_frames,
            diff_threshold=self.diff_threshold,
            warmup_frames=self.warmup_frames,
            gate_width=self.gate_width,
            gate_height=self.gate_height,
        )


class EvalConfig(BaseModel):
    """Which golden set ``make eval`` measures against.

    Versioned rather than a path, so switching the instrument is a recorded
    config change with a config_sha behind it, not an argument someone passed
    once.
    """

    model_config = ConfigDict(frozen=True)

    golden_set_version: str = "v2-indoor"
    """Active set. v1 is driving-domain and legacy: it catches pipeline
    regressions but must never back a product metric."""

    golden_sets_dir: Path = Path("configs/golden")
    allow_legacy_golden_set: bool = False
    """Escape hatch for deliberately running the legacy driving set. The eval
    entrypoint still refuses to label those numbers as product metrics."""


class EnduranceConfig(BaseModel):
    """Soak-test parameters.

    A separate section from :class:`RuntimeConfig` because these describe a test
    harness, not the inference runtime — mixing them would mean an operator
    tuning thread counts has to read past leak thresholds to find them.
    """

    model_config = ConfigDict(frozen=True)

    num_clips: int = Field(default=200, gt=0)

    warmup_clips: int = Field(default=1, ge=0)
    """Leading clips excluded from the trend fit.

    The first inference allocates framework buffers. Including it fits a step
    change as though it were a slope, which reads as a leak.
    """

    leak_mb_per_hour_max: float = Field(default=50.0, gt=0)
    """Maximum tolerated RSS growth rate, in MB per hour.

    Replaces the previous pair of magic numbers (``total_growth > 150 MB`` AND
    ``trend > 30 MB``). A leak is a rate, so the gate is a rate: growth is
    measured as a regression slope and converted to MB/hour using the run's
    measured throughput. One threshold, with units, comparable across runs of
    different lengths.
    """

    reinit_cycles: int = Field(default=20, gt=0)
    """Construct/destroy cycles for ``--mode reinit``. CI nightly uses 100."""

    tracemalloc_top: int = Field(default=10, gt=0)
    tracemalloc_interval_clips: int = Field(default=25, gt=0)

    log_filename: str = "endurance_run.log"
    metrics_filename: str = "endurance_metrics.jsonl"


class IronConfig(BaseSettings):
    """Root configuration object. Construct with :meth:`load`.

    Example:
        >>> config = IronConfig.load()
        >>> config.pipeline.clip_frames
        4
    """

    model_config = SettingsConfigDict(
        env_prefix="IRON_",
        env_nested_delimiter="__",
        frozen=True,
        extra="forbid",
    )

    paths: PathsConfig = Field(default_factory=PathsConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    cascade: CascadeConfig = Field(default_factory=CascadeConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    endurance: EnduranceConfig = Field(default_factory=EnduranceConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Put environment variables ahead of constructor values.

        :meth:`load` feeds the YAML file in through the constructor, so env must
        outrank init for ``IRON_*`` to override the file. Earlier sources win.
        """
        return (env_settings, init_settings, dotenv_settings, file_secret_settings)

    @classmethod
    def load(cls, config_path: Path | str | None = None) -> "IronConfig":
        """Build a config from YAML plus ``IRON_*`` environment overrides.

        Args:
            config_path: YAML file to read. Defaults to ``configs/default.yaml``.
                A missing default file is not an error — the field defaults are
                a complete, valid configuration on their own. A missing file
                that was named *explicitly* is an error, because the caller
                asked for something that is not there.

        Raises:
            FileNotFoundError: if ``config_path`` was given but does not exist.
            ValueError: if the YAML file's top level is not a mapping.
        """
        explicit = config_path is not None
        path = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)

        values: dict[str, Any] = {}
        if path.exists():
            loaded = yaml.safe_load(path.read_text()) or {}
            if not isinstance(loaded, dict):
                raise ValueError(
                    f"{path}: expected a YAML mapping at the top level, "
                    f"got {type(loaded).__name__}"
                )
            values = loaded
        elif explicit:
            raise FileNotFoundError(f"Config file not found: {path}")

        return cls(**values)

    def config_sha(self) -> str:
        """Stable sha256 over the fully-resolved configuration.

        Deterministic across processes and machines: the payload is JSON with
        sorted keys, and ``paths.project_root`` is excluded because it is an
        absolute machine-specific path. Two machines running the same config
        must produce the same value, otherwise the manifest rule "two runs are
        comparable iff their manifests differ only in the dimension under test"
        could never hold across machines.

        The excluded root is not lost — the run manifest records hostname and
        absolute paths separately.
        """
        payload = self.model_dump(mode="json")
        payload.get("paths", {}).pop("project_root", None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AppliedRuntime(BaseModel):
    """Record of which runtime settings actually took effect.

    Returned by :func:`apply_runtime_settings` instead of having it silently do
    its best. A run that thinks it is seeded but is not produces irreproducible
    results that look reproducible, which is worse than an obvious failure — so
    the caller gets a structured answer it can log into the manifest.
    """

    model_config = ConfigDict(frozen=True)

    seed: int
    numpy_seeded: bool
    torch_seeded: bool
    torch_num_threads: int | None
    skipped: tuple[str, ...] = ()


def apply_runtime_settings(config: IronConfig) -> AppliedRuntime:
    """Apply seeds and thread caps to the current process.

    Call once at startup, before any model loads. NumPy is always seeded.
    Torch is seeded and thread-capped when importable; when it is not, the
    omission is reported in :attr:`AppliedRuntime.skipped` rather than being
    swallowed, so callers can log or fail on it as their context requires.

    Args:
        config: The loaded configuration.

    Returns:
        A record of what was applied and what was skipped.
    """
    import numpy as np

    runtime = config.runtime
    np.random.seed(runtime.seed)

    skipped: list[str] = []
    torch_seeded = False
    torch_threads: int | None = None

    try:
        import torch
    except ImportError as exc:
        skipped.append(f"torch seeding and thread cap: torch not importable ({exc})")
    else:
        torch.manual_seed(runtime.seed)
        torch_seeded = True
        if runtime.torch_num_threads > 0:
            torch.set_num_threads(runtime.torch_num_threads)
            torch_threads = runtime.torch_num_threads
        else:
            torch_threads = torch.get_num_threads()

    return AppliedRuntime(
        seed=runtime.seed,
        numpy_seeded=True,
        torch_seeded=torch_seeded,
        torch_num_threads=torch_threads,
        skipped=tuple(skipped),
    )
