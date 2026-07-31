"""Red-first tests encoding defects found in an external audit.

These tests are written against the CURRENT code and are expected to fail.
A failure here is the deliverable, not a problem to paper over: each one
converts a suspicion into a reproducible, named defect with evidence attached.
Do not "fix" the pipeline to make them green without a measured change and
before/after numbers — that is what iron-eval-discipline forbids.

Each test carries ``@pytest.mark.known_bug`` and a comment naming the audit
finding. When a fix lands, the marker is removed in the same change so the test
survives as a permanent regression guard.

Tests marked ``requires_weights`` skip with an explicit reason when checkpoints
or runtime libraries are absent. They never fail on missing weights and they
never fake a pass — a skip is reported as "blocked", not as "clean".
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.config import IronConfig

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Skip helpers — blocked is not the same as passing.
# ---------------------------------------------------------------------------


def _require_openvino() -> Any:
    """Return the openvino module, or skip with a reason naming what is absent."""
    if importlib.util.find_spec("openvino") is None:
        pytest.skip(
            "openvino is not installed in this environment; install "
            "locking-requirements.txt to exercise the V-JEPA2 IR path"
        )
    import openvino as ov

    return ov


def _require_vjepa_ir(config: IronConfig) -> Path:
    path = config.paths.resolved_vjepa_xml
    if not path.exists():
        pytest.skip(
            f"V-JEPA2 OpenVINO IR not found at {path}. Fetch and export it with "
            "scripts/fetch_weights.py and scripts/quantize_vjepa.py, then rerun."
        )
    return path


def _require_semantic_extractor(config: IronConfig) -> Any:
    """Import SemanticExtractor, skipping if its heavy dependencies are absent."""
    for module, hint in (
        ("openvino", "install locking-requirements.txt"),
        ("torch", "install locking-requirements.txt"),
        (
            "cotracker",
            "install CoTracker3 from https://github.com/facebookresearch/co-tracker",
        ),
    ):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is not installed; {hint}")

    _require_vjepa_ir(config)
    checkpoint = config.paths.resolved_cotracker_checkpoint
    if not checkpoint.exists():
        pytest.skip(f"CoTracker3 checkpoint not found at {checkpoint}")

    from src.semantics.semantic_extractor import SemanticExtractor

    return SemanticExtractor


# ---------------------------------------------------------------------------
# Finding 1 — temporal token count contradicts the tubelet.
# ---------------------------------------------------------------------------


@pytest.mark.known_bug
@pytest.mark.requires_weights
@pytest.mark.xfail(
    strict=False, reason="AUDIT FINDING 1: unverified — needs the V-JEPA2 IR to resolve"
)
def test_vjepa_token_count_matches_tubelet() -> None:
    """AUDIT FINDING 1: V-JEPA2 token count vs. the repo's documented T*196.

    semantic_extractor.py documents its V-JEPA output as ``[B, T*196, 1024]``
    and draws a data-flow diagram with one temporal slot per input frame.
    V-JEPA2 is a tubelet encoder with tubelet 2, so a 4-frame clip yields 2
    temporal slots, not 4. Both cannot be true.

    The assertion message prints the observed shape alongside what each
    hypothesis predicts, so whoever reads the failure gets the evidence rather
    than a bare mismatch.
    """
    config = IronConfig.load()
    ov = _require_openvino()
    xml_path = _require_vjepa_ir(config)

    pipeline = config.pipeline
    core = ov.Core()
    compiled = core.compile_model(str(xml_path), config.runtime.device)

    clip = np.random.rand(*pipeline.clip_shape).astype(np.float32)
    request = compiled.create_infer_request()
    request.infer({"video": clip})
    features = request.get_output_tensor(0).data

    spatial = pipeline.spatial_patches
    tubelet_hypothesis = (pipeline.clip_frames // pipeline.tubelet) * spatial
    per_frame_hypothesis = pipeline.clip_frames * spatial
    observed_tokens = int(features.shape[1])

    message = (
        f"\n  observed output shape       : {tuple(features.shape)}"
        f"\n  observed token count        : {observed_tokens}"
        f"\n  hypothesis A (tubelet={pipeline.tubelet}) : {tubelet_hypothesis}"
        f"  = ({pipeline.clip_frames} // {pipeline.tubelet}) * {spatial}"
        f"\n  hypothesis B (tubelet=1)    : {per_frame_hypothesis}"
        f"  = {pipeline.clip_frames} * {spatial}   <- what the repo docstrings claim"
        f"\n  If B holds, the export is nonstandard and every downstream"
        f"\n  temporal index is built on an assumption the encoder does not honour."
    )
    assert observed_tokens == tubelet_hypothesis, message


# ---------------------------------------------------------------------------
# Finding 2 — per-frame semantics are attributed to the wrong frames.
# ---------------------------------------------------------------------------


@pytest.mark.known_bug
@pytest.mark.requires_weights
@pytest.mark.xfail(
    strict=False, reason="AUDIT FINDING 2: unverified — needs weights to resolve"
)
def test_patch_mapper_temporal_alignment() -> None:
    """AUDIT FINDING 2: temporal misalignment in _map_tracks_to_embeddings.

    The mapper computes ``t_out = min(t, T_out - 1)``. With 4 input frames and
    2 temporal slots that sends frames 1, 2 and 3 all to slot 1, so frame 1's
    semantics come from a slot covering frames 2-3.

    The probe clip is black for frames 0-1 and white for frames 2-3, which is
    exactly the tubelet boundary. Correct behaviour groups {0,1} together and
    {2,3} together, and makes the two groups differ. The buggy mapping splits
    frames 0 and 1 across the boundary, so this fails on the first assertion.
    """
    config = IronConfig.load()
    SemanticExtractor = _require_semantic_extractor(config)

    pipeline = config.pipeline
    if pipeline.clip_frames < 4:
        pytest.skip(f"probe needs at least 4 frames, config has {pipeline.clip_frames}")

    half = pipeline.clip_frames // 2
    video = np.zeros(pipeline.clip_shape, dtype=np.float32)
    video[:, half:] = 1.0  # first half black, second half white

    extractor = SemanticExtractor(
        vjepa_xml=str(config.paths.resolved_vjepa_xml),
        cotracker_checkpoint=str(config.paths.resolved_cotracker_checkpoint),
        grid_size=pipeline.grid_size,
        device=config.runtime.device,
    )
    semantic = extractor.extract(video)["semantic_tracks"]

    def frame_vector(index: int) -> np.ndarray:
        return semantic[0, index].mean(axis=0)

    dark = [frame_vector(i) for i in range(half)]
    light = [frame_vector(i) for i in range(half, pipeline.clip_frames)]

    within_dark = float(np.abs(dark[0] - dark[-1]).max())
    within_light = float(np.abs(light[0] - light[-1]).max())
    across = float(np.abs(dark[0] - light[0]).max())

    diagnostics = (
        f"\n  max|frame0 - frame{half - 1}| (both black) : {within_dark:.6g}"
        f"\n  max|frame{half} - frame{pipeline.clip_frames - 1}| (both white) : "
        f"{within_light:.6g}"
        f"\n  max|black - white|                        : {across:.6g}"
        f"\n  Frames inside one tubelet must share a temporal slot, so the two"
        f"\n  within-group figures should be ~0 while the across figure is large."
    )

    assert within_dark == pytest.approx(0.0, abs=1e-5), (
        "frames inside the same tubelet got different semantics — the mapper "
        "split a tubelet across temporal slots" + diagnostics
    )
    assert within_light == pytest.approx(0.0, abs=1e-5), (
        "frames inside the same tubelet got different semantics" + diagnostics
    )
    assert across > 1e-4, (
        "black and white halves produced identical semantics — the mapper is "
        "reading one slot for every frame" + diagnostics
    )


# ---------------------------------------------------------------------------
# Finding 3 — no input normalization before the encoder.
# ---------------------------------------------------------------------------

# Evidence of a mean/std STANDARDISATION step.
#
# Deliberately narrow. A first draft of this test matched the word
# "normalis/ze" and passed, because semantic_extractor.py's docstring says the
# input "Must be normalised to [0, 1]". Scaling into [0, 1] is a range
# conversion; it is not subtracting a channel mean and dividing by a channel
# standard deviation, which is what V-JEPA2 was trained with. A detector that
# conflates the two reports the bug as absent, which is worse than no test.
_STANDARDISATION_PATTERNS = (
    r"0\.485|0\.456|0\.406",  # ImageNet channel means
    r"0\.229|0\.224|0\.225",  # ImageNet channel standard deviations
    r"IMAGENET",
    r"Normalize\s*\(",
    r"\w*processor\b",  # a hub/HF preprocessor does the standardisation for you
    r"\bmean\b.*\bstd\b",  # both on one line: the subtract-and-divide itself
    r"\bmean\s*[-+*/]?=",  # assigning the constants
    r"\bstd\s*[-+*/]?=",
)

# The module that actually runs inference in production. endurance_run.py and
# integration.py both reach the encoder through this path.
_PRODUCTION_ENCODER_PATH = REPO_ROOT / "src" / "semantics" / "semantic_extractor.py"

_ALL_ENCODER_PATHS = (
    _PRODUCTION_ENCODER_PATH,
    REPO_ROOT / "src" / "models" / "vjepa_wrapper.py",
    REPO_ROOT / "scripts" / "vjepa_wrapper.py",
)


def _standardisation_evidence(path: Path) -> list[str]:
    """Return source lines in ``path`` that look like a mean/std standardisation."""
    if not path.exists():
        return []
    hits: list[str] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if line.lstrip().startswith("#"):
            continue
        if any(re.search(pattern, line) for pattern in _STANDARDISATION_PATTERNS):
            hits.append(f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()}")
    return hits


def test_standardisation_detector_ignores_range_scaling() -> None:
    """Guard the detector in the test above against its own false positive.

    "Normalised to [0, 1]" must not count as standardisation. Without this
    guard, a docstring edit could silently flip the known-bug test to green and
    report a live defect as fixed.
    """
    assert not any(
        re.search(pattern, "Must be normalised to [0, 1].")
        for pattern in _STANDARDISATION_PATTERNS
    )
    assert any(
        re.search(pattern, "video = (video - mean) / std")
        for pattern in _STANDARDISATION_PATTERNS
    )


@pytest.mark.known_bug
@pytest.mark.xfail(
    strict=False,
    reason="AUDIT FINDING 3: CONFIRMED — production encoder path applies no mean/std",
)
def test_preprocess_normalization() -> None:
    """AUDIT FINDING 3: the encoder is fed [0,1] pixels with no mean/std applied.

    ``SemanticExtractor._run_vjepa`` passes the raw array straight into the
    OpenVINO IR, and ``extract``'s docstring requires only "normalised to
    [0, 1]". V-JEPA2 was trained on ImageNet-standardised input. Feeding it
    un-standardised pixels shifts every activation, silently degrading every
    embedding. Nothing raises, the shapes are right, and the vectors look
    perfectly reasonable.

    Needs no weights: this is a source inspection, plus a numerical check on
    any importable preprocess function.
    """
    per_file = {path: _standardisation_evidence(path) for path in _ALL_ENCODER_PATHS}

    report = "\n".join(
        f"  {path.relative_to(REPO_ROOT)}: "
        + (f"{len(hits)} candidate line(s)" if hits else "NO mean/std standardisation")
        + "".join(f"\n      {hit}" for hit in hits)
        for path, hits in per_file.items()
    )

    # If a preprocess callable is reachable, prove it actually changes values.
    # A no-op preprocessor passes a source grep and fails reality.
    numerical_note = "  no importable preprocess function found to exercise"
    try:
        from src.semantics import semantic_extractor as extractor_module
    except ImportError as exc:
        numerical_note = f"  semantic_extractor not importable here: {exc}"
    else:
        preprocess = getattr(extractor_module, "preprocess", None)
        if callable(preprocess):
            ones = np.ones((1, 4, 3, 224, 224), dtype=np.float32)
            transformed = np.asarray(preprocess(ones))
            numerical_note = (
                f"  preprocess(ones) changed the input: "
                f"{not np.array_equal(transformed, ones)}"
            )
            assert not np.array_equal(transformed, ones), (
                "preprocess() returned its input unchanged, so it applies no "
                "normalization at all\n" + report
            )

    assert per_file[_PRODUCTION_ENCODER_PATH], (
        "No mean/std normalization found on the production V-JEPA2 inference "
        f"path ({_PRODUCTION_ENCODER_PATH.relative_to(REPO_ROOT)}). V-JEPA2 "
        "expects ImageNet-standardised input; this path forwards [0,1] pixels "
        "straight to the encoder.\n"
        f"{report}\n{numerical_note}"
    )


# ---------------------------------------------------------------------------
# Finding 4 — relative disparity is documented as metres.
# ---------------------------------------------------------------------------


@pytest.mark.known_bug
@pytest.mark.xfail(
    strict=False,
    reason="AUDIT FINDING 4: CONFIRMED — DA-V2 relative disparity published as metres",
)
def test_depth_output_declares_units() -> None:
    """AUDIT FINDING 4: DA-V2 relative depth is published as "meters".

    Depth-Anything-V2 emits relative inverse depth on an arbitrary per-frame
    scale. ``DAv2Wrapper.predict`` returns ``{"depth": ...}`` with no units.
    That array flows through ``projector_vectorized.project_points_to_3d``,
    which treats it as Z in camera space, into the Parquet ``z`` column — which
    the schema documents as metres.

    Every 3D coordinate downstream therefore has arbitrary scale while claiming
    to be metric. The failure names the file and line making the claim.
    """
    metric_claims: list[str] = []
    searched = (
        REPO_ROOT / "README.md",
        REPO_ROOT / "architecture.md",
        REPO_ROOT / "src" / "utils" / "parquet_writer.py",
        REPO_ROOT / "src" / "models" / "dav2_wrapper.py",
        REPO_ROOT / "src" / "geometry" / "projector_vectorized.py",
    )
    for path in searched:
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if re.search(r"\bmet(er|re)s\b", line, flags=re.IGNORECASE):
                metric_claims.append(
                    f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()}"
                )

    # Does the depth wrapper declare units at all?
    dav2_source = (REPO_ROOT / "src" / "models" / "dav2_wrapper.py").read_text()
    declares_units = "DepthField" in dav2_source or "units" in dav2_source

    detail = "\n".join(f"      {claim}" for claim in metric_claims) or "      (none)"

    assert declares_units, (
        "DAv2Wrapper.predict returns a bare ndarray with no units declaration, "
        "while the following code paths assert its values are metric:\n"
        f"{detail}\n"
        "    Depth-Anything-V2 outputs relative inverse depth with unknown "
        "scale AND unknown shift, so no constant converts it to metres. Wrap "
        "the output in src.contracts.DepthField(units='disparity_rel') and let "
        "unproject() refuse it until an anchoring step exists."
    )


# ---------------------------------------------------------------------------
# Finding 5 — "consecutive errors" counter is actually cumulative.
# ---------------------------------------------------------------------------


def _legacy_error_policy(outcomes: list[bool]) -> int:
    """Replica of the abort logic in endurance_run.py as of the audit.

    Copied verbatim in behaviour from ``endurance_run.py`` (the ``errors > 5``
    branch, whose log line reads "Aborting: too many consecutive errors"). The
    counter is never reset on success, so it counts cumulative errors across
    the whole run.

    Kept after the fix as documentation of what the defect actually was. The
    live assertion below now runs against the real harness in
    ``src/endurance``, not against this replica.

    Args:
        outcomes: One entry per clip. True means the clip raised.

    Returns:
        The clip index (1-based) at which the run aborted, or 0 if it completed.
    """
    errors = 0
    for index, failed in enumerate(outcomes, start=1):
        if failed:
            errors += 1
            if errors > 5:
                return index
            continue
        # NOTE: no `errors = 0` here. That omission is the defect.
    return 0


def test_legacy_error_counter_was_cumulative_not_consecutive() -> None:
    """AUDIT FINDING 5, confirmed: the abort counter never reset on success.

    ``endurance_run.py`` logged "Aborting: too many consecutive errors" while
    counting cumulatively. Six scattered transient failures, each immediately
    recovered from, aborted the run at clip 16.

    This records the defect against the replica above. The regression guard for
    the fix is the test below.
    """
    scattered: list[bool] = []
    for _ in range(6):
        scattered.extend([True, False, False])

    assert _legacy_error_policy(scattered) == 16, (
        "the replica no longer reproduces the audited behaviour, so it is no "
        "longer evidence of what the defect was"
    )
    assert _legacy_error_policy([True] * 6) == 6


def test_endurance_error_counter_semantics(tmp_path: Path) -> None:
    """AUDIT FINDING 5, fixed: the tolerance counter is gone entirely.

    The ``known_bug`` marker is deliberately absent. Objective 5 fixed this by
    deleting the tolerance rather than by making the counter consecutive, which
    is stronger than the originally intended semantics: per iron-testing, any
    exception fails the run. That skill also requires the marker to be removed
    in the same change as the fix, so the test guards the behaviour forever.

    Asserts against the real harness, not the replica.
    """
    import os

    from src.endurance.runner import ExitCode, execute

    class _FailsOnce:
        """Raises on its second call, and would succeed on every call after."""

        def __init__(self) -> None:
            self.calls = 0

        def extract(self, video: np.ndarray) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("single transient failure")
            return {"semantic_tracks": np.zeros((1, 2, 4, 8), dtype=np.float64)}

    os.environ["IRON_PATHS__LOG_DIR"] = str(tmp_path / "logs")
    os.environ["IRON_PIPELINE__CLIP_H"] = "16"
    os.environ["IRON_PIPELINE__CLIP_W"] = "16"
    try:
        extractor = _FailsOnce()
        code = execute(
            IronConfig.load(),
            mode="steady",
            iterations=50,
            factory=lambda: extractor,
        )
    finally:
        for key in (
            "IRON_PATHS__LOG_DIR",
            "IRON_PIPELINE__CLIP_H",
            "IRON_PIPELINE__CLIP_W",
        ):
            os.environ.pop(key, None)

    assert code == ExitCode.PIPELINE_RAISED, (
        "a single exception must fail the run; a surviving tolerance counter "
        "means the harness reports memory stability for a pipeline that is not "
        "producing output"
    )
    assert extractor.calls == 2, (
        f"the run continued past the failing clip ({extractor.calls} calls); "
        "no error tolerance may remain"
    )
