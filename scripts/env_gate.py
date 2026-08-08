"""Environment gate: can this machine produce trustworthy numbers today?

Every forensic verdict, golden vector and before/after measurement in this
project depends on the runtime and the weights being what they claim to be. A
run on a partial environment does not produce a weaker answer, it produces a
confident wrong one — a missing library becomes a skipped test that reads as
green, and a missing checkpoint becomes a verdict nobody actually observed.

So the gate is a hard stop, checked once, up front, with the whole picture
printed rather than the first failure. Six rows:

0. **Interpreter identity.** Which python is actually running this process,
   and whether it resolves inside ``.venv-pinned``. Day 16 found the default
   ``.venv`` had drifted to numpy 2.5.1 against a numpy==1.26.2 pin, and that
   mypy run through it silently reported 6 of 35 real errors — a wrong
   interpreter does not fail loudly, it produces a plausible wrong answer.
   Rows 1-5 below check whether the packages inside SOME environment are
   correct; this row checks whether the process asking the question is even
   in that environment, which every other row silently assumes.
0b. **No stray project venvs on disk.** Two environments on one machine is a
   coin flip that resolves silently — which one a bare ``python``/``pip``
   picks up depends on ``$PATH`` order, not on anything this project
   controls. One pinned environment is a structural guarantee; a second
   venv sitting on disk, however it got there, is the same latent defect
   that produced the mypy undercount, waiting for the next command run
   without ``.venv-pinned`` explicitly on the command line.
1. **Runtime imports at pinned versions.** Presence is not enough. INT8 kernel
   selection and reduction order differ between OpenVINO releases, so a golden
   vector recorded under one version is not a reference for another.
2. **Model artifacts.** Existence, sha256 and size for every path in the
   config. The sha is what lets a later run prove it used the same weights.
3. **Dependency consistency.** ``pip check``: an unsatisfied constraint means
   some import is resolving to a version nothing verified.
4. **Determinism settings.** Seeds and thread counts actually applied from
   config, since INT8 determinism holds only at a fixed thread count.

A python process outside any project venv at all (the system interpreter, or
a completely unrelated one such as a miniconda base environment) is not
caught by row 0b's disk scan — nothing on disk identifies it. Row 0 (the
interpreter-identity check) is what catches that case: it fails regardless
of what venvs exist, because it checks the *running* process, not the
filesystem. Row 0b closes the complementary gap — a second venv that exists
but was not the one invoked this time.

Exit codes:
    0  every row passed; measurements taken here are attributable
    1  at least one row failed; the checklist names exactly what is missing
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import IronConfig, apply_runtime_settings

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = REPO_ROOT / "locking-requirements.txt"

# Distribution name in requirements -> module name to import.
IMPORT_NAMES = {
    "opencv-python-headless": "cv2",
    "opencv-python": "cv2",
    "PyYAML": "yaml",
    "pyyaml": "yaml",
}

# (import name, candidate distribution names) for each required runtime.
#
# The two are not the same thing, and the difference is load-bearing. A module's
# ``__version__`` attribute is whatever its authors chose to put there:
# ``cv2.__version__`` is "4.8.1" where the distribution is 4.8.1.78, and
# ``openvino.__version__`` is
# "2024.6.0-17404-4c0f47d2335-releases/2024/6" where the distribution is
# 2024.6.0. Comparing pins against those strings reports a correctly pinned
# environment as wrong. Distribution metadata is what pip actually resolved and
# recorded, so that is what gets compared.
#
# cv2 lists two candidates because either OpenCV distribution provides it.
REQUIRED_MODULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("torch", ("torch",)),
    ("openvino", ("openvino",)),
    ("cv2", ("opencv-python-headless", "opencv-python")),
    ("numpy", ("numpy",)),
)


def installed_version(candidates: tuple[str, ...]) -> tuple[str | None, str | None]:
    """Return ``(distribution, version)`` from installed package metadata.

    Returns ``(None, None)`` when no candidate distribution is installed, which
    the caller reports rather than silently falling back to a module attribute.
    """
    from importlib.metadata import PackageNotFoundError, version

    for distribution in candidates:
        try:
            return distribution, version(distribution)
        except PackageNotFoundError:
            continue
    return None, None


@dataclass
class Row:
    """One gate check.

    ``blocks`` names what a failure actually stops. Not every row gates the
    whole day: the production-artifact rows gate the forensic verdict alone,
    because a fresh export can satisfy every other objective and must never
    satisfy that one.
    """

    name: str
    passed: bool
    detail: str
    remedy: str = ""
    blocks: str = "all"


@dataclass
class GateResult:
    rows: list[Row] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when every row passes, including the production-artifact rows."""
        return all(row.passed for row in self.rows)

    @property
    def can_proceed(self) -> bool:
        """True when everything except the production artifact is satisfied.

        This is the gate that matters for most of the work. Export, golden
        vectors and the normalization fix need a working runtime, not the
        production IR — that artifact gates only the forensic verdict on
        production, which nothing else can substitute for.
        """
        return all(row.passed for row in self.rows if row.blocks == "all")

    @property
    def failures(self) -> list[Row]:
        return [row for row in self.rows if not row.passed]

    @property
    def blocking_failures(self) -> list[Row]:
        return [row for row in self.rows if not row.passed and row.blocks == "all"]

    @property
    def parked_failures(self) -> list[Row]:
        return [row for row in self.rows if not row.passed and row.blocks != "all"]


PINNED_VENV_NAME = ".venv-pinned"


def check_interpreter_identity() -> Row:
    """Row 0: is the process asking every other question even in the right
    environment?

    Compared via ``sys.prefix``, not ``sys.executable``. A venv's
    ``bin/python`` is conventionally a *symlink* to the base interpreter it
    was created from — that is normal venv construction, not drift — so
    resolving the executable's symlink (``Path.resolve()``) walks straight
    past the venv boundary and back to the base install, defeating this
    check entirely (caught in Day 17's own dry run: it reported the pinned
    venv as "not pinned" while genuinely running inside it).
    ``sys.prefix`` is what venv activation actually sets to the venv's own
    directory regardless of how the executable itself is implemented, and
    ``sys.base_prefix`` names the base install a venv was created from —
    the two differing at all is itself the "am I in a venv" signal.
    """
    prefix = Path(sys.prefix).resolve()
    pinned_root = (REPO_ROOT / PINNED_VENV_NAME).resolve()
    inside_pinned = prefix == pinned_root

    if inside_pinned:
        return Row(
            name="interpreter identity",
            passed=True,
            detail=f"{sys.executable} (sys.prefix={prefix}, inside {PINNED_VENV_NAME})",
        )
    return Row(
        name="interpreter identity",
        passed=False,
        detail=f"{sys.executable} (sys.prefix={prefix}) is NOT {PINNED_VENV_NAME} "
        "— every check below is being asked of the wrong environment",
        remedy=f"invoke {PINNED_VENV_NAME}/bin/python explicitly, or "
        f"`source {PINNED_VENV_NAME}/bin/activate` first. A bare `python`/"
        "`python3` on $PATH is not this project's environment even when it "
        "happens to have the same packages installed by coincidence — see "
        "Day 17: this machine's default `python3` resolves to a miniconda "
        "base environment carrying its own, third, numpy version, entirely "
        "independent of this repository's pin.",
    )


def _is_venv_dir(path: Path) -> bool:
    return path.is_dir() and (path / "pyvenv.cfg").exists()


def check_no_stray_venvs() -> Row:
    """Row 0b: no OTHER venv sits on disk for a bare python/pip to find.

    Deliberately does not exempt distinctly-named ones (``.venv-infinigen``
    included): a second venv on disk is the risk this row exists to name,
    regardless of how well-motivated or clearly-labelled it is. Removing or
    formally re-justifying one that legitimately needs to stay (Infinigen's
    python 3.11 requirement, Day 11) is a decision for whoever is running
    the gate to make, not a silent exemption baked into the check.
    """
    found = sorted(
        p.name
        for p in REPO_ROOT.iterdir()
        if p.name != PINNED_VENV_NAME and _is_venv_dir(p)
    )
    if not found:
        return Row(
            name="no stray project venvs",
            passed=True,
            detail=f"only {PINNED_VENV_NAME} exists on disk",
        )
    return Row(
        name="no stray project venvs",
        passed=False,
        detail=f"{len(found)} other venv(s) on disk: {', '.join(found)}",
        remedy="delete each one, or if a workflow genuinely needs it "
        "(e.g. Infinigen's python 3.11 requirement), keep it deliberately "
        "and re-run with an explicit acknowledgement that this check is "
        "expected to fail until it is resolved — do not weaken this check "
        "to tolerate it silently",
    )


def pinned_versions() -> dict[str, str]:
    """Parse ``name==version`` pins out of locking-requirements.txt."""
    pins: dict[str, str] = {}
    if not REQUIREMENTS.exists():
        return pins
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        match = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", line)
        if match:
            pins[match.group(1)] = match.group(2)
    return pins


def module_for(distribution: str) -> str:
    return IMPORT_NAMES.get(distribution, distribution.replace("-", "_"))


def check_imports(pins: dict[str, str]) -> list[Row]:
    """Row 1: required modules import, at the pinned version where pinned.

    Two separate questions, both of which matter. Does the module import — a
    package that is installed but broken fails at inference time, which is
    worse than one that is absent. And does the *distribution* match the pin —
    compared against installed metadata rather than the module's own
    ``__version__`` string, for the reasons in :data:`REQUIRED_MODULES`.
    """
    rows: list[Row] = []

    for module, candidates in REQUIRED_MODULES:
        distribution, found = installed_version(candidates)
        wanted = next(
            (pins[name] for name in candidates if name in pins),
            None,
        )
        primary = candidates[0]

        if importlib.util.find_spec(module) is None:
            rows.append(
                Row(
                    name=f"import {module}",
                    passed=False,
                    detail="not installed",
                    remedy=(
                        f"pip install {primary}=={wanted}"
                        if wanted
                        else f"pip install {primary}"
                    )
                    + "  (or: pip install -r locking-requirements.txt)",
                )
            )
            continue

        try:
            __import__(module)
        except Exception as exc:  # noqa: BLE001 - report, never crash the gate
            rows.append(
                Row(
                    name=f"import {module}",
                    passed=False,
                    detail=f"installed but failed to import: {exc}",
                    remedy="reinstall the package; a broken install is worse "
                    "than an absent one because it fails at inference time",
                )
            )
            continue

        if found is None:
            rows.append(
                Row(
                    name=f"import {module}",
                    passed=False,
                    detail=(
                        f"imports, but no distribution metadata found for any of "
                        f"{', '.join(candidates)} — the version cannot be verified"
                    ),
                    remedy=f"pip install {primary}"
                    + (f"=={wanted}" if wanted else "")
                    + " so the installed version is recorded and checkable",
                )
            )
        elif wanted is None:
            rows.append(
                Row(f"import {module}", True, f"{distribution} {found} (unpinned)")
            )
        elif found == wanted:
            rows.append(
                Row(f"import {module}", True, f"{distribution} {found} (pinned)")
            )
        else:
            rows.append(
                Row(
                    name=f"import {module}",
                    passed=False,
                    detail=f"{distribution} {found}, pinned {wanted}",
                    remedy=(
                        f"pip install {distribution}=={wanted} — INT8 kernel "
                        "selection and reduction order differ between releases, "
                        "so golden vectors recorded under one version are not a "
                        "reference for another"
                    ),
                )
            )
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_models(config: IronConfig) -> tuple[list[Row], dict[str, Any]]:
    """Row 2: every configured model artifact exists; record sha256 and size."""
    # (label, path, what a failure blocks). The production IR gates only the
    # forensic verdict; everything else in the day runs without it.
    # (fingerprint key, display label, path, what a failure blocks).
    #
    # The key and the label are separate on purpose: the label carries the
    # PRODUCTION warning for humans reading the table, while the key stays
    # stable so the fingerprint written into a manifest does not change meaning
    # when the display text is reworded.
    targets: list[tuple[str, str, Path, str]] = [
        (
            "vjepa_xml",
            "PRODUCTION vjepa_xml",
            config.paths.resolved_vjepa_xml,
            "objective-1",
        ),
        (
            "vjepa_bin",
            "PRODUCTION vjepa_bin",
            config.paths.resolved_vjepa_xml.with_suffix(".bin"),
            "objective-1",
        ),
        (
            "cotracker_checkpoint",
            "cotracker_checkpoint",
            config.paths.resolved_cotracker_checkpoint,
            "all",
        ),
    ]

    rows: list[Row] = []
    fingerprint: dict[str, Any] = {}
    for key, label, path, blocks in targets:
        if not path.exists():
            remedy = (
                "copy the artifact production actually ran from the machine "
                "that ran it, with sha256sum taken before transfer. Do NOT "
                "export a replacement into this path: a fresh export is a "
                "different artifact, and putting it here destroys the only "
                "evidence of what the stored embeddings were computed with."
                if blocks == "objective-1"
                else "python scripts/fetch_weights.py, then export per scripts/"
            )
            rows.append(
                Row(
                    name=f"model {label}",
                    passed=False,
                    detail=f"missing at {path}",
                    remedy=remedy,
                    blocks=blocks,
                )
            )
            fingerprint[key] = {"path": str(path), "present": False}
            continue

        size = path.stat().st_size
        digest = sha256_file(path)
        rows.append(
            Row(
                name=f"model {label}",
                passed=True,
                detail=f"{size / 1e6:.1f} MB  sha256 {digest[:16]}...",
                blocks=blocks,
            )
        )
        fingerprint[key] = {
            "path": str(path),
            "present": True,
            "size_bytes": size,
            "sha256": digest,
        }
    return rows, fingerprint


def check_pip() -> Row:
    """Row 3: dependency constraints are satisfied."""
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Row("pip check", False, f"could not run: {exc}", "check the venv")

    output = completed.stdout.strip() or completed.stderr.strip()
    if completed.returncode == 0:
        return Row("pip check", True, output or "no broken requirements")
    return Row(
        name="pip check",
        passed=False,
        detail=output.replace("\n", "; ")[:300],
        remedy="pip install -r locking-requirements.txt — an unsatisfied "
        "constraint means some import resolves to a version nothing verified",
    )


def check_runtime(config: IronConfig) -> Row:
    """Row 4: seeds and thread counts actually applied."""
    applied = apply_runtime_settings(config)
    if applied.skipped:
        return Row(
            name="seeds / threads",
            passed=False,
            detail="; ".join(applied.skipped),
            remedy="install the missing runtime; INT8 determinism holds only "
            "at a fixed thread count, so an unseeded or unpinned run cannot be "
            "compared against a golden vector",
        )
    return Row(
        name="seeds / threads",
        passed=True,
        detail=(
            f"seed={applied.seed} numpy=yes torch=yes "
            f"torch_threads={applied.torch_num_threads}"
        ),
    )


def run_gate(config: IronConfig) -> tuple[GateResult, dict[str, Any]]:
    result = GateResult()
    result.rows.append(check_interpreter_identity())
    result.rows.append(check_no_stray_venvs())
    result.rows.extend(check_imports(pinned_versions()))
    model_rows, fingerprint = check_models(config)
    result.rows.extend(model_rows)
    result.rows.append(check_pip())
    result.rows.append(check_runtime(config))
    return result, fingerprint


def render(result: GateResult) -> str:
    width = max(len(row.name) for row in result.rows) + 2
    lines = [f"{'CHECK'.ljust(width)} {'STATUS':<6}  DETAIL", "-" * (width + 60)]
    for row in result.rows:
        status = "PASS" if row.passed else "FAIL"
        lines.append(f"{row.name.ljust(width)} {status:<6}  {row.detail}")
    return "\n".join(lines)


def render_checklist(result: GateResult) -> str:
    """The exact missing-item list, in the order someone would act on it."""
    lines = [
        "MISSING-ITEM CHECKLIST",
        "",
        f"{len(result.failures)} of {len(result.rows)} checks failed. Every one "
        "must pass before any",
        "forensic verdict, golden vector, or before/after measurement is "
        "trustworthy.",
        "",
    ]
    for index, row in enumerate(result.failures, start=1):
        lines.append(f"{index}. {row.name}: {row.detail}")
        if row.remedy:
            lines.append(f"   -> {row.remedy}")
        lines.append("")
    return "\n".join(lines).rstrip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="emit the fingerprint as JSON"
    )
    args = parser.parse_args(argv)

    config = IronConfig.load()
    result, fingerprint = run_gate(config)

    if args.json:
        print(
            json.dumps(
                {
                    "passed": result.passed,
                    "rows": [
                        {"name": r.name, "passed": r.passed, "detail": r.detail}
                        for r in result.rows
                    ],
                    "models": fingerprint,
                    "config_sha": config.config_sha(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if result.passed else 1

    print("=" * 78)
    print("ENVIRONMENT GATE")
    print("=" * 78)
    print(f"config_sha : {config.config_sha()}")
    print(f"python     : {sys.version.split()[0]}")
    print()
    print(render(result))
    print()

    if result.passed:
        print("GATE PASSED. Measurements taken here are attributable.")
        return 0

    if result.can_proceed:
        print("GATE: PROCEED (production artifact parked)")
        print()
        print(
            "Every row that gates general work passes. The only failures are "
            "the\nproduction-artifact rows, which gate the forensic verdict "
            "alone:"
        )
        for row in result.parked_failures:
            print(f"  - {row.name}: {row.detail}")
        print()
        print(
            "Export, golden vectors and the normalization fix can proceed "
            "against a\nfreshly exported IR. The forensic verdict on "
            "production cannot, and no\nexport substitutes for it."
        )
        return 0

    print(render_checklist(result))
    print()
    print(
        "GATE FAILED. Stopping. Producing verdicts on a partial environment "
        "yields\nconfident wrong answers, not weaker ones: a missing library "
        "becomes a skipped\ntest that reads as green, and a missing checkpoint "
        "becomes a verdict nobody\nactually observed."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
