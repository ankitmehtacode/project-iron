"""Environment gate: can this machine produce trustworthy numbers today?

Every forensic verdict, golden vector and before/after measurement in this
project depends on the runtime and the weights being what they claim to be. A
run on a partial environment does not produce a weaker answer, it produces a
confident wrong one — a missing library becomes a skipped test that reads as
green, and a missing checkpoint becomes a verdict nobody actually observed.

So the gate is a hard stop, checked once, up front, with the whole picture
printed rather than the first failure. Four rows:

1. **Runtime imports at pinned versions.** Presence is not enough. INT8 kernel
   selection and reduction order differ between OpenVINO releases, so a golden
   vector recorded under one version is not a reference for another.
2. **Model artifacts.** Existence, sha256 and size for every path in the
   config. The sha is what lets a later run prove it used the same weights.
3. **Dependency consistency.** ``pip check``: an unsatisfied constraint means
   some import is resolving to a version nothing verified.
4. **Determinism settings.** Seeds and thread counts actually applied from
   config, since INT8 determinism holds only at a fixed thread count.

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

# The four the brief names, plus the two that silently change numerics.
REQUIRED_MODULES = ("torch", "openvino", "cv2", "numpy")


@dataclass
class Row:
    """One gate check."""

    name: str
    passed: bool
    detail: str
    remedy: str = ""


@dataclass
class GateResult:
    rows: list[Row] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(row.passed for row in self.rows)

    @property
    def failures(self) -> list[Row]:
        return [row for row in self.rows if not row.passed]


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
    """Row 1: required modules importable, at the pinned version where pinned."""
    expected = {module_for(dist): version for dist, version in pins.items()}
    rows: list[Row] = []

    for module in REQUIRED_MODULES:
        if importlib.util.find_spec(module) is None:
            wanted = expected.get(module)
            rows.append(
                Row(
                    name=f"import {module}",
                    passed=False,
                    detail="not installed",
                    remedy=(
                        f"pip install {module}=={wanted}"
                        if wanted
                        else f"pip install {module}"
                    )
                    + "  (or: pip install -r locking-requirements.txt)",
                )
            )
            continue

        try:
            imported = __import__(module)
            found = str(getattr(imported, "__version__", "unknown"))
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

        wanted = expected.get(module)
        if wanted is None:
            rows.append(Row(f"import {module}", True, f"{found} (unpinned)"))
        elif found.split("+")[0] == wanted:
            rows.append(Row(f"import {module}", True, f"{found} (pinned)"))
        else:
            rows.append(
                Row(
                    name=f"import {module}",
                    passed=False,
                    detail=f"version {found}, pinned {wanted}",
                    remedy=(
                        f"pip install {module}=={wanted} — INT8 kernel selection "
                        "and reduction order differ between releases, so golden "
                        "vectors recorded under one version are not a reference "
                        "for another"
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
    targets: list[tuple[str, Path]] = [
        ("vjepa_xml", config.paths.resolved_vjepa_xml),
        ("vjepa_bin", config.paths.resolved_vjepa_xml.with_suffix(".bin")),
        ("cotracker_checkpoint", config.paths.resolved_cotracker_checkpoint),
    ]

    rows: list[Row] = []
    fingerprint: dict[str, Any] = {}
    for label, path in targets:
        if not path.exists():
            rows.append(
                Row(
                    name=f"model {label}",
                    passed=False,
                    detail=f"missing at {path}",
                    remedy=(
                        "python scripts/fetch_weights.py, then export and "
                        "quantize per scripts/ — see the checklist below"
                    ),
                )
            )
            fingerprint[label] = {"path": str(path), "present": False}
            continue

        size = path.stat().st_size
        digest = sha256_file(path)
        rows.append(
            Row(
                name=f"model {label}",
                passed=True,
                detail=f"{size / 1e6:.1f} MB  sha256 {digest[:16]}...",
            )
        )
        fingerprint[label] = {
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
