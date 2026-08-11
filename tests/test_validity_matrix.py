"""scripts/validity_matrix.py -- the capability list is DERIVED, not copied.

Day 16 and Day 23 both found the same defect: a capability gets registered
in ``src.data.validity.GATES`` and does not appear in the one script whose
job is to report every registered capability, because that script kept its
own hand-maintained tuple next to the registry. Fixed twice by editing the
tuple; fixed a third time (Day 24, Objective 3) by deleting the tuple
entirely -- ``registered_capabilities()`` reads ``validity.GATES`` live, on
every call, so there is nothing to fall out of sync.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import validity_matrix  # noqa: E402

from src.data import validity  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
_SCANNED_DIRS = ("src", "scripts", "tests", "configs")
_ALLOWED_FILES = {"src/data/validity.py"}


def test_registered_capabilities_matches_gates_registry() -> None:
    assert validity_matrix.registered_capabilities() == tuple(validity.GATES)


def test_registering_a_new_gate_appears_without_editing_this_script(
    monkeypatch,
) -> None:
    """The structural guarantee: register a dummy gate directly in
    ``validity.GATES`` and confirm it appears -- with zero edits to
    ``validity_matrix.py`` itself, which is the point of deriving the list
    instead of copying it."""
    assert "dummy_capability_day24" not in validity_matrix.registered_capabilities()

    def _dummy_gate(**kwargs: object) -> tuple[bool, str, dict[str, object]]:
        return True, "dummy gate for the structural test", {}

    patched_gates = dict(validity.GATES)
    patched_gates["dummy_capability_day24"] = _dummy_gate
    monkeypatch.setattr(validity, "GATES", patched_gates)

    capabilities = validity_matrix.registered_capabilities()
    assert "dummy_capability_day24" in capabilities
    # And every capability that was already registered is still present --
    # registering one gate must not displace another.
    for name in ("motion_geometry", "state_estimation", "depth"):
        assert name in capabilities


def test_no_hand_maintained_capability_list_elsewhere() -> None:
    """Grep-verify: the only place all of ``validity.GATES``'s capability
    names appear together as a literal collection is the registry itself.
    A second copy -- even a well-intentioned one -- is exactly how this
    defect recurred twice (Day 16, Day 23) before today's structural fix."""
    capability_names = tuple(validity.GATES)
    offenders: list[str] = []
    for dirname in _SCANNED_DIRS:
        for path in (REPO_ROOT / dirname).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            rel = str(path.relative_to(REPO_ROOT))
            if rel in _ALLOWED_FILES:
                continue
            for lineno, line in enumerate(
                path.read_text(errors="ignore").splitlines(), start=1
            ):
                hits = sum(
                    1
                    for name in capability_names
                    if f'"{name}"' in line or f"'{name}'" in line
                )
                if hits >= 4:
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, (
        "found what looks like a hand-maintained copy of the capability "
        f"list outside validity.GATES: {offenders}"
    )
