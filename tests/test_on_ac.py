"""scripts/on_ac.py: refuse on battery, stop cleanly when AC drops (Day 45)."""

from __future__ import annotations

import sys
from pathlib import Path

from scripts.on_ac import POWER_STOP, guard

QUIET = lambda _m: None  # noqa: E731


def test_refuses_to_start_on_battery(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    cmd = [sys.executable, "-c", f"open({str(marker)!r}, 'w')"]
    assert guard(cmd, power=lambda: False, log=QUIET) == POWER_STOP
    assert not marker.exists()


def test_passes_the_child_exit_code_through_on_ac() -> None:
    cmd = [sys.executable, "-c", "raise SystemExit(7)"]
    assert guard(cmd, power=lambda: True, poll_s=0.05, log=QUIET) == 7


def test_ac_loss_interrupts_the_child_which_cleans_up(tmp_path: Path) -> None:
    marker = tmp_path / "cleaned"
    child = (
        "import time\n"
        "try:\n"
        "    time.sleep(30)\n"
        "except KeyboardInterrupt:\n"
        f"    open({str(marker)!r}, 'w').write('ok')\n"
    )
    readings = iter([True, True, False])
    rc = guard(
        [sys.executable, "-c", child],
        power=lambda: next(readings, False),
        poll_s=0.2,
        grace_s=10,
        log=QUIET,
    )
    assert rc == POWER_STOP
    assert marker.read_text() == "ok"  # SIGINT, not a kill: the child finished
