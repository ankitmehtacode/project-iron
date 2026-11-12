"""Run a command only on AC power; stop it cleanly if AC drops (Day 45).

    python scripts/on_ac.py -- python -m iron run --video ... --out ...

Refuses to start on battery. While the child runs, polls the power source
every ``--poll`` seconds; on the first poll that reads battery it sends the
child SIGINT (Python raises KeyboardInterrupt; the detection cache is
fsynced per frame, so the next run resumes from the first frame not on
disk), waits up to ``--grace`` seconds, then SIGTERM. Exit codes: the
child's own on a normal finish, 3 if refused or stopped for power.

The power source is read from ``pmset -g batt`` (macOS). A reading that
cannot be parsed counts as "not AC": the guard fails closed.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
from collections.abc import Callable, Sequence

POWER_STOP = 3


def on_ac() -> bool:
    try:
        out = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "'AC Power'" in out


def guard(
    cmd: Sequence[str],
    power: Callable[[], bool] = on_ac,
    poll_s: float = 5.0,
    grace_s: float = 60.0,
    log: Callable[[str], None] = lambda m: print(m, file=sys.stderr, flush=True),
) -> int:
    if not power():
        log("on_ac: not on AC power; refusing to start")
        return POWER_STOP
    child = subprocess.Popen(list(cmd))
    while True:
        try:
            return child.wait(timeout=poll_s)
        except subprocess.TimeoutExpired:
            pass
        if not power():
            log(f"on_ac: AC lost; sending SIGINT to pid {child.pid}")
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                log("on_ac: child did not stop on SIGINT; SIGTERM")
                child.terminate()
                child.wait()
            log("on_ac: stopped for power; rerun on AC to resume from the cache")
            return POWER_STOP


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--poll", type=float, default=5.0)
    p.add_argument("--grace", type=float, default=60.0)
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        p.error("no command after --")
    return guard(cmd, poll_s=args.poll, grace_s=args.grace)


if __name__ == "__main__":
    raise SystemExit(main())
