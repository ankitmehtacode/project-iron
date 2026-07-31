"""Command-line entrypoint for the endurance / memory-stability harness.

All logic lives in :mod:`src.endurance`. This file is only argument parsing and
an exit code, so that the parts that decide whether a run passed are importable
and testable without model weights.

Usage:
    pip install -e .
    python endurance_run.py                     # steady-state, config default
    python endurance_run.py --clips 3           # short smoke run
    python endurance_run.py --mode reinit       # load/unload leak hunt
    python endurance_run.py --config other.yaml

Exit codes:
    0  gates passed (or were inconclusive on a run too short to judge)
    1  a gate failed: memory growth over the limit, or non-determinism
    2  prerequisites missing: model weights or runtime libraries absent
    3  the pipeline raised; see the traceback in the log
"""

from __future__ import annotations

import argparse

from src.config import IronConfig
from src.endurance.runner import execute


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("steady", "reinit"),
        default="steady",
        help=(
            "steady: one extractor, many clips — finds per-inference retention. "
            "reinit: construct and destroy the extractor each cycle — finds "
            "load/unload leaks, which is where framework leaks actually live."
        ),
    )
    parser.add_argument(
        "--clips",
        type=int,
        default=None,
        help="iterations to run; defaults to the configured value for the mode",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="path to a YAML config; defaults to configs/default.yaml",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = IronConfig.load(args.config)

    if args.clips is not None:
        iterations = args.clips
    elif args.mode == "reinit":
        iterations = config.endurance.reinit_cycles
    else:
        iterations = config.endurance.num_clips

    return int(execute(config, mode=args.mode, iterations=iterations))


if __name__ == "__main__":
    raise SystemExit(main())
