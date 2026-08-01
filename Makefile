# Convenience targets. Everything here is a thin wrapper over a script that
# works standalone — the Makefile is for muscle memory, never the only way in.
#
# PYTHON defaults to the pinned virtualenv built per the environment gate.
# Override for a different interpreter: make eval PYTHON=python3

PYTHON ?= .venv-pinned/bin/python

.PHONY: help gate eval eval-legacy test lint types bench

help:
	@echo "gate         environment gate: runtime, weights, seeds"
	@echo "eval         report the active golden set and its gaps"
	@echo "eval-legacy  same, against the driving-domain legacy set"
	@echo "test         pytest, excluding weights-dependent tests"
	@echo "lint         black --check and flake8"
	@echo "types        mypy --strict, scoped by mypy.ini"
	@echo "bench        cascade wake-parity and budget benchmark"

gate:
	$(PYTHON) scripts/env_gate.py

# Golden-set version comes from configs/default.yaml (eval.golden_set_version),
# so switching the instrument is a recorded config change with a config_sha
# behind it rather than an argument someone passed once.
eval:
	$(PYTHON) scripts/eval_report.py

eval-legacy:
	$(PYTHON) scripts/eval_report.py --version v1-driving

test:
	$(PYTHON) -m pytest -m "not requires_weights" -q

lint:
	$(PYTHON) -m black --check .
	$(PYTHON) -m flake8

types:
	$(PYTHON) -m mypy

bench:
	$(PYTHON) scripts/cascade_bench.py

inspect:
	$(PYTHON) -m src.inspector.server
