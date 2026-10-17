# arm — agent-runtime-min
#
# One command runs the tests and the benchmark: `make bench`.
#
# The acceptance criteria are numbers (replay fidelity, token reduction, fault
# recovery, regression detection), so `make bench` exits non-zero when one is
# missed. A benchmark that always succeeds measures nothing.

PY      := .venv/bin/python
PYTEST  := $(PY) -m pytest
SRC     := src

.PHONY: help venv install test bench lint typecheck clean fmt check all

help:
	@echo "arm targets:"
	@echo "  venv       create the virtualenv"
	@echo "  install    install dependencies"
	@echo "  test       run the test suite"
	@echo "  bench      run tests + the four acceptance metrics (one command)"
	@echo "  typecheck  run pyright"
	@echo "  check      test + typecheck"
	@echo "  clean      remove caches and build artefacts"

venv:
	uv venv --python 3.11 .venv

install: venv
	uv pip install --python $(PY) pydantic httpx pyyaml pytest pytest-asyncio

test:
	PYTHONPATH=$(SRC) $(PYTEST) tests -q

# `bench` runs the tests first, then the metrics. A failing test suite makes the
# metrics meaningless, so it is a prerequisite rather than a separate step.
bench: test
	@PYTHONPATH=$(SRC):. $(PY) -m arm.bench.main

# Code that nothing calls. This project hit the same defect five times -- a
# normalizer, a pre-empted validation, a scheduler, a delivery confirmation and an
# approval service -- and every one was invisible to a passing test suite. The
# check compares against a recorded baseline, so a *new* uncalled definition fails.
deadcode:
	@PYTHONPATH=$(SRC):. $(PY) -c "from pathlib import Path; from arm.bench.deadcode import run, render, load_baseline; f, ok = run(Path('src')); print(render(f, load_baseline())); raise SystemExit(0 if ok else 1)"

# The claims the design documents left unverified, now measured. Separate from
# `bench` because it answers a different question: `bench` asks whether the
# acceptance criteria hold, this asks whether the design's own assumptions were
# right.
claims:
	@PYTHONPATH=$(SRC):. $(PY) -c "from arm.bench.claims import run_measurements, render_report; m, ok = run_measurements(); print(render_report(m, ok)); raise SystemExit(0 if ok else 1)"

bench-json:
	@PYTHONPATH=$(SRC):. $(PY) -m arm.bench.main --json

# Does the assembled runtime work? The unit suite tests the parts by building
# them itself; this builds them the way the CLI does and asks whether one call
# through each subsystem succeeds. Every defect found by *running* this project
# was in the wiring, which is the class no unit test can see.
#
# Offline by default so it can run anywhere. Set ARM_SMOKE_MODEL_KEY to add a
# real provider turn and a real MCP call, which is the only way to catch defects
# in the contract with an external system rather than in our own code.
smoke:
	@PYTHONPATH=$(SRC):. $(PY) -m arm.bench.smoke

typecheck:
	pyright

check: test typecheck smoke

clean:
	rm -rf .pytest_cache .pyright .coverage htmlcov
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete

all: check bench
