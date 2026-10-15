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

bench-json:
	@PYTHONPATH=$(SRC):. $(PY) -m arm.bench.main --json

typecheck:
	pyright

check: test typecheck

clean:
	rm -rf .pytest_cache .pyright .coverage htmlcov
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete

all: check bench
