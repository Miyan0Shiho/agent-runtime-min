"""Make ``arm`` importable without touching the repo's packaging.

``spikes/conftest.py`` puts each spike directory on ``sys.path`` but knows
nothing about ``src/``; the repo is not installed into the venv. Every module
here that imports ``arm`` imports this first, so the spike runs identically
under ``pytest`` and as a plain script::

    cd spikes && ../.venv/bin/python q2_schema/differential.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

__all__ = ["REPO_ROOT", "SRC"]
