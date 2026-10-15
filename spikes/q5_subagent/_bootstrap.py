"""Make ``arm`` importable without touching the repo's packaging."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

__all__ = ["REPO_ROOT", "SRC"]
