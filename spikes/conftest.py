"""Make each spike directory importable regardless of where pytest is invoked."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

for sub in ("q1_cancellation", "q2_schema", "q3_waterfall"):
    p = str(HERE / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
