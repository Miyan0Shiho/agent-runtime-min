"""Dead-code census for the three target modules.

For each public symbol, count references under ``src/`` excluding the defining
file, and separately under ``tests/`` and ``spikes/``. A symbol with zero ``src/``
references outside its own module is dead: the feature is configurable but not
usable.

Run:  cd spikes && ../.venv/bin/python q5_subagent/probe_deadcode.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
TESTS = REPO / "tests"


def grep_count(pattern: str, root: Path, *, include_pyc: bool = False) -> list[str]:
    cmd = ["grep", "-rn", "--include=*.py", pattern, str(root)]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return lines


def report(symbol: str, defining: str, pattern: str | None = None) -> None:
    pat = pattern or rf"\b{re.escape(symbol)}\b"
    src_hits = grep_count(pat, SRC)
    test_hits = grep_count(pat, TESTS)
    outside = [
        ln
        for ln in src_hits
        if not ln.startswith(f"{SRC}/arm/{defining}")
    ]
    verdict = "LIVE" if outside else "DEAD"
    print(f"[{verdict:4}] {symbol:34} src-outside={len(outside):2}  tests={len(test_hits):3}")
    for ln in outside[:6]:
        print("          ", ln.replace(str(REPO) + "/", ""))
    if not outside:
        for ln in src_hits[:4]:
            print("           (only in defining module)", ln.replace(str(REPO) + "/", ""))


TARGETS = [
    # (symbol, defining path relative to src/arm)
    ("capabilities_for", "subagent/contracts.py"),
    ("ProviderCapabilities", "subagent/contracts.py"),
    ("STOP_REASONS", "subagent/contracts.py"),
    ("to_stop_reason", "subagent/contracts.py"),
    ("DelegationResult", "subagent/contracts.py"),
    ("DelegationRequest", "subagent/contracts.py"),
    ("assert_max_depth", "subagent/contracts.py"),
    ("assert_capabilities", "subagent/contracts.py"),
    ("NoProviderError", "subagent/contracts.py"),
    ("UnsupportedCapabilityError", "subagent/contracts.py"),
    ("DepthLimitError", "subagent/contracts.py"),
    ("validate_output_schema", "subagent/runtime.py"),
    ("SubagentRuntime", "subagent/runtime.py"),
    ("final_assistant_output", "subagent/runtime.py"),
    ("turn_stop_reason", "subagent/runtime.py"),
    ("consumed_work", "subagent/runtime.py"),
    ("RunningSubagent", "subagent/runtime.py"),
    ("SubagentTool", "subagent/tool.py"),
    ("InProcessProvider", "subagent/inprocess.py"),
    ("ChildFactory", "subagent/inprocess.py"),
    ("IN_PROCESS_CAPABILITIES", "subagent/inprocess.py"),
    ("result_key", "subagent/inprocess.py"),
    ("TeamJournal", "team/journal.py"),
    ("TeamMember", "team/journal.py"),
    ("TeamMessage", "team/journal.py"),
    ("TeamTask", "team/journal.py"),
    ("assert_task_graph", "team/journal.py"),
    ("validate_member_name", "team/journal.py"),
    ("TeamTools", "team/tools.py"),
    ("ScheduleStore", "schedule/runtime.py"),
    ("ScheduleRuntime", "schedule/runtime.py"),
    ("ScheduleRecord", "schedule/runtime.py"),
    ("DeliveryReceipt", "schedule/runtime.py"),
    ("Occurrence", "schedule/runtime.py"),
    ("next_occurrence", "schedule/runtime.py"),
    ("ScheduleTools", "schedule/tools.py"),
    ("retry_failed", "schedule/runtime.py"),
    ("request_drive", "schedule/runtime.py"),
    ("confirm_delivery", "team/journal.py"),
    ("pending_for", "team/journal.py"),
    ("summary", "team/journal.py"),
    ("active_members", "team/journal.py"),
    ("live_tasks", "team/journal.py"),
    ("for_session", "schedule/runtime.py"),
    ("history", "schedule/runtime.py"),
    ("_timer", "schedule/runtime.py"),
    ("last_delivery", "schedule/runtime.py"),
]


def main() -> int:
    print(f"repo: {REPO}")
    print(f"src hash snapshot taken at review start; modules are being edited live.\n")
    for symbol, defining in TARGETS:
        report(symbol, defining)
    return 0


if __name__ == "__main__":
    sys.exit(main())
