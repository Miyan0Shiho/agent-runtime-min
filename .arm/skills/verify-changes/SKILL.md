---
name: verify-changes
description: Verify a change actually works before reporting it done, by running the project's tests and reading the output rather than assuming.
whenToUse: Before claiming any code change is complete.
---

# Verify changes before reporting them

Do not report a change as done because the edit looked right. Run it.

## Steps

1. Identify the narrowest command that exercises the change.
   Prefer a single test file over the whole suite; a fast check gets run.

2. Run it, and **read the output**. A zero exit code with a traceback in the
   captured output is still a failure.

3. If the change touched a shared interface, run the tests that depend on it
   too. A passing unit test next to a broken caller is not a passing change.

4. Report what you ran and what it printed. "Tests pass" is not evidence;
   the command and its output are.

## What not to do

- Do not run the full suite when one file would answer the question. It is
  slower, and it buries the relevant failure in noise.
- Do not re-run the same command unchanged after a failure. Read the error and
  change something.
