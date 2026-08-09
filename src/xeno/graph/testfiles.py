"""Shared test-file detection.

Used two ways (PRD S10): Talos flags `EvalReport.touched_test_files` when a
write landed in what looks like a test file, and Chiron's hard rule ("MUST
NOT modify test files to make tests pass") is enforced against the same
definition — one drifting out of sync with the other would let a patch slip
past the rule it's supposed to be checked against.
"""

from __future__ import annotations

TEST_FILE_MARKERS = ("test_", "_test.py", "/tests/", "conftest.py")


def is_test_file(rel_path: str) -> bool:
    return any(marker in rel_path for marker in TEST_FILE_MARKERS)
