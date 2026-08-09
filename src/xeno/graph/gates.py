"""Talos's deterministic quality gates (PRD S10): parse, lint, type, test,
coverage.

"THE GATES ARE DETERMINISTIC TOOLS, not model calls" (PRD S8.2) — nothing in
this module calls an LLM. Every gate runs via a fixed argv list built by a
`LanguageAdapter` (PRD S12), executed inside a `Sandbox` acquired from the
warm pool (PRD S11, S13 Phase 2) — never on the host. Phase 1's host
subprocess version of this module is gone; the Phase 2 exit criterion is
"zero host execution of generated code."

Lint and type checks are scoped to the files Daedalus/Chiron touched (the
diff is what changed); the test suite runs over the whole worktree, because a
single-file change can break tests elsewhere and "tests_must_still_pass" is
part of the suite's own acceptance criteria (XENO-SUITE-30). Coverage only
runs on an otherwise-green result — it is advisory (PRD S10 gate 5) and
doubling test runtime to measure coverage of a suite that is still red buys
nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from xeno.adapters.base import LanguageAdapter, TestSummary
from xeno.sandbox.pool import Sandbox

#: Generous but bounded — an evaluation that never returns is worse than one
#: that fails fast and lets CB-1 do its job.
GATE_TIMEOUT_SECONDS = 120
TEST_TIMEOUT_SECONDS = 300


@dataclass(frozen=True, slots=True)
class GateOutcome:
    parse_ok: bool
    lint_errors: int
    type_errors: int
    tests_run: int
    tests_failed: int
    failing_test_ids: tuple[str, ...]
    exception_type: str
    failing_location: str
    infrastructure_failure: bool
    #: PRD S10 gate 5. None when the gate did not run (a failing gate chain,
    #: or an adapter with no coverage tool).
    coverage_percent: float | None = None
    log: str = field(default="", repr=False)
    #: Identity of the lint/type findings (rule codes + locations), not just
    #: their count — CB-4's `failure_signature` (PRD S7.3) needs this because
    #: `exception_type`/`failing_location` above are test-shaped and stay
    #: empty for a lint- or type-only failure. Empty string when the
    #: respective gate found nothing to report.
    lint_signature: str = ""
    type_signature: str = ""


def run_gates(
    sandbox: Sandbox,
    adapter: LanguageAdapter,
    worktree: Path,
    touched_files: Sequence[Path],
) -> GateOutcome:
    """Run the gate chain in order, fail-fast on parse (PRD S10 gate order).

    A parse failure short-circuits everything after it: there is nothing for
    a linter or type checker to say about a file that does not parse, and
    running pytest against a broken worktree would just report collection
    errors that duplicate the same information less precisely.
    """
    py_files = [p.relative_to(worktree).as_posix() for p in touched_files if p.suffix == ".py"]

    parse_ok, parse_log, parse_infra = _parse_gate(sandbox, adapter, py_files)
    if not parse_ok:
        return GateOutcome(
            parse_ok=False,
            lint_errors=0,
            type_errors=0,
            tests_run=0,
            tests_failed=0,
            failing_test_ids=(),
            exception_type="" if parse_infra else "SyntaxError",
            failing_location=parse_log,
            infrastructure_failure=parse_infra,
            log=parse_log,
        )

    lint_errors, lint_signature, lint_log, lint_infra = _lint_gate(sandbox, adapter, py_files)
    type_errors, type_signature, type_log, type_infra = _type_gate(sandbox, adapter, py_files)
    summary, test_log, test_infra = _test_gate(sandbox, adapter)

    infra = lint_infra or type_infra or test_infra
    log = "\n---- lint ----\n" + lint_log + "\n---- types ----\n" + type_log
    log += "\n---- tests ----\n" + test_log

    exc_type = summary.exception_type
    location = summary.failing_location
    if not exc_type and (lint_errors or type_errors):
        exc_type = "LintOrTypeError"
        location = f"{lint_errors} lint, {type_errors} type error(s)"

    coverage_percent: float | None = None
    green = not infra and lint_errors == 0 and type_errors == 0 and summary.tests_failed == 0
    if green:
        coverage_percent, cov_log = _coverage_gate(sandbox, adapter)
        log += "\n---- coverage ----\n" + cov_log

    return GateOutcome(
        parse_ok=True,
        lint_errors=lint_errors,
        type_errors=type_errors,
        tests_run=summary.tests_run,
        tests_failed=summary.tests_failed,
        failing_test_ids=summary.failing_test_ids,
        exception_type=exc_type,
        failing_location=location,
        infrastructure_failure=infra,
        coverage_percent=coverage_percent,
        log=log,
        lint_signature=lint_signature,
        type_signature=type_signature,
    )


def _parse_gate(
    sandbox: Sandbox, adapter: LanguageAdapter, py_files: list[str]
) -> tuple[bool, str, bool]:
    cmd = adapter.parse_cmd(py_files)
    if cmd is None:
        return True, "", False
    code, output = sandbox.exec(cmd, timeout=GATE_TIMEOUT_SECONDS)
    if code == -1:
        return False, output, True
    return code == 0, output, False


def _lint_gate(
    sandbox: Sandbox, adapter: LanguageAdapter, py_files: list[str]
) -> tuple[int, str, str, bool]:
    if not py_files:
        return 0, "", "", False
    code, output = sandbox.exec(adapter.lint_cmd(py_files), timeout=GATE_TIMEOUT_SECONDS)
    if code == -1:
        return 0, "", output, True
    try:
        return adapter.parse_lint(output), adapter.lint_signature(output), output, False
    except ValueError:
        # The adapter got output it could not make sense of — an adapter
        # fault, not a lint finding attributable to the code under test.
        return 0, "", output, True


def _type_gate(
    sandbox: Sandbox, adapter: LanguageAdapter, py_files: list[str]
) -> tuple[int, str, str, bool]:
    if not py_files:
        return 0, "", "", False
    code, output = sandbox.exec(adapter.typecheck_cmd(py_files), timeout=GATE_TIMEOUT_SECONDS)
    if code == -1:
        return 0, "", output, True
    return adapter.parse_typecheck(output), adapter.type_signature(output), output, False


def _test_gate(sandbox: Sandbox, adapter: LanguageAdapter) -> tuple[TestSummary, str, bool]:
    code, output = sandbox.exec(adapter.test_cmd(), timeout=TEST_TIMEOUT_SECONDS)
    if code == -1:
        return TestSummary(0, 0, (), "", ""), output, True
    return adapter.parse_test_failures(output), output, False


def _coverage_gate(sandbox: Sandbox, adapter: LanguageAdapter) -> tuple[float | None, str]:
    cmd = adapter.coverage_cmd()
    if cmd is None:
        return None, ""
    # Best-effort and genuinely non-blocking (PRD S10 gate 5): a timeout or a
    # coverage-tool crash here must never turn a green gate chain red.
    code, output = sandbox.exec(cmd, timeout=TEST_TIMEOUT_SECONDS)
    if code == -1:
        return None, output
    return adapter.parse_coverage(output), output
