"""Talos's deterministic quality gates (PRD S10, S8.2 revised): run a
discovered command list, decide pass/fail purely on exit code.

"THE GATES ARE DETERMINISTIC TOOLS, not model calls" (PRD S8.2) — nothing in
this module calls an LLM. The commands themselves come from a
`DiscoveredToolchain` (`xeno.adapters.discovery`, a model call made once per
repo and cached), but by the time this module runs, that decision is already
frozen into a fixed argv list — the same "fixed argument vector, never a
shell string" contract every gate has always had. Every command executes
inside a `Sandbox` acquired from the warm pool (PRD S11, S13 Phase 2) — never
on the host.

`toolchain.required` runs in declared order, fail-fast: there is nothing for
a later command to say once an earlier one has failed, and not every
ecosystem separates parse/lint/type/test into distinct commands the way the
old Python-only gate chain assumed. `toolchain.advisory` runs only once every
required command has passed, and never flips the verdict — the same
"advisory, non-blocking" contract PRD S10 gate 5 (coverage) always had,
generalized to any advisory command instead of one hardcoded coverage step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from xeno.adapters.generic import DiscoveredCommand, DiscoveredToolchain

#: Generous but bounded — a command that never returns is worse than one
#: that fails fast and lets CB-1 do its job. One budget for every discovered
#: command: which of them is "the test one" is no longer something this
#: module knows (PRD S12 revised), so the ceiling is set by the slowest thing
#: a repo might plausibly gate on.
COMMAND_TIMEOUT_SECONDS = 300

#: `Sandbox.exec`'s out-of-band signal that the command never returned, as
#: opposed to returning a real non-zero status (`xeno.sandbox.pool`). Kept
#: distinct from a code defect all the way into `EvalReport.
#: infrastructure_failure`, because the escalation ladder must not send
#: Chiron chasing a bug that isn't there (PRD S10).
_INFRASTRUCTURE_EXIT = -1


class _Execable(Protocol):
    """The one method this module actually needs from `xeno.sandbox.pool.
    Sandbox` — a `Protocol` rather than the concrete class so a test fake can
    satisfy it structurally instead of standing up a real Docker container."""

    def exec(self, argv: Sequence[str], *, timeout: float) -> tuple[int, str]: ...


@dataclass(frozen=True, slots=True)
class GateOutcome:
    passed: bool
    #: Name of the first required command that failed, or "" when `passed`
    #: is True or the failure was infrastructure (nothing code-shaped to
    #: name).
    failed_command: str
    exit_code: int | None
    infrastructure_failure: bool
    log: str = field(default="", repr=False)


def run_gates(sandbox: _Execable, toolchain: DiscoveredToolchain) -> GateOutcome:
    """Run `toolchain.required` in order (fail-fast), then `toolchain.
    advisory` best-effort if every required command passed."""
    log_chunks: list[str] = []

    def run(command: DiscoveredCommand, label: str) -> int:
        exit_code, output = sandbox.exec(command.argv, timeout=COMMAND_TIMEOUT_SECONDS)
        log_chunks.append(f"---- {label} ----\n{output}")
        return exit_code

    for command in toolchain.required:
        exit_code = run(command, command.name)
        if exit_code != 0:
            infrastructure = exit_code == _INFRASTRUCTURE_EXIT
            return GateOutcome(
                passed=False,
                failed_command=command.name,
                exit_code=None if infrastructure else exit_code,
                infrastructure_failure=infrastructure,
                log="\n".join(log_chunks),
            )

    for command in toolchain.advisory:
        # Best-effort and genuinely non-blocking (PRD S10 gate 5): a timeout
        # or a crash here must never turn a green required chain red, so the
        # exit code is logged and discarded rather than inspected.
        run(command, f"{command.name} (advisory)")

    return GateOutcome(
        passed=True,
        failed_command="",
        exit_code=0,
        infrastructure_failure=False,
        log="\n".join(log_chunks),
    )
