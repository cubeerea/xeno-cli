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
from xeno.core.types import GateProfile

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


#: What Talos reports when there is no toolchain to gate on. Named like a
#: command so it flows through `EvalReport.failed_command`, the CB-4 failure
#: signature, and Chiron's prompt exactly like any other gate failure.
_NO_TOOLCHAIN_COMMAND = "toolchain-discovery"

_NO_TOOLCHAIN_LOG = """\
No toolchain is established for this repository: it declares no manifest or
build file, so there is no lint, type-check, or test command to run, and
nothing here can be verified.

This is a code defect, not an infrastructure fault: the fix is to add the
manifest that declares this project's dependencies and its lint/type-check/
test configuration. Once a manifest exists, the toolchain is re-discovered
automatically on the next evaluation.
"""


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
    #: Set when the failure is "there was nothing to run", as opposed to
    #: something having run and failed. Distinct from `passed=False` alone so
    #: a caller can tell a broken build from an absent one.
    toolchain_unestablished: bool = False


def run_gates(
    sandbox: _Execable,
    toolchain: DiscoveredToolchain,
    *,
    profile: GateProfile = GateProfile.FULL,
) -> GateOutcome:
    """Run this profile's required commands in order (fail-fast), then
    `toolchain.advisory` best-effort if every one of them passed.

    `profile` selects which required commands run, and only ever narrows the
    list — see `DiscoveredToolchain.required_for`, which also guarantees the
    narrowed list is never empty. Advisory commands are unaffected: they never
    decide anything, so holding them back would only cost log detail.

    An unestablished toolchain (no required commands at all) is a FAILURE,
    never a pass. Falling through the loops below would otherwise return
    `passed=True` having executed nothing — a silent green that would carry a
    greenfield run all the way to Cerberus with every task "verified" by
    zero commands. It is reported as an ordinary code defect rather than an
    infrastructure fault on purpose: a missing manifest is something Chiron
    can actually fix, so the ladder should engage instead of escalating.
    """
    if not toolchain.established:
        return GateOutcome(
            passed=False,
            failed_command=_NO_TOOLCHAIN_COMMAND,
            exit_code=1,
            infrastructure_failure=False,
            log=_NO_TOOLCHAIN_LOG,
            toolchain_unestablished=True,
        )

    log_chunks: list[str] = []

    def run(command: DiscoveredCommand, label: str) -> int:
        exit_code, output = sandbox.exec(command.argv, timeout=COMMAND_TIMEOUT_SECONDS)
        log_chunks.append(f"---- {label} ----\n{output}")
        return exit_code

    for command in toolchain.required_for(profile):
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
