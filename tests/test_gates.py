"""Talos's gate chain (PRD S10, S8.2 revised): fail-fast over required
commands, best-effort advisory commands, exit-code-only verdicts. Faked at
the `Sandbox.exec` boundary (`_Execable`, a `Protocol` in `xeno.graph.gates`)
rather than a real Docker container — the same "fake only the network/process
boundary" philosophy `ScriptedProvider` uses for the Router in `test_graph.py`.
"""

from __future__ import annotations

from collections.abc import Sequence

from xeno.adapters.generic import DiscoveredCommand, DiscoveredToolchain
from xeno.graph.gates import run_gates


class _FakeSandbox:
    def __init__(self, responses: dict[tuple[str, ...], tuple[int, str]]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, ...]] = []

    def exec(self, argv: Sequence[str], *, timeout: float) -> tuple[int, str]:
        del timeout
        key = tuple(argv)
        self.calls.append(key)
        return self._responses[key]


def _toolchain(
    required: tuple[DiscoveredCommand, ...], advisory: tuple[DiscoveredCommand, ...] = ()
) -> DiscoveredToolchain:
    return DiscoveredToolchain(install=None, required=required, advisory=advisory, fingerprint="x")


def test_all_required_commands_passing_yields_passed() -> None:
    lint = DiscoveredCommand(name="lint", argv=("ruff", "check", "."))
    test = DiscoveredCommand(name="test", argv=("pytest", "-q"))
    sandbox = _FakeSandbox({lint.argv: (0, "clean"), test.argv: (0, "5 passed")})

    outcome = run_gates(sandbox, _toolchain((lint, test)))

    assert outcome.passed
    assert outcome.failed_command == ""
    assert outcome.exit_code == 0
    assert not outcome.infrastructure_failure
    assert sandbox.calls == [lint.argv, test.argv]


def test_fails_fast_on_first_required_command_and_skips_the_rest() -> None:
    lint = DiscoveredCommand(name="lint", argv=("ruff", "check", "."))
    test = DiscoveredCommand(name="test", argv=("pytest", "-q"))
    sandbox = _FakeSandbox({lint.argv: (1, "F401 unused import"), test.argv: (0, "5 passed")})

    outcome = run_gates(sandbox, _toolchain((lint, test)))

    assert not outcome.passed
    assert outcome.failed_command == "lint"
    assert outcome.exit_code == 1
    assert not outcome.infrastructure_failure
    assert sandbox.calls == [lint.argv]  # test never ran


def test_infrastructure_failure_has_no_exit_code() -> None:
    test = DiscoveredCommand(name="test", argv=("pytest", "-q"))
    sandbox = _FakeSandbox({test.argv: (-1, "INFRASTRUCTURE: pytest timed out after 300s")})

    outcome = run_gates(sandbox, _toolchain((test,)))

    assert not outcome.passed
    assert outcome.failed_command == "test"
    assert outcome.exit_code is None
    assert outcome.infrastructure_failure


def test_advisory_runs_only_when_every_required_command_passed() -> None:
    test = DiscoveredCommand(name="test", argv=("pytest", "-q"))
    coverage = DiscoveredCommand(name="coverage", argv=("pytest", "--cov=."))
    sandbox = _FakeSandbox({test.argv: (0, "5 passed"), coverage.argv: (0, "TOTAL 80%")})

    outcome = run_gates(sandbox, _toolchain((test,), (coverage,)))

    assert outcome.passed
    assert sandbox.calls == [test.argv, coverage.argv]
    assert "coverage" in outcome.log


def test_advisory_command_never_runs_after_a_required_failure() -> None:
    test = DiscoveredCommand(name="test", argv=("pytest", "-q"))
    coverage = DiscoveredCommand(name="coverage", argv=("pytest", "--cov=."))
    sandbox = _FakeSandbox({test.argv: (1, "1 failed"), coverage.argv: (0, "TOTAL 80%")})

    outcome = run_gates(sandbox, _toolchain((test,), (coverage,)))

    assert not outcome.passed
    assert sandbox.calls == [test.argv]  # coverage never ran


def test_advisory_command_failure_never_flips_a_passing_verdict() -> None:
    test = DiscoveredCommand(name="test", argv=("pytest", "-q"))
    coverage = DiscoveredCommand(name="coverage", argv=("pytest", "--cov=."))
    sandbox = _FakeSandbox(
        {test.argv: (0, "5 passed"), coverage.argv: (1, "coverage tool crashed")}
    )

    outcome = run_gates(sandbox, _toolchain((test,), (coverage,)))

    assert outcome.passed
    assert outcome.failed_command == ""
