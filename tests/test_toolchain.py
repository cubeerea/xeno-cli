"""Mid-run toolchain re-derivation (`xeno.graph.toolchain`).

The staleness check is a local hash, so these tests need neither Docker nor a
model: the session is constructed without `start()` (leaving `_pool` unset,
which short-circuits the image/pool work) and `discover_toolchain` is
monkeypatched at the module boundary — the same "fake only the expensive
boundary" approach `test_gates.py` takes with `Sandbox.exec`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xeno.adapters.discovery import DiscoveryError, manifest_fingerprint
from xeno.adapters.generic import DiscoveredCommand, DiscoveredToolchain
from xeno.core.runlog import NullRunLog
from xeno.core.state import AgentState
from xeno.graph.toolchain import ToolchainSession

_PYPROJECT = """\
[project]
name = "demo"
version = "0.1.0"
"""


def _established(fingerprint: str, *, name: str = "test") -> DiscoveredToolchain:
    return DiscoveredToolchain(
        install=None,
        required=(DiscoveredCommand(name=name, argv=("pytest", "-q")),),
        advisory=(),
        fingerprint=fingerprint,
    )


def _session(worktree: Path, toolchain: DiscoveredToolchain) -> ToolchainSession:
    return ToolchainSession(
        router=None,  # type: ignore[arg-type]
        config=None,  # type: ignore[arg-type]
        keyring=None,  # type: ignore[arg-type]
        paths=None,  # type: ignore[arg-type]
        repo_root=worktree,
        worktree=worktree,
        runlog=NullRunLog(),
        toolchain=toolchain,
    )


def _patch_discovery(monkeypatch: pytest.MonkeyPatch, result: object) -> list[int]:
    """Returns a call counter; `result` is either a toolchain or an
    exception instance to raise."""
    calls: list[int] = []

    def fake(**kwargs: object) -> DiscoveredToolchain:
        calls.append(1)
        if isinstance(result, Exception):
            raise result
        assert isinstance(result, DiscoveredToolchain)
        return result

    monkeypatch.setattr("xeno.graph.toolchain.discover_toolchain", fake)
    return calls


def test_no_refresh_when_manifests_are_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The common case must not cost a model call — the check is a hash."""
    (tmp_path / "pyproject.toml").write_text(_PYPROJECT)
    session = _session(tmp_path, _established(manifest_fingerprint(tmp_path)))
    calls = _patch_discovery(monkeypatch, _established("other"))

    assert session.refresh_if_stale(AgentState(run_id="t", goal="g")) is False
    assert calls == []


def test_a_scaffold_that_adds_a_manifest_establishes_the_toolchain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The greenfield path end to end: a run starts with nothing to gate on,
    the first task writes a manifest, and the next evaluation gates on it."""
    session = _session(tmp_path, DiscoveredToolchain.unestablished(manifest_fingerprint(tmp_path)))
    assert session.toolchain.established is False

    (tmp_path / "pyproject.toml").write_text(_PYPROJECT)
    discovered = _established(manifest_fingerprint(tmp_path), name="lint")
    calls = _patch_discovery(monkeypatch, discovered)

    assert session.refresh_if_stale(AgentState(run_id="t", goal="g")) is True
    assert calls == [1]
    assert session.toolchain.established is True
    assert session.toolchain.required[0].name == "lint"
    # The adapter must follow the toolchain, or gates keep running the old argv.
    assert session.adapter.toolchain is session.toolchain


def test_a_new_dependency_retriggers_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not just greenfield: editing an existing manifest mid-run is the same
    bug, and previously left the gates running against a repo that no longer
    existed."""
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(_PYPROJECT)
    session = _session(tmp_path, _established(manifest_fingerprint(tmp_path)))

    manifest.write_text(_PYPROJECT + '\ndependencies = ["httpx"]\n')
    updated = _established(manifest_fingerprint(tmp_path), name="test-with-httpx")
    _patch_discovery(monkeypatch, updated)

    assert session.refresh_if_stale(AgentState(run_id="t", goal="g")) is True
    assert session.toolchain.required[0].name == "test-with-httpx"


def test_failed_rediscovery_keeps_the_previous_toolchain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that was gating on something a moment ago is better served by
    continuing to gate on it than by collapsing mid-run."""
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(_PYPROJECT)
    original = _established(manifest_fingerprint(tmp_path), name="original")
    session = _session(tmp_path, original)

    manifest.write_text("this is not valid toml {{{")
    _patch_discovery(monkeypatch, DiscoveryError("model said something unusable"))

    assert session.refresh_if_stale(AgentState(run_id="t", goal="g")) is False
    assert session.toolchain is original


def test_a_cosmetic_manifest_edit_rediscovers_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A comment added to a manifest moves the fingerprint but not the
    commands. The new fingerprint must still be adopted, or this check
    re-discovers on every evaluation for the rest of the run."""
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(_PYPROJECT)
    session = _session(tmp_path, _established(manifest_fingerprint(tmp_path)))

    manifest.write_text(_PYPROJECT + "\n# a comment\n")
    calls = _patch_discovery(monkeypatch, _established(manifest_fingerprint(tmp_path)))
    state = AgentState(run_id="t", goal="g")

    assert session.refresh_if_stale(state) is True
    assert session.refresh_if_stale(state) is False
    assert session.refresh_if_stale(state) is False
    assert calls == [1], "the fingerprint must be adopted so this settles"


def test_a_scaffold_that_fails_to_add_a_manifest_stays_unestablished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bad-scaffold case. `run_gates` turns this into a failure the
    ladder acts on, rather than a vacuous pass."""
    session = _session(tmp_path, DiscoveredToolchain.unestablished(manifest_fingerprint(tmp_path)))
    (tmp_path / "notes.md").write_text("I wrote a readme instead of a manifest\n")
    calls = _patch_discovery(monkeypatch, DiscoveredToolchain.unestablished("x"))

    # A non-manifest file does not move the manifest fingerprint at all, so
    # there is nothing to re-discover and no call to spend.
    assert session.refresh_if_stale(AgentState(run_id="t", goal="g")) is False
    assert calls == []
    assert session.toolchain.established is False
