"""Toolchain discovery (PRD S8.2, S12 revised): fingerprinting, cache I/O,
and the executable allowlist. No subprocess, no model call — those are
covered live; this module is about the pure, deterministic logic being
correct on its own (same philosophy the old PythonAdapter tests used)."""

from __future__ import annotations

from pathlib import Path

import pytest

from xeno.adapters.discovery import (
    DiscoveryError,
    _validate,
    load_cached_toolchain,
    manifest_fingerprint,
    save_toolchain,
)
from xeno.adapters.generic import DiscoveredCommand, DiscoveredToolchain


def _toolchain(fingerprint: str = "abc123", *, argv0: str = "pytest") -> DiscoveredToolchain:
    return DiscoveredToolchain(
        install=("pip", "install", "-e", "."),
        required=(DiscoveredCommand(name="test", argv=(argv0, "-q")),),
        advisory=(),
        fingerprint=fingerprint,
    )


def test_fingerprint_is_stable_across_calls(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    assert manifest_fingerprint(tmp_path) == manifest_fingerprint(tmp_path)


def test_fingerprint_changes_when_manifest_content_changes(tmp_path: Path) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text("[project]\nname = 'x'\n")
    before = manifest_fingerprint(tmp_path)
    manifest.write_text("[project]\nname = 'y'\n")
    assert manifest_fingerprint(tmp_path) != before


def test_fingerprint_ignores_unrelated_source_edits(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    before = manifest_fingerprint(tmp_path)
    (tmp_path / "app.py").write_text("print('hello')\n")
    assert manifest_fingerprint(tmp_path) == before


def test_fingerprint_is_empty_hash_with_no_manifests_present(tmp_path: Path) -> None:
    # Still deterministic and non-empty (sha256 of nothing added), just the
    # same for every manifest-less repo.
    assert manifest_fingerprint(tmp_path) == manifest_fingerprint(tmp_path / "sub")


def test_cache_round_trip(tmp_path: Path) -> None:
    toolchain = _toolchain()
    save_toolchain(tmp_path, toolchain)
    loaded = load_cached_toolchain(tmp_path, toolchain.fingerprint)
    assert loaded == toolchain


def test_cache_miss_returns_none_when_file_absent(tmp_path: Path) -> None:
    assert load_cached_toolchain(tmp_path, "nonexistent") is None


def test_cache_miss_returns_none_on_fingerprint_mismatch(tmp_path: Path) -> None:
    save_toolchain(tmp_path, _toolchain(fingerprint="abc123"))
    assert load_cached_toolchain(tmp_path, "different") is None


def test_cache_miss_returns_none_on_corrupt_json(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".xeno" / "discovery"
    cache_dir.mkdir(parents=True)
    (cache_dir / "abc123.json").write_text("not valid json {")
    assert load_cached_toolchain(tmp_path, "abc123") is None


def test_validate_accepts_a_known_executable() -> None:
    _validate(_toolchain(argv0="pytest"))  # must not raise


def test_validate_rejects_a_disallowed_executable() -> None:
    with pytest.raises(DiscoveryError, match="disallowed executable"):
        _validate(_toolchain(argv0="rm"))


@pytest.mark.parametrize("executable", ["curl", "sudo", "wget", "dd", "chmod"])
def test_validate_rejects_common_destructive_or_network_tools(executable: str) -> None:
    with pytest.raises(DiscoveryError, match="disallowed executable"):
        _validate(_toolchain(argv0=executable))


def test_validate_rejects_empty_required_list() -> None:
    empty = DiscoveredToolchain(install=None, required=(), advisory=(), fingerprint="x")
    with pytest.raises(DiscoveryError, match="no required commands"):
        _validate(empty)


def test_validate_rejects_disallowed_install_executable() -> None:
    toolchain = DiscoveredToolchain(
        install=("curl", "evil.sh"),
        required=(DiscoveredCommand(name="test", argv=("pytest", "-q")),),
        advisory=(),
        fingerprint="x",
    )
    with pytest.raises(DiscoveryError, match="disallowed executable"):
        _validate(toolchain)


def test_validate_rejects_advisory_command_with_disallowed_executable() -> None:
    toolchain = DiscoveredToolchain(
        install=None,
        required=(DiscoveredCommand(name="test", argv=("pytest", "-q")),),
        advisory=(DiscoveredCommand(name="coverage", argv=("sudo", "pytest", "--cov")),),
        fingerprint="x",
    )
    with pytest.raises(DiscoveryError, match="disallowed executable"):
        _validate(toolchain)
