"""Toolchain discovery (PRD S8.2, S12 revised): fingerprinting, cache I/O,
and the executable allowlist. No subprocess, no model call — those are
covered live; this module is about the pure, deterministic logic being
correct on its own (same philosophy the old PythonAdapter tests used)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xeno.adapters.discovery import (
    DiscoveryError,
    _command,
    _validate,
    discover_toolchain,
    has_manifests,
    load_cached_toolchain,
    manifest_fingerprint,
    save_toolchain,
)
from xeno.adapters.generic import DiscoveredCommand, DiscoveredToolchain
from xeno.core.config import STATE_DIRNAME
from xeno.core.types import GateProfile


def _toolchain(fingerprint: str = "abc123", *, argv0: str = "pytest") -> DiscoveredToolchain:
    return DiscoveredToolchain(
        install=("pip", "install", "-e", "."),
        #: `is_test` is derived from the label and argv on every path a
        #: toolchain can be built by, so the fixture carries what
        #: `_classify` would derive — otherwise the round-trip below would
        #: read as a mismatch when it is actually the classifier working.
        required=(DiscoveredCommand(name="test", argv=(argv0, "-q"), is_test=True),),
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


# ---- greenfield -----------------------------------------------------------


class _ExplodingRouter:
    """Any attribute access is a test failure: the greenfield path must reach
    its answer deterministically, without spending a model call."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"discovery must not call the router on a manifest-less repo ({name})")


def test_has_manifests_is_false_for_an_empty_directory(tmp_path: Path) -> None:
    assert has_manifests(tmp_path) is False


def test_has_manifests_is_false_when_only_source_is_present(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('hi')\n")
    assert has_manifests(tmp_path) is False


@pytest.mark.parametrize("name", ["pyproject.toml", "package.json", "go.mod", "Cargo.toml"])
def test_has_manifests_detects_each_ecosystem(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_text("{}")
    assert has_manifests(tmp_path) is True


def test_greenfield_resolves_to_unestablished_without_a_model_call(tmp_path: Path) -> None:
    toolchain = discover_toolchain(
        router=_ExplodingRouter(),  # type: ignore[arg-type]
        config=None,  # type: ignore[arg-type]
        keyring=None,  # type: ignore[arg-type]
        state=None,  # type: ignore[arg-type]
        paths=None,  # type: ignore[arg-type]
        repo_root=tmp_path,
        worktree=tmp_path,
        allow_unestablished=True,
    )

    assert toolchain.established is False
    assert toolchain.required == ()


def test_greenfield_result_is_not_cached(tmp_path: Path) -> None:
    """Every manifest-less repo shares one fingerprint, so persisting
    "nothing here" would outlive the very first commit that fixes it."""
    discover_toolchain(
        router=_ExplodingRouter(),  # type: ignore[arg-type]
        config=None,  # type: ignore[arg-type]
        keyring=None,  # type: ignore[arg-type]
        state=None,  # type: ignore[arg-type]
        paths=None,  # type: ignore[arg-type]
        repo_root=tmp_path,
        worktree=tmp_path,
        allow_unestablished=True,
    )

    assert not list((tmp_path / ".xeno" / "discovery").glob("*.json"))


def test_greenfield_still_raises_when_unestablished_is_not_allowed(tmp_path: Path) -> None:
    with pytest.raises(DiscoveryError, match="no manifest"):
        discover_toolchain(
            router=_ExplodingRouter(),  # type: ignore[arg-type]
            config=None,  # type: ignore[arg-type]
            keyring=None,  # type: ignore[arg-type]
            state=None,  # type: ignore[arg-type]
            paths=None,  # type: ignore[arg-type]
            repo_root=tmp_path,
            worktree=tmp_path,
        )


def test_unestablished_toolchain_reports_itself_as_such() -> None:
    toolchain = DiscoveredToolchain.unestablished("deadbeef")
    assert toolchain.established is False
    assert toolchain.fingerprint == "deadbeef"


def test_a_toolchain_with_required_commands_is_established() -> None:
    assert _toolchain().established is True


# ---- the cache is an input, and gets the same allowlist -------------------


def test_cache_load_rejects_a_disallowed_executable(tmp_path: Path) -> None:
    """The cache file is advertised as human-editable, so it is an untrusted
    input like any other — skipping `_validate` here would let the allowlist
    be bypassed by writing the command to disk instead of proposing it."""
    cache_dir = tmp_path / ".xeno" / "discovery"
    cache_dir.mkdir(parents=True)
    (cache_dir / "ff.json").write_text(
        json.dumps(
            {
                "fingerprint": "ff",
                "install": None,
                "required": [{"name": "evil", "argv": ["rm", "-rf", "/"]}],
                "advisory": [],
            }
        )
    )

    with pytest.raises(DiscoveryError, match="disallowed executable"):
        load_cached_toolchain(tmp_path, "ff")


def test_cache_load_rejects_an_empty_required_list(tmp_path: Path) -> None:
    """Otherwise a truncated cache file turns Talos into a no-op that reports
    success for every task."""
    cache_dir = tmp_path / ".xeno" / "discovery"
    cache_dir.mkdir(parents=True)
    (cache_dir / "ee.json").write_text(
        json.dumps({"fingerprint": "ee", "install": None, "required": [], "advisory": []})
    )

    with pytest.raises(DiscoveryError, match="no required commands"):
        load_cached_toolchain(tmp_path, "ee")


# ---- test-command classification (xeno.adapters.discovery._classify) -------


@pytest.mark.parametrize(
    ("name", "argv"),
    [
        ("test", ("python3", "-m", "pytest")),
        ("unit tests", ("make", "check")),
        ("specs", ("bundle", "exec", "rspec")),
        ("check", ("pytest", "-q")),
        ("check", ("npm", "test")),
        ("check", ("npm", "run", "test")),
        ("check", ("cargo", "test")),
        ("check", ("go", "test", "./...")),
        ("check", ("npx", "jest")),
        ("check", ("gradle", "test")),
    ],
)
def test_classified_as_a_test_command(name: str, argv: tuple[str, ...]) -> None:
    toolchain = DiscoveredToolchain(
        install=None,
        required=(_command(name, argv),),
        advisory=(),
        fingerprint="f",
    )
    assert toolchain.required[0].is_test


@pytest.mark.parametrize(
    ("name", "argv"),
    [
        # The reason argv is read POSITIONALLY: this type-checks a directory
        # that happens to be called tests, and a whole-argv scan for "test"
        # would hold the type checker back from every implementation gate.
        ("typecheck", ("mypy", "src", "tests")),
        ("lint", ("ruff", "check", ".")),
        ("build", ("cargo", "build")),
        ("install", ("pip", "install", "-e", ".")),
    ],
)
def test_not_classified_as_a_test_command(name: str, argv: tuple[str, ...]) -> None:
    toolchain = DiscoveredToolchain(
        install=None,
        required=(_command(name, argv),),
        advisory=(),
        fingerprint="f",
    )
    assert not toolchain.required[0].is_test


def test_classification_is_derived_when_an_older_cache_entry_is_loaded(tmp_path: Path) -> None:
    """The field is derived rather than stored precisely so the on-disk
    format did not have to change — every discovery file written before it
    existed still loads, and still classifies correctly."""
    path = tmp_path / STATE_DIRNAME / "discovery" / "old.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "fingerprint": "old",
                "install": None,
                "required": [
                    {"name": "lint", "argv": ["ruff", "check", "."]},
                    {"name": "test", "argv": ["pytest", "-q"]},
                ],
                "advisory": [],
            }
        )
    )

    loaded = load_cached_toolchain(tmp_path, "old")

    assert loaded is not None
    assert [c.is_test for c in loaded.required] == [False, True]
    assert len(loaded.required_for(GateProfile.IMPLEMENTATION)) == 1
