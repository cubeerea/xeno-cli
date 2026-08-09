"""Sandbox profile and pool logic that does not need a live Docker daemon
(that coverage lives in test_graph.py's real-sandbox test). This module is
about the pure argument-shaping and hashing logic being correct on its own."""

from __future__ import annotations

from pathlib import Path

from xeno.core.config import SandboxConfig
from xeno.sandbox.pool import _deps_hash
from xeno.sandbox.profile import (
    SANDBOX_UID,
    WORKSPACE_MOUNT,
    container_kwargs,
    install_container_kwargs,
)

CONFIG = SandboxConfig(memory="1g", cpus=1.5, pids_limit=256, network="none")


def test_container_kwargs_is_hardened(tmp_path: Path) -> None:
    kwargs = container_kwargs(
        image="img:tag", scratch_dir=tmp_path, config=CONFIG, network_disabled=True
    )
    assert kwargs["read_only"] is True
    assert kwargs["cap_drop"] == ["ALL"]
    assert kwargs["security_opt"] == ["no-new-privileges"]
    assert kwargs["network_disabled"] is True
    assert kwargs["user"] == f"{SANDBOX_UID}:{SANDBOX_UID}"
    assert kwargs["mem_limit"] == "1g"
    assert kwargs["pids_limit"] == 256


def test_container_kwargs_mounts_the_scratch_dir_at_workspace(tmp_path: Path) -> None:
    kwargs = container_kwargs(
        image="img:tag", scratch_dir=tmp_path, config=CONFIG, network_disabled=True
    )
    assert kwargs["volumes"] == {str(tmp_path): {"bind": WORKSPACE_MOUNT, "mode": "rw"}}
    assert kwargs["working_dir"] == WORKSPACE_MOUNT


def test_container_kwargs_nano_cpus_scales_from_config(tmp_path: Path) -> None:
    kwargs = container_kwargs(
        image="img:tag", scratch_dir=tmp_path, config=CONFIG, network_disabled=True
    )
    assert kwargs["nano_cpus"] == 1_500_000_000


def test_install_container_kwargs_is_writable_and_networked(tmp_path: Path) -> None:
    kwargs = install_container_kwargs(image="img:tag", scratch_dir=tmp_path, config=CONFIG)
    assert kwargs["read_only"] is False
    assert kwargs["network_disabled"] is False
    # Still hardened everywhere it can afford to be: no capabilities, no
    # privilege escalation, even though this container is temporarily
    # writable and networked for the install step (PRD S11.2).
    assert kwargs["cap_drop"] == ["ALL"]
    assert kwargs["security_opt"] == ["no-new-privileges"]
    assert "user" not in kwargs  # runs as root, unlike the hardened profile


def test_deps_hash_is_stable_for_identical_content(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("six==1.16.0\n")
    assert _deps_hash(tmp_path) == _deps_hash(tmp_path)


def test_deps_hash_changes_with_content(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "requirements.txt").write_text("six==1.16.0\n")
    (b / "requirements.txt").write_text("six==1.17.0\n")
    assert _deps_hash(a) != _deps_hash(b)


def test_deps_hash_is_stable_with_no_manifest(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert _deps_hash(a) == _deps_hash(b)
