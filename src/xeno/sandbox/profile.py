"""Container security profile (PRD S11.1, S11.2).

Two shapes, both derived from `SandboxConfig`:

  `container_kwargs`          the hardened profile every gate-run container
                               uses: non-root, read-only root filesystem,
                               all capabilities dropped, no new privileges,
                               resource-capped, network disabled by default.
  `install_container_kwargs`  the narrow exception for the S11.2 'install'
                               network mode: a writable root filesystem and
                               network access, run as root so `pip install`
                               can write into site-packages the same way the
                               base image's own Dockerfile does — but ONLY
                               for the duration of the install step. The
                               container this produces is committed to an
                               image and destroyed; every container that
                               actually executes generated code still uses
                               the hardened profile above.

.git is never mounted (PRD S11.1) — not because anything here excludes it,
but because the worktree synced into the sandbox never contained it in the
first place. `xeno.prompt.keys.DEFAULT_IGNORES` drops `.git` at every
worktree-copy site (host worktree creation and the sandbox sync alike), so
there is nothing for this module to filter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xeno.core.config import SandboxConfig

#: Where the synced worktree lands inside every sandbox container.
WORKSPACE_MOUNT = "/workspace"

#: PRD S11.1: non-root. Matches the `useradd --uid 65532` in the adapter's
#: Dockerfile (xeno.adapters.generic.DOCKERFILE) — an arbitrary but fixed and
#: unprivileged UID, chosen above the usual 1-fixed-digit system range.
SANDBOX_UID = 65532


def _base_kwargs(image: str, scratch_dir: Path, config: SandboxConfig) -> dict[str, Any]:
    """The settings both profiles share. Kept in one place so a hardening
    change (a dropped capability, a resource cap) cannot land on the gate
    profile and silently miss the install profile — the two differ in exactly
    three keys, and those three are the only ones stated at the call sites
    below."""
    return {
        "image": image,
        "command": ["sleep", "infinity"],
        "detach": True,
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges"],
        "tmpfs": {"/tmp": "rw,size=256m"},
        "mem_limit": config.memory,
        "nano_cpus": int(config.cpus * 1_000_000_000),
        "pids_limit": config.pids_limit,
        "volumes": {str(scratch_dir): {"bind": WORKSPACE_MOUNT, "mode": "rw"}},
        "working_dir": WORKSPACE_MOUNT,
    }


def container_kwargs(
    *,
    image: str,
    scratch_dir: Path,
    config: SandboxConfig,
    network_disabled: bool,
) -> dict[str, Any]:
    """The hardened profile: what every Talos/Chiron gate-run container uses."""
    return {
        **_base_kwargs(image, scratch_dir, config),
        "user": f"{SANDBOX_UID}:{SANDBOX_UID}",
        "read_only": True,
        "network_disabled": network_disabled,
    }


def install_container_kwargs(
    *,
    image: str,
    scratch_dir: Path,
    config: SandboxConfig,
) -> dict[str, Any]:
    """PRD S11.2 'install' mode's writable, networked, root-run exception.

    Root and a writable filesystem are both required for `pip install` to
    write into site-packages (the same reason the base image's own Dockerfile
    installs ruff/mypy/pytest as root before switching to the sandbox user).
    Network is left enabled here — the caller is responsible for committing
    the result and discarding this container before any generated code runs
    against it.

    Note the three differences from `container_kwargs`, and only those three:
    no `user` (so root), a writable root filesystem, and network on.
    """
    return {
        **_base_kwargs(image, scratch_dir, config),
        "read_only": False,
        "network_disabled": False,
    }
