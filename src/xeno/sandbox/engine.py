"""Reaching the container engine, and saying something useful when we cannot.

Every gate Talos runs executes inside a container, so an unreachable engine
is not a degraded run — it is no run at all. Two things were wrong with how
that surfaced.

The first is *when*. `docker.from_env()` lived in `ToolchainSession.start`,
which the CLI reaches only after copying the repo into a worktree, holding a
whole spec conversation with Odysseus, and spending a discovery call. The
engine's availability is knowable at second zero, so every token spent before
the failure was spent on a run that could never have gated anything.

The second is *what it says*. `docker-py` connects to a Unix socket, so a
stopped Docker Desktop surfaces as `FileNotFoundError: [Errno 2] No such file
or directory` at the bottom of a thirty-frame urllib3 traceback — no path, no
mention of Docker. Run that in a directory you are actively working in and
the obvious reading is that the harness cannot find *your files*. It is a
convincing misreading and it costs real time.

So this module answers the question the traceback does not: which endpoint we
tried, what we found there, and what to do about it.
"""

from __future__ import annotations

import os
from pathlib import Path

import docker
from docker.context import ContextAPI
from docker.errors import DockerException

#: What `docker-py` falls back to when neither `DOCKER_HOST` nor a context
#: says otherwise.
_DEFAULT_ENDPOINT = "unix:///var/run/docker.sock"


class EngineUnavailable(RuntimeError):
    """The container engine could not be reached.

    Carries a diagnosed, human-readable message — the caller is expected to
    print it verbatim rather than wrap it in more prose.
    """


def endpoint() -> str:
    """The endpoint `docker.from_env()` will actually try.

    `DOCKER_HOST` wins; otherwise the active context decides, which is how a
    Mac ends up on `~/.docker/run/docker.sock` rather than the `/var/run` path
    everyone quotes. Never raises: this exists to make an error message
    concrete, and an error message that itself fails is worse than a vague one.
    """
    if host := os.environ.get("DOCKER_HOST"):
        return host
    try:
        current = ContextAPI.get_current_context()
    except Exception:
        return _DEFAULT_ENDPOINT
    return current.Host or _DEFAULT_ENDPOINT


def _socket_finding(where: str) -> str:
    """What is actually at a `unix://` endpoint, in one clause.

    The dangling-symlink case is called out by name because it is the one a
    user cannot see with `ls`: Docker Desktop leaves `/var/run/docker.sock`
    behind as a symlink to a socket under `~/.docker` that only exists while
    the VM is up, so the path looks present and resolves to nothing.
    """
    path = Path(where)
    if path.is_socket():
        return f"the socket at {where} exists but refused the connection"
    if path.is_symlink():
        return f"{where} is a symlink to {os.readlink(where)}, which does not exist"
    if path.exists():
        return f"{where} exists but is not a socket"
    return f"there is no socket at {where}"


def _diagnose(exc: DockerException) -> str:
    where = endpoint()
    if where.startswith("unix://"):
        finding = _socket_finding(where.removeprefix("unix://"))
    else:
        finding = f"{where} did not answer ({exc})"
    return (
        f"cannot reach the container engine — {finding}.\n"
        "Every gate Talos runs (lint, types, tests) executes inside a container, "
        "so there is nothing this run could evaluate.\n"
        "Start Docker Desktop (or another engine — Colima, OrbStack, Podman) and "
        "try again; `xeno doctor` will confirm once it is up."
    )


def connect() -> docker.DockerClient:
    """A connected client, or `EngineUnavailable` with a message worth reading."""
    try:
        client = docker.from_env()
    except DockerException as exc:
        raise EngineUnavailable(_diagnose(exc)) from exc
    return client


def probe() -> tuple[bool, str]:
    """`(reachable, detail)` for `xeno doctor` and the pre-run check.

    Mirrors `Provider.health_check`'s shape so the doctor table can hold a
    sandbox row beside the provider rows without a second code path.
    """
    try:
        client = connect()
    except EngineUnavailable as exc:
        return False, str(exc)
    try:
        version = client.version()
        server = version.get("Version", "unknown")
        return True, f"{endpoint()} (engine {server})"
    except DockerException as exc:
        return False, f"connected to {endpoint()} but the daemon did not answer: {exc}"
    finally:
        client.close()
