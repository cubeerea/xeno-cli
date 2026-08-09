"""Docker container lifecycle and the warm pool (PRD S11, S13 Phase 2).

M3.1 requires p50 sandbox turnaround under 3s against a warm-pool container,
where turnaround is defined as "Talos invocation to EvalReport returned,
excluding model inference time" (PRD S15.3) — i.e. it includes whatever the
pool does on Talos's behalf. A "warm" container is therefore one that is
ALREADY RUNNING when `acquire()` is called: no `docker create`/`docker
start` on the critical path. `release()` destroys the used container and
replenishes the pool on a background thread, so that cost lands on the NEXT
acquire()'s slack time instead of the current caller's turnaround.

Containers are destroyed rather than wiped and reused: the sandbox user may
have left cache directories (.mypy_cache, .ruff_cache, __pycache__, a stray
.coverage file) under the bind-mounted scratch dir, and reliably
distinguishing "safe to keep" leftover state from "could leak into the next
evaluation" is not worth the bookkeeping. A fresh container per acquisition
is simpler and no less correct; what makes the pool WARM is that the
replacement is built ahead of the next request, off the critical path.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import queue
import shutil
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path

import docker
from docker.errors import DockerException, ImageNotFound
from docker.models.containers import Container

from xeno.adapters.base import LanguageAdapter
from xeno.core.config import SandboxConfig, SecretsConfig
from xeno.sandbox.profile import WORKSPACE_MOUNT, container_kwargs, install_container_kwargs
from xeno.security.mounts import mount_ignore

logger = logging.getLogger(__name__)

#: How long acquire() waits for a pooled container before falling back to a
#: synchronous (cold) create. PRD M3.3: that fallback is a pool-sizing
#: defect, not the intended path — it exists so a misconfigured pool
#: degrades to "slow" instead of "hangs".
_ACQUIRE_TIMEOUT_SECONDS = 30.0

#: Time budget for building the adapter's image or running an install step,
#: both one-time-ish costs that are allowed to be much slower than a gate.
_BUILD_TIMEOUT_SECONDS = 600


@dataclass
class Sandbox:
    """One running container plus the host-side directory bind-mounted into
    it as `/workspace` (PRD S11.1)."""

    container: Container
    scratch_dir: Path

    def sync_worktree(self, worktree: Path, *, secrets: SecretsConfig | None = None) -> None:
        """Replace the container's `/workspace` contents with `worktree`'s.

        This is an ordinary host filesystem copy, not `docker cp` — the
        container's bind-mount source IS `scratch_dir`, so writing here is
        writing there. `secrets` re-applies the S11.3 denylist at the mount
        boundary (defense in depth: the harness worktree this copies from
        should already exclude denylisted paths, but this method has no way
        to know that of an arbitrary caller).
        """
        for entry in self.scratch_dir.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        ignore = mount_ignore(secrets) if secrets is not None else None
        shutil.copytree(worktree, self.scratch_dir, dirs_exist_ok=True, ignore=ignore)

    def exec(self, argv: Sequence[str], *, timeout: float) -> tuple[int, str]:
        """Run a fixed argv inside the container (PRD S10: no model-authored
        shell strings — every caller passes an adapter-built vector).

        docker-py's `exec_run` has no built-in timeout, so this enforces one
        with a helper thread. A timeout leaves that thread to finish on its
        own; the container is never reused after this Sandbox is released
        (see module docstring), so the orphaned exec has nothing left to
        affect.
        """
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                self.container.exec_run, list(argv), workdir=WORKSPACE_MOUNT, demux=False
            )
            try:
                result = future.result(timeout=timeout)
            except FutureTimeoutError:
                return -1, f"INFRASTRUCTURE: {argv[0]} timed out after {timeout}s in sandbox"
        raw = result.output
        assert isinstance(raw, bytes | type(None)), "demux=False, stream unset: bytes or None"
        output = (raw or b"").decode(errors="replace")
        return result.exit_code or 0, output


class WarmPool:
    """A ring of pre-started, hardened containers for one adapter image."""

    def __init__(
        self,
        client: docker.DockerClient,
        adapter: LanguageAdapter,
        config: SandboxConfig,
        *,
        workspace_root: Path,
        image: str | None = None,
        network_disabled: bool = True,
    ) -> None:
        self._client = client
        self._adapter = adapter
        self._config = config
        self._workspace_root = workspace_root
        #: Overridable so the S11.2 'install' path can hand the pool a
        #: post-install committed image instead of the adapter's base image.
        self._image = image or adapter.image()
        self._network_disabled = network_disabled
        self._ready: queue.Queue[Sandbox] = queue.Queue()
        self._replenisher = ThreadPoolExecutor(max_workers=max(1, config.warm_pool_size))
        self._counter = 0
        self._counter_lock = threading.Lock()
        self._closed = False

    def ensure_image(self) -> None:
        """Build the adapter's base image if it is not already cached
        locally (PRD S11.1: 'Base image built per LanguageAdapter, cached
        locally'). A no-op on every run after the first."""
        _ensure_image(self._client, self._adapter.image(), self._adapter.dockerfile())

    def fill(self) -> None:
        """Pool fill (PRD M3.3): pay the cold-start cost once, up front, so
        every Talos invocation during the run gets a warm container."""
        for _ in range(self._config.warm_pool_size):
            self._ready.put(self._create_one())

    def acquire(self) -> Sandbox:
        try:
            return self._ready.get(timeout=_ACQUIRE_TIMEOUT_SECONDS)
        except queue.Empty:
            logger.warning(
                "sandbox pool exhausted after %.0fs; falling back to a cold "
                "container (PRD M3.3: this is a pool-sizing defect)",
                _ACQUIRE_TIMEOUT_SECONDS,
            )
            return self._create_one()

    def release(self, sandbox: Sandbox) -> None:
        """Destroy the used container and replenish in the background, so
        the cost never lands on the caller's own turnaround (PRD M3.1)."""
        if self._closed:
            _destroy(sandbox)
            return
        self._replenisher.submit(self._retire_and_replenish, sandbox)

    def shutdown(self) -> None:
        """Drain and destroy every container — pooled or in flight — then
        stop accepting new background work. Call once per run, in a
        finally."""
        self._closed = True
        self._replenisher.shutdown(wait=True)
        while True:
            try:
                sandbox = self._ready.get_nowait()
            except queue.Empty:
                break
            _destroy(sandbox)

    def _retire_and_replenish(self, sandbox: Sandbox) -> None:
        _destroy(sandbox)
        if self._closed:
            return
        try:
            self._ready.put(self._create_one())
        except DockerException:
            # A failed replenishment shrinks the pool rather than crashing a
            # background thread silently; the next acquire()'s cold-start
            # fallback covers the shortfall.
            logger.exception("sandbox pool replenishment failed")

    def _new_scratch_dir(self) -> Path:
        with self._counter_lock:
            self._counter += 1
            n = self._counter
        scratch = self._workspace_root / f"sandbox-{n}"
        scratch.mkdir(parents=True, exist_ok=True)
        return scratch

    def _create_one(self) -> Sandbox:
        scratch = self._new_scratch_dir()
        kwargs = container_kwargs(
            image=self._image,
            scratch_dir=scratch,
            config=self._config,
            network_disabled=self._network_disabled,
        )
        container = self._client.containers.create(**kwargs)
        container.start()
        return Sandbox(container=container, scratch_dir=scratch)


def _ensure_image(client: docker.DockerClient, image: str, dockerfile: str | None) -> None:
    try:
        client.images.get(image)
        return
    except ImageNotFound:
        pass
    if dockerfile is None:
        client.images.pull(image)
        return
    client.images.build(fileobj=io.BytesIO(dockerfile.encode()), tag=image, rm=True)


def _destroy(sandbox: Sandbox) -> None:
    with contextlib.suppress(DockerException):
        sandbox.container.remove(force=True)
    shutil.rmtree(sandbox.scratch_dir, ignore_errors=True)


def prepare_gate_image(
    client: docker.DockerClient,
    adapter: LanguageAdapter,
    config: SandboxConfig,
    *,
    worktree: Path,
    workspace_root: Path,
    secrets: SecretsConfig,
) -> tuple[str, bool]:
    """Resolve the image and network policy the gate-run WarmPool should use
    (PRD S11.2).

    Returns `(image, network_disabled)`. For `network: install`, this runs
    the adapter's install command against a temporary, writable, networked
    container (PRD: "the container is then committed and the network
    dropped for the test run"), then hands back the committed image with
    `network_disabled=True` — every container Talos or Chiron actually
    execute code in stays hardened and offline, matching `network: none`.
    """
    _ensure_image(client, adapter.image(), adapter.dockerfile())

    if config.network == "none":
        return adapter.image(), True
    if config.network == "open":
        logger.warning(
            "sandbox network policy is 'open' (PRD S11.2) — generated code has "
            "unrestricted egress for this run"
        )
        return adapter.image(), False

    # network == "install"
    install_cmd = adapter.install_cmd(worktree)
    if install_cmd is None:
        return adapter.image(), True

    tag = _run_install(client, adapter, config, worktree, workspace_root, install_cmd, secrets)
    return tag, True


def _run_install(
    client: docker.DockerClient,
    adapter: LanguageAdapter,
    config: SandboxConfig,
    worktree: Path,
    workspace_root: Path,
    install_cmd: Sequence[str],
    secrets: SecretsConfig,
) -> str:
    tag = f"{adapter.image()}-installed-{_deps_hash(worktree)}"
    try:
        client.images.get(tag)
        return tag
    except ImageNotFound:
        pass

    scratch = workspace_root / "install-scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    shutil.copytree(worktree, scratch, dirs_exist_ok=True, ignore=mount_ignore(secrets))

    kwargs = install_container_kwargs(image=adapter.image(), scratch_dir=scratch, config=config)
    container = client.containers.create(**kwargs)
    try:
        container.start()
        sandbox = Sandbox(container=container, scratch_dir=scratch)
        code, output = sandbox.exec(install_cmd, timeout=_BUILD_TIMEOUT_SECONDS)
        if code != 0:
            raise DockerException(f"dependency install failed (exit {code}): {output[-2000:]}")
        container.commit(repository=tag.split(":")[0], tag=tag.split(":")[1])
        return tag
    finally:
        container.remove(force=True)
        shutil.rmtree(scratch, ignore_errors=True)


def _deps_hash(worktree: Path) -> str:
    hasher = hashlib.sha256()
    for name in ("requirements.txt", "pyproject.toml"):
        path = worktree / name
        if path.is_file():
            hasher.update(name.encode())
            hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]
