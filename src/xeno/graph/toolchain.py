"""The toolchain as run-time state rather than pre-run configuration.

The original design resolved a repository's toolchain exactly once, before
the graph existed, and froze it for the whole run. That holds only while the
repository's build declaration is something the run *reads* — and it stops
holding the moment the run also *writes* it:

- **Greenfield.** An empty directory declares no toolchain, so there is
  nothing to discover before the run. The toolchain is an OUTPUT of the
  run's first task (scaffold the project), not an input to it.
- **A new dependency.** Even against an existing repo, the moment Daedalus
  adds a package to `pyproject.toml`/`package.json`, the frozen toolchain is
  describing a repository that no longer exists. Gates keep executing the
  originally-discovered argv against dependencies that were never installed;
  those come back as ordinary non-zero exits rather than the infrastructure
  sentinel, so the ladder classifies an environment problem as a code defect
  and burns L0 through L4 chasing a bug that is not there.

Both are the same bug — a cached fact outliving the thing it described — so
both get the same fix. This module owns the mutable `(toolchain, adapter,
pool)` triple and re-derives it whenever the worktree's manifest fingerprint
stops matching the one the current toolchain was discovered from. Talos asks
before every evaluation (`xeno.graph.talos`), which is the only point in a
run where the answer can matter.

The check itself is a cheap local hash (`manifest_fingerprint`); a model call
happens only when that hash actually moved, and the sandbox image is rebuilt
only when the re-discovered toolchain resolves to a different image than the
one already running.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import docker

from xeno.adapters.discovery import DiscoveryError, discover_toolchain, manifest_fingerprint
from xeno.adapters.generic import DiscoveredToolchain, GenericAdapter
from xeno.core.config import XenoConfig
from xeno.core.paths import RunPaths
from xeno.core.runlog import EventKind, RunLog
from xeno.core.state import AgentState
from xeno.prompt.keys import CacheKeyring
from xeno.router.router import Router
from xeno.sandbox.engine import connect as connect_engine
from xeno.sandbox.pool import WarmPool, prepare_gate_image


class ToolchainSession:
    """Owns the toolchain and the sandbox resources derived from it.

    Constructed by `xeno.cli` before the graph is built; `start` brings up
    the first pool and registers teardown. Exactly one teardown callback is
    registered for the whole session regardless of how many times the pool is
    later swapped, because it closes over `self` rather than a specific pool
    instance.
    """

    def __init__(
        self,
        *,
        router: Router,
        config: XenoConfig,
        keyring: CacheKeyring,
        paths: RunPaths,
        repo_root: Path,
        worktree: Path,
        runlog: RunLog,
        toolchain: DiscoveredToolchain,
    ) -> None:
        self._router = router
        self._config = config
        self._keyring = keyring
        self._paths = paths
        self._repo_root = repo_root
        self._worktree = worktree
        self._runlog = runlog

        self.toolchain = toolchain
        self.adapter = GenericAdapter(toolchain)
        self._pool: WarmPool | None = None
        self._client: docker.DockerClient | None = None
        self._image = ""
        self._network_disabled = True

    @property
    def pool(self) -> WarmPool:
        assert self._pool is not None, "ToolchainSession.start() must run before the graph"
        return self._pool

    def start(self, stack: ExitStack) -> None:
        """Bring up the Docker client and the first warm pool.

        Teardown is registered against the session, not the pool, so a
        mid-run swap cannot strand the pool that replaced the one the stack
        was originally told about.

        The CLI checks the engine in `_preflight`, long before this runs, so
        reaching a failure HERE means the engine went away between that check
        and now — a Docker Desktop quit or an OOM kill mid-run. Rare, but it
        raises `EngineUnavailable` rather than a urllib3 traceback either way.
        """
        self._client = connect_engine()
        stack.callback(self._client.close)
        stack.callback(self.shutdown)
        self._build_pool()

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def refresh_if_stale(self, state: AgentState) -> bool:
        """Re-derive the toolchain if the worktree's manifests have moved.

        Returns whether anything changed. Never raises: a re-discovery that
        fails leaves the previous toolchain in place, because a run that was
        gating on something a moment ago is better served by continuing to
        gate on it than by collapsing. A genuinely unestablished result still
        propagates — `run_gates` reports that as a failure the ladder can act
        on, which is how a bad scaffold gets repaired rather than ignored.
        """
        fingerprint = manifest_fingerprint(self._worktree)
        if fingerprint == self.toolchain.fingerprint:
            return False

        try:
            discovered = discover_toolchain(
                router=self._router,
                config=self._config,
                keyring=self._keyring,
                state=state,
                paths=self._paths,
                repo_root=self._repo_root,
                worktree=self._worktree,
                allow_unestablished=True,
            )
        except DiscoveryError as exc:
            self._runlog.event(
                EventKind.TOOLCHAIN_REFRESH,
                ok=False,
                detail=f"re-discovery failed, keeping previous toolchain: {exc}",
            )
            return False

        # Always adopt the result, even when the command list is identical:
        # it carries the NEW fingerprint, and without storing that this same
        # check would re-discover on every single evaluation for the rest of
        # the run. Whether anything expensive follows from the swap is
        # decided below, by comparing the resolved image rather than the
        # commands.
        was_established = self.toolchain.established
        self.toolchain = discovered
        self.adapter = GenericAdapter(discovered)
        self._runlog.event(
            EventKind.TOOLCHAIN_REFRESH,
            ok=True,
            established=discovered.established,
            newly_established=discovered.established and not was_established,
            commands=[c.name for c in discovered.required],
            fingerprint=discovered.fingerprint,
        )
        self._rebuild_pool_if_image_changed()
        return True

    def _rebuild_pool_if_image_changed(self) -> None:
        """Only pay for a pool swap when the image actually differs.

        Under `network: none` (the default) `prepare_gate_image` never runs
        an install step, so a re-discovered toolchain almost always resolves
        to the same base image and the running containers stay valid — the
        new commands simply execute in them. Under `network: install` a new
        dependency set produces a new committed tag, and then the pool does
        have to be replaced.
        """
        if self._pool is None:
            return
        image, network_disabled = self._resolve_image()
        if image == self._image and network_disabled == self._network_disabled:
            return
        self._pool.shutdown()
        self._pool = None
        self._build_pool()

    def _resolve_image(self) -> tuple[str, bool]:
        assert self._client is not None, "start() sets the client before any image work"
        return prepare_gate_image(
            self._client,
            self.adapter,
            self._config.sandbox,
            worktree=self._worktree,
            workspace_root=self._paths.workspace,
            secrets=self._config.secrets,
        )

    def _build_pool(self) -> None:
        assert self._client is not None, "start() sets the client before any pool work"
        self._image, self._network_disabled = self._resolve_image()
        pool = WarmPool(
            self._client,
            self.adapter,
            self._config.sandbox,
            workspace_root=self._paths.workspace / "sandboxes",
            image=self._image,
            network_disabled=self._network_disabled,
        )
        pool.fill()
        self._pool = pool
