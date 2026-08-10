"""On-disk layout for .xeno/ (PRD T2, S13 Phase 0).

    .xeno/
      runs/<run_id>/
        events.jsonl     append-only, replayable run log
        cost.json        the per-run cost ledger (M1.1-M1.4 instrument)
        workspace/       filesystem-as-memory: handle targets
      worktrees/<run_id>/  isolated git worktree for the run
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from xeno.core.config import STATE_DIRNAME

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def new_run_id() -> str:
    """Sortable, collision-resistant, and readable in a directory listing."""
    return f"{time.strftime('%Y%m%dT%H%M%S')}-{os.urandom(3).hex()}"


def short_run_id(run_id: str) -> str:
    """The suffix used in branch names: xeno/<slug>-<short_run_id> (PRD S8.4)."""
    return run_id.rsplit("-", 1)[-1]


def slugify(goal: str, *, max_len: int = 32) -> str:
    slug = _SLUG_RE.sub("-", goal.lower()).strip("-")
    return (slug[:max_len].rstrip("-")) or "run"


def run_branch_name(prefix: str, goal: str, run_id: str) -> str:
    """The dedicated branch a run lives on (PRD S8.4).

    Lives here because two modules need to agree on it exactly and neither
    can ask the other: `xeno.graph.build` CREATES the branch at the start of
    a run, and `xeno.cli` NAMES it again afterwards to report and push it.
    It is a pure function of inputs both already hold, so recomputing beats
    threading it back through the graph's return value — but only as long as
    there is one formula rather than two copies that can drift.
    """
    return f"{prefix}{slugify(goal)}-{short_run_id(run_id)}"


@dataclass(frozen=True, slots=True)
class RunPaths:
    repo_root: Path
    run_id: str

    @property
    def state(self) -> Path:
        return self.repo_root / STATE_DIRNAME

    @property
    def run_dir(self) -> Path:
        return self.state / "runs" / self.run_id

    @property
    def events(self) -> Path:
        return self.run_dir / "events.jsonl"

    @property
    def cost(self) -> Path:
        return self.run_dir / "cost.json"

    @property
    def workspace(self) -> Path:
        return self.run_dir / "workspace"

    @property
    def worktree(self) -> Path:
        return self.state / "worktrees" / self.run_id

    def ensure(self) -> RunPaths:
        for directory in (self.run_dir, self.workspace):
            directory.mkdir(parents=True, exist_ok=True)
        return self
