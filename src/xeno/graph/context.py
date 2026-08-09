"""Building the CODEBASE MAP breakpoint (PRD S9.6.1), used by Daedalus and
Chiron alike.

There is no Argus until Phase 3 (PRD S13). The PRD's OQ-10 names a manual
`@file` context mechanism as a scaffold standing in for it until Argus
exists; this module is that scaffold's harness-side half — a plain
file-tree-plus-contents dump, not a summarized index.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from xeno.prompt.keys import DEFAULT_IGNORES

#: Directories whose content is skipped by default (still listed in the file
#: tree). PRD OQ-10's manual `@file` scaffold exists precisely so a local
#: model is not handed the whole repository, including its own test suite,
#: as system-prompt filler on every call — a large local model already
#: struggles to hold instructions under a big context; a full-tree content
#: dump made that worse in practice (see xeno.graph.daedalus's format-retry).
_CONTENT_EXCLUDED_DIRS = frozenset({"tests", "tests_pending", "test"})


def _walk(root: Path, ignore: frozenset[str]) -> Iterable[Path]:
    for entry in sorted(root.iterdir()):
        if entry.name in ignore or entry.is_symlink():
            continue
        if entry.is_dir():
            yield from _walk(entry, ignore)
        elif entry.is_file():
            yield entry


def build_codebase_map(
    worktree: Path,
    *,
    focus: Sequence[Path] | None = None,
    max_content_bytes: int = 16_000,
) -> str:
    """A file tree plus file content, budget-capped.

    `focus` is the manual `@file` context (PRD OQ-10, this project's Argus
    stand-in until Phase 3): when given, ONLY those files' content is shown —
    the tree still lists everything else for orientation, but a user who
    named the relevant file(s) has already done Argus's job, and dumping the
    rest of the tree's content on top only spends context for no benefit.

    Without `focus`, every Python file is a candidate, but content under a
    test directory is skipped (still listed in the tree) since neither
    Daedalus nor Chiron can run tests anyway, and the acceptance tests for
    the task at hand would otherwise be handed to them as an answer key
    alongside the codebase.
    """
    files = list(_walk(worktree, DEFAULT_IGNORES))
    lines = ["Repository file tree:"]
    lines.extend(f"  {f.relative_to(worktree).as_posix()}" for f in files)
    lines.append("")
    lines.append("File contents:")

    focus_resolved = {p.resolve() for p in focus} if focus else None

    budget = max_content_bytes
    for f in files:
        if f.suffix != ".py" or budget <= 0:
            continue
        if focus_resolved is not None:
            if f.resolve() not in focus_resolved:
                continue
        elif any(part in _CONTENT_EXCLUDED_DIRS for part in f.relative_to(worktree).parts[:-1]):
            continue
        try:
            text = f.read_text()
        except OSError:
            continue
        block = f"\n--- {f.relative_to(worktree).as_posix()} ---\n{text}"
        if len(block) > budget:
            block = block[:budget] + "\n...(truncated)"
        lines.append(block)
        budget -= len(block)
    return "\n".join(lines)
