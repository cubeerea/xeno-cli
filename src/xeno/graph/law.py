"""PROJECT LAW: the documents a run is bound by, rendered into breakpoint 2.

Three artifacts, one block:

  * the SPEC — what the run was started to build (Odysseus JOB 2, or `--file`);
  * the ROADMAP — the milestones, and which of them are done;
  * MEMORY — standing human preferences, one entry per rejection.

They are grouped because they share the property that makes them law: none of
them is derivable from the worktree. The codebase map answers "what does this
code do"; no amount of reading source answers "what was this supposed to be",
"which milestone are we on", or "the human rejected this approach last time".
Re-deriving the first from a 16 KB budgeted map is guesswork at flagship
prices; the third cannot be re-derived at all, because a rejected approach is
precisely the thing that never reached disk.

Why this is not just more `context_handles`
-------------------------------------------
It used to be. `xeno.cli` seeded the spec as a Handle and appended it to
`state.context_handles`, which every node then passed to `build_codebase_map`
as `focus`. That path silently dropped it: `focus` is a FILTER over a walk of
the worktree, and the spec lives in `.xeno/runs/<id>/workspace/`, outside it.
So the handle was recorded in state, counted in the run summary as a file in
context, and matched nothing — Daedalus and Chiron never saw the spec at all,
while the accounting said otherwise. Law gets its own breakpoint and its own
reader precisely so "is it in the prompt" stops depending on where on disk it
happens to live.

Budget
------
Law has a reserved budget rather than sharing the map's. The map spends its
16 KB in walk order, so a shared pool would let an alphabetically-early source
file evict the project's own specification — and the failure would be silent,
since nothing downstream can tell a spec that was truncated from one that was
never written.

Within that budget the order is memory, then roadmap, then spec, smallest and
least compressible first. Memory entries are a few hundred bytes and carry the
highest value per byte in the whole prompt (a preference no other artifact
records); the spec is the largest and the most tolerant of a tail truncation,
since its opening paragraphs carry the intent. The one thing that must never
happen is a memory entry being dropped to fit a spec appendix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xeno.core.config import STATE_DIRNAME
from xeno.core.state import AgentState, Handle
from xeno.prompt.delimit import as_law

#: Filename of the project-level memory file, alongside `runs/` and
#: `worktrees/` under `.xeno/`. Project-level rather than run-level because a
#: preference that expired when the run did would never be worth stating.
MEMORY_FILENAME = "memory.md"

#: Total bytes of law content per prompt, before block overhead. Sized so the
#: common case (a spec of a few KB, a roadmap of one, a handful of memory
#: entries) is never truncated at all, while a pathological spec cannot crowd
#: out the rest of the prompt.
DEFAULT_LAW_BUDGET = 12_000

#: Per-section floors, subtracted from the shared budget before the spec is
#: allowed to draw on it. Guarantees the two small high-value sections always
#: fit, whatever the spec's size.
_MEMORY_FLOOR = 4_000
_ROADMAP_FLOOR = 2_000


def memory_path(repo_root: Path) -> Path:
    """Where a project's memory lives. One file, at a fixed location, so a
    human can open it without asking the harness where it put it."""
    return repo_root / STATE_DIRNAME / MEMORY_FILENAME


@dataclass(frozen=True, slots=True)
class ProjectLaw:
    """Renders the law block for a run. Built once, rendered per call.

    Per call rather than once, because law is not constant within a run: a
    re-plan rewrites the roadmap, and an accepted memory entry appends to
    memory.md. Rendering is a few file reads of a few kilobytes, which is
    cheaper than the bug where a node acts on a preference that was withdrawn
    twenty turns ago.
    """

    repo_root: Path
    budget: int = DEFAULT_LAW_BUDGET

    def render(self, state: AgentState) -> str:
        """The PROJECT LAW block text, or "" when the project has no law yet.

        Empty is a legitimate result — a `xeno run` on a fresh directory has no
        memory and no roadmap until Odysseus writes one — and the builder omits
        the whole block rather than sending an empty heading.
        """
        sections: list[str] = []
        budget = self.budget

        memory = _read_memory(self.repo_root)
        if memory:
            room = min(len(memory), max(_MEMORY_FLOOR, budget - _ROADMAP_FLOOR))
            block = as_law(
                memory,
                label="project memory",
                source=f"{STATE_DIRNAME}/{MEMORY_FILENAME}",
                truncate_to=room,
            )
            sections.append(block)
            budget -= len(block)

        roadmap = _read_handle(state.roadmap)
        if roadmap:
            room = min(len(roadmap), max(_ROADMAP_FLOOR, budget))
            block = as_law(roadmap, label="roadmap", truncate_to=room)
            sections.append(block)
            budget -= len(block)

        spec = _read_spec(state)
        if spec and budget > 0:
            block = as_law(spec, label="specification", truncate_to=budget)
            sections.append(block)

        if not sections:
            return ""
        return "PROJECT LAW\n\n" + "\n\n".join(sections)


def _read_memory(repo_root: Path) -> str:
    path = memory_path(repo_root)
    try:
        return path.read_text().strip()
    except (OSError, UnicodeDecodeError):
        # A missing memory.md is the normal case on a new project, and an
        # unreadable one must not take a run down: law is additive context,
        # never a precondition.
        return ""


def _read_handle(handle: Handle | None) -> str:
    if handle is None:
        return ""
    try:
        return handle.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return ""


def _read_spec(state: AgentState) -> str:
    """The spec, found among the context handles by its summary prefix.

    `xeno.cli._converse_to_spec` tags it `spec: <title>` when it seeds the
    handle, which is the only marker distinguishing it from Argus's file
    selections in the same list.
    """
    for handle in state.context_handles:
        if handle.summary.startswith("spec:"):
            return _read_handle(handle)
    return ""
