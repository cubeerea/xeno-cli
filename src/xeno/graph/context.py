"""Building the CODEBASE MAP breakpoint (PRD S9.6.1), used by Daedalus and
Chiron alike.

Argus (`xeno.graph.argus`) is what normally selects the focus files; the
PRD's OQ-10 `@file` mechanism, surfaced as `xeno run --file`, lets a user add
to that selection by hand. Either way this module renders the result the same
way — a plain file-tree-plus-contents dump, not a summarized index.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from xeno.prompt.delimit import as_data
from xeno.prompt.keys import DEFAULT_IGNORES

#: Fixed cost of `as_data`'s header, warning line, and footer, excluding the
#: `source=` path. Measured once rather than guessed, so the content budget
#: stays honest as the wrapper's wording changes.
_DATA_BLOCK_OVERHEAD = len(as_data("", label="repository file", source="", truncate_to=0)) + len(
    " truncated=true"
)

#: Directories whose content is skipped by default (still listed in the file
#: tree). PRD OQ-10's manual `@file` scaffold exists precisely so a local
#: model is not handed the whole repository, including its own test suite,
#: as system-prompt filler on every call — a large local model already
#: struggles to hold instructions under a big context; a full-tree content
#: dump made that worse in practice (hence the corrective retry every node
#: shares via `xeno.graph.nodeops.complete_with_format_retry`).
_CONTENT_EXCLUDED_DIRS = frozenset({"tests", "tests_pending", "test"})

#: Which files are worth spending context budget on. Deliberately an
#: allowlist of source-and-config text, not "anything that decodes as UTF-8":
#: the budget is small enough that one stray fixture or generated artifact
#: can crowd out the file the model actually needs.
#:
#: This used to be `.py` alone, which quietly made the whole harness
#: Python-only — `GenericAdapter` and toolchain discovery are language
#: agnostic, but a Rust or TypeScript repo still handed Daedalus and Chiron a
#: bare file tree with zero content, so they were writing blind against every
#: language but one.
_SOURCE_SUFFIXES = frozenset(
    {
        ".py", ".pyi",
        ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".svelte", ".vue",
        ".rs", ".go", ".rb", ".java", ".kt", ".kts", ".scala", ".swift",
        ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".php",
        ".sh", ".bash", ".sql", ".proto", ".graphql",
        ".html", ".css", ".scss",
        ".toml", ".json", ".yaml", ".yml", ".ini", ".cfg", ".md",
    }
)

#: Lockfiles are machine-generated, enormous, and tell the model nothing it
#: cannot read from the manifest beside them. They match `_SOURCE_SUFFIXES`
#: by extension, so without this a `package-lock.json` sorting early in the
#: walk would consume the entire content budget on its own.
_LOCKFILE_NAMES = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "Cargo.lock",
        "Gemfile.lock",
        "composer.lock",
        "go.sum",
        "uv.lock",
    }
)


def _walk(root: Path, ignore: frozenset[str]) -> Iterable[Path]:
    for entry in sorted(root.iterdir()):
        if entry.name in ignore or entry.is_symlink():
            continue
        if entry.is_dir():
            yield from _walk(entry, ignore)
        elif entry.is_file():
            yield entry


#: Cap on the file-tree listing, separately from file CONTENT.
#:
#: The listing was the one part of this block with no bound at all. Content is
#: budgeted, but the tree prints every path in the worktree, so its size is set
#: by file COUNT rather than by anything the budget controls — measured at
#: ~8.9 KB on this repo's 55 source files, and a repository an order of
#: magnitude larger would spend more prompt on a list of paths than the content
#: budget allows for the code itself. Truncation is by whole lines with an
#: explicit count of what was dropped, because a tree silently missing its tail
#: reads exactly like a project that does not contain those files.
DEFAULT_MAX_TREE_BYTES = 6_000


def focus_paths(handles: Iterable[Path], written: Iterable[Path]) -> list[Path]:
    """What a writing node should see the contents of: Argus's selection plus
    everything this run has already written.

    The second half is not a nicety. `focus` is a filter, so a non-empty
    selection that omits a file makes that file's content invisible — and
    Argus selects for RESEARCH, before the write exists. Daedalus would write
    `src/foo.py`, come back on the next ladder rung, and find it gone from the
    map, having been narrowed out by a selection made before it existed.

    That was survivable while the model's own response still carried the file
    verbatim in history. `condense_file_blocks` removes that second copy on
    the grounds that the map holds the first, so this is what makes the
    grounds true.
    """
    ordered = {p.resolve(): p for p in handles}
    ordered.update({p.resolve(): p for p in written})
    return list(ordered.values())


def _tree_lines(files: Sequence[Path], worktree: Path, budget: int) -> list[str]:
    """The file-tree listing, capped by bytes and truncated by whole lines.

    Says how many paths it dropped rather than trailing off. A node that reads
    a short tree and concludes the project has no `tests/` directory will act
    on that, and the acting is the expensive part.
    """
    rendered: list[str] = []
    spent = 0
    for index, path in enumerate(files):
        line = f"  {path.relative_to(worktree).as_posix()}"
        if spent + len(line) + 1 > budget:
            return [*rendered, f"  ... {len(files) - index} more path(s) not listed"]
        rendered.append(line)
        spent += len(line) + 1
    return rendered


def build_codebase_map(
    worktree: Path,
    *,
    focus: Sequence[Path] | None = None,
    max_content_bytes: int = 16_000,
    max_tree_bytes: int = DEFAULT_MAX_TREE_BYTES,
) -> str:
    """A file tree plus file content, budget-capped.

    `focus` is whatever narrowed the context — Argus's own file selection or
    the user's `--file` (PRD OQ-10). When given, ONLY those files' content is
    shown —
    the tree still lists everything else for orientation, but a user who
    named the relevant file(s) has already done Argus's job, and dumping the
    rest of the tree's content on top only spends context for no benefit.

    Without `focus`, every source or config file is a candidate (see
    `_SOURCE_SUFFIXES`), but content under a test directory is skipped (still
    listed in the tree) since neither Daedalus nor Chiron can run tests
    anyway, and the acceptance tests for the task at hand would otherwise be
    handed to them as an answer key alongside the codebase.
    """
    files = list(_walk(worktree, DEFAULT_IGNORES))
    lines = ["Repository file tree:"]
    lines.extend(_tree_lines(files, worktree, max_tree_bytes))
    lines.append("")
    lines.append("File contents:")

    focus_resolved = {p.resolve() for p in focus} if focus else None

    budget = max_content_bytes
    for f in files:
        if budget <= 0:
            continue
        if focus_resolved is not None:
            # An explicit pick — Argus's or the user's `--file` — overrides
            # the suffix allowlist: naming a file IS the argument for
            # showing it, whatever it is called.
            if f.resolve() not in focus_resolved:
                continue
        else:
            if f.suffix not in _SOURCE_SUFFIXES or f.name in _LOCKFILE_NAMES:
                continue
            if any(part in _CONTENT_EXCLUDED_DIRS for part in f.relative_to(worktree).parts[:-1]):
                continue
        try:
            text = f.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        rel = f.relative_to(worktree).as_posix()
        # PRD S11.4: repository files are untrusted input. Every retrieved
        # file goes into the prompt inside a labelled, guarded DATA block, so
        # a file containing text aimed at whichever node reads it has been
        # explicitly framed as content rather than instruction. The guard is
        # derived from the content (not random), so an unchanged file still
        # produces byte-identical CODEBASE MAP text and keeps its cache hit.
        room = budget - _DATA_BLOCK_OVERHEAD - len(rel)
        if room <= 0:
            continue
        block = "\n" + as_data(text, label="repository file", source=rel, truncate_to=room)
        lines.append(block)
        budget -= len(block)
    return "\n".join(lines)
