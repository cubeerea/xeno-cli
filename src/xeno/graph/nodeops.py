"""Operations every model-calling node performs identically.

Three things were copy-pasted across `argus`, `odysseus`, `daedalus`,
`chiron`, `cerberus`, and `adapters.discovery` before this module existed:
the one-corrective-retry wire-format loop, the worktree-containment-checked
file write, and the unified-diff computation over what was written. They are
behaviourally identical at every site, and each is load-bearing enough
(prompt-cache turn bookkeeping; path-traversal containment) that six drifting
copies were a real risk rather than a stylistic one.

Nothing here decides policy. Each caller still owns what a failure MEANS for
it — `ChainExhaustedError` and `WorktreeEscape` both propagate, because a
halt reason is node-specific prose (and Cerberus, uniquely, must also set a
verdict) that does not belong in a shared helper.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Protocol, TypeVar

from xeno.core.state import AgentState
from xeno.core.types import NodeRole
from xeno.graph.prompts import FileBlock
from xeno.prompt.assembly import PromptBuilder
from xeno.router.router import Router

#: One corrective nudge, not an open-ended retry loop: CB-1 bounds the graph
#: as a whole, and a format failure that survives a single explicit
#: correction is more likely a capability gap than a one-off slip.
MAX_FORMAT_ATTEMPTS = 2


class ParsedOutput(Protocol):
    """What every `parse_*_output` in `xeno.graph.prompts` returns: a result
    that knows whether it is the harness's own "found no usable tags"
    fallback. Structural rather than a base class, so the parse results stay
    plain frozen dataclasses."""

    @property
    def malformed(self) -> bool: ...


_OutputT = TypeVar("_OutputT", bound=ParsedOutput)


def complete_with_format_retry(
    *,
    router: Router,
    builder: PromptBuilder,
    node: NodeRole,
    state: AgentState,
    current_turn: str,
    correction: str,
    parse: Callable[[str], _OutputT],
) -> _OutputT:
    """Call `node`'s model, re-prompting once with `correction` if the
    response does not parse.

    Every turn is appended to `builder` whether or not it parsed: the
    correction attempt has to see the malformed response it is correcting,
    and the prompt cache's turn history must reflect what was actually sent
    (PRD T8).

    Returns the last parse result, which the caller must still check for
    `malformed` — surviving the corrective retry is a node-specific decision
    (Daedalus halts, Cerberus escalates, discovery raises), not this
    helper's. `ChainExhaustedError` propagates for the same reason.
    """
    output: _OutputT | None = None
    for attempt in range(MAX_FORMAT_ATTEMPTS):
        turn_text = current_turn if attempt == 0 else correction
        result = router.complete(node, builder.build(turn_text), state=state)
        builder.append_turn("user", turn_text)
        builder.append_turn("assistant", result.text)

        output = parse(result.text)
        if not output.malformed:
            break

    assert output is not None, "MAX_FORMAT_ATTEMPTS is never zero"
    return output


class WorktreeEscape(Exception):
    """A model-authored path resolved outside the worktree. Carries the
    offending path so each node can phrase its own halt reason."""

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.path = path


def write_file_blocks(worktree: Path, blocks: Iterable[FileBlock]) -> tuple[list[Path], str]:
    """Write model-authored file blocks into the worktree, returning the
    paths written and a unified diff of the change.

    Escaping the worktree is checked via the resolved path (so `../`-style
    traversal cannot hide behind a symlink), but the UNRESOLVED path is what
    gets written and returned: `worktree` itself may be reached through a
    symlink (e.g. macOS's /tmp -> /private/tmp), and resolving only the
    target while every other path in the codebase stays unresolved would
    silently break every later `relative_to(worktree)` call downstream
    (`gates.py`, `talos.py`'s touched-file scoping).

    Blocks are written in order and `WorktreeEscape` is raised the moment one
    is caught, leaving earlier blocks on disk. That is deliberate: the caller
    halts the run, and a half-written worktree that a human can inspect beats
    a silent rollback of writes the model believed it had made.
    """
    resolved_worktree = worktree.resolve()
    before: dict[Path, str] = {}
    written: list[Path] = []

    for block in blocks:
        target = worktree / block.path
        try:
            target.resolve().relative_to(resolved_worktree)
        except ValueError:
            raise WorktreeEscape(block.path) from None
        before[target] = target.read_text() if target.exists() else ""
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(block.content)
        written.append(target)

    return written, _unified_diff(worktree, written, before)


def _unified_diff(worktree: Path, touched: Sequence[Path], before: dict[Path, str]) -> str:
    chunks: list[str] = []
    for path in touched:
        rel = path.relative_to(worktree).as_posix()
        new = path.read_text() if path.exists() else ""
        diff = difflib.unified_diff(
            before.get(path, "").splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )
        chunks.append("".join(diff))
    return "\n".join(chunks)
