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

What IS settled here is that a response nobody could parse never disappears.
It is written to the run workspace before this module returns, so whichever
decision the caller then makes — halt, escalate, raise, carry on — the text
behind that decision is still on disk when someone comes to read it.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Protocol, TypeVar

from xeno.core.paths import RunPaths
from xeno.core.runlog import EventKind
from xeno.core.state import AgentState, Handle
from xeno.core.types import NodeRole
from xeno.graph.prompts import FileBlock
from xeno.prompt.assembly import PromptBuilder
from xeno.router.router import Router, RouterResult

#: One corrective nudge, not an open-ended retry loop: CB-1 bounds the graph
#: as a whole, and a format failure that survives a single explicit
#: correction is more likely a capability gap than a one-off slip.
MAX_FORMAT_ATTEMPTS = 2

#: Filename stem for a saved unparseable exchange, shared by the writer and
#: the counter beside it so the two cannot drift on what one is called.
_UNPARSED_STEM = "unparsed"


class ParsedOutput(Protocol):
    """What every `parse_*_output` in `xeno.graph.prompts` returns: a result
    that knows whether it is the harness's own "found no usable tags"
    fallback. Structural rather than a base class, so the parse results stay
    plain frozen dataclasses."""

    @property
    def malformed(self) -> bool: ...

    @property
    def format_hint(self) -> str: ...


_OutputT = TypeVar("_OutputT", bound=ParsedOutput)


def complete_with_format_retry(
    *,
    router: Router,
    builder: PromptBuilder,
    node: NodeRole,
    state: AgentState,
    paths: RunPaths,
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

    An exchange that never parsed is written to `paths.workspace` first and
    pointed at by `state.unparsed_response`, so every one of those decisions
    is made about a response someone can still read afterwards.
    """
    output: _OutputT | None = None
    attempts: list[RouterResult] = []
    for attempt in range(MAX_FORMAT_ATTEMPTS):
        turn_text = current_turn if attempt == 0 else _corrective_turn(correction, output)
        result = router.complete(node, builder.build(turn_text), state=state)
        builder.append_turn("user", turn_text)
        # An empty assistant turn is rejected outright by some providers, and
        # replaying one would poison every later call on this builder.
        builder.append_turn("assistant", result.text or "(the model returned no text)")
        attempts.append(result)

        output = parse(result.text)
        if not output.malformed:
            break

    assert output is not None, "MAX_FORMAT_ATTEMPTS is never zero"
    # Set on failure and CLEARED on success, in one assignment. A pointer that
    # was only ever set would still be naming some earlier node's recovered
    # slip by the time an unrelated halt printed it, and a halt panel naming
    # the wrong file is worse than one naming none.
    state.unparsed_response = (
        _save_unparsed(router, paths, node, attempts) if output.malformed else None
    )
    return output


def _corrective_turn(correction: str, previous: ParsedOutput | None) -> str:
    """The generic correction, plus what was actually wrong when the parser
    can say.

    Without the second half, a model that emitted well-formed tags carrying
    one unreadable attribute is told "no tag was found" — a correction it
    cannot act on, because the thing it is being asked to fix was already
    right. That is how run 20260811T124116 spent its retry.
    """
    hint = previous.format_hint if previous is not None else ""
    return f"{correction}\nSpecifically: {hint}\n" if hint else correction


def _save_unparsed(
    router: Router, paths: RunPaths, node: NodeRole, attempts: Sequence[RouterResult]
) -> Handle:
    """Write down an exchange that never parsed, and say so in the run log.

    Only the FAILED exchanges are written. A response that parsed has already
    been persisted in the form the run actually uses — a roadmap, a plan, a
    diff, a set of notes — so keeping the raw text of every call as well would
    double the workspace to duplicate what is already in it. The unparseable
    ones are the only model output in the harness that otherwise leaves
    nothing behind, which is exactly why they were the only ones nobody could
    debug after the fact.

    Deliberately NOT re-scanned for secrets. Every outbound prompt already
    passed `xeno.security.outbound.sanitize`, so a response echoing the
    codebase map echoes redaction MARKERS rather than credentials, and the
    workspace this lands in already holds verbatim worktree diffs, which are
    strictly more sensitive. Redacting here would mangle the one artifact
    whose whole value is being byte-exact about what the model said.
    """
    # No closure to hang a counter on — this is a free function shared by
    # every node — so the workspace is the counter, and it is the only one
    # that cannot lose count across a node's several calls.
    index = len(list(paths.workspace.glob(f"{_UNPARSED_STEM}_{node.value}_*.txt"))) + 1
    path = paths.workspace / f"{_UNPARSED_STEM}_{node.value}_{index}.txt"

    chunks = [
        f"{node.value}: {len(attempts)} attempt(s), none of which parsed. "
        f"Nothing below was acted on."
    ]
    for number, result in enumerate(attempts, start=1):
        origin = "original turn" if number == 1 else "after the format correction"
        header = f"\n--- attempt {number}/{len(attempts)} ({origin})"
        if result.truncated:
            header += ", CUT OFF at the output token limit"
        elif result.silent:
            header += ", NO TEXT returned"
        chunks.append(f"{header}; finish_reason={result.finish_reason!r} ---\n{result.text}")
        if result.reasoning:
            # The half of a reasoning model's output the parser never sees.
            # Kept because it is usually the only place the model says WHY it
            # answered as it did — and on the run this was built for, it was
            # roughly three quarters of what the call was billed for.
            chunks.append(
                f"\n--- attempt {number} reasoning (never parsed) ---\n{result.reasoning}"
            )
    path.write_text("\n".join(chunks))

    handle = Handle.for_file(
        path, summary=f"{node.value}: unparseable response, {len(attempts)} attempt(s)"
    )
    # Through the router rather than a runlog parameter of this module's own:
    # the router already holds this run's log, and threading a second
    # reference through ten call sites to say one more sentence about a call
    # the router itself just made is the copy-paste this module exists to end.
    router.runlog.event(
        EventKind.RESPONSE_UNPARSED,
        node=node.value,
        attempts=len(attempts),
        path=str(path),
        bytes=handle.bytes,
    )
    return handle


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
