"""The outbound-context choke point (PRD S11.3).

"Before any model call, outbound context passes a secret scanner." One function
does that, and the router calls it on the assembled prompt rather than each
node scanning its own inputs — six scan sites is five chances to forget one.

Redaction is deterministic, which matters more here than it looks: an
identical prompt must redact to identical bytes on every call, or the SYSTEM
and CODEBASE MAP breakpoints would drift and lose their cache hits (PRD T8).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from xeno.core.types import Breakpoint
from xeno.prompt.assembly import AssembledPrompt, Block, Turn
from xeno.security.scanner import Finding, SecretScanner


@dataclass(frozen=True, slots=True)
class SanitizedPrompt:
    prompt: AssembledPrompt
    findings: tuple[Finding, ...]
    #: Which layers contained a redaction. A hit in SYSTEM is a harness bug
    #: worth surfacing loudly — the system prompt is ours, so a credential in
    #: it did not come from the repository.
    breakpoints_hit: frozenset[Breakpoint]

    @property
    def clean(self) -> bool:
        return not self.findings


def sanitize(prompt: AssembledPrompt, scanner: SecretScanner) -> SanitizedPrompt:
    """Redact every layer of an assembled prompt, preserving its structure."""
    findings: list[Finding] = []
    hit: set[Breakpoint] = set()

    blocks: list[Block] = []
    for block in prompt.blocks:
        if not block.text:
            blocks.append(block)
            continue
        result = scanner.scan(block.text)
        if result.findings:
            findings.extend(result.findings)
            hit.add(block.breakpoint)
        blocks.append(replace(block, text=result.text))

    history: list[Turn] = []
    for turn in prompt.history:
        result = scanner.scan(turn.content)
        if result.findings:
            findings.extend(result.findings)
            hit.add(Breakpoint.ACCUMULATED_HISTORY)
        history.append(Turn(role=turn.role, content=result.text))

    sanitized = AssembledPrompt(
        node=prompt.node, blocks=tuple(blocks), history=tuple(history)
    )
    return SanitizedPrompt(
        prompt=sanitized, findings=tuple(findings), breakpoints_hit=frozenset(hit)
    )
