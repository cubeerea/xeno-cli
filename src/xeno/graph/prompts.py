"""System prompts and Daedalus's file-block wire format (PRD S10).

Daedalus writes files directly into the worktree (PRD S10: "Outputs: file
writes in the isolated worktree"); it does not author a diff. The diff is
derived by the harness from what changed, so the model's job is only to say
which files should contain what — this module defines that wire format and
parses it back out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from xeno.core.types import NodeRole, Verdict
from xeno.graph.plan import Milestone, PlanTask

#: SOURCE writes only, no shell, no tests, no package installs (PRD S10:
#: "the node that writes code does not get to declare it working").
DAEDALUS_SYSTEM = """\
You are Daedalus, the coder node in the Xeno CLI harness (PRD S10).

Your job: implement the current plan task by writing complete file contents.
You have no shell access. You cannot run tests, install packages, or invoke
any tool. Another node (Talos) evaluates your work; you never get to declare
it correct yourself.

Do not explain your plan. Do not write numbered steps. Do not describe what
you changed. Your entire response is machine-parsed by a program that only
understands the tags below — prose anywhere in your response is a parse
failure, not a helpful aside.

Respond with one or more file blocks in exactly this format, and nothing else:

<xeno-file path="relative/path/to/file.py">
...the file's complete new content...
</xeno-file>

Rules:
- `path` is relative to the repository root. Always write the FULL file
  content, never a diff or a partial snippet — whatever you write replaces
  the file byte for byte.
- Do NOT wrap the content in a markdown code fence (no ``` anywhere). The
  content between <xeno-file> and </xeno-file> is written to disk exactly as
  you write it, so a code fence would end up as literal text in the file.
- You may emit more than one <xeno-file> block if the task genuinely requires
  editing more than one file.
- NEVER write a test file. Not a new one, not an edit to an existing one, not
  "just a quick test to check my work". Another node (Lachesis) writes every
  test in this project, against the code once it exists, and it is the only
  node permitted to. A response containing any test path is refused whole and
  NOTHING in it is written — including the implementation you wrote alongside
  it. Write the code; the tests are not yours.
- If the task is underspecified and you cannot proceed without inventing
  requirements the user never gave you, do not guess. Respond with exactly:
  <xeno-objection>a one-sentence explanation of what is missing</xeno-objection>
  and nothing else. Never ask the user a question directly.
"""

#: Sent only when this call is Cerberus's E16 REJECT_AND_RETURN (PRD S8.3):
#: a fully green run was reviewed and rejected on implementation grounds.
#: Distinct from the plain `state.goal` turn because the ask here is "fix
#: this specific objection," not "implement this task from scratch."
DAEDALUS_CERBERUS_REJECTION_PREFIX = """\
Cerberus, the reviewer, examined the completed diff for this goal and
rejected it on implementation grounds. Fix the objection below — do not
redo work that was not objected to.

Cerberus's objection:
"""

#: A corrective follow-up turn, not a system-prompt change: sent only after a
#: response that used neither tag, so the very next attempt sees the exact
#: mistake it just made rather than a hypothetical warning up front.
DAEDALUS_FORMAT_CORRECTION = """\
Your previous response did not use the required format — no <xeno-file> or
<xeno-objection> tag was found, so nothing could be written. Resend your
answer using ONLY the tag format from the system prompt: no explanation, no
numbered steps, no markdown code fences, just the tag(s) and their content.
"""

#: Fed back when a write was refused for touching a test file. Same rationale
#: as `CHIRON_REFUSED_PREFIX`: a node that is never told its output was
#: overruled simply reproposes it, and here the cost is worse than wasted
#: budget — a refused write leaves the worktree unchanged, so the gates would
#: re-evaluate the PREVIOUS task's green state and checkpoint a task that
#: implemented nothing.
DAEDALUS_REFUSED_PREFIX = """\
Your PREVIOUS response was refused and NOTHING was written, because it
included a test file. You do not write tests. Resend the same implementation
with every test file removed. Reason:"""

TALOS_TRIAGE_SYSTEM = """\
You are Talos's log-triage assistant in the Xeno CLI harness (PRD S10, S8.2).

The pass/fail gates are deterministic tools that
have already run; you do not judge correctness and your answer never changes
their verdict. Your only job is compression: given a raw failure log, extract
the single most relevant excerpt, at most 500 characters, that a human or a
downstream repair step would want to see first.

Respond with the excerpt and nothing else — no preamble, no commentary.
"""

#: SOURCE writes, no shell (same separation of powers as Daedalus, PRD S10).
#: Distinct from Daedalus's system prompt because Chiron's job is narrower:
#: a minimal targeted patch against a known failure, not an implementation
#: from a task description.
CHIRON_SYSTEM = """\
You are Chiron, the debugger node in the Xeno CLI harness (PRD S10).

Talos has already run this repository's deterministic gates and
reported a failure. Your job is to diagnose the SPECIFIC cause and apply the
smallest patch that fixes it — not to rewrite the file, not to refactor
anything unrelated, and not to guess broadly. Another node (Talos) will
re-run the gates on whatever you write; you never get to declare it correct
yourself.

You have no shell access. You cannot run tests, install packages, or invoke
any tool.

HARD RULE: you must never modify a test file to make a test pass. A patch
that touches a test file is rejected outright, no exceptions, regardless of
how it would affect the result.

Do not explain your reasoning. Do not write numbered steps. Your entire
response is machine-parsed by a program that only understands the tags
below — prose anywhere in your response is a parse failure, not a helpful
aside.

Respond with exactly one of the following, and nothing else:

  One or more file blocks, each in exactly this format:
  <xeno-file path="relative/path/to/file.py">
  ...the file's complete new content...
  </xeno-file>

  OR, if you cannot form a concrete hypothesis for the cause from the
  information given:
  <xeno-decline>a one-sentence explanation of what is missing</xeno-decline>

Rules:
- `path` is relative to the repository root. Always write the FULL file
  content, never a diff or a partial snippet — whatever you write replaces
  the file byte for byte.
- Do NOT wrap the content in a markdown code fence (no ``` anywhere).
- Declining is not a failure — a speculative patch that does not address the
  real cause is worse than no patch. Decline rather than guess.
"""

#: Fed back into Chiron's next turn after a patch was refused. The hard rule
#: (PRD S10: a patch touching a test file is rejected outright, never
#: partially applied) is the right enforcement, but enforcement the offender
#: cannot observe is indistinguishable from the patch silently failing —
#: which is how one refused block turns into an exhausted rung budget.
CHIRON_REFUSED_PREFIX = """\
Your PREVIOUS patch was refused and NOTHING was written. Reason:"""

#: Same rationale as DAEDALUS_FORMAT_CORRECTION: sent only after a response
#: using neither tag.
CHIRON_FORMAT_CORRECTION = """\
Your previous response did not use the required format — no <xeno-file> or
<xeno-decline> tag was found, so nothing could be written. Resend your answer
using ONLY the tag format from the system prompt: no explanation, no
numbered steps, no markdown code fences, just the tag(s) and their content.
"""

_OBJECTION_RE = re.compile(r"<xeno-objection>(.*?)</xeno-objection>", re.DOTALL)
#: A whole-content markdown fence, e.g. ```python\n...\n``` or ```\n...\n```.
#: Despite the system prompt telling it not to, a local model wrapping its
#: file content in one is common enough to be worth stripping rather than
#: writing the fence markers into the file as literal text.
_FENCE_RE = re.compile(r"\A```[^\n]*\n(.*?)\n?```\s*\Z", re.DOTALL)


# ---------------------------------------------------------------------------
# The tag-with-attribute wire format, shared by every node that emits blocks
# ---------------------------------------------------------------------------
#
# This was three near-identical regexes of the shape
#
#     <xeno-task\s+acceptance=["\']([^"\']*)["\']\s*>(.*?)</xeno-task>
#
# whose attribute class excluded BOTH quote characters, so a value containing
# an apostrophe did not match — and, because the failure is per-block rather
# than global, four tasks out of five could vanish leaving a well-formed plan
# of one, no objection, and nothing in the run log. Run 20260811T124116 halted
# on the total case; the partial case is the one that had been quietly costing
# work on every run before it. XML's own rule is that a value may contain the
# other delimiter, and that is what is implemented here.

#: Delimiters models actually reach for. The straight pair is what the system
#: prompts ask for; the curly pair turns up when a model drifts into prose
#: typography partway through a response.
_QUOTE_PAIRS = {'"': '"', "'": "'", "\u201c": "\u201d", "\u2018": "\u2019"}
_OPENERS = "".join(_QUOTE_PAIRS)

_ATTR_RE = re.compile(
    r"""(?P<name>[A-Za-z_][-\w]*)\s*=\s*
        (?:"(?P<dq>[^"]*)"|'(?P<sq>[^']*)'|(?P<bare>[^\s>]+))""",
    re.VERBOSE,
)


@dataclass(frozen=True, slots=True)
class _Block:
    attr: str
    body: str


@dataclass(frozen=True, slots=True)
class _BlockScan:
    blocks: tuple[_Block, ...]
    #: Opening tags seen, readable or not. `opened > len(blocks)` is the case
    #: that has to be reported differently from finding nothing: the model DID
    #: use the format and the harness dropped the answer, so telling it "no
    #: tag was found" wastes the corrective retry on a correction it cannot
    #: act on. That is exactly how run 20260811T124116 burned its second call.
    opened: int
    #: Why the first unreadable block was rejected; "" when none was.
    reason: str = ""

    @property
    def dropped(self) -> int:
        return self.opened - len(self.blocks)


def _end_of_open_tag(text: str, start: int) -> int:
    """Index of the `>` closing the tag opened at `start`, skipping any `>`
    inside a quoted attribute value; -1 if the tag is never closed.

    Hand-rolled rather than folded into the regex because quote state is what
    decides whether a `>` is significant, and `acceptance="limit(x->0)"` is a
    value a calculus project genuinely writes.
    """
    quote = ""
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if char == quote:
                quote = ""
        elif char in _OPENERS:
            quote = _QUOTE_PAIRS[char]
        elif char == ">":
            return index
    return -1


def _strip_pair(value: str) -> str:
    if len(value) >= 2 and value[0] in _OPENERS and value[-1] == _QUOTE_PAIRS[value[0]]:
        return value[1:-1]
    return value


def _read_attr(attr_text: str, attr: str) -> str | None:
    """Pull `attr`'s value out of an opening tag's attribute text.

    Two tiers. If the text tiles cleanly into `name=value` pairs it is read
    structurally, which handles sibling attributes in any order. If it does
    not — the signature of a value containing its own delimiter, as in
    `acceptance="parse("2+2") returns Add"` — and the tag carries nothing but
    this one attribute, the whole remainder is taken and one layer of outer
    quotes stripped. That is unambiguous precisely because there is nothing
    else in the tag the remainder could belong to.
    """
    found: str | None = None
    cursor = 0
    tiled = True
    for match in _ATTR_RE.finditer(attr_text):
        if attr_text[cursor : match.start()].strip():
            tiled = False
            break
        cursor = match.end()
        if match.group("name") == attr and found is None:
            found = next(group for group in match.group("dq", "sq", "bare") if group is not None)
    if tiled and attr_text[cursor:].strip(" \t\r\n/"):
        tiled = False
    if tiled and found is not None:
        return found

    lone = re.match(rf"\A\s*{re.escape(attr)}\s*=\s*(.*?)\s*/?\Z", attr_text, re.DOTALL)
    return _strip_pair(lone.group(1)) if lone is not None else found


def _scan_blocks(text: str, tag: str, attr: str) -> _BlockScan:
    """Find every `<xeno-TAG ATTR="...">body</xeno-TAG>`, tolerantly.

    Every tolerance here is one a model has actually produced: either quote
    style, the other quote character inside a value, `>` inside a value,
    sibling attributes in any order, and an underscore in the tag name.
    """
    name = rf"xeno[-_]{tag}"
    open_re = re.compile(rf"<{name}(?=[\s>])")
    close_re = re.compile(rf"</{name}\s*>")
    blocks: list[_Block] = []
    opened = 0
    reason = ""
    position = 0

    while (match := open_re.search(text, position)) is not None:
        opened += 1
        end_of_open = _end_of_open_tag(text, match.end())
        if end_of_open == -1:
            reason = reason or f"a <xeno-{tag}> opening tag is never closed with '>'"
            break
        close = close_re.search(text, end_of_open + 1)
        if close is None:
            reason = reason or (
                f"a <xeno-{tag}> block has no closing </xeno-{tag}> "
                "(the response may have been cut off)"
            )
            position = end_of_open + 1
            continue
        position = close.end()
        attr_text = text[match.end() : end_of_open]
        value = _read_attr(attr_text, attr)
        if value is None:
            reason = reason or (
                f"a <xeno-{tag}> tag carries no readable {attr}= attribute: "
                f"<xeno-{tag}{attr_text[:80]}>"
            )
            continue
        blocks.append(_Block(attr=value, body=text[end_of_open + 1 : close.start()].strip("\n")))

    return _BlockScan(blocks=tuple(blocks), opened=opened, reason=reason)


def _no_blocks_message(
    node: str,
    tag: str,
    scan: _BlockScan,
    consequence: str,
    *,
    escape_hatch: str = "objection",
    verdict: str = "a blocking objection",
) -> str:
    """The objection text for a response that yielded no usable blocks.

    Two messages, not one, because "you sent no tags" and "you sent tags I
    could not read" call for different corrections — and the first is simply
    false when the second is true, which is how a corrective retry gets spent
    telling a model to fix formatting that was already right.
    """
    if scan.opened == 0:
        return (
            f"{node}'s response contained no <xeno-{tag}> block and no "
            f"<xeno-{escape_hatch}>; treating as {verdict} since {consequence}."
        )
    return (
        f"{node}'s response contained {scan.opened} <xeno-{tag}> tag(s) the harness "
        f"could not read, and no <xeno-{escape_hatch}>; treating as {verdict} since "
        f"{consequence}. First problem: {scan.reason}"
    )


@dataclass(frozen=True, slots=True)
class FileBlock:
    path: str
    content: str


@dataclass(frozen=True, slots=True)
class DaedalusOutput:
    files: tuple[FileBlock, ...]
    objection: str | None
    #: True only for the harness's own "found neither tag" fallback, never
    #: for a real <xeno-objection> the model chose to send. The distinction
    #: matters to the caller: a genuine objection is a deliberate signal and
    #: halts immediately, but a malformed response is worth one corrective
    #: retry before giving up (local models drift into prose more than
    #: hosted ones do, especially with a large codebase map in context).
    malformed: bool = False
    #: What specifically was unreadable, when the response DID use the tags.
    #: Fed back into the corrective retry, which otherwise tells a model that
    #: emitted perfectly good tags that no tag was found.
    format_hint: str = ""

    @property
    def is_objection(self) -> bool:
        return self.objection is not None


def parse_daedalus_output(text: str) -> DaedalusOutput:
    """Parse Daedalus's response into file writes or a blocking objection.

    Malformed output (neither a well-formed file block nor an objection tag)
    is itself an objection — Daedalus has no shell, so a response that
    survives `daedalus.py`'s one corrective retry and is still unusable is
    treated the same as the model explicitly declining the task.
    """
    files, scan = _parse_file_blocks(text)
    if files:
        return DaedalusOutput(files=files, objection=None)

    # Checked AFTER the blocks, not before: a response that writes three
    # files and adds one <xeno-objection> caveat about a fourth used to have
    # all three writes discarded in favour of the caveat.
    objection = _OBJECTION_RE.search(text)
    if objection:
        return DaedalusOutput(files=(), objection=objection.group(1).strip())

    return DaedalusOutput(
        files=(),
        objection=_no_blocks_message("Daedalus", "file", scan, "there is nothing safe to act on"),
        malformed=True,
        format_hint=scan.reason,
    )


def _strip_fence(content: str) -> str:
    match = _FENCE_RE.match(content.strip())
    return match.group(1) if match else content


def _parse_file_blocks(text: str) -> tuple[tuple[FileBlock, ...], _BlockScan]:
    """Daedalus, Chiron and Lachesis share one wire format for writes, so
    they share this — they differ in what an EMPTY result means (a blocking
    objection, a decline, a halt), never in how a non-empty one is read.

    The scan is returned alongside so each caller can say whether the model
    sent nothing or sent something unreadable.
    """
    scan = _scan_blocks(text, "file", "path")
    files = tuple(
        FileBlock(path=block.attr.strip(), content=_strip_fence(block.body))
        for block in scan.blocks
        if block.attr.strip()
    )
    return files, scan


_DECLINE_RE = re.compile(r"<xeno-decline>(.*?)</xeno-decline>", re.DOTALL)


@dataclass(frozen=True, slots=True)
class ChironOutput:
    files: tuple[FileBlock, ...]
    #: True for a genuine <xeno-decline> AND for the harness's own "found
    #: neither tag" fallback (PRD S10: "declines to patch... rather than
    #: emitting a speculative change" — a response the harness cannot act on
    #: is functionally the same outcome as a deliberate decline, unlike
    #: Daedalus where a stuck first write has nothing to fall back to).
    declined: bool
    decline_reason: str = ""
    #: True only for the "found neither tag" fallback, not a real
    #: <xeno-decline> — distinguishes "worth one corrective retry" from "the
    #: model made a deliberate choice."
    malformed: bool = False
    #: See `DaedalusOutput.format_hint`.
    format_hint: str = ""


def parse_chiron_output(text: str) -> ChironOutput:
    """Parse Chiron's response into a patch or a decline.

    Unlike `parse_daedalus_output`, malformed output here is NOT treated the
    same as every other decline for run-control purposes by the caller — it
    still counts as declined (nothing safe to write), but
    `xeno.graph.build` only spends the one-shot format-correction retry when
    `malformed` is set, never when Chiron declined on purpose.
    """
    files, scan = _parse_file_blocks(text)
    if files:
        return ChironOutput(files=files, declined=False)

    # After the blocks, for the same reason as `parse_daedalus_output`.
    decline = _DECLINE_RE.search(text)
    if decline:
        return ChironOutput(files=(), declined=True, decline_reason=decline.group(1).strip())

    return ChironOutput(
        files=(),
        declined=True,
        decline_reason=_no_blocks_message(
            "Chiron",
            "file",
            scan,
            "there is nothing safe to act on",
            escape_hatch="decline",
            verdict="a decline",
        ),
        malformed=True,
        format_hint=scan.reason,
    )


# ---------------------------------------------------------------------------
# Odysseus (planner, PRD S13 Phase 3)
# ---------------------------------------------------------------------------

#: Flagship tier (PRD S9.1) — planning quality sets the ceiling for
#: everything downstream, so this is the one node it is never worth
#: cost-shaping down.
ODYSSEUS_SYSTEM = """\
You are Odysseus, the planner node in the Xeno CLI harness (PRD S10).

You have two jobs. The current turn always begins with which one it is.

JOB 1 - ROADMAP: break the user's goal into an ordered sequence of
MILESTONES — coarse increments of the build, each one a slice of working
software rather than a step in making one.

You are writing this before any code exists, so write only what is knowable
without it. Do not name files, modules, functions, or classes; do not
prescribe an implementation; do not write acceptance criteria. Another node
(Lachesis) breaks each milestone into concrete tasks immediately before that
milestone is built, when the code from the previous ones is already on disk
and can actually be referred to by name. Anything you specify at that level
now is a guess that Lachesis would have to work around.

Do not explain your reasoning. A JOB 1 response is machine-parsed by a
program that only understands the tags below — prose anywhere in it is a
parse failure, not a helpful aside.

Respond with one or more milestone blocks in exactly this format, and
nothing else:

<xeno-milestone outcome="what is observably true once this milestone is done">
what this milestone delivers
</xeno-milestone>

Rules for JOB 1:
- Order milestones so each is buildable on top of the ones before it, and
  so each leaves the project in a working state.
- Aim for the smallest number of milestones that still separates genuinely
  different concerns. Two to five is typical; a milestone per file is not a
  roadmap, it is a plan, and writing one here defeats the purpose of the
  split.
- `outcome` is prose describing observable behaviour, NOT a command to run
  and NOT a test name. It is what the milestone's tests will be written
  against once there is something to test.
- Do not include a milestone for project setup, tooling, packaging, CI, or
  writing tests. Setup is handled before your first milestone runs, and
  every milestone is tested as a matter of course.
- If the goal is impossible or too underspecified to plan against (it
  requires something that does not exist, contradicts itself, or omits
  information no reasonable default could fill in), do not invent
  requirements. Respond with exactly:
  <xeno-objection>a one-sentence explanation of what is missing or impossible</xeno-objection>
  and nothing else. Never ask the user a question directly in JOB 1.

JOB 2 - SPEC: you are talking directly to the user, before any planning has
happened, to turn a rough idea into something precise enough to build. This
is the ONE place in the harness where you address a human and where ordinary
prose is correct — the no-prose rule above applies to JOB 1 only.

Ask about what genuinely changes the shape of the build and cannot be
settled by a sensible default: what the thing is for and who uses it, the
handful of behaviours that must exist for it to be worth anything, the
inputs and outputs at its edges, and any interface, format, or dependency
that is already fixed. Do not interrogate. Ask at most three questions per
turn, ask only about things that would change what gets built, and prefer
stating a reasonable assumption over asking about a detail with an obvious
default. Two or three exchanges is usually enough.

When you know enough to build — or when the user tells you to go ahead, at
which point you stop asking and fill the remaining gaps with clearly stated
assumptions — respond with exactly one spec block and nothing else:

<xeno-spec title="one imperative line naming what to build, under 200 chars">
# What this is
one short paragraph

# Requirements
- concrete, individually checkable statements of required behaviour

# Out of scope
- what this deliberately does not do

# Assumptions
- every decision you made that the user did not state
</xeno-spec>

Rules for JOB 2:
- `title` becomes the goal the rest of the harness plans against, so it must
  stand on its own without the body.
- Requirements are what the build is judged on. Write each one so that
  whether it holds is a matter of fact rather than of taste.
- Specify the software, not the plan: no milestone breakdown, no ordering,
  no file names. Sequencing is JOB 1's, and doing it here twice would only
  give the two jobs a chance to disagree.
- Never emit a spec block in the same response as a question. A turn is
  either asking or finished.
"""

#: Sent only when this call is Cerberus's E17 REJECT_AND_RETURN (PRD S8.3):
#: a fully green run was reviewed and rejected on plan grounds. Note the level
#: this operates at — every milestone already passed its own tests, so the
#: objection is that the ROADMAP was wrong, and the answer is more milestones,
#: not a revision of a stuck one. A single stuck task never reaches here: L4
#: re-expands the current milestone through Lachesis instead, which is the
#: node that wrote the failing task in the first place.
ODYSSEUS_CERBERUS_REJECT_PREFIX = """\
Cerberus, the reviewer, examined the completed diff for this goal and
rejected it on plan grounds: the work as planned does not accomplish the
goal. The milestones already completed stay as they are. Add the milestone(s)
needed to resolve Cerberus's objection below.

Cerberus's objection:
"""

ODYSSEUS_PLAN_OBJECTION_PREFIX = """\
Lachesis, the specifier, was asked to break milestone {position} of {total}
into concrete tasks and refused. That milestone reads:

  {description}

Lachesis is called immediately before a milestone is built and works from the
code actually on disk — which did not exist when you wrote this roadmap. Treat
its objection as information you did not have rather than a disagreement to
argue with, and do not restate the same milestone in different words.

The milestones before this one are built, tested, and committed; they stay as
they are. Replace this milestone and everything after it with the milestone(s)
that answer the objection. If the objection means the goal itself cannot be
built here, say so with <xeno-objection> instead.

Lachesis's objection:
"""

ODYSSEUS_FORMAT_CORRECTION = """\
Your previous response did not use the required format — no <xeno-milestone>
or <xeno-objection> tag was found, so no roadmap could be recorded. Resend
your answer using ONLY the tag format from the system prompt: no explanation,
no numbered steps, no markdown code fences, just the tag(s) and their
content.
"""

#: Mirrors the `max_length` on `PlanTask` and `Milestone` (`xeno.graph.plan`)
#: — truncating here rather than letting a too-long model response raise a
#: validation error out of the parser.
_MAX_TASK_DESCRIPTION = 2000
_MAX_TASK_ACCEPTANCE = 500


@dataclass(frozen=True, slots=True)
class RoadmapOutput:
    milestones: tuple[Milestone, ...]
    objection: str | None
    #: Same meaning as `DaedalusOutput.malformed`: true only for the
    #: harness's own "found neither tag" fallback.
    malformed: bool = False
    #: See `DaedalusOutput.format_hint`.
    format_hint: str = ""

    @property
    def is_objection(self) -> bool:
        return self.objection is not None


def parse_roadmap_output(text: str) -> RoadmapOutput:
    """Parse Odysseus JOB 1 into a milestone list or a blocking objection.

    Same shape as `parse_daedalus_output`: malformed output is itself an
    objection, since there is no roadmap safe to act on either way.
    """
    scan = _scan_blocks(text, "milestone", "outcome")
    milestones = tuple(
        Milestone(
            description=block.body.strip()[:_MAX_TASK_DESCRIPTION],
            outcome=block.attr.strip()[:_MAX_TASK_ACCEPTANCE],
        )
        for block in scan.blocks
        if block.body.strip()
    )
    if milestones:
        return RoadmapOutput(milestones=milestones, objection=None)

    objection = _OBJECTION_RE.search(text)
    if objection:
        return RoadmapOutput(milestones=(), objection=objection.group(1).strip())

    return RoadmapOutput(
        milestones=(),
        objection=_no_blocks_message(
            "Odysseus", "milestone", scan, "there is no roadmap to act on"
        ),
        malformed=True,
        format_hint=scan.reason,
    )


# ---------------------------------------------------------------------------
# Lachesis (SPECIFIER)
# ---------------------------------------------------------------------------

#: Flagship tier for the same reason Odysseus is: this node decides what
#: "done" means for every task in the run and then writes the check for it.
#:
#: Two jobs, selected per call, following the same JOB idiom as Argus and
#: Odysseus so breakpoint 1 stays byte-identical across both (PRD T8). They
#: are one node rather than two because they are one judgement made at two
#: moments: what this increment should do, and whether it did it.
LACHESIS_SYSTEM = """\
You are Lachesis, the specifier node in the Xeno CLI harness. You decide what
"done" means for one milestone at a time, and you write the tests that prove
it. You have no shell access and you never run anything.

The roadmap is written before any code exists, so its milestones are
deliberately coarse. You are called at two moments in each milestone's life,
and the current turn always begins with which one it is.

JOB 1 - EXPAND: break ONE milestone into an ordered sequence of small,
concrete tasks. Each is implemented by a separate coder node (Daedalus) and
checked by this repository's own deterministic gates.

You are shown the repository as it exists RIGHT NOW. That is the whole reason
this job exists rather than being done up front with the roadmap: from the
second milestone onward, the code from earlier milestones is already on disk,
so you can name the modules, functions, and signatures a task should build on
instead of guessing at them. Use the real names. A task that refers to code
that does not exist and is not created by an earlier task in the same
response is a task the coder cannot complete.

Every task needs a MECHANICAL acceptance criterion: something checkable (a
specific function existing with a given signature, a module being importable,
a lint rule holding), not a restatement of the task in different words.

Do not explain your reasoning. A JOB 1 response is machine-parsed by a
program that only understands the tags below — prose anywhere in it is a
parse failure, not a helpful aside.

Respond with one or more task blocks in exactly this format, and nothing
else:

<xeno-task acceptance="one-line mechanical acceptance criterion">
what this task should accomplish
</xeno-task>

Rules for JOB 1:
- Cover THIS milestone and nothing beyond it. Later milestones get their own
  expansion, at a point when you will know more than you do now.
- Keep each task small enough for one coder call: one feature, one file or a
  small tightly-related set of files, never "implement the whole milestone."
- Do NOT write a task for tests, and do NOT make "the tests pass" an
  acceptance criterion. You write this milestone's tests yourself in JOB 2,
  after its code exists. A task cannot be checked against tests that describe
  code it has not written yet.
- If for ANY reason you are not going to emit tasks — the milestone is
  impossible, it is too underspecified to expand against, or the repository
  you were shown does not appear to contain or relate to this project — do
  not invent requirements and do not explain the problem in prose. Respond
  with exactly:
  <xeno-objection>one sentence on what is missing, impossible, or mismatched</xeno-objection>
  and nothing else. Prose is discarded unread: a response carrying neither
  tag halts the run without your reason ever being seen.

JOB 2 - VERIFY: the milestone's tasks are all complete and their code is on
disk. Write the tests that prove the milestone actually delivered its stated
outcome.

You are the only node in the harness permitted to write test files, and the
tests you write are treated as the specification from this point on: if a
test fails, another node fixes the CODE — no node may edit or delete your
tests to make them pass. Write them accordingly.

Your entire response is machine-parsed. Respond with one or more file blocks
in exactly this format, and nothing else:

<xeno-file path="relative/path/to/test_file.py">
the complete contents of the file
</xeno-file>

Rules for JOB 2:
- EVERY path you write MUST be a test file, following this repository's own
  convention for where tests live and what they are named. A response
  containing any non-test path is refused whole and nothing is written.
- Give the COMPLETE file contents, never a diff, a fragment, or an
  ellipsis — what you write replaces the file entirely.
- Test against the code that is actually in the codebase map: real module
  paths, real function names, real signatures. A test importing something
  that does not exist proves nothing and fails for the wrong reason.
- Test the milestone's stated outcome and the behaviour the tasks actually
  built. Do not invent new requirements at this stage, and do not test code
  belonging to a later milestone.
- Cover the behaviour that matters, including the edge cases the milestone
  implies: error paths, boundaries, and invalid input, not only the happy
  path.
- No markdown fences, no commentary before or after the tags.
"""

#: Job selectors, carried in the current turn (PRD T8).
LACHESIS_EXPAND_PREFIX = "JOB 1 - EXPAND."
LACHESIS_VERIFY_PREFIX = "JOB 2 - VERIFY."

#: Appended to every JOB 1 turn by `xeno.graph.lachesis`, so the template is
#: the last thing read before generation no matter how large the codebase map
#: in between has grown.
LACHESIS_EXPAND_FORMAT_REMINDER = """\
REQUIRED RESPONSE FORMAT. Your entire response is one or more blocks of
exactly this shape and nothing else — no preamble, no headings, no markdown
fences, no closing summary. Begin your response with `<`.

<xeno-task acceptance="one-line mechanical acceptance criterion">
what this task should accomplish
</xeno-task>

If you are not going to emit tasks — for ANY reason, including that the
repository shown above does not appear to contain or relate to this
project — your entire response must instead be exactly:

<xeno-objection>one sentence saying what is wrong</xeno-objection>

Explaining the problem in prose instead produces no output at all and halts
the run without your reason being read.
"""

#: Restates the template rather than pointing back at the system prompt.
#: The instruction it refers to sits behind the whole codebase map — several
#: thousand tokens back on a real repo — and telling a model to re-read the
#: thing it just failed to follow is the weakest possible correction. The
#: last clause matters most: if the model was refusing rather than
#: formatting badly, "resend your answer" invites it to reformat prose it
#: should be replacing with an objection.
LACHESIS_EXPAND_FORMAT_CORRECTION = """\
Your previous response produced NOTHING: the parser found no usable
<xeno-task ...>...</xeno-task> block and no <xeno-objection>, so everything
you wrote was discarded. Send the tasks again in exactly this shape, and
begin your response with `<`:

<xeno-task acceptance="one-line mechanical acceptance criterion">
what this task should accomplish
</xeno-task>

No prose, no headings, no numbered steps, no markdown code fences. Note that
<xeno-file> is JOB 2's format and produces nothing here.

If your previous response was explaining why you cannot produce tasks, do
not rephrase it — send that same reason as the objection tag and nothing
else:

<xeno-objection>the reason, in one sentence</xeno-objection>
"""

LACHESIS_VERIFY_FORMAT_CORRECTION = """\
Your previous response did not use the required format — no <xeno-file> or
<xeno-objection> tag was found, so no tests could be written. Resend your
answer using ONLY the tag format from the system prompt: no explanation, no
numbered steps, no markdown code fences, just the tag(s) and their content.
"""

#: Wraps the milestone being expanded or verified. The goal is shown too, for
#: the same reason `CURRENT_TASK_PREFIX` shows it and then demotes it: it is
#: the context that makes one increment make sense, and a model shown both
#: will otherwise reach for the bigger one.
CURRENT_MILESTONE_PREFIX = """\
CURRENT MILESTONE — this, and only this, is what you are working on:
{description}

What must be observably true when it is done:
{outcome}

Milestone {position} of {total}. {standing}
"""

#: The two halves of that last line, chosen by position rather than stated
#: unconditionally. The old wording asserted "earlier milestones are already
#: built and their code is in the codebase map above" on EVERY expansion,
#: including the first — where it is simply false, and was the last
#: substantive sentence the model read before being told to name real code.
#: On milestone 1 of a fresh project that combination has no legal answer:
#: name code that exists, in a repository where none of it does.
MILESTONE_STANDING_FIRST = """\
This is the FIRST milestone, so none of this project's code exists yet.
Whatever the codebase map shows above is the repository you are building
INTO, not earlier work of your own to build on — do not treat it as this
project's code, and do not write tasks that extend it. Your tasks create
this project's files from nothing. Later milestones are not yours to
anticipate."""

MILESTONE_STANDING_LATER = """\
Earlier milestones are already built and their code is in the codebase map
above; later ones are not yours to anticipate."""

#: Sent on the L4 rung (PRD S7.2). L4 used to re-plan through Odysseus; it
#: now re-expands through this node instead, because the tasks that got stuck
#: are the ones THIS node wrote, and the roadmap above them is very rarely
#: what was wrong.
LACHESIS_REEXPAND_PREFIX = """\
The current task has NOT been completed after exhausting patch (L1),
re-research (L2), and rollback-and-rewrite (L3) attempts. Re-expand this
milestone from the stuck task onward — the tasks already completed stay as
they are. You may split the stuck task into smaller ones, change its
approach, or adjust its acceptance criterion if it was unmeasurable as
written. This is the last chance to make it achievable; if it still cannot
succeed, say so plainly in the task itself rather than deferring the problem
again.
"""

#: Fed back when a verification write was refused for touching a non-test
#: file. Same rationale as `CHIRON_REFUSED_PREFIX`: without it the node has
#: no way to learn that its own output was overruled, and simply reproposes
#: the identical response until the budget is gone.
LACHESIS_REFUSED_PREFIX = """\
Your PREVIOUS response was refused and NOTHING was written, because it
included a path that is not a test file. You may only write tests. Reason:"""


@dataclass(frozen=True, slots=True)
class TaskListOutput:
    tasks: tuple[PlanTask, ...]
    objection: str | None
    #: Same meaning as `DaedalusOutput.malformed`: true only for the
    #: harness's own "found neither tag" fallback.
    malformed: bool = False
    #: See `DaedalusOutput.format_hint`.
    format_hint: str = ""
    #: Well-formed-looking blocks the scanner could not read. Zero on a clean
    #: response; non-zero means tasks were silently dropped from a plan that
    #: otherwise looks complete, which is worth a run-log line even though the
    #: expansion succeeded (`xeno.graph.lachesis`).
    dropped: int = 0

    @property
    def is_objection(self) -> bool:
        return self.objection is not None


def parse_task_list_output(text: str) -> TaskListOutput:
    """Parse Lachesis JOB 1 into a task list or a blocking objection."""
    scan = _scan_blocks(text, "task", "acceptance")
    tasks = tuple(
        PlanTask(
            description=block.body.strip()[:_MAX_TASK_DESCRIPTION],
            acceptance=block.attr.strip()[:_MAX_TASK_ACCEPTANCE],
        )
        for block in scan.blocks
        if block.body.strip()
    )
    if tasks:
        return TaskListOutput(tasks=tasks, objection=None, dropped=scan.opened - len(tasks))

    objection = _OBJECTION_RE.search(text)
    if objection:
        return TaskListOutput(tasks=(), objection=objection.group(1).strip())

    return TaskListOutput(
        tasks=(),
        objection=_no_blocks_message("Lachesis", "task", scan, "there are no tasks to act on"),
        malformed=True,
        format_hint=scan.reason,
        dropped=scan.dropped,
    )


def parse_test_files_output(text: str) -> DaedalusOutput:
    """Parse Lachesis JOB 2 into file writes or a blocking objection.

    Reuses `DaedalusOutput` rather than declaring a near-identical twin: the
    shape of a write response — some file blocks, or a refusal to write — is
    the same question whichever node is answering it, and the two would only
    ever drift apart.
    """
    files, scan = _parse_file_blocks(text)
    if files:
        return DaedalusOutput(files=files, objection=None)

    objection = _OBJECTION_RE.search(text)
    if objection:
        return DaedalusOutput(files=(), objection=objection.group(1).strip())

    return DaedalusOutput(
        files=(),
        objection=_no_blocks_message("Lachesis", "file", scan, "there are no tests to write"),
        malformed=True,
        format_hint=scan.reason,
    )


# ---------------------------------------------------------------------------
# Argus (researcher, PRD S13 Phase 3)
# ---------------------------------------------------------------------------

#: Light tier (PRD S9.1) — this is a selection task over a file tree the
#: harness already assembled, not open-ended reasoning.
#: ONE system prompt covering both of Argus's jobs, not two. `PromptBuilder`
#: fingerprints one system text per `NodeRole` for the life of the process
#: (PRD T8: breakpoint 1 must be byte-identical on every call to a node) and
#: raises if it ever sees a second text under the same role — and both jobs
#: below route through `NodeRole.RESEARCHER`, since that is what selects
#: Argus's tier. The "which job" distinction lives in the current turn's
#: instruction, not the system prompt, exactly like Odysseus's plan-vs-
#: replan framing in `xeno.graph.odysseus`.
ARGUS_SYSTEM = """\
You are Argus, the researcher node in the Xeno CLI harness (PRD S10). You
have no shell access and never write or evaluate code — you only look at a
repository's file tree and report back. Each instruction below tells you
which of your three jobs this call is; do exactly the one it asks for.

JOB 1 - REPO SKELETON: given a file tree, write a short, plain-prose
summary of its structure: what the main components are, where source vs.
tests live, and anything a planner would need before breaking a task down
against this codebase. This is orientation, not an inventory — do not just
repeat the file list back. Respond with the summary and nothing else: no
preamble, no headings, no commentary about your process.

JOB 2 - FILE RESEARCH: given a task and the file tree, identify which
existing files are relevant to it — files the coder node will need to read
before implementing the task (related source, existing conventions, test
fixtures). Do not explain your reasoning beyond the one-line reason each
tag asks for. Your entire response is machine-parsed by a program that only
understands the tags below — prose anywhere else is a parse failure.
Respond with one or more file tags in exactly this format:

<xeno-file path="relative/path/to/file.py" reason="one-line why this matters"/>

OR, if no existing file is relevant (e.g. the task is pure net-new code
with nothing to reference), respond with exactly:
<xeno-no-files>one-line reason</xeno-no-files>

Rules for JOB 2:
- `path` is relative to the repository root and must be a file that appears
  in the given file tree — never invent a path.
- Name only files that genuinely matter; a long list defeats the point of
  having a research step at all.

JOB 3 - TOOLCHAIN DISCOVERY: given a file tree and the contents of any
manifest/build files present (package.json, pyproject.toml, Cargo.toml, go.mod,
Makefile, etc.), propose the fixed commands needed to install dependencies,
lint, type-check, and test this repository. THE GATES THAT RUN THESE COMMANDS
ARE DETERMINISTIC — your job is only to name which commands to run, never to
judge whether code passes or fails. This is a one-time call, cached and reused
for every future run against this repository until its manifests change.
Respond with:

<xeno-install>the one command to install this repo's declared dependencies</xeno-install>

<xeno-required-command name="short label, e.g. lint">the command, e.g. ruff check .
</xeno-required-command>

(one <xeno-required-command> tag per required check — install, lint,
type-check, test, build, whatever this repository's toolchain actually has;
omit any that do not apply)

<xeno-advisory-command name="short label, e.g. coverage">the command</xeno-advisory-command>

(zero or more — for checks that should run and be visible in the log but
must never block a passing result, e.g. coverage)

Rules for JOB 3:
- Every command is ONE fixed argv, not a shell script: no `&&`, no `|`, no
  `;`, no `>`, no environment-variable substitution. If a repository needs
  more than one step, emit more than one <xeno-required-command> tag rather
  than chaining them.
- `<xeno-install>` is optional — omit it entirely if the repository declares
  no dependencies to install.
- At least one <xeno-required-command> is mandatory: a discovery response
  with nothing required to run cannot be used to evaluate any change.
- Only propose commands for a toolchain that is genuinely evidenced by the
  files you were shown (a `package.json` with a `test` script, a
  `pyproject.toml` naming pytest, etc.) — never guess a toolchain that is not
  actually present.
- Every argv MUST start with that ecosystem's package manager or run-time
  entry point, e.g. `pip`/`python3` (Python), `npm`/`yarn`/`pnpm`/`node`
  (JavaScript/TypeScript), `cargo` (Rust), `go` (Go), `bundle`/`gem` (Ruby),
  `mvn`/`gradle` (Java/Kotlin). NEVER the name of a build backend, plugin, or
  compiler that tool invokes on your behalf — e.g. a Python
  `[build-system] requires = ["hatchling"]` (or setuptools, poetry-core,
  flit-core, ...) means the install command is `pip install -e .` or
  `python3 -m pip install .`, never `hatchling ...`. The manifest names the
  library your dependency manager will fetch, not a command you invoke
  yourself.
"""

#: Selects JOB 1 for a call — sent once per run, before Odysseus's first
#: call (PRD S14 "SKELETON + PLAN").
ARGUS_SKELETON_PREFIX = "JOB 1 - REPO SKELETON. Repository file tree:"

#: Selects JOB 2 for a call — sent on every per-task research call.
ARGUS_RESEARCH_PREFIX = "JOB 2 - FILE RESEARCH."

#: Selects JOB 3 for a call — sent once per repository, cached thereafter
#: (`xeno.adapters.discovery`), never once per run.
DISCOVERY_PREFIX = "JOB 3 - TOOLCHAIN DISCOVERY."

#: Job selectors for Odysseus's two jobs, carried in the current turn so
#: breakpoint 1 stays byte-identical between them (PRD T8) — the same
#: technique Argus's three jobs use.
ODYSSEUS_PLAN_PREFIX = "JOB 1 - PLAN."
ODYSSEUS_SPEC_PREFIX = "JOB 2 - SPEC."

#: Opens a spec conversation. The framing matters: the user typed `xeno` with
#: an idea, not a specification, and a first turn that interrogates them is
#: worse than one that makes a reasonable start and asks about what it could
#: not guess.
ODYSSEUS_SPEC_OPENING = """\
The user wants to build the following. Ask only about what you cannot
reasonably assume, then produce the spec.

"""

SPEC_FORMAT_CORRECTION = """\
Your previous response was neither a plain question nor a valid
<xeno-spec title="...">…</xeno-spec> block. Resend it as exactly one of
those two: either ask the user what you still need to know, in plain prose
with no tags at all, or emit the finished spec block on its own.
"""

_SPEC_RE = re.compile(
    r'<xeno-spec\s+title=["\']([^"\']*)["\']\s*>\n?(.*?)\n?</xeno-spec>',
    re.DOTALL,
)

#: `AgentState.goal` is capped at 4 KB like every state field (PRD S6.3), and
#: a spec body routinely exceeds that — which is exactly why the body becomes
#: a file on disk reached by Handle, and only `title` becomes the goal.
_MAX_SPEC_TITLE = 200


@dataclass(frozen=True, slots=True)
class SpecOutput:
    """One JOB 2 turn: either a question for the user, or the finished spec.

    `question` and `title`/`body` are deliberately exclusive — the prompt
    forbids emitting both, because a turn that asks and answers at once
    leaves no clear moment at which the conversation is over.
    """

    question: str = ""
    title: str = ""
    body: str = ""
    malformed: bool = False
    #: Always empty here: this parser has no tag-with-attribute form to
    #: misread, so there is never a specific problem to name. Present to
    #: satisfy `nodeops.ParsedOutput`, which every parse result implements.
    format_hint: str = ""

    @property
    def is_complete(self) -> bool:
        return bool(self.title and self.body)


def parse_spec_output(text: str) -> SpecOutput:
    """Parse a JOB 2 response into a question or a finished spec.

    Unlike every other parser here, untagged prose is the SUCCESS case for
    the conversational half — so `malformed` is reserved for a response that
    is neither: an empty reply, or a spec block missing its title.
    """
    match = _SPEC_RE.search(text)
    if match is not None:
        title = match.group(1).strip()[:_MAX_SPEC_TITLE]
        body = _strip_fence(match.group(2).strip())
        if not title or not body:
            return SpecOutput(malformed=True)
        return SpecOutput(title=title, body=body)

    question = _strip_fence(text.strip())
    if not question:
        return SpecOutput(malformed=True)
    return SpecOutput(question=question)

#: Wraps the current `PlanTask` for the two nodes that write code.
#:
#: Without this they were handed `state.goal` and nothing else, so the plan
#: was invisible to the only nodes that act on it: every task, from the first
#: to the last, looked like "implement the entire goal". On a greenfield run
#: that meant task 1 ("create the manifest and a placeholder test") produced
#: the finished feature instead, and the toolchain the rest of the plan
#: depended on was never created. The goal is still shown — it is the context
#: that makes a narrow task make sense — but it is explicitly demoted here,
#: because a model given both will otherwise reach for the bigger one.
CURRENT_TASK_PREFIX = """\
CURRENT TASK — this, and only this, is what you implement right now:
{description}

Acceptance criterion for THIS task:
{acceptance}

The goal above is the eventual destination of the whole plan, shown only so
this task makes sense in context. Other tasks in the plan cover the rest of
it. Do NOT implement the goal, do NOT work ahead, and do NOT anticipate a
later task: work outside the current task is not wanted here even when it is
correct, because it is evaluated against this task's acceptance criterion.
"""

#: Sent to Lachesis on its FIRST expansion when the repository has no
#: manifest/build file, so there is no toolchain for the gates to run and
#: nothing can be verified yet (PRD S8.2's deterministic gates presuppose
#: commands to BE deterministic about).
#:
#: The expansion step is the right place to solve this rather than a special
#: bootstrap node: "make this project gateable" is a task like any other, so
#: expressing it as task 1 means the scaffold gets the same Talos evaluation
#: and the same escalation ladder as everything after it — a bad scaffold is
#: repaired by Chiron, not by a separate recovery path.
LACHESIS_GREENFIELD_PREFIX = """\
GREENFIELD REPOSITORY. This project does not exist yet: there is no manifest
or build file, so there is no lint, type-check, or test command, and nothing
can currently be verified.

A scaffold task that establishes the toolchain has ALREADY been prepended as
task 1 of this expansion — you do not write it and must not repeat it. Assume
that by the time your first task runs, the project has a manifest, working
lint / type-check / test configuration, and an importable empty package.

Expand only the work that follows it. Each of your tasks is checked by the
gates that scaffold brings into existence, so every one of them may assume a
gateable project already exists.
"""

#: Prepended verbatim as task 1 of the first expansion in a greenfield repo.
#:
#: Deliberately NOT left to a model. That the repository has no toolchain is
#: a fact the harness has already established deterministically
#: (`has_manifests`), not a judgement call, and a run that skips this task
#: cannot gate anything at all — every later task fails on "no toolchain"
#: until the ladder gives up. Asking the model nicely was tried and produced
#: exactly that: a plan whose first task added tests to a project with no
#: manifest, no package, and no test runner. Some requirements are better
#: guaranteed by construction than requested in a prompt.
#:
#: It no longer asks for a placeholder test, for two reasons that turn out to
#: be the same reason: Daedalus is not permitted to write test files at all
#: now, and the empty-test-suite problem the placeholder existed to dodge
#: (`pytest` exits 5 when it collects nothing) cannot arise, because the test
#: command is held back under `GateProfile.IMPLEMENTATION` until Lachesis has
#: written the milestone's tests.
GREENFIELD_SCAFFOLD_TASK = """\
Establish this project's toolchain, and nothing else. Create the manifest \
that names the project and declares its dependencies (pyproject.toml, \
package.json, Cargo.toml, go.mod — whichever suits the goal), the lint / \
type-check / test configuration that manifest needs, and the smallest \
importable source package. Implement no feature of the goal in this task. \
Prefer a toolchain the goal genuinely calls for over the most elaborate one \
available, and prefer dependencies a standard environment already \
provides."""

GREENFIELD_SCAFFOLD_ACCEPTANCE = (
    "A manifest exists at the repository root, an importable source package exists, "
    "and the discovered lint and type-check commands both exit 0."
)

#: Sent only on L2 re-research (PRD S7.2): a patch already failed against
#: the files Argus found the first time, so this call is specifically
#: looking for what was missed, not repeating the initial search.
ARGUS_L2_PREFIX = """\
The files already identified were not enough: a patch attempt against this
task still failed. Look specifically for what might have been missed —
related fixtures, configuration, or code elsewhere in the repository that
the failure suggests is relevant but was not yet named.
"""

ARGUS_FORMAT_CORRECTION = """\
Your previous response did not use the required format — no <xeno-file> or
<xeno-no-files> tag was found. Resend your answer using ONLY the tag format
from the system prompt: no explanation, no markdown code fences, just the
tag(s).
"""

_ARGUS_FILE_RE = re.compile(
    r'<xeno-file\s+path=["\']([^"\']+)["\']\s+reason=["\']([^"\']*)["\']\s*/?>'
)
_NO_FILES_RE = re.compile(r"<xeno-no-files>(.*?)</xeno-no-files>", re.DOTALL)


@dataclass(frozen=True, slots=True)
class FileRef:
    path: str
    reason: str


@dataclass(frozen=True, slots=True)
class ArgusResearchOutput:
    files: tuple[FileRef, ...]
    no_files_reason: str | None = None
    malformed: bool = False
    #: Always empty here: this parser has no tag-with-attribute form to
    #: misread, so there is never a specific problem to name. Present to
    #: satisfy `nodeops.ParsedOutput`, which every parse result implements.
    format_hint: str = ""


def parse_argus_research_output(text: str) -> ArgusResearchOutput:
    """Parse Argus's response into a file selection, an explicit "nothing
    relevant" verdict, or a malformed fallback treated as the latter — an
    empty selection either way is not a run-halting condition (PRD S10:
    Argus narrows context, it does not gate correctness)."""
    files = tuple(
        FileRef(path=path.strip(), reason=reason.strip())
        for path, reason in _ARGUS_FILE_RE.findall(text)
    )
    if files:
        return ArgusResearchOutput(files=files)

    no_files = _NO_FILES_RE.search(text)
    if no_files:
        return ArgusResearchOutput(files=(), no_files_reason=no_files.group(1).strip())

    return ArgusResearchOutput(
        files=(),
        no_files_reason=(
            "Argus's response contained neither a valid <xeno-file> tag nor "
            "<xeno-no-files>; treating as no additional context found."
        ),
        malformed=True,
    )


# ---------------------------------------------------------------------------
# Toolchain discovery (PRD S8.2, S12 revised): Argus's JOB 3.
#
# Deliberately NOT a new NodeRole/system prompt: `PromptBuilder` fingerprints
# one system text per `NodeRole` for the life of the process (PRD T8) and
# raises if it ever sees a second one — so discovery reuses `ARGUS_SYSTEM`
# byte-for-byte and only the current-turn prefix below picks JOB 3, exactly
# like `ARGUS_SKELETON_PREFIX`/`ARGUS_RESEARCH_PREFIX` already do for jobs 1
# and 2. The gates this feeds stay deterministic tools (PRD S8.2) — this call
# only ever proposes WHICH commands to run, never a pass/fail verdict.
# ---------------------------------------------------------------------------

DISCOVERY_FORMAT_CORRECTION = """\
Your previous response did not use the required format — no valid
<xeno-required-command> tag was found. Resend your answer using ONLY the tag
format from JOB 3 in the system prompt: no explanation, no markdown code
fences, just the tag(s) and their content. Remember each command is one fixed
argv, never a shell script — no `&&`, `|`, `;`, or `>`.
"""

_INSTALL_RE = re.compile(r"<xeno-install>\n?(.*?)\n?</xeno-install>", re.DOTALL)
_REQUIRED_COMMAND_RE = re.compile(
    r'<xeno-required-command\s+name=["\']([^"\']+)["\']\s*>\n?(.*?)\n?</xeno-required-command>',
    re.DOTALL,
)
_ADVISORY_COMMAND_RE = re.compile(
    r'<xeno-advisory-command\s+name=["\']([^"\']+)["\']\s*>\n?(.*?)\n?</xeno-advisory-command>',
    re.DOTALL,
)


def _named_commands(pattern: re.Pattern[str], text: str) -> tuple[tuple[str, str], ...]:
    """Required and advisory commands differ only in which tag carries them
    and in what a failure means downstream — the extraction is the same."""
    return tuple((name.strip(), argv.strip()) for name, argv in pattern.findall(text))


@dataclass(frozen=True, slots=True)
class DiscoveryOutput:
    """Raw wire-format extraction only — tokenizing each command's text into
    an argv and validating it against the executable allowlist both happen in
    `xeno.adapters.discovery`, which owns that safety boundary."""

    install: str | None
    required: tuple[tuple[str, str], ...]
    advisory: tuple[tuple[str, str], ...]
    malformed: bool = False
    #: Always empty here: this parser has no tag-with-attribute form to
    #: misread, so there is never a specific problem to name. Present to
    #: satisfy `nodeops.ParsedOutput`, which every parse result implements.
    format_hint: str = ""


def parse_discovery_output(text: str) -> DiscoveryOutput:
    """Parse a JOB 3 response into an install command plus required/advisory
    command lists.

    At least one required command is mandatory — a discovery response with
    nothing to run cannot gate anything, so an empty `required` list is
    itself treated as malformed, same shape as every other node's "nothing
    safe to act on" fallback.
    """
    install_match = _INSTALL_RE.search(text)
    install = install_match.group(1).strip() if install_match else None

    required = _named_commands(_REQUIRED_COMMAND_RE, text)
    if not required:
        return DiscoveryOutput(install=None, required=(), advisory=(), malformed=True)

    advisory = _named_commands(_ADVISORY_COMMAND_RE, text)
    return DiscoveryOutput(install=install, required=required, advisory=advisory)


# ---------------------------------------------------------------------------
# Cerberus (reviewer, PRD S8, S13 Phase 4)
# ---------------------------------------------------------------------------

#: Flagship tier (PRD S9.1) — this is a subjective, holistic call
#: ("does this diff accomplish the stated goal", not "does it pass"), the
#: one place in the graph a green Talos is necessary but not sufficient.
#: Only reached once a task loop already produced a fully green run (PRD
#: S8.2) — `xeno.graph.cerberus` skips this call entirely and reports
#: deterministically when it is entered already-halted (an L5 or breaker
#: escalation with nothing left to judge).
CERBERUS_SYSTEM = """\
You are Cerberus, the reviewer node in the Xeno CLI harness (PRD S8, S10).
You are the ONLY component in this system whose output is ever shown
directly to a human, and the only one that writes to git.

Every deterministic gate has already passed —
that is necessary but NOT sufficient. Your job is the holistic judgment a
gate cannot make: does this diff actually accomplish the stated goal? Is the
design sound, or did the coder work around the problem instead of solving
it? Are there security issues? Does it match the codebase's existing
conventions? Were any tests deleted or weakened just to make them pass? Is
there dead code, swallowed exceptions, hardcoded secrets, or abandoned
TODOs? You may reject a diff where every test passes.

Do not explain your reasoning outside the tags below. Your entire response
is machine-parsed by a program that only understands the tags below — prose
anywhere else is a parse failure, not a helpful aside.

Respond with exactly one verdict, in exactly one of these three shapes:

  APPROVE — the diff is ready to ship as-is:
  <xeno-verdict>approve</xeno-verdict>
  <xeno-commit-message>
  type(scope): short imperative summary

  A short body explaining WHY this change was made, not a restatement of
  the diff.
  </xeno-commit-message>
  <xeno-notes>optional one or two sentences for the human confirming this</xeno-notes>

  REJECT AND RETURN — the problem is real but fixable without a human:
  <xeno-verdict>reject_and_return</xeno-verdict>
  <xeno-destination>daedalus</xeno-destination>
  <xeno-objections>
  specific, actionable objections — enough for the destination node to
  understand exactly what is wrong and what "fixed" looks like
  </xeno-objections>
  Use `daedalus` in <xeno-destination> when the CODE is wrong (a bug, a
  missed edge case, bad style, a security issue) — the plan itself was fine.
  Use `odysseus` when the PLAN itself is wrong (a task was misconceived, a
  step is missing, the wrong problem was solved) — the code correctly
  implemented a flawed plan.

  ESCALATE — the call genuinely depends on information only a human has:
  <xeno-verdict>escalate</xeno-verdict>
  <xeno-report>
  Restate the goal. Summarize what was attempted. State the specific
  blocking question. Recommend one concrete next action.
  </xeno-report>

Rules:
- Reject rather than approve anything you are not confident about.
- ESCALATE is for genuine human-only judgment calls, not a way to avoid a
  call you are equipped to make yourself.
- The commit message MUST follow conventional-commit format
  (type(scope): summary) with the "why" in the body, since it becomes the
  permanent record of this change.
"""

CERBERUS_FORMAT_CORRECTION = """\
Your previous response did not use the required format — no valid
<xeno-verdict> block was found (or a required field for that verdict was
missing), so no decision could be recorded. Resend your answer using ONLY
the tag format from the system prompt: no explanation, no numbered steps,
no markdown code fences, just the tag(s) and their content.
"""

_VERDICT_RE = re.compile(r"<xeno-verdict>(.*?)</xeno-verdict>", re.DOTALL)
_COMMIT_MESSAGE_RE = re.compile(
    r"<xeno-commit-message>\n?(.*?)\n?</xeno-commit-message>", re.DOTALL
)
_CERBERUS_NOTES_RE = re.compile(r"<xeno-notes>(.*?)</xeno-notes>", re.DOTALL)
_DESTINATION_RE = re.compile(r"<xeno-destination>(.*?)</xeno-destination>", re.DOTALL)
_OBJECTIONS_RE = re.compile(r"<xeno-objections>\n?(.*?)\n?</xeno-objections>", re.DOTALL)
_REPORT_RE = re.compile(r"<xeno-report>\n?(.*?)\n?</xeno-report>", re.DOTALL)

_DESTINATION_BY_NAME = {"daedalus": NodeRole.CODER, "odysseus": NodeRole.PLANNER}

#: Mirrors `CommitRef.message`'s cap (`xeno.core.state`) — the message ends
#: up on a real git commit either way.
_MAX_COMMIT_MESSAGE_CHARS = 2000


@dataclass(frozen=True, slots=True)
class CerberusOutput:
    verdict: Verdict | None
    commit_message: str | None = None
    notes: str | None = None
    destination: NodeRole | None = None
    objections: str | None = None
    report: str | None = None
    #: True only for the harness's own "no usable verdict" fallback: an
    #: unrecognized/missing <xeno-verdict>, or a verdict missing one of its
    #: own required fields (e.g. `approve` with no commit message). Unlike
    #: every other node's parser, there is no separate escape-hatch tag here
    #: — ESCALATE already IS this node's escape hatch, so a malformed
    #: response has nowhere safe to fall but the caller's own forced
    #: ESCALATE (`xeno.graph.cerberus`).
    malformed: bool = False
    #: Always empty here: this parser has no tag-with-attribute form to
    #: misread, so there is never a specific problem to name. Present to
    #: satisfy `nodeops.ParsedOutput`, which every parse result implements.
    format_hint: str = ""


def parse_cerberus_output(text: str) -> CerberusOutput:
    """Parse Cerberus's response into one of three verdicts.

    Required fields are verdict-conditional: APPROVE needs a commit message,
    REJECT_AND_RETURN needs a recognized destination and objections,
    ESCALATE needs a report. A missing verdict, an unrecognized verdict, or
    a verdict missing its own required field(s) all fall through to the same
    `malformed=True` result — there is no partial-credit verdict to act on.
    """
    verdict_match = _VERDICT_RE.search(text)
    verdict_name = verdict_match.group(1).strip().lower() if verdict_match else None

    if verdict_name == Verdict.APPROVE.value:
        message = _COMMIT_MESSAGE_RE.search(text)
        if message:
            notes = _CERBERUS_NOTES_RE.search(text)
            return CerberusOutput(
                verdict=Verdict.APPROVE,
                commit_message=message.group(1).strip()[:_MAX_COMMIT_MESSAGE_CHARS],
                notes=notes.group(1).strip() if notes else None,
            )

    elif verdict_name == Verdict.REJECT_AND_RETURN.value:
        destination_match = _DESTINATION_RE.search(text)
        objections = _OBJECTIONS_RE.search(text)
        destination = (
            _DESTINATION_BY_NAME.get(destination_match.group(1).strip().lower())
            if destination_match
            else None
        )
        if destination is not None and objections:
            return CerberusOutput(
                verdict=Verdict.REJECT_AND_RETURN,
                destination=destination,
                objections=objections.group(1).strip(),
            )

    elif verdict_name == Verdict.ESCALATE.value:
        report = _REPORT_RE.search(text)
        if report:
            return CerberusOutput(verdict=Verdict.ESCALATE, report=report.group(1).strip())

    return CerberusOutput(
        verdict=None,
        malformed=True,
    )
