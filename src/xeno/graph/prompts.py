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

from xeno.graph.plan import PlanTask

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
- If the task is underspecified and you cannot proceed without inventing
  requirements the user never gave you, do not guess. Respond with exactly:
  <xeno-objection>a one-sentence explanation of what is missing</xeno-objection>
  and nothing else. Never ask the user a question directly.
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

TALOS_TRIAGE_SYSTEM = """\
You are Talos's log-triage assistant in the Xeno CLI harness (PRD S10, S8.2).

The pass/fail gates (parse, lint, types, tests) are deterministic tools that
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

Talos has already run the deterministic gates (parse, lint, types, tests) and
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

#: Same rationale as DAEDALUS_FORMAT_CORRECTION: sent only after a response
#: using neither tag.
CHIRON_FORMAT_CORRECTION = """\
Your previous response did not use the required format — no <xeno-file> or
<xeno-decline> tag was found, so nothing could be written. Resend your answer
using ONLY the tag format from the system prompt: no explanation, no
numbered steps, no markdown code fences, just the tag(s) and their content.
"""

#: Path quoting accepts single or double quotes: local models follow the
#: double-quote instruction inconsistently, and rejecting an otherwise
#: well-formed block over quote style would trade a real write for an
#: objection.
_FILE_BLOCK_RE = re.compile(
    r'<xeno-file\s+path=["\']([^"\']+)["\']\s*>\n?(.*?)\n?</xeno-file>', re.DOTALL
)
_OBJECTION_RE = re.compile(r"<xeno-objection>(.*?)</xeno-objection>", re.DOTALL)
#: A whole-content markdown fence, e.g. ```python\n...\n``` or ```\n...\n```.
#: Despite the system prompt telling it not to, a local model wrapping its
#: file content in one is common enough to be worth stripping rather than
#: writing the fence markers into the file as literal text.
_FENCE_RE = re.compile(r"\A```[^\n]*\n(.*?)\n?```\s*\Z", re.DOTALL)


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
    objection = _OBJECTION_RE.search(text)
    if objection:
        return DaedalusOutput(files=(), objection=objection.group(1).strip())

    files = tuple(
        FileBlock(path=path.strip(), content=_strip_fence(content))
        for path, content in _FILE_BLOCK_RE.findall(text)
    )
    if not files:
        return DaedalusOutput(
            files=(),
            objection=(
                "Daedalus's response contained neither a valid <xeno-file> block "
                "nor an <xeno-objection>; treating as a blocking objection since "
                "there is nothing safe to act on."
            ),
            malformed=True,
        )
    return DaedalusOutput(files=files, objection=None)


def _strip_fence(content: str) -> str:
    match = _FENCE_RE.match(content.strip())
    return match.group(1) if match else content


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


def parse_chiron_output(text: str) -> ChironOutput:
    """Parse Chiron's response into a patch or a decline.

    Unlike `parse_daedalus_output`, malformed output here is NOT treated the
    same as every other decline for run-control purposes by the caller — it
    still counts as declined (nothing safe to write), but
    `xeno.graph.build` only spends the one-shot format-correction retry when
    `malformed` is set, never when Chiron declined on purpose.
    """
    decline = _DECLINE_RE.search(text)
    if decline:
        return ChironOutput(files=(), declined=True, decline_reason=decline.group(1).strip())

    files = tuple(
        FileBlock(path=path.strip(), content=_strip_fence(content))
        for path, content in _FILE_BLOCK_RE.findall(text)
    )
    if not files:
        return ChironOutput(
            files=(),
            declined=True,
            decline_reason=(
                "Chiron's response contained neither a valid <xeno-file> block nor "
                "an <xeno-decline>; treating as a decline since there is nothing "
                "safe to act on."
            ),
            malformed=True,
        )
    return ChironOutput(files=files, declined=False)


# ---------------------------------------------------------------------------
# Odysseus (planner, PRD S13 Phase 3)
# ---------------------------------------------------------------------------

#: Flagship tier (PRD S9.1) — planning quality sets the ceiling for
#: everything downstream, so this is the one node it is never worth
#: cost-shaping down.
ODYSSEUS_SYSTEM = """\
You are Odysseus, the planner node in the Xeno CLI harness (PRD S10).

Your job: break the user's goal into an ordered sequence of small, concrete
tasks. Each task will be implemented by a separate coder node (Daedalus) and
verified by deterministic gates (parse, lint, types, tests) — you never
implement anything yourself and you have no shell access.

Every task needs a MECHANICAL acceptance criterion: something a gate can
check (a test passing, a specific function existing, a lint rule holding),
not a restatement of the task in different words.

Do not explain your reasoning. Your entire response is machine-parsed by a
program that only understands the tags below — prose anywhere in your
response is a parse failure, not a helpful aside.

Respond with one or more task blocks in exactly this format, and nothing
else:

<xeno-task acceptance="one-line mechanical acceptance criterion">
what this task should accomplish
</xeno-task>

Rules:
- Order tasks so each one is implementable on its own, building on the
  tasks before it.
- Keep each task small enough for one coder call: one feature, one file or
  a small tightly-related set of files, never "implement the whole thing."
- If the goal is impossible or too underspecified to plan against (it
  requires something that does not exist, contradicts itself, or omits
  information no reasonable default could fill in), do not invent
  requirements. Respond with exactly:
  <xeno-objection>a one-sentence explanation of what is missing or impossible</xeno-objection>
  and nothing else. Never ask the user a question directly.
"""

#: Sent only on a REVISION call (PRD S7.2 L4): the prior plan reached its
#: L1-L3 budgets without producing a passing task. Distinct from the initial
#: system prompt's framing because the input here is a specific failure, not
#: a blank goal.
ODYSSEUS_REPLAN_PREFIX = """\
The current task has NOT been completed after exhausting patch (L1),
re-research (L2), and rollback-and-rewrite (L3) attempts. Revise the plan
starting from this task onward — the tasks already completed stay as they
are. You may split the stuck task into smaller ones, change its approach, or
adjust its acceptance criterion if it was unmeasurable as written. This is
your last chance to make it achievable; if it still cannot succeed, say so
plainly in the task itself rather than deferring the problem again.
"""

ODYSSEUS_FORMAT_CORRECTION = """\
Your previous response did not use the required format — no <xeno-task> or
<xeno-objection> tag was found, so no plan could be recorded. Resend your
answer using ONLY the tag format from the system prompt: no explanation, no
numbered steps, no markdown code fences, just the tag(s) and their content.
"""

#: Lenient like `_FILE_BLOCK_RE`: local models are inconsistent about quote
#: style, so both are accepted rather than rejecting an otherwise well-formed
#: block over it.
_TASK_BLOCK_RE = re.compile(
    r'<xeno-task\s+acceptance=["\']([^"\']*)["\']\s*>\n?(.*?)\n?</xeno-task>', re.DOTALL
)
#: Mirrors `PlanTask.description`/`PlanTask.acceptance`'s `max_length`
#: (`xeno.graph.plan`) — truncating here rather than letting a too-long
#: model response raise a validation error out of the parser.
_MAX_TASK_DESCRIPTION = 2000
_MAX_TASK_ACCEPTANCE = 500


@dataclass(frozen=True, slots=True)
class OdysseusOutput:
    tasks: tuple[PlanTask, ...]
    objection: str | None
    #: Same meaning as `DaedalusOutput.malformed`: true only for the
    #: harness's own "found neither tag" fallback.
    malformed: bool = False

    @property
    def is_objection(self) -> bool:
        return self.objection is not None


def parse_odysseus_output(text: str) -> OdysseusOutput:
    """Parse Odysseus's response into a task list or a blocking objection.

    Same shape as `parse_daedalus_output`: malformed output is itself an
    objection, since there is no plan safe to act on either way.
    """
    objection = _OBJECTION_RE.search(text)
    if objection:
        return OdysseusOutput(tasks=(), objection=objection.group(1).strip())

    tasks = tuple(
        PlanTask(
            description=description.strip()[:_MAX_TASK_DESCRIPTION],
            acceptance=acceptance.strip()[:_MAX_TASK_ACCEPTANCE],
        )
        for acceptance, description in _TASK_BLOCK_RE.findall(text)
        if description.strip()
    )
    if not tasks:
        return OdysseusOutput(
            tasks=(),
            objection=(
                "Odysseus's response contained neither a valid <xeno-task> block "
                "nor an <xeno-objection>; treating as a blocking objection since "
                "there is no plan to act on."
            ),
            malformed=True,
        )
    return OdysseusOutput(tasks=tasks, objection=None)


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
which of your two jobs this call is; do exactly the one it asks for.

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
"""

#: Selects JOB 1 for a call — sent once per run, before Odysseus's first
#: call (PRD S14 "SKELETON + PLAN").
ARGUS_SKELETON_PREFIX = "JOB 1 - REPO SKELETON. Repository file tree:"

#: Selects JOB 2 for a call — sent on every per-task research call.
ARGUS_RESEARCH_PREFIX = "JOB 2 - FILE RESEARCH."

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
