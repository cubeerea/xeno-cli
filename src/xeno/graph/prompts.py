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

#: Sent only when this call is Cerberus's E17 REJECT_AND_RETURN (PRD S8.3):
#: a fully green run was reviewed and rejected on plan grounds. Distinct from
#: `ODYSSEUS_REPLAN_PREFIX` (L4, a single stuck task exhausted its ladder
#: budget): here EVERY task already passed Talos's gates, and the problem is
#: that the plan as a whole did not accomplish the goal Cerberus judged it
#: against — this call appends task(s), it does not revise a stuck one.
ODYSSEUS_CERBERUS_REJECT_PREFIX = """\
Cerberus, the reviewer, examined the completed diff for this goal and
rejected it on plan grounds: the work as planned does not accomplish the
goal. The tasks already completed stay as they are. Add the task(s) needed
to resolve Cerberus's objection below.

Cerberus's objection:
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

Every deterministic gate (parse, lint, types, tests) has already passed —
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
