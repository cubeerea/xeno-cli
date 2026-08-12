"""Odysseus JOB 2: the spec conversation (`xeno.graph.spec`).

Driven end to end with a scripted provider and a scripted `ask`, so the whole
loop — questions, replies, convergence, and the forced close — is exercised
without a terminal or a model. That is the reason `run_spec_conversation`
takes `ask`/`show` as parameters rather than importing a console.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from xeno.core.config import Limits, ModelSpec, NodeSpec, ProviderSpec, XenoConfig
from xeno.core.ledger import CostLedger
from xeno.core.paths import RunPaths
from xeno.core.state import AgentState
from xeno.core.types import DEFAULT_NODE_TIERS, Tier
from xeno.core.usage import Usage
from xeno.graph.prompts import parse_spec_output
from xeno.graph.spec import SpecAbandoned, run_spec_conversation
from xeno.prompt.keys import CacheKeyring
from xeno.router.providers.base import CompletionResult, Provider
from xeno.router.router import Router

SPEC_BLOCK = """\
<xeno-spec title="Build a CLI todo list with add, list, and done">
# What this is
A small command line todo list.

# Requirements
- `add <text>` appends an item
- `list` prints every item with its index

# Out of scope
- syncing

# Assumptions
- storage is a local JSON file
</xeno-spec>"""


class _Scripted(Provider):
    """Returns queued responses in order; records the turns it was sent."""

    def __init__(self, name: str, spec: ProviderSpec, *, replies: list[str]) -> None:
        super().__init__(name, spec)
        self.replies = list(replies)
        self.turns: list[str] = []
        self.cache_capable = True

    def complete(self, prompt, model, *, max_tokens, temperature=0.0):  # type: ignore[no-untyped-def]
        self.turns.append(prompt.current_turn)
        text = self.replies.pop(0) if self.replies else SPEC_BLOCK
        return CompletionResult(
            text=text, model=model.model, usage=Usage(input_tokens=100, output_tokens=20),
            latency_ms=1.0,
        )

    def health_check(self) -> tuple[bool, str]:
        return True, "fake"


@pytest.fixture
def config() -> XenoConfig:
    return XenoConfig(
        providers={"ollama": ProviderSpec(family="ollama", base_url="http://x")},
        tiers={t: (ModelSpec(provider="ollama", model="m"),) for t in Tier},
        nodes={role: NodeSpec(tier=tier) for role, tier in DEFAULT_NODE_TIERS.items()},
        limits=Limits(),
    )


def _converse(
    config: XenoConfig,
    tmp_path: Path,
    replies: list[str],
    answers: list[str],
    **kwargs: object,
) -> tuple[object, _Scripted, list[str]]:
    router = Router(config, ledger=CostLedger(run_id="t"))
    fake = _Scripted("ollama", config.providers["ollama"], replies=replies)
    router._providers["ollama"] = fake
    shown: list[str] = []
    queue: Iterator[str] = iter(answers)

    def ask(_label: str) -> str:
        return next(queue, "")

    spec = run_spec_conversation(
        router=router,
        config=config,
        keyring=CacheKeyring(run_id="t", worktree_root=tmp_path),
        state=AgentState(run_id="t", goal="idea"),
        paths=RunPaths(repo_root=tmp_path, run_id="t").ensure(),
        idea="a todo list",
        ask=ask,
        show=shown.append,
        **kwargs,  # type: ignore[arg-type]
    )
    return spec, fake, shown


# ---- the parser -----------------------------------------------------------


def test_untagged_prose_is_a_question_not_a_parse_failure() -> None:
    """The conversational half inverts every other parser here: plain prose
    is the success case, because that is how a question arrives."""
    out = parse_spec_output("Where should the todos be stored?")
    assert out.question == "Where should the todos be stored?"
    assert not out.malformed
    assert not out.is_complete


def test_a_spec_block_parses_into_title_and_body() -> None:
    out = parse_spec_output(SPEC_BLOCK)
    assert out.is_complete
    assert out.title == "Build a CLI todo list with add, list, and done"
    assert "`add <text>` appends an item" in out.body
    assert not out.question


def test_an_empty_response_is_malformed() -> None:
    assert parse_spec_output("   \n  ").malformed


def test_a_spec_block_without_a_title_is_malformed() -> None:
    """The title becomes the run's goal, so a spec without one is unusable
    even though its body may be perfectly good."""
    assert parse_spec_output('<xeno-spec title="">body</xeno-spec>').malformed


# ---- the conversation -----------------------------------------------------


def test_a_spec_on_the_first_turn_ends_the_conversation(
    config: XenoConfig, tmp_path: Path
) -> None:
    spec, _fake, shown = _converse(config, tmp_path, [SPEC_BLOCK], [])
    assert spec.title.startswith("Build a CLI todo list")  # type: ignore[attr-defined]
    assert shown == [], "nothing to ask, so nothing shown"


def test_questions_are_shown_and_answers_are_sent_back(
    config: XenoConfig, tmp_path: Path
) -> None:
    spec, fake, shown = _converse(
        config,
        tmp_path,
        ["Where should todos be stored?", SPEC_BLOCK],
        ["a json file"],
    )
    assert shown == ["Where should todos be stored?"]
    assert fake.turns[-1] == "a json file", "the reply is the next turn verbatim"
    assert spec.body.startswith("# What this is")  # type: ignore[attr-defined]


def test_the_first_turn_carries_the_job_selector_and_the_idea(
    config: XenoConfig, tmp_path: Path
) -> None:
    """JOB 2 is selected in the current turn so breakpoint 1 stays identical
    to JOB 1's (PRD T8) — the same technique Argus's three jobs use."""
    _spec, fake, _shown = _converse(config, tmp_path, [SPEC_BLOCK], [])
    assert fake.turns[0].startswith("JOB 2 - SPEC.")
    assert "a todo list" in fake.turns[0]


def test_an_empty_answer_abandons_rather_than_inventing_a_spec(
    config: XenoConfig, tmp_path: Path
) -> None:
    with pytest.raises(SpecAbandoned):
        _converse(config, tmp_path, ["What should it do?"], [""])


def test_a_conversation_that_will_not_converge_is_forced_to_close(
    config: XenoConfig, tmp_path: Path
) -> None:
    """A model that only ever asks would otherwise loop forever in front of
    a user. After the budget, it is told to emit the spec with assumptions."""
    spec, fake, _shown = _converse(
        config,
        tmp_path,
        ["q1", "q2", SPEC_BLOCK],
        ["a1", "a2", "a3"],
        max_exchanges=2,
    )
    assert "Stop asking questions" in fake.turns[-1]
    assert spec.title  # type: ignore[attr-defined]
    assert spec.body  # type: ignore[attr-defined]


def test_a_forced_close_that_still_fails_raises(config: XenoConfig, tmp_path: Path) -> None:
    with pytest.raises(SpecAbandoned):
        _converse(config, tmp_path, ["q1", "q2", "q3", "q4"], ["a1", "a2", "a3"], max_exchanges=2)


def test_the_spec_is_written_as_markdown_with_its_title(
    config: XenoConfig, tmp_path: Path
) -> None:
    spec, _fake, _shown = _converse(config, tmp_path, [SPEC_BLOCK], [])
    path = spec.write(tmp_path)  # type: ignore[attr-defined]
    text = path.read_text()
    assert path.name == "SPEC.md"
    assert text.startswith("# Build a CLI todo list")
    assert "# Requirements" in text
