"""The spec conversation: Odysseus JOB 2, the one place a human is talked to.

`xeno run "some goal"` asks the harness to plan against a single sentence.
That works when you already know what you want; it is a poor way to start a
project, because everything downstream — the plan, every acceptance
criterion, Cerberus's review — inherits whatever ambiguity was in that
sentence, and the first time anyone notices is when the wrong thing has been
built and gated green.

So this runs first: a short back-and-forth that ends in a written spec,
which then becomes the run's goal. It is deliberately the *only* interactive
node. Every other node in the harness is forbidden from addressing the user
(PRD S10) and expresses "I cannot proceed" as a blocking objection instead,
precisely so that questions happen here, once, before any money is spent on
implementation.

The spec body is written to disk and seeded as a context Handle rather than
stuffed into `AgentState.goal`: state fields are capped at 4 KB (PRD S6.3)
and a real spec exceeds that, so the goal carries the spec's one-line title
and every node reads the full text through the same context mechanism it
already uses for source files.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from xeno.core.config import XenoConfig
from xeno.core.paths import RunPaths
from xeno.core.state import AgentState
from xeno.core.types import NodeRole
from xeno.graph.nodeops import complete_with_format_retry
from xeno.graph.prompts import (
    ODYSSEUS_SPEC_OPENING,
    ODYSSEUS_SPEC_PREFIX,
    ODYSSEUS_SYSTEM,
    SPEC_FORMAT_CORRECTION,
    parse_spec_output,
)
from xeno.prompt.assembly import PromptBuilder
from xeno.prompt.keys import CacheKeyring
from xeno.router.router import ChainExhaustedError, Router

#: A conversation that will not converge is a conversation to stop having.
#: On the last exchange the user is told to answer or say "go", and the
#: prompt's own rule ("when the user tells you to go ahead, fill the gaps
#: with stated assumptions") is what closes it out.
MAX_EXCHANGES = 6


class SpecAbandoned(Exception):
    """The user left the conversation without a spec."""


@dataclass(frozen=True, slots=True)
class Spec:
    title: str
    body: str

    def write(self, directory: Path) -> Path:
        path = directory / "SPEC.md"
        path.write_text(f"# {self.title}\n\n{self.body}\n")
        return path


def run_spec_conversation(
    *,
    router: Router,
    config: XenoConfig,
    keyring: CacheKeyring,
    state: AgentState,
    paths: RunPaths,
    idea: str,
    ask: Callable[[str], str],
    show: Callable[[str], None],
    max_exchanges: int = MAX_EXCHANGES,
) -> Spec:
    """Talk until there is a spec, then return it.

    `ask` and `show` are injected rather than importing a console here: this
    module decides what the conversation IS, and the CLI decides what it
    looks like. It also means the whole loop is testable by passing a
    scripted `ask`, with no terminal involved.

    Raises `SpecAbandoned` if the user backs out, and `ChainExhaustedError`
    propagates for the same reason it does in every node — a model chain
    that is exhausted is not something to paper over with a made-up spec.
    """
    builder = PromptBuilder(
        node=NodeRole.PLANNER,
        keyring=keyring,
        system_text=ODYSSEUS_SYSTEM,
        caching_enabled=config.caching.enabled,
    )

    current_turn = f"{ODYSSEUS_SPEC_PREFIX}\n{ODYSSEUS_SPEC_OPENING}{idea}"

    for exchange in range(max_exchanges):
        output = complete_with_format_retry(
            router=router,
            builder=builder,
            node=NodeRole.PLANNER,
            state=state,
            paths=paths,
            current_turn=current_turn,
            correction=SPEC_FORMAT_CORRECTION,
            parse=parse_spec_output,
        )

        if output.is_complete:
            return Spec(title=output.title, body=output.body)

        if output.malformed:
            # The corrective retry inside `complete_with_format_retry` is
            # already spent by now. Rather than halt a conversation the user
            # is sitting in front of, fall back to asking them directly.
            question = "I did not follow that. Could you say a bit more about what you want?"
        else:
            question = output.question

        show(question)
        if exchange == max_exchanges - 2:
            show("(answer, or say 'go' to build with the assumptions so far)")

        reply = ask("you").strip()
        if not reply:
            raise SpecAbandoned("no answer given")
        current_turn = reply

    # Out of exchanges: demand the spec rather than returning without one.
    output = complete_with_format_retry(
        router=router,
        builder=builder,
        node=NodeRole.PLANNER,
        state=state,
        paths=paths,
        current_turn=(
            "Stop asking questions. Emit the <xeno-spec> block now, filling every "
            "remaining gap with an assumption recorded under # Assumptions."
        ),
        correction=SPEC_FORMAT_CORRECTION,
        parse=parse_spec_output,
    )
    if not output.is_complete:
        raise SpecAbandoned("the conversation did not converge on a spec")
    return Spec(title=output.title, body=output.body)


def spec_chain_error(exc: ChainExhaustedError) -> str:
    """Phrase a chain exhaustion for a human who is mid-conversation."""
    return f"the planner model is unavailable: {exc}"
