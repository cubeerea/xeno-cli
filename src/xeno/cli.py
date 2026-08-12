"""The `xeno` command (PRD S13).

`xeno run` drives the full seven-node graph (PRD S13 Phase 3-4): Odysseus
plans the goal into tasks, Argus researches each one, Daedalus implements
it, Talos's sandboxed gates evaluate it, a bounded L0-L5 escalation ladder
(re-run, patch, re-research, roll back and rewrite, re-plan, halt) handles
failure, and Cerberus reviews the completed diff — the sole human gate
(PRD S8.1) — all on a throwaway worktree that is never the user's working
tree. Everything needed to trust the numbers behind that — config, routing,
prompt construction, secret scanning, the cost ledger — is exercised
standalone by `xeno models test`, Phase 0's exit criterion.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from xeno import __version__
from xeno.adapters.discovery import DiscoveryError, discover_toolchain
from xeno.core import vcs
from xeno.core.config import (
    CONFIG_FILENAME,
    ConfigError,
    XenoConfig,
    load_config,
)
from xeno.core.ledger import CostLedger
from xeno.core.paths import RunPaths, new_run_id, run_branch_name
from xeno.core.runlog import EventKind, RunLog
from xeno.core.state import AgentState, Handle
from xeno.core.types import CALLSIGNS, NodeRole, Tier, Verdict
from xeno.graph.build import run_graph
from xeno.graph.spec import SpecAbandoned, run_spec_conversation, spec_chain_error
from xeno.graph.toolchain import ToolchainSession
from xeno.prompt.assembly import PromptBuilder
from xeno.prompt.keys import CacheKeyring
from xeno.router.providers import UnsupportedProviderError
from xeno.router.router import ChainExhaustedError, Router
from xeno.security.mounts import mount_ignore
from xeno.security.scanner import SecretScanner
from xeno.ui.live import LiveRunView, NullRunView, RunView

app = typer.Typer(
    name="xeno",
    help="Terminal-native multi-agent coding harness. One human gate, tier-routed models.",
    add_completion=False,
)
models_app = typer.Typer(name="models", help="Inspect and exercise configured model tiers.")
config_app = typer.Typer(name="config", help="Inspect configuration.")
app.add_typer(models_app)
app.add_typer(config_app)

console = Console()

#: Declared once and shared: five commands take the same `--config` flag, and
#: Typer resolves an `Annotated` alias identically to an inline annotation.
ConfigOption = Annotated[Path | None, typer.Option("--config", "-c")]


def _warn(message: str) -> None:
    console.print(f"[yellow]warning:[/] {message}")


def _error(message: str) -> None:
    console.print(f"[red]error:[/] {message}")


#: Exercised on every `models test` call so the caching path is measured
#: through the real assembly code rather than a bespoke test prompt. Long
#: enough to clear OpenAI's 1,024-token minimum cacheable prefix (PRD S9.6.3).
_TEST_SYSTEM_PROMPT = (
    "You are a component of the Xeno CLI harness running a connectivity and "
    "cost-accounting self-test. Answer with a single short line and nothing else. "
    "This system block is deliberately verbose so that it exceeds the minimum "
    "cacheable prefix length that OpenAI-compatible providers require before they "
    "will serve a request prefix from cache. Its content is inert filler and "
    "carries no instructions beyond the first sentence. "
) * 12

_TEST_TURN = "Reply with the single word: ok"


def _resolve_config(path: Path | None) -> XenoConfig:
    try:
        return load_config(path)
    except ConfigError as exc:
        console.print(f"[bold red]config error[/]\n{exc}")
        raise typer.Exit(code=2) from exc


def _print_capability_warnings(config: XenoConfig) -> None:
    """PRD S9.4. The flagship warning is unconditional whenever the tier
    resolves to any locally-served model, at any size — there is no parameter
    threshold that makes a local model flagship-class on this project's target
    hardware, so there is no threshold in the check."""
    warnings: list[str] = []
    if config.flagship_is_local():
        warnings.append(
            "FLAGSHIP tier resolves to a locally-served model. Fully local is a "
            "supported mode, but plans (Odysseus) and reviews (Cerberus) will be "
            "materially worse than a hybrid setup. Xeno makes no "
            "'local with no quality loss' claim (PRD S9.4)."
        )
    if config.allow_unpriced_models:
        warnings.append(
            "allow_unpriced_models is set: CB-3 cannot enforce max_usd_per_run for "
            "unpriced remote models, and cost.json totals will be a lower bound."
        )
    if not config.caching.enabled:
        warnings.append("prompt caching is disabled; M1.4 will not be measurable this run.")
    for warning in warnings:
        _warn(warning)


@dataclass(frozen=True, slots=True)
class _RunContext:
    """Everything both `models test` and `run` stand up before doing any
    work. Bundled so the two commands cannot drift on the order they build
    it in — the ledger has to exist before the router that feeds it, and the
    run log before the router that writes to it."""

    run_id: str
    paths: RunPaths
    ledger: CostLedger
    router: Router
    state: AgentState
    keyring: CacheKeyring


def _start_run(
    config: XenoConfig,
    stack: ExitStack,
    *,
    repo_root: Path,
    state_goal: str,
    succeeded: Callable[[], bool],
    keyring_on_worktree: bool = False,
    observer: Callable[[dict[str, Any]], None] | None = None,
    **event_fields: Any,
) -> tuple[_RunContext, RunLog]:
    """Open a run's log, ledger, router, and state, registering teardown on
    `stack` as each is acquired.

    The `ExitStack` is the caller's, not this function's: an error anywhere
    downstream — image build, pool fill, the graph itself — must still close
    the router and the log, and only the caller knows where that scope ends.
    Registration order is LIFO and load-bearing: RUN_END is registered
    immediately after the log so it unwinds LAST, after every resource the
    caller adds later has already been released — a run's final event should
    describe a run that is actually over.

    `succeeded` is a callable rather than a value because the outcome is only
    known at teardown time, long after this function has returned.
    """
    run_id = new_run_id()
    paths = RunPaths(repo_root=repo_root, run_id=run_id).ensure()
    ledger = CostLedger(run_id=run_id, caching_enabled=config.caching.enabled)
    scanner = SecretScanner(entropy_threshold=config.secrets.entropy_threshold)

    runlog = stack.enter_context(RunLog(paths.events, run_id=run_id, observer=observer))
    runlog.event(
        EventKind.RUN_START,
        xeno_version=__version__,
        config_source=str(config.source_path or "defaults"),
        **event_fields,
    )
    stack.callback(lambda: runlog.event(EventKind.RUN_END, ok=succeeded()))

    router = Router(config, ledger=ledger, runlog=runlog, scanner=scanner)
    stack.callback(router.close)

    return (
        _RunContext(
            run_id=run_id,
            paths=paths,
            ledger=ledger,
            router=router,
            state=AgentState(run_id=run_id, goal=state_goal),
            keyring=CacheKeyring(
                run_id=run_id,
                worktree_root=paths.worktree if keyring_on_worktree else paths.workspace,
            ),
        ),
        runlog,
    )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool, typer.Option("--version", help="Print the version and exit.")
    ] = False,
    config_path: ConfigOption = None,
    repo: Annotated[
        Path, typer.Option("--repo", help="Repository to work in.")
    ] = Path("."),
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress the live graph and narration.")
    ] = False,
) -> None:
    """Bare `xeno` opens a spec conversation, then builds what you agreed.

    `xeno run "<goal>"` remains the non-interactive form for when the goal
    is already settled and for scripting.
    """
    if version:
        console.print(f"xeno-cli {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is not None:
        return
    if not console.is_terminal:
        _error("`xeno` with no arguments needs a terminal — use `xeno run \"<goal>\"` instead.")
        raise typer.Exit(code=2)
    _interactive(config_path, repo=repo, quiet=quiet)


def _interactive(config_path: Path | None, *, repo: Path, quiet: bool) -> None:
    """The `xeno`-with-no-arguments session."""
    config = _resolve_config(config_path)
    console.print(Panel(_SPEC_BANNER.strip(), border_style="cyan"))
    try:
        idea = Prompt.ask("\n[bold cyan]what are we building[/]").strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]nothing to do.[/]")
        raise typer.Exit(code=1) from None
    if not idea:
        console.print("[dim]nothing to do.[/]")
        raise typer.Exit(code=1)

    try:
        _execute(config, goal=idea, repo=repo, quiet=quiet, context_files=None, idea=idea)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]interrupted — nothing was squashed or pushed.[/]")
        raise typer.Exit(code=130) from None


@app.command()
def init(
    directory: Annotated[Path, typer.Argument(help="Where to write xeno.yaml.")] = Path("."),
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Write a starter xeno.yaml targeting Hardware Tier 1."""
    target = directory / CONFIG_FILENAME
    if target.exists() and not force:
        console.print(f"[yellow]{target} already exists.[/] Pass --force to overwrite.")
        raise typer.Exit(code=1)
    target.write_text(_STARTER_CONFIG)
    console.print(f"[green]wrote[/] {target}")
    console.print(
        "Hardware Tier 0 (8 GB VRAM): point medium's first entry at an API provider — "
        "a 14B local model will not fit (PRD S9.3)."
    )


@config_app.command("show")
def config_show(
    config_path: ConfigOption = None,
) -> None:
    """Render the resolved configuration and any capability warnings."""
    config = _resolve_config(config_path)
    source = config.source_path or "<built-in defaults>"
    console.print(Panel(str(source), title="config source", expand=False))

    table = Table("node", "callsign", "tier", "max tokens", title="node routing (static, v1)")
    for role in NodeRole:
        spec = config.nodes[role]
        table.add_row(role.value, CALLSIGNS[role], spec.tier.value, str(spec.max_tokens))
    console.print(table)

    chains = Table(
        "tier", "#", "provider", "model", "escalation", "$/1M in", title="fallback chains"
    )
    for tier in (Tier.FLAGSHIP, Tier.MEDIUM, Tier.LIGHT):
        for index, entry in enumerate(config.tiers.get(tier, ())):
            chains.add_row(
                tier.value if index == 0 else "",
                str(index),
                entry.provider,
                entry.model,
                "yes" if entry.escalation else "",
                f"{entry.usd_per_1m_input:.2f}" if entry.usd_per_1m_input is not None else "local",
            )
    console.print(chains)

    limits = Table("limit", "value", title="run caps (PRD S7.3)")
    for name, value in config.limits.model_dump().items():
        limits.add_row(name, str(value))
    console.print(limits)

    _print_capability_warnings(config)


@app.command()
def doctor(
    config_path: ConfigOption = None,
) -> None:
    """Check that every configured provider is reachable and usable."""
    config = _resolve_config(config_path)
    ledger = CostLedger(run_id="doctor")
    router = Router(config, ledger=ledger)

    table = Table("provider", "family", "status", "detail", title="providers")
    ok = True
    try:
        for name in sorted(config.providers):
            try:
                provider = router.provider(name)
            except UnsupportedProviderError as exc:
                table.add_row(
                    name,
                    config.providers[name].family.value,
                    "[red]unsupported[/]",
                    str(exc),
                )
                ok = False
                continue
            reachable, detail = provider.health_check()
            ok = ok and reachable
            table.add_row(
                name,
                provider.family.value,
                "[green]ok[/]" if reachable else "[red]fail[/]",
                detail,
            )
        console.print(table)
        for warning in router.warn_local_backends_without_prefix_cache():
            _warn(warning)
    finally:
        router.close()

    _print_capability_warnings(config)

    if not ok:
        raise typer.Exit(code=1)


@models_app.command("list")
def models_list(
    config_path: ConfigOption = None,
) -> None:
    """Show which model each node resolves to."""
    config = _resolve_config(config_path)
    table = Table("node", "callsign", "tier", "primary model", "fallbacks")
    for role in NodeRole:
        chain = config.chain_for(config.tier_for(role))
        table.add_row(
            role.value,
            CALLSIGNS[role],
            config.tier_for(role).value,
            chain[0].ref,
            ", ".join(e.ref for e in chain[1:]) or "—",
        )
    console.print(table)


@models_app.command("test")
def models_test(
    config_path: ConfigOption = None,
    repo: Annotated[Path, typer.Option("--repo", help="Repository root for .xeno/.")] = Path("."),
    skip_probe: Annotated[bool, typer.Option("--skip-probe")] = False,
) -> None:
    """Exercise every configured tier and emit cost.json.

    This is Phase 0's exit criterion (PRD S13). Each tier is called TWICE with
    an identical static prefix — one call would report connectivity but could
    say nothing at all about caching, and a cached/fresh breakdown is half of
    what the exit criterion asks for.
    """
    config = _resolve_config(config_path)
    _print_capability_warnings(config)
    failures: list[str] = []

    with ExitStack() as stack:
        ctx, _ = _start_run(
            config,
            stack,
            repo_root=repo.resolve(),
            state_goal="models test",
            succeeded=lambda: not ctx.ledger.has_unpriced_calls,
            command="models test",
        )

        if not skip_probe:
            for result in ctx.router.probe_caching():
                marker = "[green]yes[/]" if result.cache_capable else "[yellow]no[/]"
                console.print(
                    f"cache probe {result.provider}/{result.model}: {marker} — {result.evidence}"
                )

        table = Table(
            "tier",
            "node",
            "model",
            "call",
            "latency",
            "in / cached",
            "out",
            "usd",
            title="tier exercise",
        )

        for tier in (Tier.FLAGSHIP, Tier.MEDIUM, Tier.LIGHT):
            if tier not in config.tiers:
                continue
            role = _representative_node(config, tier)
            builder = PromptBuilder(
                node=role,
                keyring=ctx.keyring,
                system_text=_TEST_SYSTEM_PROMPT,
                caching_enabled=config.caching.enabled,
            )
            for call_index in (1, 2):
                prompt = builder.build(_TEST_TURN)
                try:
                    call = ctx.router.complete(role, prompt, state=ctx.state)
                except (ChainExhaustedError, UnsupportedProviderError) as exc:
                    failures.append(f"{tier.value}: {exc}")
                    table.add_row(
                        tier.value, role.value, "—", str(call_index), "—", "—", "—", "[red]FAIL[/]"
                    )
                    break
                usage = call.record.usage
                table.add_row(
                    tier.value if call_index == 1 else "",
                    role.value if call_index == 1 else "",
                    call.record.model_ref if call_index == 1 else "",
                    str(call_index),
                    f"{call.record.latency_ms:.0f}ms",
                    f"{usage.input_tokens} / {usage.cache_read_tokens}",
                    str(usage.output_tokens),
                    "local" if call.record.usd == 0 else f"${call.record.usd:.5f}",
                )
        console.print(table)

    _print_ledger_summary(ctx.ledger, ctx.ledger.write(ctx.paths.cost))

    if failures:
        for failure in failures:
            console.print(f"[red]tier failed:[/] {failure}")
        raise typer.Exit(code=1)


@app.command()
def run(
    goal: Annotated[str, typer.Argument(help="What the harness should accomplish.")],
    config_path: ConfigOption = None,
    repo: Annotated[
        Path, typer.Option("--repo", help="Source repository copied into a throwaway worktree.")
    ] = Path("."),
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Suppress the live graph and per-node narration. Implied when "
            "stdout is not a terminal, so piping or redirecting a run always "
            "produces a clean transcript.",
        ),
    ] = False,
    context_files: Annotated[
        list[str] | None,
        typer.Option(
            "--file",
            help="Existing file(s) known to be relevant, seeded into context ahead of "
            "Argus's own research. Repeatable. Optional — Argus (PRD S13 Phase 3) finds "
            "context on its own; this only adds to what it finds.",
        ),
    ] = None,
) -> None:
    """Plan and execute `goal` with the full seven-node graph (PRD S13 Phase 3-4).

    Odysseus breaks the goal into a plan of tasks; each task is researched by
    Argus, implemented by Daedalus, and evaluated by Talos's sandboxed gates
    (PRD S11) — zero host execution of generated code. A failing task climbs
    a bounded ladder (patch, re-research, roll back and rewrite, re-plan)
    before the run halts. Once every task is done (or the run halts),
    Cerberus reviews the diff and either approves it (you confirm, then it's
    squashed onto a dedicated branch and optionally opened as a PR),
    escalates it to you with a report, or sends it back into the loop with
    written objections. Operates on a throwaway copy of `repo` under
    .xeno/worktrees/<run_id>, never the user's working tree.
    """
    _execute(
        _resolve_config(config_path),
        goal=goal,
        repo=repo,
        quiet=quiet,
        context_files=context_files,
        idea=None,
    )


def _execute(
    config: XenoConfig,
    *,
    goal: str,
    repo: Path,
    quiet: bool,
    context_files: list[str] | None,
    idea: str | None,
) -> None:
    """One run, shared by `xeno run` and the interactive session.

    `idea` selects the interactive front half: when set, Odysseus JOB 2
    converses to a written spec first and its title replaces `goal`. Both
    halves live in ONE run context on purpose — the conversation's tokens
    are part of what the run cost, and splitting them across two ledgers
    would understate every figure the harness reports about itself.
    """
    _preflight(config)

    repo_root = repo.resolve()
    final_state: AgentState | None = None
    view = _make_view(quiet)

    with ExitStack() as stack:
        ctx, runlog = _start_run(
            config,
            stack,
            repo_root=repo_root,
            state_goal=goal,
            succeeded=lambda: bool(final_state and final_state.review_verdict is Verdict.APPROVE),
            keyring_on_worktree=True,
            observer=view.handle,
            command="run",
            goal=goal,
        )
        worktree = ctx.paths.worktree
        _copy_into_worktree(repo_root, worktree, config)

        if idea is not None:
            goal = _converse_to_spec(ctx, config, idea)

        # PRD S9.6.3: without this, `provider.cache_capable` stays None for
        # the whole run, `supports_explicit_cache_markers()` is False for
        # every aggregator, and no cache_control marker is ever emitted — so
        # the caching design would be inert in exactly the mode M1.4
        # measures. `probe_caching` is itself gated on
        # `caching.probe_aggregators_at_startup`, and only aggregators are
        # probed, so a purely local or Anthropic setup pays nothing here.
        for probe in ctx.router.probe_caching():
            if not probe.cache_capable:
                _warn(f"cache probe {probe.provider}/{probe.model}: {probe.evidence}")

        # Discovery needs `router`/`state` (PRD S8.2 revised: the one model
        # call that proposes which commands to run, cost-tracked like every
        # other node call) BEFORE the sandbox pool can be built — the pool
        # needs to know the image, and the image now depends on what
        # discovery finds. A cache hit (the common case after the first run
        # against a repo) costs zero model calls.
        try:
            toolchain = discover_toolchain(
                router=ctx.router,
                config=config,
                keyring=ctx.keyring,
                state=ctx.state,
                paths=ctx.paths,
                repo_root=repo_root,
                worktree=worktree,
                # Greenfield is a supported starting point, not a usage
                # error: a repo with no manifest resolves to an
                # unestablished toolchain and the run's first task is to
                # scaffold one (`xeno.graph.toolchain`). A repo that DOES
                # declare a toolchain we then fail to read still raises.
                allow_unestablished=True,
            )
        except DiscoveryError as exc:
            _error(f"toolchain discovery failed: {exc}")
            raise typer.Exit(code=2) from exc

        if not toolchain.established:
            _warn(
                "no manifest found — treating this as a new project. The first planned "
                "task will scaffold the toolchain (manifest, lint/type/test config); "
                "gates start running once it exists."
            )

        session = ToolchainSession(
            router=ctx.router,
            config=config,
            keyring=ctx.keyring,
            paths=ctx.paths,
            repo_root=repo_root,
            worktree=worktree,
            runlog=runlog,
            toolchain=toolchain,
        )
        session.start(stack)
        _seed_context_files(ctx.state, worktree, context_files)

        # The live view is only pinned around the graph itself: warnings and
        # preflight output above it are ordinary scrollback, and the human
        # gate below it needs the terminal back to prompt.
        with view.running():
            final_state = run_graph(
                router=ctx.router,
                config=config,
                keyring=ctx.keyring,
                paths=ctx.paths,
                worktree=worktree,
                runlog=runlog,
                state=ctx.state,
                session=session,
                repo_root=repo_root,
            )

    assert final_state is not None
    _print_run_summary(final_state, worktree)
    _print_ledger_summary(ctx.ledger, ctx.ledger.write(ctx.paths.cost))

    branch = run_branch_name(config.git.branch_prefix, goal, ctx.run_id)
    raise typer.Exit(code=_finalize(final_state, config, worktree, branch))


_SPEC_BANNER = """\
[bold]xeno[/] — describe what you want to build.

Odysseus will ask about anything it cannot reasonably assume, then write a
spec and hand it to the seven-node graph. Say 'go' at any point to build with
the assumptions so far, or press Ctrl-C to leave.
"""


def _converse_to_spec(ctx: _RunContext, config: XenoConfig, idea: str) -> str:
    """Run Odysseus JOB 2 to a written spec and make it this run's goal.

    The spec body is seeded as a context Handle rather than assigned to
    `state.goal`: `AgentState` fields are capped at 4 KB (PRD S6.3) and a
    real spec exceeds that, so the goal carries the one-line title and every
    node reads the full text through the same context mechanism it already
    uses for source files.
    """

    def show(text: str) -> None:
        console.print()
        console.print(Panel(text, title="odysseus", border_style="cyan", title_align="left"))

    try:
        spec = run_spec_conversation(
            router=ctx.router,
            config=config,
            keyring=ctx.keyring,
            state=ctx.state,
            paths=ctx.paths,
            idea=idea,
            ask=lambda label: Prompt.ask(f"\n[bold cyan]{label}[/]"),
            show=show,
        )
    except SpecAbandoned as exc:
        _error(f"no spec was produced: {exc}")
        raise typer.Exit(code=1) from exc
    except ChainExhaustedError as exc:
        _error(spec_chain_error(exc))
        raise typer.Exit(code=2) from exc

    spec_path = spec.write(ctx.paths.workspace)
    ctx.state.goal = spec.title
    ctx.state.context_handles = [
        *ctx.state.context_handles,
        Handle.for_file(spec_path, summary=f"spec: {spec.title}"),
    ]

    console.print()
    console.print(Panel(spec.body, title=spec.title, border_style="green", title_align="left"))
    console.print(f"[dim]spec written to {spec_path}[/]")
    if not Confirm.ask("\nBuild this?", default=True):
        console.print("[yellow]stopped — nothing was built.[/]")
        raise typer.Exit(code=1)
    return spec.title


def _make_view(quiet: bool) -> RunView:
    """The live view, or a no-op stand-in.

    Not a terminal means not a view, regardless of `--quiet`: a run whose
    output is being piped, redirected, or captured by CI should produce a
    plain transcript, and a `Live` region redrawing into a file produces
    neither a usable log nor a usable picture.
    """
    if quiet or not console.is_terminal:
        return NullRunView()
    return LiveRunView(console)


def _preflight(config: XenoConfig) -> None:
    """Environment and configuration checks that must happen before a run
    creates anything — a missing `git` is worth catching before a worktree
    copy, not after."""
    if shutil.which("git") is None:
        _error(
            "git was not found on PATH — required for the checkpoint substrate "
            "(PRD S13 Phase 3: every completed task is a commit, and a failed one "
            "rolls back to the last one)"
        )
        raise typer.Exit(code=2)

    _print_capability_warnings(config)
    if config.sandbox.network == "open":
        _warn(
            "sandbox network policy is 'open' (PRD S11.2) — generated code has "
            "unrestricted egress for this run"
        )
    if config.git.open_pr and shutil.which("gh") is None:
        _warn(
            "git.open_pr is true but gh was not found on PATH — PR creation will be "
            "skipped at the end of this run; the squashed commit will still land on "
            "the preserved local branch."
        )


def _seed_context_files(state: AgentState, worktree: Path, rels: list[str] | None) -> None:
    """PRD S13 Phase 3: `--file` only ADDS to what Argus finds on its own, so
    a bad path is a usage error worth failing on rather than quietly
    dropping."""
    for rel in rels or []:
        target = worktree / rel
        if not target.exists():
            _error(f"--file {rel} not found under {worktree}")
            raise typer.Exit(code=2)
        state.context_handles = [
            *state.context_handles,
            Handle.for_file(target, summary=f"user-provided context: {rel}"),
        ]


def _finalize(
    state: AgentState, config: XenoConfig, worktree: Path, branch: str
) -> int:
    """The post-graph disposition (PRD S8.1, S8.4), returning the process
    exit code rather than raising, so the decision stays testable
    independently of Typer."""
    if state.review_verdict is Verdict.APPROVE:
        if _human_gate(state, worktree, branch) != "approve":
            console.print(f"[yellow]declined by human — branch preserved, un-squashed:[/] {branch}")
            return 1
        assert state.commit_message is not None
        vcs.squash_to_one_commit(
            worktree, since=vcs.root_commit(worktree), message=state.commit_message
        )
        console.print(f"[green]squashed[/] onto {branch}")
        if config.git.open_pr:
            _open_pr(state, worktree, branch)
        return 0

    if state.review_verdict is Verdict.ESCALATE:
        _print_escalate_report(state, branch)
        return 1
    return 0 if _run_succeeded(state) else 1


def _open_pr(state: AgentState, worktree: Path, branch: str) -> None:
    assert state.commit_message is not None
    pushed = vcs.push_branch(worktree, branch)
    pr_url = (
        vcs.open_pr(
            worktree,
            branch=branch,
            title=state.commit_message.splitlines()[0],
            body=state.commit_message,
        )
        if pushed
        else None
    )
    if pr_url:
        console.print(f"[green]opened PR:[/] {pr_url}")
    else:
        _warn(
            "PR creation was skipped or failed — the squashed commit is still on "
            f"the preserved branch {branch}."
        )


def _run_succeeded(state: AgentState) -> bool:
    """Every plan task checkpointed and the run never halted.

    Distinct from Phase 2's only definition of success ("the last
    eval_report passed"): the last report reflects whichever task the run
    stopped on — the final task on a genuine multi-task success, but
    whatever task got stuck on a halt. Comparing `task_cursor` against
    `task_count` is what actually answers "did the whole plan complete."
    """
    return not state.halted and state.task_count > 0 and state.task_cursor >= state.task_count


def _human_gate(state: AgentState, worktree: Path, branch: str) -> Literal["approve", "reject"]:
    """The one human confirmation the whole system ever asks for (PRD S8.1):
    reached only on Cerberus's own APPROVE. This is a genuine veto distinct
    from Cerberus's REJECT_AND_RETURN — REJECT_AND_RETURN already happened
    automatically, before the human was ever consulted, and re-enters the
    graph; a human "reject" here does not — it leaves the branch preserved
    exactly like an ESCALATE, since APPROVE having already happened means
    there is nothing left to route back into.
    """
    assert state.review_diff_handle is not None
    assert state.cerberus_notes is not None
    assert state.commit_message is not None

    console.print(Panel(state.commit_message, title="proposed commit message"))
    console.print(Panel(state.cerberus_notes.read_text(), title="Cerberus's notes"))
    _print_diff_line(state.review_diff_handle)
    console.print(f"branch: {branch}")

    while True:
        choice = Prompt.ask(
            "Approve this change?", choices=["approve", "reject", "inspect"], default="inspect"
        )
        if choice == "inspect":
            with console.pager():
                console.print(state.review_diff_handle.read_text())
            continue
        return "approve" if choice == "approve" else "reject"


def _print_diff_line(handle: Handle) -> None:
    console.print(f"diff: {handle.path} ({handle.bytes} bytes)")


def _print_escalate_report(state: AgentState, branch: str) -> None:
    """PRD S8.3 ESCALATE: no action prompt beyond acknowledgment — the human
    decides offline whether to intervene, redirect, or abandon. `None`
    `cerberus_notes` means Cerberus itself failed (PRD S8.2 "Failure"): the
    harness never auto-approves in the absence of a review, so this is
    labeled UNREVIEWED rather than presented as a normal report.
    """
    if state.cerberus_notes is None:
        console.print(
            Panel(
                f"halt reason: {state.halt_reason}\n\n"
                "Cerberus itself could not complete a review (its model chain was "
                "exhausted). This diff has NOT been reviewed by anyone.",
                title="[bold red]UNREVIEWED[/]",
            )
        )
    else:
        console.print(Panel(state.cerberus_notes.read_text(), title="Cerberus's escalation report"))

    if state.review_diff_handle is not None:
        _print_diff_line(state.review_diff_handle)
    if state.eval_report is not None:
        console.print(f"last evaluation: {state.eval_report}")
    if state.checkpoints:
        last = state.checkpoints[-1]
        console.print(f"last green checkpoint: {last.sha[:12]} (task {last.task_index})")
    console.print(f"branch preserved for inspection: {branch}")


def _copy_into_worktree(repo_root: Path, worktree: Path, config: XenoConfig) -> None:
    """A throwaway copy Daedalus writes into — never the user's working tree
    (PRD S13 Phase 1 carve-out, still true in Phase 2's sandboxed gates)."""
    worktree.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo_root, worktree, ignore=mount_ignore(config.secrets))


def _print_run_summary(state: AgentState, worktree: Path) -> None:
    report = state.eval_report
    if state.halted:
        halted = f"[red]halted:[/] {state.halt_reason}"
        if state.unparsed_response is not None:
            # Printed with the halt rather than filed with the artifacts
            # below, because for a format failure this panel IS the end of
            # the trail: the objection says the response was unusable, and
            # nothing else on screen says what it actually was.
            halted += f"\n\nraw response saved to {state.unparsed_response.path}"
        console.print(Panel(halted, title="run result"))
    elif _run_succeeded(state):
        console.print(
            Panel(
                f"[green]passed[/] — {state.task_count}/{state.task_count} plan task(s) complete",
                title="run result",
            )
        )

    console.print(f"plan progress: {state.task_cursor}/{state.task_count} task(s) checkpointed")
    if state.checkpoints:
        cp_table = Table("task", "commit", "message", title="checkpoints")
        for cp in state.checkpoints:
            cp_table.add_row(str(cp.task_index), cp.sha[:12], cp.message)
        console.print(cp_table)

    if report is not None:
        table = Table("gate", "result", title="evaluation")
        if report.passed:
            table.add_row("required commands", "[green]all passed[/]")
        else:
            table.add_row("failed command", f"[red]{report.failed_command or 'infrastructure'}[/]")
        console.print(table)
        if report.first_failure:
            console.print(Panel(report.first_failure, title="first failure"))
        if report.full_log_handle:
            console.print(f"full log: {report.full_log_handle.path}")

    console.print(
        f"ladder_rung={state.ladder_rung} rung_attempts={state.rung_attempts} "
        f"iterations={state.iteration_count} signature_streak={state.signature_streak}"
    )
    if state.diff_handle:
        console.print(f"diff: {state.diff_handle.path}")
    console.print(f"worktree: {worktree}")


def _representative_node(config: XenoConfig, tier: Tier) -> NodeRole:
    """The first node declaring this tier.

    Using a real node rather than a synthetic one means the test exercises the
    same PromptBuilder, node spec, and token limits that a real run would.
    """
    for role in NodeRole:
        if config.tier_for(role) is tier:
            return role
    return NodeRole.EVALUATOR


def _print_ledger_summary(ledger: CostLedger, cost_path: Path) -> None:
    metrics = ledger.metrics()
    summary = Table("metric", "value", title="cost ledger")
    summary.add_row("calls", str(len(ledger.calls)))
    summary.add_row("usd", f"${ledger.usd_spent:.5f}")
    if ledger.has_unpriced_calls:
        summary.add_row("usd is a lower bound", "[yellow]yes — unpriced calls present[/]")
    share = metrics["M1.1_non_flagship_token_share"]
    summary.add_row(
        "M1.1 non-flagship token share",
        f"{share:.1%} (target ≥ 70%)" if share is not None else "n/a",
    )
    hit_rate = metrics["M1.4_cache_hit_rate"]
    summary.add_row(
        "M1.4 cache hit rate",
        f"{hit_rate:.1%} (target ≥ 40%)"
        if hit_rate is not None
        else "n/a — no cache-capable calls",
    )
    for tier, stats in ledger.latency_summary().items():
        # Square brackets are Rich markup, not literal text — "[flagship]" is
        # parsed as an (unrecognized, silently dropped) style tag rather than
        # rendered. Parentheses sidestep the collision instead of relying on
        # rich.markup.escape() being remembered at every call site.
        summary.add_row(f"latency p50 ({tier})", f"{stats['p50']:.0f}ms (n={stats['n']})")
    console.print(summary)
    console.print(f"[green]wrote[/] {cost_path}")

    breakdown = ledger.cache_stats_by_breakpoint()
    if breakdown:
        table = Table("breakpoint", "hit", "miss", "write", "hit rate", title="cached vs fresh")
        for name, stats in breakdown.items():
            rate = stats.hit_rate
            table.add_row(
                name,
                str(stats.hit_tokens),
                str(stats.miss_tokens),
                str(stats.write_tokens),
                f"{rate:.1%}" if rate is not None else "—",
            )
        console.print(table)


@app.command("show-cost")
def show_cost(
    path: Annotated[Path, typer.Argument(help="Path to a cost.json.")],
) -> None:
    """Pretty-print a cost.json from a previous run."""
    data = json.loads(path.read_text())
    console.print(Panel(json.dumps(data["metrics"], indent=2), title=f"metrics — {data['run_id']}"))
    console.print(Panel(json.dumps(data["totals"], indent=2), title="totals"))


_STARTER_CONFIG = """\
# Xeno CLI configuration (PRD S9.5). Defaults target Hardware Tier 1
# (Apple Silicon, 32 GB unified memory).
#
# Hardware Tier 0 (8 GB VRAM): set medium's first entry to an API provider.
# A 14B local model will not fit. See PRD S9.3.

tiers:
  flagship:
    - {provider: openrouter, model: z-ai/glm-5.2, usd_per_1m_input: 1.40, usd_per_1m_output: 2.20}
  medium:
    - {provider: ollama, model: "qwen2.5-coder:14b"}
    # Upward fallback must be declared explicitly. It is logged and counted
    # against M1.1 by the model actually billed (PRD S9.5).
    - {provider: openrouter, model: z-ai/glm-5.2, escalation: true,
       usd_per_1m_input: 1.40, usd_per_1m_output: 2.20}
  light:
    - {provider: ollama, model: "qwen2.5-coder:7b"}

nodes:
  planner:    {tier: flagship, max_tokens: 32000}   # Odysseus
  researcher: {tier: light}                         # Argus
  specifier:  {tier: flagship, max_tokens: 32000}   # Lachesis
  coder:      {tier: medium}                        # Daedalus
  evaluator:  {tier: light}                         # Talos
  debugger:   {tier: medium}                        # Chiron
  reviewer:   {tier: flagship, max_tokens: 32000}   # Cerberus

limits:
  max_usd_per_run: 10.00
  max_runtime_minutes: 120
  max_iterations_per_task: 25
  max_iterations_per_run: 200
  max_rejections_per_run: 2
  max_deleted_lines: 200

git:
  branch_prefix: "xeno/"
  open_pr: false          # set true + install `gh` to open a PR on APPROVE

caching:
  enabled: true
  probe_aggregators_at_startup: true

sandbox:
  network: none          # none | install | open  (PRD S11.2)
  memory: 2g
  cpus: 2
  warm_pool_size: 2

secrets:
  # Appended to the built-in denylist; never replaces it.
  extra_denylist: []
  scan_outbound_context: true
"""


if __name__ == "__main__":
    app()
