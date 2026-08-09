"""The `xeno` command (PRD S13).

`xeno run` drives the full six-node graph (PRD S13 Phase 3): Odysseus plans
the goal into tasks, Argus researches each one, Daedalus implements it,
Talos's sandboxed gates evaluate it, and a bounded L0-L5 escalation ladder
(re-run, patch, re-research, roll back and rewrite, re-plan, halt) handles
failure — all on a throwaway worktree that is never the user's working
tree. Everything needed to trust the numbers behind that — config, routing,
prompt construction, secret scanning, the cost ledger — is exercised
standalone by `xeno models test`, Phase 0's exit criterion.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

import docker
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from xeno import __version__
from xeno.adapters.python import PythonAdapter
from xeno.core.config import (
    CONFIG_FILENAME,
    ConfigError,
    XenoConfig,
    load_config,
)
from xeno.core.ledger import CostLedger
from xeno.core.paths import RunPaths, new_run_id
from xeno.core.runlog import EventKind, RunLog
from xeno.core.state import AgentState, Handle
from xeno.core.types import CALLSIGNS, NodeRole, Tier
from xeno.graph import run_graph
from xeno.prompt.assembly import PromptBuilder
from xeno.prompt.keys import CacheKeyring
from xeno.router.providers import UnsupportedProviderError
from xeno.router.router import ChainExhaustedError, Router
from xeno.sandbox.pool import WarmPool, prepare_gate_image
from xeno.security.mounts import mount_ignore
from xeno.security.scanner import SecretScanner

app = typer.Typer(
    name="xeno",
    help="Terminal-native multi-agent coding harness. One human gate, tier-routed models.",
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(name="models", help="Inspect and exercise configured model tiers.")
config_app = typer.Typer(name="config", help="Inspect configuration.")
app.add_typer(models_app)
app.add_typer(config_app)

console = Console()

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


def _capability_warnings(config: XenoConfig) -> list[str]:
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
    return warnings


@app.callback(invoke_without_command=True)
def main(
    version: Annotated[
        bool, typer.Option("--version", help="Print the version and exit.")
    ] = False,
) -> None:
    if version:
        console.print(f"xeno-cli {__version__}")
        raise typer.Exit()


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
    config_path: Annotated[Path | None, typer.Option("--config", "-c")] = None,
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

    for warning in _capability_warnings(config):
        console.print(f"[yellow]warning:[/] {warning}")


@app.command()
def doctor(
    config_path: Annotated[Path | None, typer.Option("--config", "-c")] = None,
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
            console.print(f"[yellow]warning:[/] {warning}")
    finally:
        router.close()

    for warning in _capability_warnings(config):
        console.print(f"[yellow]warning:[/] {warning}")

    if not ok:
        raise typer.Exit(code=1)


@models_app.command("list")
def models_list(
    config_path: Annotated[Path | None, typer.Option("--config", "-c")] = None,
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
    config_path: Annotated[Path | None, typer.Option("--config", "-c")] = None,
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
    for warning in _capability_warnings(config):
        console.print(f"[yellow]warning:[/] {warning}")

    run_id = new_run_id()
    paths = RunPaths(repo_root=repo.resolve(), run_id=run_id).ensure()
    ledger = CostLedger(run_id=run_id, caching_enabled=config.caching.enabled)
    scanner = SecretScanner(entropy_threshold=config.secrets.entropy_threshold)
    failures: list[str] = []

    with RunLog(paths.events, run_id=run_id) as runlog:
        runlog.event(
            EventKind.RUN_START,
            command="models test",
            xeno_version=__version__,
            config_source=str(config.source_path or "defaults"),
        )
        router = Router(config, ledger=ledger, runlog=runlog, scanner=scanner)
        state = AgentState(run_id=run_id, goal="models test")
        keyring = CacheKeyring(run_id=run_id, worktree_root=paths.workspace)

        try:
            if not skip_probe:
                for result in router.probe_caching():
                    marker = "[green]yes[/]" if result.cache_capable else "[yellow]no[/]"
                    console.print(
                        f"cache probe {result.provider}/{result.model}: "
                        f"{marker} — {result.evidence}"
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
                    keyring=keyring,
                    system_text=_TEST_SYSTEM_PROMPT,
                    caching_enabled=config.caching.enabled,
                )
                for call_index in (1, 2):
                    prompt = builder.build(_TEST_TURN)
                    try:
                        call = router.complete(role, prompt, state=state)
                    except (ChainExhaustedError, UnsupportedProviderError) as exc:
                        failures.append(f"{tier.value}: {exc}")
                        table.add_row(
                            tier.value,
                            role.value,
                            "—",
                            str(call_index),
                            "—",
                            "—",
                            "—",
                            "[red]FAIL[/]",
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
        finally:
            router.close()
            runlog.event(EventKind.RUN_END, ok=not ledger.has_unpriced_calls)

    cost_path = ledger.write(paths.cost)
    _print_ledger_summary(ledger, cost_path)

    if failures:
        for failure in failures:
            console.print(f"[red]tier failed:[/] {failure}")
        raise typer.Exit(code=1)


@app.command()
def run(
    goal: Annotated[str, typer.Argument(help="What the harness should accomplish.")],
    config_path: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    repo: Annotated[
        Path, typer.Option("--repo", help="Source repository copied into a throwaway worktree.")
    ] = Path("."),
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
    """Plan and execute `goal` with the full six-node graph (PRD S13 Phase 3).

    Odysseus breaks the goal into a plan of tasks; each task is researched by
    Argus, implemented by Daedalus, and evaluated by Talos's sandboxed gates
    (PRD S11) — zero host execution of generated code. A failing task climbs
    a bounded ladder (patch, re-research, roll back and rewrite, re-plan)
    before the run halts with a report. Operates on a throwaway copy of
    `repo` under .xeno/worktrees/<run_id>, never the user's working tree.
    """
    if shutil.which("git") is None:
        console.print(
            "[red]error:[/] git was not found on PATH — required for the checkpoint "
            "substrate (PRD S13 Phase 3: every completed task is a commit, and a failed "
            "one rolls back to the last one)"
        )
        raise typer.Exit(code=2)

    config = _resolve_config(config_path)
    for warning in _capability_warnings(config):
        console.print(f"[yellow]warning:[/] {warning}")
    if config.sandbox.network == "open":
        console.print(
            "[yellow]warning:[/] sandbox network policy is 'open' (PRD S11.2) — "
            "generated code has unrestricted egress for this run"
        )

    run_id = new_run_id()
    repo_root = repo.resolve()
    paths = RunPaths(repo_root=repo_root, run_id=run_id).ensure()
    worktree = paths.worktree
    _copy_into_worktree(repo_root, worktree, config)

    ledger = CostLedger(run_id=run_id, caching_enabled=config.caching.enabled)
    scanner = SecretScanner(entropy_threshold=config.secrets.entropy_threshold)
    final_state: AgentState | None = None

    adapter = PythonAdapter()
    docker_client = docker.from_env()
    image, network_disabled = prepare_gate_image(
        docker_client,
        adapter,
        config.sandbox,
        worktree=worktree,
        workspace_root=paths.workspace,
        secrets=config.secrets,
    )
    pool = WarmPool(
        docker_client,
        adapter,
        config.sandbox,
        workspace_root=paths.workspace / "sandboxes",
        image=image,
        network_disabled=network_disabled,
    )
    pool.fill()

    with RunLog(paths.events, run_id=run_id) as runlog:
        runlog.event(
            EventKind.RUN_START,
            command="run",
            goal=goal,
            xeno_version=__version__,
            config_source=str(config.source_path or "defaults"),
        )
        router = Router(config, ledger=ledger, runlog=runlog, scanner=scanner)
        state = AgentState(run_id=run_id, goal=goal)
        keyring = CacheKeyring(run_id=run_id, worktree_root=worktree)

        for rel in context_files or []:
            target = worktree / rel
            if not target.exists():
                console.print(f"[red]error:[/] --file {rel} not found under {worktree}")
                router.close()
                pool.shutdown()
                docker_client.close()
                raise typer.Exit(code=2)
            state.context_handles = [
                *state.context_handles,
                Handle.for_file(target, summary=f"user-provided context: {rel}"),
            ]

        try:
            final_state = run_graph(
                router=router,
                config=config,
                keyring=keyring,
                paths=paths,
                worktree=worktree,
                runlog=runlog,
                state=state,
                pool=pool,
                adapter=adapter,
            )
        finally:
            router.close()
            pool.shutdown()
            docker_client.close()
            ok = bool(final_state and _run_succeeded(final_state))
            runlog.event(EventKind.RUN_END, ok=ok)

    cost_path = ledger.write(paths.cost)
    assert final_state is not None
    _print_run_summary(final_state, worktree)
    _print_ledger_summary(ledger, cost_path)

    if not _run_succeeded(final_state):
        raise typer.Exit(code=1)


def _run_succeeded(state: AgentState) -> bool:
    """Every plan task checkpointed and the run never halted.

    Distinct from Phase 2's only definition of success ("the last
    eval_report passed"): the last report reflects whichever task the run
    stopped on — the final task on a genuine multi-task success, but
    whatever task got stuck on a halt. Comparing `task_cursor` against
    `task_count` is what actually answers "did the whole plan complete."
    """
    return not state.halted and state.task_count > 0 and state.task_cursor >= state.task_count


def _copy_into_worktree(repo_root: Path, worktree: Path, config: XenoConfig) -> None:
    """A throwaway copy Daedalus writes into — never the user's working tree
    (PRD S13 Phase 1 carve-out, still true in Phase 2's sandboxed gates)."""
    worktree.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo_root, worktree, ignore=mount_ignore(config.secrets))


def _print_run_summary(state: AgentState, worktree: Path) -> None:
    report = state.eval_report
    if state.halted:
        console.print(Panel(f"[red]halted:[/] {state.halt_reason}", title="run result"))
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
        table.add_row("parse", "[green]ok[/]" if report.parse_ok else "[red]fail[/]")
        table.add_row(
            "lint", "[green]0[/]" if report.lint_errors == 0 else f"[red]{report.lint_errors}[/]"
        )
        table.add_row(
            "types", "[green]0[/]" if report.type_errors == 0 else f"[red]{report.type_errors}[/]"
        )
        table.add_row(
            "tests",
            f"[green]{report.tests_run}/{report.tests_run}[/]"
            if report.tests_failed == 0
            else f"[red]{report.tests_failed} failed[/] of {report.tests_run}",
        )
        if report.coverage_delta is not None:
            arrow = "+" if report.coverage_delta >= 0 else ""
            table.add_row("coverage delta", f"{arrow}{report.coverage_delta:.1f}pp")
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
  coder:      {tier: medium}                        # Daedalus
  evaluator:  {tier: light}                         # Talos
  debugger:   {tier: medium}                        # Chiron
  reviewer:   {tier: flagship, max_tokens: 32000}   # Cerberus

limits:
  max_usd_per_run: 2.00
  max_runtime_minutes: 45
  max_iterations_per_task: 12
  max_iterations_per_run: 60
  max_rejections_per_run: 2
  max_deleted_lines: 200

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
