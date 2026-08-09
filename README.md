# Xeno CLI

A terminal-native, open-source multi-agent coding harness. It runs a six-node
graph over your repository — plan, research, write, test, debug, review — and
stops exactly once, at a human review gate, before anything reaches your
default branch.

> **Status: Phase 2 (sandbox and gates).** `xeno run <goal>` drives a real
> Daedalus → Talos → Chiron loop: the coder writes, the evaluator runs
> parse/lint/type/test/coverage gates inside a sandboxed, network-isolated
> container from a warm pool, and the debugger patches a specific failure
> (bounded by the L0/L1 ladder rungs and CB-1 through CB-6). Odysseus, Argus,
> and Cerberus — planning, research, and the human review gate — do not exist
> yet; `xeno run` operates on a single task against a throwaway worktree,
> never your working tree.

Built from `XenoCLI-PRD-v2.2.txt`. Section references throughout the code point
back at it.

## The wedge

Not "cheaper than a subscription" — Aider, Cline, OpenHands, and opencode are
all free. Four things, together:

- **D1. Cost-shaped routing as a feature, not a model dropdown.** A *different*
  model per node, under a declared budget ceiling.
- **D2. The graph is the product surface.** Add a node or change an edge
  condition by editing YAML. The orchestration is the API, not the moat.
- **D3. One well-defined autonomy boundary.** Not "auto-approve everything,"
  not "confirm every write." One gate, where review has leverage: the completed
  diff.
- **D4. Failure is a designed subsystem.** A bounded escalation ladder with
  circuit breakers, not an unbounded retry loop. Most agent harnesses fail by
  looping; this one is built to notice and stop.

## The Mortal Forge

Six nodes. Roles are primary; callsigns are shorthand used in logs and the TUI.

| Role | Callsign | Tier | Does |
| --- | --- | --- | --- |
| Planner | Odysseus | flagship | Decomposes the goal into verifiable tasks |
| Researcher | Argus | light | Returns exactly the context others need, as handles |
| Coder | Daedalus | medium | Implements the current plan task |
| Evaluator | Talos | light | Objective gates: parse, lint, types, tests |
| Debugger | Chiron | medium | Diagnoses a failure and patches it |
| Reviewer | Cerberus | flagship | Holistic review, the human gate, git authority |

Separation of powers is load-bearing: the nodes that write code (Daedalus,
Chiron) have no shell access, so the node that writes code does not get to
declare it working.

## Install

Requires Python 3.11+. The package is `xeno-cli`; the command is `xeno`.

PRD OQ-6, re-checked after the rename: bare `xeno` is taken on PyPI by
[lainproliant/xeno](https://pypi.org/project/xeno/), a Python IOC and build
framework. `xeno-cli` is free on PyPI and is what this package claims. (`xeno-cli`
is taken on npm by an unrelated file-generation tool — irrelevant for a Python
package, but worth knowing before any npm distribution.)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Try it

```bash
xeno init                 # write a starter xeno.yaml (Hardware Tier 1 defaults)
xeno config show          # resolved config, routing table, run caps, warnings
xeno doctor               # provider reachability and capability checks
xeno models test          # Phase 0 exit criterion: exercise every tier, emit cost.json
xeno run "add a --priority filter to the list command" --file src/pkg/cli.py
xeno show-cost .xeno/runs/<run_id>/cost.json
```

`xeno models test` calls each configured tier **twice** with an identical static
prefix. One call would prove connectivity but say nothing about caching, and a
cached-vs-fresh breakdown is half of what the exit criterion asks for.

`xeno run` needs a reachable Docker daemon — gates execute inside a sandboxed
container (PRD S11), never on the host. `--file` is repeatable and is the
manual stand-in for Argus (PRD OQ-10, Argus itself lands in Phase 3): name
the file(s) relevant to the task and Daedalus sees only their content; leave
it unset and it sees every non-test file's content, which costs more context
for no benefit once the task is scoped to a few files.

## Configuration

`xeno.yaml` declares providers, per-tier fallback chains, per-node tiers, and
run caps. Defaults target Hardware Tier 1 (Apple Silicon, 32 GB). On Hardware
Tier 0 (8 GB VRAM) point medium's first entry at an API provider — a 14B local
model will not fit.

Two routing rules the loader enforces rather than documents:

- **Downward fallback is forbidden.** An exhausted chain halts the run. Quietly
  serving a weaker model produces worse output *and* hides the cost problem.
- **Upward fallback must be declared** (`escalation: true`), is logged, and is
  counted against M1.1 by the model actually billed — so an escalation cannot
  hide behind the node's declared tier.

Unpriced remote models are rejected at load: CB-3 cannot enforce a USD ceiling
it cannot compute. Override with `allow_unpriced_models: true` and accept that
cost totals become a lower bound.

`sandbox:` controls the container every gate runs in (PRD S11): non-root,
all capabilities dropped, read-only root filesystem, resource-capped, and a
`warm_pool_size` of pre-started containers so no `xeno run` pays a
per-evaluation cold start. `network` defaults to `none`; `install` briefly
enables network for a declared-dependency install step, then commits and
drops it before any generated code runs; `open` leaves it on for the whole
run and prints a warning.

## Local-only, honestly

Fully local is a supported mode. "Fully local with no quality loss" is a claim
this project will not make. On Tier 0/1 hardware the two nodes that most need
frontier reasoning — Odysseus and Cerberus — would run a 7B–32B model. The CLI
prints a capability warning whenever the flagship tier resolves to any locally
served model, at any size. There is no parameter threshold that makes a local
model flagship-class on this hardware, so there is no threshold in the check.

## Architecture notes

**Filesystem as working memory (T2).** Large artifacts never enter shared
state. They are written to `.xeno/` and passed as handles — `{path, sha256,
summary, bytes}`. `AgentState` enforces a hard 4 KB per-field limit at
construction, because context-window exhaustion surfaces very far from the line
of code that caused it.

**Static-first prompt assembly (T8).** Every prompt is built
`SYSTEM → CODEBASE MAP → ACCUMULATED HISTORY → CURRENT TURN`. This is a
construction rule, not an optimization pass: caching only pays off if nothing
upstream of a breakpoint ever changes. The builder owns ordering, history is
append-only and verified as such, and a node's system text is fingerprinted so
that interpolating anything dynamic into it raises instead of silently costing
the run its cache hits.

**Cache invalidation is derived, not managed.** The codebase map is invalidated
the moment a node writes to the worktree. A stale map served from cache is
worse than a miss — Argus's summaries would describe pre-edit code with full
confidence — so `CacheKeyring` refuses to serve a key it knows to be stale.

**Untrusted repository content.** Retrieved file content is wrapped in labelled
DATA blocks with content-derived guards, and handle summaries are generated by
the harness from counted facts, never copied from file text. A file cannot
inject through the line that describes it.

**Aggregator caching is unverified (OQ-11).** Support and pricing for
open-weight models served via OpenRouter were not established. The router probes
at startup — duplicate prefix, inspect reported cached tokens — and reports
capability only on positive evidence. Latency is recorded but never used to
declare success on its own, because a false positive would put fabricated
savings straight into `cost.json`. Providers that fail the probe get
cache-dependent projections disabled rather than estimated.

## Measurement

Two frozen populations, committed before any measurement:

- **XENO-SUITE-30** — 30 tasks (10 single-file, 15 multi-file, 5 deliberately
  infeasible) with mechanical acceptance criteria.
- **SWEBV-50** — 50 SWE-bench Verified instances, every 10th of the
  lexicographically sorted 500.

`cost.json` is the instrument for M1.1–M1.4 and is written from day one so the
metrics stay falsifiable:

| Metric | Target |
| --- | --- |
| M1.1 non-flagship share of billed tokens | ≥ 70% |
| M1.2 USD reduction vs. all-flagship baseline | ≥ 50% |
| M1.3 median USD per completed task | ≤ $0.50 |
| M1.4 cache hit rate on eligible input tokens | ≥ 40% |

M1.4 is a lever *inside* M1.2, not an additive claim — do not sum them.

## Roadmap

| Phase | Delivers | Exit |
| --- | --- | --- |
| **0** ✅ | Config, router, prompt caching, secret scanner, cost ledger, CB-1/3/4, frozen suite | `xeno models test` emits a valid `cost.json` |
| **1** ✅ | LangGraph state machine, Daedalus + Talos | Single-file task end to end, medium tier on a **local** model (the R1 test) |
| **2** ✅ | Docker sandbox, warm pool, full gate chain, Chiron, L0/L1, CB-2/5/6 | Failing test diagnosed and patched with zero host execution |
| 3 | Odysseus + Argus, filesystem-as-memory, L2/L4/L5, checkpoints | Multi-file task autonomous; infeasible task halts cleanly; 6-node beats 2-node baseline (the R3 test) |
| 4 | Cerberus, git layer, reject-and-return loop, the gate UI | One reviewable branch, human consulted exactly once, working tree untouched |

## License

MIT (PRD OQ-5 — pending confirmation before public release).
