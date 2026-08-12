# Xeno CLI

**A terminal-native multi-agent coding harness that knows when to stop.**

Six specialist agents — plan, research, write, test, debug, review — run as a
LangGraph state machine over a throwaway copy of your repository. Every model
call is routed to the cheapest tier that can do the job, every line of
generated code is executed inside a network-isolated container, and the whole
run interrupts you exactly once: at a human review gate, on a finished diff,
before anything touches your working tree.

Most agent harnesses fail by looping. This one is built to notice and halt.

```
  Argus  ─▶  Odysseus  ─▶  Lachesis
 skeleton     roadmap       expand
                              │
   ┌──── once per milestone ──┘
   ▼
  Argus  ─▶  Daedalus  ─▶  Talos  ─┬─▶  ckpt  ─▶  Lachesis  ─▶  Cerberus  ─▶  you
 research      write       gates   │              verify         review
   ▲                               │
   └────── Chiron ◀────────────────┘
          patch   (bounded ladder: L0…L5)
```

Two nested loops, because the work is knowable at two different times.
Odysseus maps the goal into coarse **milestones** before any code exists.
Lachesis expands one milestone into concrete tasks immediately before it is
built — with the previous milestones already on disk, so it names real
modules instead of guessing at them — and closes the milestone by writing the
tests that prove it.

**Status:** release v1 complete — all seven nodes, sandboxed gates, the
escalation ladder, checkpoints, the git layer, and the human gate are
implemented and wired. Built from `XenoCLI-PRD-v2.2.txt`; section references
throughout the code point back at it.

## Table of contents

- [Why this exists](#why-this-exists)
- [The Mortal Forge](#the-mortal-forge)
- [Prerequisites and installation](#prerequisites-and-installation)
  - [How much machine you actually need](#how-much-machine-you-actually-need)
- [Usage](#usage)
  - [First run](#first-run)
  - [Running a goal](#running-a-goal)
  - [What happens when a task fails](#what-happens-when-a-task-fails)
  - [The human gate](#the-human-gate)
  - [Cost reporting](#cost-reporting)
- [Configuration](#configuration)
  - [Routing and tiers](#routing-and-tiers)
  - [Sandbox](#sandbox)
  - [Run caps and circuit breakers](#run-caps-and-circuit-breakers)
  - [Full reference](#full-reference)
- [Tech stack](#tech-stack)
- [Architecture notes](#architecture-notes)
- [License](#license)

## Why this exists

Not "cheaper than a subscription" — Aider, Cline, OpenHands, and opencode are
all free. Four things, together:

- **Cost-shaped routing as a feature, not a model dropdown.** A *different*
  model per node, under a declared budget ceiling.
- **The graph is the product surface.** Add a node or change an edge condition
  by editing YAML. The orchestration is the API, not the moat.
- **One well-defined autonomy boundary.** Not "auto-approve everything," not
  "confirm every write." One gate, where review has leverage: the completed
  diff.
- **Failure is a designed subsystem.** A bounded escalation ladder with
  circuit breakers, not an unbounded retry loop.

## The Mortal Forge

Roles are primary; callsigns are shorthand used in logs and output.

| Role | Callsign | Tier | Does |
| --- | --- | --- | --- |
| Planner | Odysseus | flagship | Maps the goal into coarse milestones, once, up front |
| Researcher | Argus | light | Returns exactly the context others need, as handles |
| Specifier | Lachesis | flagship | Expands the next milestone into tasks; writes its tests |
| Coder | Daedalus | medium | Implements the current plan task |
| Evaluator | Talos | light | Runs the discovered gate commands; reports exit codes |
| Debugger | Chiron | medium | Diagnoses a specific failure and patches it |
| Reviewer | Cerberus | flagship | Holistic review, the human gate, git authority |

Separation of powers is load-bearing: the nodes that write code (Daedalus,
Chiron) have no shell access, so the node that writes code does not get to
declare it working. It cuts the other way too — Lachesis is the *only* node
allowed to write a test file, and Daedalus and Chiron are refused any
response that touches one. A failing test therefore has no path to being
weakened; the ladder can only fix the code. Talos's verdict is decided entirely by process exit codes
in plain Python — its single model call only compresses a failure log into a
≤500-character excerpt and can never flip a pass into a fail.

## Prerequisites and installation

| Requirement | Why |
| --- | --- |
| **Python 3.11+** | `StrEnum`, `X \| Y` unions at runtime |
| **Docker** (running daemon) | Every gate command executes in a container; nothing generated runs on the host |
| **At least one model provider** | Ollama on `localhost:11434`, or an API key for OpenRouter / OpenAI-compatible endpoint |
| `git` on PATH | Checkpoints, the review diff, the squash onto a branch |
| `gh` on PATH *(optional)* | Only if you set `git.open_pr: true` |

### How much machine you actually need

Docker is the floor and it is not optional: the warm pool holds
`warm_pool_size` containers at `sandbox.memory` each — 4 GB with the shipped
defaults — for the whole run, on top of whatever the models cost.

| Setup | Memory | What runs where |
| --- | --- | --- |
| **All API** *(recommended)* | ~8 GB | Every node on OpenRouter. Memory is Docker plus the OS; the machine barely notices the run |
| **Hybrid** *(shipped default)* | 16 GB | flagship on API; `medium`/`light` local. Roughly 13 GB of weights *if both local models stay resident*, so 16 GB is workable and 24 GB is comfortable |
| **Fully local** | 32 GB+ | See the caveats below before choosing this |

Two things about the local numbers are worth knowing before you plan around
them. Neither is something you configure — both are handled — but both shape
how much memory a local run actually wants.

**Context is the real constraint, not parameter count.** On this project's own
runs the specifier sent 8.3k-token prompts against a nearly empty repository,
and that figure grows with the codebase. A local model serving a flagship node
therefore needs a 16k–32k window, and the KV cache for it is charged on top of
the weights: a 14B at Q4 costs about 9.5 GB at 4k context and roughly 15 GB at
32k. Xeno derives the window per call from the prompt it just assembled,
rounds it up to a bucket so the model is not reloaded on every call, and
clamps it to what the model reports. You are never asked to guess a number.

This matters because Ollama otherwise serves its own default window — 4096 on
current daemons — regardless of what the model supports, and **silently
discards the front of any prompt that exceeds it**, which is where the system
prompt lives. A model that appears to ignore the output format may simply
never have been shown it. If a prompt genuinely cannot fit the model's window,
the run stops with the two numbers in the message rather than sending a call
that would be answered from half a prompt.

**Models are released when the run ends.** Weights are held resident between
calls, because reloading them between ladder iterations would dominate the
latency the local tier exists to keep low — so during a run both the `medium`
and `light` models stay in memory at once, which is where the 13 GB figure
above comes from. A daemon has no way to know a run finished, so Xeno unloads
them explicitly at teardown; you should get the memory back within a second of
the run ending, not when a keep-alive timer expires. Set
`release_on_exit: false` on the provider if you run back-to-back and would
rather keep them warm.

None of the above is a quality claim; see
[Routing and tiers](#routing-and-tiers) for that. Enough memory makes a local
model *run*. It does not make it flagship-class.

The distribution is `xeno-cli`; the command is `xeno`.

```bash
git clone <this-repo> && cd zeno_cli
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

xeno --version
xeno doctor        # provider reachability + capability warnings
```

For a fully local setup, pull the models your config names:

```bash
ollama pull qwen2.5-coder:14b   # medium tier — Daedalus, Chiron
ollama pull qwen2.5-coder:7b    # light tier  — Argus, Talos
```

> PyPI's bare `xeno` is taken by [lainproliant/xeno](https://pypi.org/project/xeno/),
> an unrelated IOC and build framework, so the package claims `xeno-cli` while
> `xeno` stays the invoked command.

## Usage

### First run

```bash
xeno init                 # write a starter xeno.yaml (Hardware Tier 1 defaults)
xeno config show          # resolved config, routing table, run caps, warnings
xeno doctor               # provider reachability and capability checks
xeno models list          # which model each node actually resolves to
xeno models test          # exercise every tier, emit a cost.json
```

`xeno models test` calls each configured tier **twice** with an identical
static prefix. One call would prove connectivity but say nothing about
caching, and a cached-vs-fresh breakdown is half of what this check is for.

### Running a goal

```bash
xeno run "add a --priority filter to the list command"
```

The run copies your repo into `.xeno/worktrees/<run_id>/` and works only
there. In order: Odysseus maps out milestones; Lachesis expands the next one
into tasks; then for each task Argus selects context, Daedalus writes code,
and Talos runs the gates in a container. Each green task is checkpointed as a
commit. When a milestone's tasks are done, Lachesis writes its tests and the
gates run in full — implementation tasks are gated on everything *except* the
test command, since a task cannot be checked against tests describing code it
has not written yet. When the last milestone is verified, Cerberus reviews
the whole accumulated diff.

Useful flags:

```bash
# Seed context by hand — repeatable. This only ADDS to what Argus finds.
xeno run "fix the off-by-one in pagination" --file src/pkg/list.py

# Point at a different repository (default: cwd)
xeno run "add retries to the HTTP client" --repo ../other-project

# Use a config outside the repo
xeno run "bump the parser" --config ~/configs/xeno.yaml
```

**Gate commands are discovered, not hardcoded.** On the first run against a
repo, a light-tier model reads the manifest files (`pyproject.toml`,
`package.json`, `Cargo.toml`, `go.mod`, `Makefile`, …) and *proposes* the
install / lint / typecheck / test commands. Every proposed `argv[0]` is
checked against an executable allowlist before it is cached or run, and
everything executes as a fixed argument vector — never a shell string. The
result is cached at `.xeno/discovery/<fingerprint>.json`, so subsequent runs
cost **zero** discovery calls until a manifest actually changes. The file is
plain JSON: edit it if the model picked wrong.

Required commands run in order and fail fast on the first non-zero exit.
Advisory commands (coverage, and anything else marked advisory) run only when
everything required is green and can never flip the verdict.

### What happens when a task fails

Failure is a state machine, not a retry loop. A failing task climbs a bounded
ladder and stops:

| Rung | Response |
| --- | --- |
| **L0** | Retry as-is |
| **L1** | Chiron patches the specific failure |
| **L2** | Argus re-researches — the context was probably wrong |
| **L3** | Roll back to the last checkpoint and rewrite |
| **L4** | Lachesis re-expands the milestone around the stuck task |
| **L5** | Halt and escalate to a human |

Six circuit breakers cut across the ladder and can halt a run at any rung:
**CB-1** iteration cap, **CB-2** wall clock, **CB-3** USD budget, **CB-4** no
progress (the same failure signature surviving an intervention), **CB-5** diff
thrash, **CB-6** destructive action.

### The human gate

Reached only on Cerberus's own APPROVE — the single confirmation the system
ever asks for. You get the proposed commit message, Cerberus's notes, and the
full diff (`inspect` opens it in a pager), then choose:

- **approve** → squashed into one commit on `xeno/<goal-slug>-<short-run-id>`, and opened
  as a PR if `git.open_pr` is set. Exit 0.
- **reject** → the branch is preserved un-squashed for you to inspect. Exit 1.

Cerberus's other two verdicts never reach you as a prompt. `REJECT_AND_RETURN`
re-enters the graph automatically with written objections (budgeted by
`max_rejections_per_run`); `ESCALATE` prints a report and preserves the
branch. If Cerberus's own model chain is exhausted, the run is labelled
**UNREVIEWED** rather than approved — the harness never auto-approves in the
absence of a review.

### Cost reporting

Every run writes `.xeno/runs/<run_id>/cost.json` alongside a `events.jsonl`
trace.

```bash
xeno show-cost .xeno/runs/<run_id>/cost.json
```

It is the instrument for the project's four target metrics, written from day
one so they stay falsifiable:

| Metric | Target |
| --- | --- |
| M1.1 non-flagship share of billed tokens | ≥ 70% |
| M1.2 USD reduction vs. all-flagship baseline | ≥ 50% |
| M1.3 median USD per completed task | ≤ $0.50 |
| M1.4 cache hit rate on eligible input tokens | ≥ 40% |

M1.4 is a lever *inside* M1.2, not an additive claim — do not sum them.

## Configuration

`xeno init` writes a starter `xeno.yaml`. Defaults target Hardware Tier 1
(Apple Silicon, 32 GB unified memory). On Hardware Tier 0 (8 GB VRAM), point
medium's first entry at an API provider — a 14B local model will not fit.

### Routing and tiers

```yaml
tiers:
  flagship:
    - {provider: openrouter, model: z-ai/glm-5.2,
       usd_per_1m_input: 1.40, usd_per_1m_output: 2.20}
  medium:
    - {provider: ollama, model: "qwen2.5-coder:14b"}
    # Upward fallback must be declared explicitly. It is logged and counted
    # against M1.1 by the model actually billed.
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
```

Two rules the loader enforces rather than documents:

- **Downward fallback is forbidden.** An exhausted chain halts the run.
  Quietly serving a weaker model produces worse output *and* hides the cost
  problem.
- **Upward fallback must be declared** (`escalation: true`), is logged, and is
  billed to the model actually used — so an escalation cannot hide behind the
  node's declared tier.

Unpriced remote models are rejected at load: the budget breaker cannot enforce
a USD ceiling it cannot compute. Override with `allow_unpriced_models: true`
and accept that cost totals become a lower bound.

**Local-only, honestly.** Fully local is a supported mode; "fully local with
no quality loss" is a claim this project will not make. The CLI prints a
capability warning whenever the flagship tier resolves to any locally served
model, at any size. There is no parameter threshold that makes a local model
flagship-class on this hardware, so there is no threshold in the check.

### Sandbox

```yaml
sandbox:
  network: none          # none | install | open
  memory: 2g
  cpus: 2
  warm_pool_size: 2
```

Every gate runs non-root, with all capabilities dropped, a read-only root
filesystem, and resource caps. `warm_pool_size` pre-starts containers so no
evaluation pays a cold start. `network: none` is the default; `install`
briefly enables the network for the discovered dependency-install step, then
commits the image and drops it before any generated code runs; `open` leaves
it on for the whole run and prints a warning.

### Run caps and circuit breakers

```yaml
limits:
  max_usd_per_run: 2.00
  max_runtime_minutes: 45
  max_iterations_per_task: 12
  max_iterations_per_run: 60
  max_rejections_per_run: 2
  max_deleted_lines: 200
```

### Full reference

```yaml
git:
  branch_prefix: "xeno/"
  open_pr: false          # set true + install `gh` to open a PR on APPROVE

caching:
  enabled: true
  probe_aggregators_at_startup: true

secrets:
  extra_denylist: []      # appended to the built-in denylist; never replaces it
  scan_outbound_context: true

providers:                # every key below is optional and has a default
  ollama:
    base_url: http://localhost:11434
    keep_alive: "30m"     # how long a local model stays resident BETWEEN calls
    release_on_exit: true # unload when the run ends, rather than on the timer
    max_context: null     # force a SMALLER window than the model allows
```

The three local-only keys exist as escape hatches, not as settings you are
expected to tune. `max_context` in particular is not how the window gets
chosen — that is derived per call from the assembled prompt and clamped to
what the model reports. Set it only to cap KV-cache memory below what the
model would otherwise permit.

`probe_aggregators_at_startup` costs two short calls per aggregator provider
at run start, and is what makes explicit cache markers (and therefore M1.4)
work at all. Purely local or Anthropic-only setups pay nothing for it. Turn it
off if you would rather not pay the probe.

## Tech stack

| Layer | Choice |
| --- | --- |
| Language | Python 3.11+ |
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) — the state machine and its conditional edges |
| CLI | [Typer](https://typer.tiangolo.com/) + [Rich](https://rich.readthedocs.io/) for tables, panels, and the review pager |
| Data model | [Pydantic v2](https://docs.pydantic.dev/) — `AgentState`, config schema, and the 4 KB per-field limit |
| Config | YAML via PyYAML, validated on load |
| HTTP | [httpx](https://www.python-httpx.org/) against provider APIs directly — no vendor SDK, no LangChain model wrappers |
| Sandbox | Docker via [docker-py](https://docker-py.readthedocs.io/), with a warm container pool |
| VCS | `git` and `gh` as fixed argument vectors — never a shell string |
| Tests | pytest — 264 tests, no mocking framework; fakes are structural `Protocol`s |
| Lint / types | [ruff](https://docs.astral.sh/ruff/) (E, F, I, UP, B, SIM, RUF) and mypy `--strict` |
| Packaging | hatchling |

Providers are spoken to over their wire formats directly: Ollama's `/api/chat`
and the OpenAI-compatible `/chat/completions` (OpenAI, OpenRouter, and
anything that mimics it). Adding a provider means one `Provider` subclass with
a `complete` and a `health_check`, plus a usage parser — not a new dependency.

## Architecture notes

**Filesystem as working memory.** Large artifacts never enter shared state.
They are written under `.xeno/` and passed as handles — `{path, sha256,
summary, bytes}`. `AgentState` enforces a hard 4 KB per-field limit at
construction, because context-window exhaustion surfaces very far from the
line of code that caused it.

**Static-first prompt assembly.** Every prompt is built
`SYSTEM → CODEBASE MAP → ACCUMULATED HISTORY → CURRENT TURN`. This is a
construction rule, not an optimization pass: caching only pays off if nothing
upstream of a breakpoint ever changes. The builder owns ordering, history is
append-only and verified as such, and a node's system text is fingerprinted so
that interpolating anything dynamic into it raises instead of silently costing
the run its cache hits.

**Cache invalidation is derived, not managed.** The codebase map is
invalidated the moment a node writes to the worktree. A stale map served from
cache is worse than a miss — Argus's summaries would describe pre-edit code
with full confidence — so `CacheKeyring` refuses to serve a key it knows to be
stale.

**Untrusted repository content.** Retrieved file content, gate output, and the
review diff are each wrapped in labelled DATA blocks with content-derived
guards, and handle summaries are generated by the harness from counted facts,
never copied from file text. A file cannot inject through the line that
describes it. Guards are derived from content rather than randomness, so an
unchanged file still produces byte-identical prompt text and keeps its cache
hit.

**Aggregator caching is unverified.** Support and pricing for open-weight
models served via OpenRouter were not established upstream. The router probes
at startup — duplicate prefix, inspect reported cached tokens — and claims
capability only on positive evidence. Latency is recorded but never used to
declare success on its own, because a false positive would put fabricated
savings straight into `cost.json`. Providers that fail the probe get
cache-dependent projections disabled rather than estimated.

## License

MIT.
