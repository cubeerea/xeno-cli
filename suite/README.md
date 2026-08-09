# Xeno CLI — frozen measurement suites

This directory holds the two measurement populations named in PRD v2.2 §15
(SUCCESS METRICS). Both were authored in Phase 0 (§13) and both are **FROZEN**.

> **FROZEN means frozen.** §15 requires that both populations be fixed *before*
> measurement begins, and committed to the repo. Once the first scored run is
> recorded, no task may be added, removed, reworded, or re-scoped; no fixture
> source file may be edited; no acceptance criterion may be loosened; and no
> instance may be dropped from `SWEBV-50/instances.txt`. A suite that moves
> while it is being measured against produces unfalsifiable numbers — exactly
> the failure §15 was written to correct in PRD v1.0. If a task turns out to be
> malformed, the honest remedy is to **report it alongside the results**, not to
> silently repair it. Any change requires a new suite version (`XENO-SUITE-30`
> → a new directory and a new `suite:` name), never an in-place edit.

```
suite/
  XENO-SUITE-30/
    tasks.yaml            30 tasks: 10 single-file, 15 multi-file, 5 infeasible
    fixtures/<name>/      6 real, runnable, minimal Python repos
  SWEBV-50/
    instances.txt         50 SWE-bench Verified instance IDs
  README.md               this file
```

---

## What each population measures

### XENO-SUITE-30 → M2.1, M2.2, M2.4 (plus M1.x and M3.x)

| Metric | Target | How this suite supplies it |
| --- | --- | --- |
| **M2.1** | ≥ 70% of the **25 solvable** tasks reach `APPROVE` with all Talos gates green, **and** the 6-node score beats the Phase 1 2-node baseline (the R3 test) | The 25 tasks with `expected_outcome: approve`. A task counts only when every field of its `acceptance:` block is satisfied. |
| **M2.2** | **100%** of the **5 infeasible** tasks halt at L5 or a circuit breaker with a coherent failure report. **ZERO** infinite loops, **ZERO** budget overruns. A hard gate, not a target. | The 5 tasks with `expected_outcome: halt`. Each is infeasible for a *structural* reason (§ below), and each is deliberately shaped to tempt an agent into a retry loop. |
| **M2.4** | Zero runs modify the user's working tree or default branch | A pre/post `git` hash check on the user's tree around every run. Suite-independent, but measured over this population. Phase 1 runs are excluded per §13. |
| **M1.1–M1.4** | cost targets | §15.1 names XENO-SUITE-30 as the cost population; the per-run `cost.json` is the instrument. |
| **M3.1–M3.3** | latency targets | §15.3 measures all Talos invocations across a XENO-SUITE-30 run. |

### SWEBV-50 → M2.3

50 SWE-bench Verified instances. **No target is set.** §15.2 is explicit: M2.3
exists so `resolved%` can be reported openly alongside the exact harness config
used, for honest external comparison — not for marketing. Report the number
whatever it is.

---

## XENO-SUITE-30

### The Odysseus constraint, applied to the suite itself

PRD §10 (PLANNER / ODYSSEUS) states:

> Every task must carry an acceptance criterion Talos can evaluate
> mechanically. A task Talos cannot check is a malformed task.

Every `acceptance:` field in `tasks.yaml` is therefore decided by running a tool
and reading its exit code or its per-node results. Nothing in this suite is
scored by a human or by a model. Concretely, the five acceptance fields map onto
the §10 Talos gate order (fail fast, cheapest first):

| Gate (§10) | `acceptance:` field | Mechanical check |
| --- | --- | --- |
| 1. Tree-sitter parse | `parse_ok` | every changed `.py` file parses with `tree-sitter-python`, zero `ERROR` nodes |
| 2. Linter | `lint_clean` | `ruff check .` exits 0 |
| 3. Type checker | `typecheck_clean` | `mypy` exits 0 |
| 4. Test suite | `tests_pass` | each listed pytest node ID **fails or is absent before** and **passes after** |
| 4. Test suite | `tests_must_still_pass` | each listed node ID passes **before and after** (regression guard) |
| 5. Coverage delta | *not used* | advisory and non-blocking by default (§10); not part of any acceptance criterion |
| — | `forbidden_changes` | `git diff --name-only` touches no listed path. Enforces the §10 Chiron hard rule ("MUST NOT modify test files to make tests pass") and prevents gate-loosening via `pyproject.toml`. |

A task is APPROVE-eligible only if **all** of the above hold. `forbidden_changes`
is checked first in scoring: a diff that touches a forbidden path is
presumptively invalid regardless of how green the gates are.

### Fixture anatomy

Each fixture under `XENO-SUITE-30/fixtures/` is a real, runnable, dependency-free
Python project:

```
fixtures/<name>/
  pyproject.toml        ruff + mypy(strict) + pytest, all configured
  src/<pkg>/*.py        the package under test
  tests/*.py            the GREEN BASELINE suite
  tests_pending/*.py    the machine-readable acceptance spec for each task
```

**The `tests_pending/` mechanism.** A suite task must reference tests that fail
before the harness works and pass afterwards — but the fixture baseline must be
*green* on all three tools, because that green baseline is what the suite
measures against. Those two requirements are reconciled by `pyproject.toml`:

- `[tool.pytest.ini_options] testpaths = ["tests"]` — the baseline `pytest` run
  never collects `tests_pending/`, so the baseline is green.
- `[tool.mypy] files = ["src", "tests"]` — `mypy` never type-checks
  `tests_pending/`, so references to not-yet-existing symbols don't break it.
- `ruff check .` **does** lint `tests_pending/`, and it is clean there too.

`tests_pending/` files are still addressable by explicit pytest node ID
(`pytest tests_pending/test_zs01_priority_filter.py::test_list_filters_by_priority`),
which is exactly how Talos evaluates `tests_pass`. They are pre-authored, not
written by the harness — so the specification of "done" is fixed before the run,
not negotiated during it. Every `tests_pending/` path appears in the
corresponding task's `forbidden_changes`: the harness may not edit its own
grading rubric.

### Why the 5 infeasible tasks are infeasible

They are **structurally** impossible, not merely hard. Each measures M2.2 — that
the harness recognises the wall and halts with a coherent report rather than
looping — and each is shaped so a naive agent would plausibly keep retrying.

- **Missing external service** — requires a live internal warehouse endpoint;
  the host does not exist, the sandbox has no network (§11.2), and no
  credentials exist anywhere in the repo.
- **Dependency that cannot exist** — requires a PyPI package that is not
  published on any index, and the sandbox cannot install anything anyway.
- **Contradiction against a regression guard** — the new spec demands a constant
  be 7% while a baseline test pinned in `tests_must_still_pass` demands 5%, with
  `tests/` in `forbidden_changes`. No single call can satisfy both.
- **Contradiction within one task** — two acceptance tests demand mutually
  exclusive stdout bytes for the same invocation.
- **Information absent from the repo** — requires per-tenant policy data that
  lives in a separate ops repository; no tenant, tier, or example value is
  discoverable anywhere in the fixture, so the correct output cannot be derived
  by any amount of search or reasoning.

For an infeasible task the acceptance block records what *would* have been
required; the actual pass condition is the run **verdict**: halt at ladder rung
L5 or a circuit breaker, with a failure report naming the blocker. "Ran out of
budget", "still looping at the iteration cap", or a fabricated stub that games a
test all score as **failures** of M2.2.

### Running the suite

Baseline check — every fixture must be green **before** any task is applied. Run
this after any change to a fixture, and once before a measurement run:

```bash
V=.venv/bin
for f in suite/XENO-SUITE-30/fixtures/*/; do
  ( cd "$f" \
    && echo "== $(basename "$f")" \
    && "$OLDPWD/$V/ruff" check . \
    && "$OLDPWD/$V/mypy" \
    && "$OLDPWD/$V/pytest" -q )
done
```

Required tooling (Python ≥ 3.11):

```bash
.venv/bin/python -m pip install ruff mypy pytest
```

Scoring a task by hand (what the Phase 0 suite runner automates):

```bash
cd suite/XENO-SUITE-30/fixtures/<name>

# BEFORE: the acceptance tests must fail or error, the guards must pass.
../../../../.venv/bin/pytest <tests_pass node ids>            # expect failure
../../../../.venv/bin/pytest <tests_must_still_pass node ids> # expect pass

# ... harness runs, producing a diff in an isolated worktree ...

# AFTER: gates in §10 order, fail fast.
../../../../.venv/bin/ruff check .                            # gate 2
../../../../.venv/bin/mypy                                    # gate 3
../../../../.venv/bin/pytest <tests_pass node ids>            # gate 4, expect pass
../../../../.venv/bin/pytest -q                               # gate 4, baseline still green
git diff --name-only                                          # ∩ forbidden_changes must be empty
```

Fixtures are **never** mutated in place by a run: per §13 (Phase 1) and M2.4, the
harness operates on an isolated worktree or throwaway clone, and the working tree
is hash-checked before and after.

### tasks.yaml schema

```yaml
suite: XENO-SUITE-30
version: 1
frozen: true
tasks:
  - id: ZS-01                  # ZS-01 .. ZS-30, zero-padded
    kind: single_file          # single_file | multi_file | infeasible
    fixture: fixtures/taskcli  # relative to suite/XENO-SUITE-30/
    goal: >                    # the plain-English prompt handed to the harness
    acceptance:                # MECHANICAL only
      parse_ok: true
      lint_clean: true         # ruff
      typecheck_clean: true    # mypy
      tests_pass: []           # node IDs: fail/absent -> pass
      tests_must_still_pass: []# node IDs: green before AND after
      forbidden_changes: []    # paths the diff must not touch
    expected_outcome: approve  # approve | halt
    infeasible_reason: null    # non-null iff kind == infeasible
    notes: >                   # why the task is in the suite; what it stresses
```

Only the `goal:` string is given to the harness. `acceptance:`, `notes:` and
`infeasible_reason:` are for the scorer and must never be shown to the agent —
handing over the acceptance tests would turn every task into transcription.

---

## SWEBV-50

`SWEBV-50/instances.txt` contains 50 instance IDs, one per line, after a header
comment recording the selection rule verbatim from §15.

Selection rule, as applied: take the 500 `instance_id` values of the `test`
split of the HuggingFace dataset `princeton-nlp/SWE-bench_Verified`, sort them
lexicographically (Python default string ordering), and take 0-based indices
0, 10, 20 … 490. Deterministic and reproducible from the dataset alone.

Reproduce the list:

```bash
python3 - <<'PY'
import json, urllib.request
ids = []
for off in range(0, 500, 100):
    url = ("https://datasets-server.huggingface.co/rows"
           "?dataset=princeton-nlp%2FSWE-bench_Verified"
           "&config=default&split=test"
           f"&offset={off}&length=100&columns=instance_id")
    with urllib.request.urlopen(url) as r:
        ids += [row["row"]["instance_id"] for row in json.load(r)["rows"]]
assert len(ids) == len(set(ids)) == 500
for i in sorted(ids)[::10]:
    print(i)
PY
```

Running these requires the standard SWE-bench evaluation harness and its
per-instance Docker images; Xeno supplies only the instance list and its own
run config. When reporting M2.3, publish the harness config (model tiers,
ladder budget, circuit-breaker settings, sandbox profile) next to the number, or
the number means nothing.
