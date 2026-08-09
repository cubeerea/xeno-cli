"""State, config, ledger, and run-log invariants (PRD S6, S9.5, S15.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from xeno.core.config import (
    ConfigError,
    Limits,
    ModelSpec,
    NodeSpec,
    ProviderSpec,
    XenoConfig,
    default_config,
    load_config,
)
from xeno.core.ledger import CallRecord, CostLedger
from xeno.core.runlog import EventKind, RunLog, read_events
from xeno.core.state import MAX_FIELD_BYTES, AgentState, EvalReport, Handle
from xeno.core.types import DEFAULT_NODE_TIERS, Breakpoint, NodeRole, ProviderFamily, Tier
from xeno.core.usage import Usage

# ---- state (PRD S6.3) ------------------------------------------------------


def test_oversized_field_is_rejected_at_construction() -> None:
    """T2: anything larger than 4 KB is a Handle, not a state field. Enforced
    at construction because the failure it prevents surfaces very far away."""
    with pytest.raises(ValidationError, match="over the 4096-byte limit"):
        AgentState(run_id="r", goal="x" * 5000)


def test_oversized_field_is_rejected_on_assignment_too() -> None:
    state = AgentState(run_id="r", goal="fine")
    with pytest.raises(ValidationError):
        state.goal = "y" * 5000


def test_a_realistic_state_fits_comfortably() -> None:
    state = AgentState(
        run_id="20260805T101500-ab12cd",
        goal="Add rate limiting to the public API endpoints, 100 req/min per key",
        task_cursor=3,
        tokens_by_model={"openrouter/glm-5.2": 120_000, "ollama/qwen2.5-coder:14b": 80_000},
    )
    assert len(state.model_dump_json()) < MAX_FIELD_BYTES


def test_handle_hashes_content_and_detects_drift(tmp_path: Path) -> None:
    target = tmp_path / "PLAN.md"
    target.write_text("# Plan\n1. do the thing\n")
    handle = Handle.for_file(target, summary="plan with 1 task")
    assert handle.verify()
    assert handle.read_text().startswith("# Plan")

    target.write_text("# Plan\n1. do something else\n")
    assert not handle.verify()


def test_handle_summary_is_truncated_not_rejected(tmp_path: Path) -> None:
    target = tmp_path / "f.py"
    target.write_text("x = 1\n")
    handle = Handle.for_file(target, summary="s" * 500)
    assert len(handle.summary) <= 200


def test_eval_report_passes_only_when_every_gate_is_green() -> None:
    assert EvalReport(parse_ok=True).passed
    assert not EvalReport(parse_ok=False).passed
    assert not EvalReport(parse_ok=True, lint_errors=1).passed
    assert not EvalReport(parse_ok=True, type_errors=1).passed
    assert not EvalReport(parse_ok=True, tests_failed=1).passed


def test_infrastructure_failure_is_never_a_pass() -> None:
    """A sandbox that failed to provision is not a green run (PRD S10, Talos)."""
    assert not EvalReport(parse_ok=True, infrastructure_failure=True).passed


# ---- config (PRD S9.5) -----------------------------------------------------


def test_default_config_is_valid_and_targets_hardware_tier_1() -> None:
    config = default_config()
    assert config.chain_for(Tier.MEDIUM)[0].model == "qwen2.5-coder:14b"
    assert config.tier_for(NodeRole.PLANNER) is Tier.FLAGSHIP
    assert config.tier_for(NodeRole.RESEARCHER) is Tier.LIGHT


def test_argus_is_light_in_all_v1_configurations() -> None:
    """PRD S9.1: the 'medium for large-repo synthesis' idea is deferred to
    v1.1 and is not available in release v1."""
    assert DEFAULT_NODE_TIERS[NodeRole.RESEARCHER] is Tier.LIGHT


def test_unpriced_remote_model_is_rejected() -> None:
    """CB-3 cannot enforce a USD ceiling it cannot compute."""
    with pytest.raises(ConfigError, match="no price"):
        _config_with(
            {Tier.FLAGSHIP: (ModelSpec(provider="openrouter", model="glm-5.2"),)}
        )


def test_unpriced_remote_model_is_allowed_with_the_explicit_opt_in() -> None:
    config = _config_with(
        {Tier.FLAGSHIP: (ModelSpec(provider="openrouter", model="glm-5.2"),)},
        allow_unpriced_models=True,
    )
    assert config.allow_unpriced_models


def test_local_models_need_no_price() -> None:
    config = _config_with({Tier.FLAGSHIP: (ModelSpec(provider="ollama", model="big"),)})
    assert config.flagship_is_local()


def test_missing_node_tier_is_rejected() -> None:
    with pytest.raises(ConfigError, match="no tier declared"):
        XenoConfig(
            providers=_providers(),
            tiers={t: (ModelSpec(provider="ollama", model="m"),) for t in Tier},
            nodes={NodeRole.CODER: NodeSpec(tier=Tier.MEDIUM)},
        )


def test_unknown_provider_reference_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown provider"):
        _config_with({Tier.LIGHT: (ModelSpec(provider="nope", model="m"),)})


def test_empty_chain_is_rejected() -> None:
    with pytest.raises(ConfigError, match="empty fallback chain"):
        _config_with({Tier.LIGHT: ()})


def test_loading_from_yaml_merges_default_providers(tmp_path: Path) -> None:
    path = tmp_path / "xeno.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "tiers": {
                    "flagship": [
                        {
                            "provider": "openrouter",
                            "model": "glm-5.2",
                            "usd_per_1m_input": 1.4,
                            "usd_per_1m_output": 2.2,
                        }
                    ],
                    "medium": [{"provider": "ollama", "model": "qwen2.5-coder:14b"}],
                    "light": [{"provider": "ollama", "model": "qwen2.5-coder:7b"}],
                },
                "nodes": {r.value: {"tier": t.value} for r, t in DEFAULT_NODE_TIERS.items()},
                "limits": {"max_usd_per_run": 5.0},
            }
        )
    )
    config = load_config(path)
    assert config.limits.max_usd_per_run == 5.0
    assert "ollama" in config.providers  # merged from defaults, not declared
    assert config.source_path == path


def test_malformed_yaml_reports_the_file(tmp_path: Path) -> None:
    path = tmp_path / "xeno.yaml"
    path.write_text("tiers: [unclosed\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)


def test_secrets_denylist_is_extended_never_replaced() -> None:
    from xeno.core.config import SecretsConfig

    secrets = SecretsConfig(extra_denylist=("custom.secret",))
    assert "custom.secret" in secrets.effective_denylist
    assert ".env" in secrets.effective_denylist


def test_limits_have_no_unlimited_default() -> None:
    """PRD T5/CB-2: 'No "unlimited" default.'"""
    limits = Limits()
    assert limits.max_runtime_minutes == 45.0
    assert limits.max_usd_per_run == 2.00
    assert limits.max_iterations_per_task == 12
    assert limits.max_rejections_per_run == 2


# ---- ledger (PRD S15.1) ----------------------------------------------------


def _record(**kwargs: object) -> CallRecord:
    base: dict[str, object] = {
        "ts": 0.0,
        "node": "coder",
        "declared_tier": Tier.MEDIUM,
        "billed_tier": Tier.MEDIUM,
        "provider": "ollama",
        "model": "qwen2.5-coder:14b",
        "escalation": False,
        "usage": Usage(input_tokens=1000, output_tokens=100),
        "by_breakpoint": {},
        "usd": 0.0,
        "usd_uncached": 0.0,
        "latency_ms": 10.0,
        "cache_capable": False,
    }
    base.update(kwargs)
    return CallRecord(**base)  # type: ignore[arg-type]


def test_m1_1_measures_the_model_billed_not_the_tier_declared() -> None:
    ledger = CostLedger(run_id="t")
    ledger.record(_record())  # 1100 tokens on medium
    ledger.record(
        _record(
            declared_tier=Tier.MEDIUM,
            billed_tier=Tier.FLAGSHIP,
            provider="openrouter",
            model="glm-5.2",
            escalation=True,
            usage=Usage(input_tokens=900, output_tokens=100),
        )
    )
    share = ledger.metrics()["M1.1_non_flagship_token_share"]
    assert share == pytest.approx(1100 / 2100)


def test_unpriced_call_marks_the_total_as_a_lower_bound() -> None:
    ledger = CostLedger(run_id="t")
    ledger.record(_record(usd=None, usd_uncached=None))
    assert ledger.has_unpriced_calls
    assert ledger.to_dict()["totals"]["usd_is_lower_bound"] is True


def test_cache_stats_exclude_providers_that_failed_the_probe() -> None:
    """Folding an unknown into the denominator would make M1.4 look worse than
    measured rather than honestly narrower (OQ-11)."""
    from xeno.core.state import BreakpointStats

    ledger = CostLedger(run_id="t")
    ledger.record(
        _record(
            cache_capable=False,
            by_breakpoint={Breakpoint.SYSTEM: BreakpointStats(miss_tokens=1000)},
        )
    )
    ledger.record(
        _record(
            cache_capable=True,
            by_breakpoint={Breakpoint.SYSTEM: BreakpointStats(hit_tokens=800, miss_tokens=200)},
        )
    )
    metrics = ledger.metrics()
    assert metrics["M1.4_cache_hit_rate"] == pytest.approx(0.8)
    assert metrics["M1.4_excluded_calls_not_cache_capable"] == 1


def test_m1_3_reports_the_median_over_completed_tasks() -> None:
    ledger = CostLedger(run_id="t")
    for usd in (0.10, 0.40, 0.90):
        ledger.mark_task_completed(usd)
    assert ledger.metrics()["M1.3_median_usd_per_completed_task"] == 0.40


def test_metrics_are_none_rather_than_zero_when_nothing_was_measured() -> None:
    """A 0% that means 'no data' would read as a failed target."""
    metrics = CostLedger(run_id="t").metrics()
    assert metrics["M1.1_non_flagship_token_share"] is None
    assert metrics["M1.4_cache_hit_rate"] is None
    assert metrics["M1.3_median_usd_per_completed_task"] is None


def test_cost_json_is_valid_and_carries_the_exit_criterion_fields(tmp_path: Path) -> None:
    """PRD S13 Phase 0 EXIT: per-tier latency, USD, and a cached/fresh
    token breakdown."""
    from xeno.core.state import BreakpointStats

    ledger = CostLedger(run_id="run-1")
    ledger.record(
        _record(
            cache_capable=True,
            by_breakpoint={Breakpoint.SYSTEM: BreakpointStats(hit_tokens=500, miss_tokens=500)},
        )
    )
    path = ledger.write(tmp_path / "cost.json")
    data = json.loads(path.read_text())

    assert data["run_id"] == "run-1"
    assert "medium" in data["latency_ms"]
    assert data["latency_ms"]["medium"]["p50"] == 10.0
    assert data["cache_by_breakpoint"]["system"]["hit_tokens"] == 500
    assert data["cache_by_breakpoint"]["system"]["hit_rate"] == pytest.approx(0.5)
    assert "usd" in data["totals"]
    assert data["tokens_by_model"] == {"ollama/qwen2.5-coder:14b": 1100}


# ---- run log ---------------------------------------------------------------


def test_events_are_flushed_immediately_and_readable(tmp_path: Path) -> None:
    """The runs worth inspecting are the ones that halted; a buffered final
    event is exactly the one that gets lost."""
    path = tmp_path / "events.jsonl"
    with RunLog(path, run_id="r") as log:
        log.event(EventKind.RUN_START, command="test")
        assert read_events(path)  # readable before close
        log.event(EventKind.BREAKER_FIRED, code="CB-4")

    events = read_events(path)
    assert [e["kind"] for e in events] == ["run.start", "breaker.fired"]
    assert [e["seq"] for e in events] == [1, 2]


def test_a_truncated_final_line_does_not_lose_earlier_events(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with RunLog(path, run_id="r") as log:
        log.event(EventKind.RUN_START)
    with path.open("a") as fh:
        fh.write('{"seq": 2, "kind": "mod')  # killed mid-write
    assert len(read_events(path)) == 1


def test_span_records_duration_and_reraises(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with RunLog(path, run_id="r") as log:  # noqa: SIM117
        with pytest.raises(RuntimeError):
            with log.span(EventKind.NODE_ENTER, EventKind.NODE_EXIT, node="coder"):
                raise RuntimeError("boom")

    exit_event = read_events(path)[-1]
    assert exit_event["ok"] is False
    assert "boom" in exit_event["error"]
    assert "duration_ms" in exit_event


# ---- helpers ---------------------------------------------------------------


def _providers() -> dict[str, ProviderSpec]:
    return {
        "ollama": ProviderSpec(family=ProviderFamily.OLLAMA, base_url="http://localhost:11434"),
        "openrouter": ProviderSpec(
            family=ProviderFamily.AGGREGATOR, base_url="https://openrouter.ai/api/v1"
        ),
    }


def _config_with(
    tiers: dict[Tier, tuple[ModelSpec, ...]], **kwargs: object
) -> XenoConfig:
    full: dict[Tier, tuple[ModelSpec, ...]] = {
        Tier.FLAGSHIP: (ModelSpec(provider="ollama", model="big"),),
        Tier.MEDIUM: (ModelSpec(provider="ollama", model="mid"),),
        Tier.LIGHT: (ModelSpec(provider="ollama", model="small"),),
    }
    full.update(tiers)
    return XenoConfig(
        providers=_providers(),
        tiers=full,
        nodes={role: NodeSpec(tier=tier) for role, tier in DEFAULT_NODE_TIERS.items()},
        **kwargs,  # type: ignore[arg-type]
    )
