"""xeno.yaml schema and loader (PRD S9.5, S11, S13 Phase 0).

Config is a static artifact. No node may mutate it at runtime — that is a
prompt-injection invariant, not a style preference (PRD S11.4), so the models
here are frozen after load.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from xeno.core.types import (
    DEFAULT_NODE_TIERS,
    TIER_RANK,
    NodeRole,
    ProviderFamily,
    Tier,
)

CONFIG_FILENAME = "xeno.yaml"
STATE_DIRNAME = ".xeno"


class ConfigError(Exception):
    """Raised for a malformed or unusable xeno.yaml."""


class ProviderSpec(BaseModel):
    """An inference endpoint. Keys are read from the environment, never stored
    in config, and never enter a container or a prompt (PRD S11.3)."""

    model_config = ConfigDict(frozen=True)

    family: ProviderFamily
    base_url: str
    api_key_env: str | None = None
    timeout_seconds: float = Field(default=120.0, gt=0)

    #: How long a LOCAL backend holds a model resident between calls.
    #: Reloading weights between ladder iterations would dominate the latency
    #: the local tier exists to keep low, so the default is generous. Ignored
    #: by remote providers, which have no weights to hold.
    keep_alive: str = "30m"
    #: Unload local models when the run ends. On by default because the
    #: daemon has no way to know a run finished — `keep_alive` is a timer, so
    #: without this the weights outlive the run by design. Turn it off only if
    #: you run back-to-back and would rather keep the models warm.
    release_on_exit: bool = True
    #: Override the context window discovered from the backend, in tokens.
    #: An escape hatch, not a required setting: the window is derived per call
    #: from the assembled prompt and clamped to what the model reports. Set it
    #: only to force a smaller window than the model allows — to cap KV-cache
    #: memory on a constrained machine, say.
    max_context: int | None = Field(default=None, gt=0)

    @property
    def is_local(self) -> bool:
        return self.family is ProviderFamily.OLLAMA

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None

    def key_available(self) -> bool:
        return self.is_local or bool(self.api_key())


class ModelSpec(BaseModel):
    """One entry in a tier's ordered fallback chain (PRD S9.5)."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str

    #: Marks this entry as a deliberate UPWARD fallback — a medium chain
    #: terminating in a flagship model. Must be declared explicitly; it is
    #: logged and surfaced in the cost ledger as a tier escalation (PRD S9.5).
    escalation: bool = False

    usd_per_1m_input: float | None = None
    usd_per_1m_output: float | None = None
    #: Anthropic-style caching multipliers (PRD S9.6.3): write 1.25x base
    #: input, read 0.10x. Overridable per model as real pricing is confirmed.
    cache_write_multiplier: float = 1.25
    cache_read_multiplier: float = 0.10

    max_tokens: int | None = None

    @property
    def ref(self) -> str:
        """Stable identity for tokens_by_model, so M1.1 counts the model that
        was actually billed rather than the tier that was declared."""
        return f"{self.provider}/{self.model}"

    @property
    def priced(self) -> bool:
        return self.usd_per_1m_input is not None and self.usd_per_1m_output is not None


class NodeSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    tier: Tier
    max_tokens: int = 8192
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class Limits(BaseModel):
    """Run-level caps. Every one of these has a default; "unlimited" is not an
    option anywhere in this system (PRD T5, S7.3)."""

    model_config = ConfigDict(frozen=True)

    #: Sized for building a project, not for patching one. The original
    #: values ($2, 45 min, 12/task, 60/run) were set against XENO-SUITE-30's
    #: single-file and multi-file tasks; a greenfield run plans a scaffold
    #: task plus one task per feature and exhausts a 60-iteration run budget
    #: long before it finishes. Every one of these is still a hard cap —
    #: "unlimited" remains unexpressible — and every one is overridable in
    #: `xeno.yaml`, which is where a cost-sensitive run should lower them.
    max_usd_per_run: float = Field(default=10.00, gt=0)
    max_runtime_minutes: float = Field(default=120.0, gt=0)
    max_iterations_per_task: int = Field(default=25, gt=0)
    max_iterations_per_run: int = Field(default=200, gt=0)
    max_rejections_per_run: int = Field(default=2, ge=0)
    max_deleted_lines: int = Field(default=200, gt=0)

    #: Per-tier token ceilings (PRD CB-3). Absent tier == no separate ceiling;
    #: the USD cap still applies.
    max_tokens_per_tier: dict[Tier, int] = Field(default_factory=dict)


class CacheConfig(BaseModel):
    """Prompt caching (PRD S9.6)."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True

    #: Send a duplicate prefix at startup and compare cost/latency, so we do
    #: not assume aggregator parity with Anthropic or OpenAI (PRD S9.6.3,
    #: OQ-11). Providers that fail the probe get cache-dependent cost
    #: projections disabled rather than estimated.
    probe_aggregators_at_startup: bool = True


class SandboxConfig(BaseModel):
    """Container profile (PRD S11.1, S11.2). Consumed from Phase 2 onward; the
    schema lands in Phase 0 so config does not change shape mid-project."""

    model_config = ConfigDict(frozen=True)

    network: str = Field(default="none", pattern="^(none|install|open)$")
    memory: str = "2g"
    cpus: float = Field(default=2.0, gt=0)
    pids_limit: int = Field(default=512, gt=0)
    warm_pool_size: int = Field(default=2, ge=0)


class SecretsConfig(BaseModel):
    """PRD S11.3. The denylist is explicit and user-extensible."""

    model_config = ConfigDict(frozen=True)

    #: Never mounted into the sandbox. `extra_denylist` appends; it never
    #: replaces, so a user cannot accidentally widen the mount surface.
    denylist: tuple[str, ...] = (
        ".env",
        ".env.*",
        ".git/config",
        ".netrc",
        "~/.aws",
        "~/.ssh",
        "~/.config/gh",
        "~/.docker/config.json",
        "~/.kube",
        "*.pem",
        "*.key",
        "id_rsa*",
    )
    extra_denylist: tuple[str, ...] = ()

    #: Shannon-entropy threshold (bits/char) for flagging an unrecognised
    #: token as a probable secret.
    entropy_threshold: float = 4.0
    scan_outbound_context: bool = True

    @property
    def effective_denylist(self) -> tuple[str, ...]:
        return (*self.denylist, *self.extra_denylist)


class GitConfig(BaseModel):
    """PRD S8.4. Consumed in Phase 4."""

    model_config = ConfigDict(frozen=True)

    branch_prefix: str = "xeno/"
    open_pr: bool = False


DEFAULT_PROVIDERS: dict[str, ProviderSpec] = {
    "ollama": ProviderSpec(
        family=ProviderFamily.OLLAMA,
        base_url="http://localhost:11434",
    ),
    "openrouter": ProviderSpec(
        family=ProviderFamily.AGGREGATOR,
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
    ),
}


class XenoConfig(BaseModel):
    """The whole of xeno.yaml."""

    model_config = ConfigDict(frozen=True)

    providers: dict[str, ProviderSpec] = Field(default_factory=lambda: dict(DEFAULT_PROVIDERS))
    tiers: dict[Tier, tuple[ModelSpec, ...]]
    nodes: dict[NodeRole, NodeSpec]
    limits: Limits = Field(default_factory=Limits)
    caching: CacheConfig = Field(default_factory=CacheConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    secrets: SecretsConfig = Field(default_factory=SecretsConfig)
    git: GitConfig = Field(default_factory=GitConfig)

    #: Refuse to run a model whose price is unknown, because CB-3 cannot
    #: enforce a USD ceiling it cannot compute. Set true only when every
    #: unpriced model in the chain is local and therefore free.
    allow_unpriced_models: bool = False

    source_path: Path | None = None

    @model_validator(mode="before")
    @classmethod
    def _fill_default_node_tiers(cls, data: Any) -> Any:
        """Supply a default tier for any role the config does not name.

        Every role still has to END UP with a tier — `_validate_node_tiers`
        below is unchanged and still refuses a config that lacks one. What
        this avoids is a config file becoming invalid because the HARNESS grew
        a node: a user's working xeno.yaml should not start erroring the day
        a seventh role is added, when there is a perfectly good default for it
        sitting in `DEFAULT_NODE_TIERS`. An explicitly declared tier always
        wins; this only fills gaps.
        """
        if not isinstance(data, dict):
            return data
        nodes = data.get("nodes")
        if not isinstance(nodes, dict):
            return data
        declared = {NodeRole(k) if not isinstance(k, NodeRole) else k for k in nodes}
        filled = dict(nodes)
        for role, tier in DEFAULT_NODE_TIERS.items():
            if role not in declared:
                filled[role] = NodeSpec(tier=tier)
        return {**data, "nodes": filled}

    @model_validator(mode="after")
    def _validate(self) -> Self:
        self._validate_chains()
        self._validate_node_tiers()
        self._validate_pricing()
        offenders = self.downward_fallbacks()
        if offenders:
            listed = ", ".join(f"{tier}->{ref}" for tier, ref in offenders)
            raise ConfigError(
                f"downward fallback is forbidden (PRD S9.5): {listed}. An exhausted "
                f"chain must halt the run, not silently serve a weaker model."
            )
        return self

    def _validate_chains(self) -> None:
        for tier, chain in self.tiers.items():
            if not chain:
                raise ConfigError(f"tier '{tier}' has an empty fallback chain")
            for entry in chain:
                if entry.provider not in self.providers:
                    raise ConfigError(
                        f"tier '{tier}' references unknown provider '{entry.provider}'. "
                        f"Known: {sorted(self.providers)}"
                    )

    def _validate_node_tiers(self) -> None:
        for role, spec in self.nodes.items():
            if spec.tier not in self.tiers:
                raise ConfigError(
                    f"node '{role}' declares tier '{spec.tier}', which has no chain defined"
                )
        missing = set(NodeRole) - set(self.nodes)
        if missing:
            raise ConfigError(f"no tier declared for node(s): {sorted(m.value for m in missing)}")

    def _validate_pricing(self) -> None:
        if self.allow_unpriced_models:
            return
        unpriced = [
            f"{tier}:{entry.ref}"
            for tier, chain in self.tiers.items()
            for entry in chain
            if not entry.priced and not self.providers[entry.provider].is_local
        ]
        if unpriced:
            raise ConfigError(
                "these remote models have no price, so CB-3 cannot enforce "
                f"max_usd_per_run: {unpriced}. Add usd_per_1m_input/output, or set "
                "allow_unpriced_models: true to run without a budget guarantee."
            )

    def chain_for(self, tier: Tier) -> tuple[ModelSpec, ...]:
        return self.tiers[tier]

    def tier_for(self, role: NodeRole) -> Tier:
        return self.nodes[role].tier

    def provider_for(self, entry: ModelSpec) -> ProviderSpec:
        return self.providers[entry.provider]

    def flagship_is_local(self) -> bool:
        """PRD S9.4: the capability warning fires whenever the FLAGSHIP tier
        resolves to any locally-served model, at any size. There is no
        parameter threshold that makes a local model flagship-class on this
        project's target hardware, so there is no threshold in this check."""
        chain = self.tiers.get(Tier.FLAGSHIP, ())
        return any(self.providers[e.provider].is_local for e in chain)

    def local_models(self) -> list[tuple[Tier, str]]:
        """Every `(tier, model)` this config will serve from a local backend.

        Weights are the one configuration choice that spends a resource the
        harness cannot meter, throttle, or roll back. A remote model that is
        too big for the budget shows up as a number in cost.json; a local one
        that is too big for the machine shows up as swap thrash, and past a
        point the window server stops answering its watchdog and the OS
        restarts. Anything that reports what a run is about to do needs to be
        able to name them.
        """
        return [
            (tier, entry.model)
            for tier, chain in self.tiers.items()
            for entry in chain
            if self.providers[entry.provider].is_local
        ]

    def downward_fallbacks(self) -> list[tuple[Tier, str]]:
        """Fallback entries that resolve to a model serving a WEAKER tier.

        PRD S9.5 forbids downward fallback outright: an exhausted chain halts
        the run rather than quietly producing worse output at a lower tier.
        A model has no intrinsic tier, so the only evidence available is where
        else it appears — if flagship's second entry is the same model light
        uses, that entry is a downgrade wearing a flagship label.

        Only positions after the first are fallbacks; a chain's head is the
        tier's primary and defines it.
        """
        offenders: list[tuple[Tier, str]] = []
        for tier, chain in self.tiers.items():
            for entry in chain[1:]:
                serves_weaker = any(
                    TIER_RANK[other] < TIER_RANK[tier]
                    and any(e.ref == entry.ref for e in other_chain)
                    for other, other_chain in self.tiers.items()
                )
                if serves_weaker:
                    offenders.append((tier, entry.ref))
        return offenders


def user_config_path() -> Path:
    """`~/.config/xeno/xeno.yaml`, or the XDG equivalent when set.

    Not merged into a project config — see `find_config`.
    """
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "xeno" / CONFIG_FILENAME


def find_config(start: Path | None = None) -> Path | None:
    """The xeno.yaml that governs a run: nearest project file, else the user's.

    The upward walk finds the config belonging to the repository you are
    working in, which is the right answer whenever you are working in one. It
    is the *fallback* that was wrong: outside such a repository the walk found
    nothing and the built-in defaults took over, silently replacing the entire
    routing table — including with locally-served weights, which is the one
    substitution that can cost more than money (`XenoConfig.local_models`).

    A user-level file makes "whatever I normally use" the answer to that,
    instead of "whatever this project ships". The defaults then mean what they
    should: the routing for someone who has never configured anything.

    This is a fallback, not a layer. A project config is complete by
    construction — `tiers` and `nodes` are both required — so there is no
    partial file for a user-level one to fill in, and inventing a merge would
    only make it harder to answer "which model is about to run" by reading one
    file. `xeno config show` names the file that won.
    """
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    user = user_config_path()
    return user if user.is_file() else None


def load_config(path: Path | None = None) -> XenoConfig:
    """Load and validate xeno.yaml, falling back to built-in defaults."""
    resolved = path or find_config()
    if resolved is None:
        return default_config()
    try:
        raw = yaml.safe_load(resolved.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{resolved}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{resolved}: top level must be a mapping")

    raw.setdefault("providers", {})
    merged: dict[str, Any] = {**DEFAULT_PROVIDERS, **raw["providers"]}
    raw["providers"] = merged
    raw["source_path"] = resolved

    try:
        return XenoConfig.model_validate(raw)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"{resolved}: {exc}") from exc


def default_config() -> XenoConfig:
    """Built-in defaults targeting Hardware Tier 1 (PRD S9.3, S9.5).

    Hardware Tier 0 (8 GB VRAM) users must point medium's first entry at an API
    provider — a 14B local model will not fit.
    """
    return XenoConfig(
        tiers={
            Tier.FLAGSHIP: (
                ModelSpec(
                    provider="openrouter",
                    model="z-ai/glm-5.2",
                    usd_per_1m_input=1.40,
                    usd_per_1m_output=2.20,
                ),
            ),
            Tier.MEDIUM: (
                ModelSpec(provider="ollama", model="qwen2.5-coder:14b"),
                ModelSpec(
                    provider="openrouter",
                    model="z-ai/glm-5.2",
                    escalation=True,
                    usd_per_1m_input=1.40,
                    usd_per_1m_output=2.20,
                ),
            ),
            Tier.LIGHT: (ModelSpec(provider="ollama", model="qwen2.5-coder:7b"),),
        },
        nodes={
            role: NodeSpec(tier=tier, max_tokens=32000 if tier is Tier.FLAGSHIP else 8192)
            for role, tier in DEFAULT_NODE_TIERS.items()
        },
    )
