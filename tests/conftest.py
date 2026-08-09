from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from xeno.core.config import ModelSpec, NodeSpec, ProviderSpec, XenoConfig
from xeno.core.types import DEFAULT_NODE_TIERS, ProviderFamily, Tier
from xeno.prompt.assembly import reset_system_fingerprints
from xeno.prompt.keys import CacheKeyring


@pytest.fixture(autouse=True)
def _clean_fingerprints() -> Iterator[None]:
    """The system-prompt fingerprint registry is process-global by design.

    Tests deliberately build the same node with different system text, which
    would otherwise trip the drift guard for reasons unrelated to what is being
    tested.
    """
    reset_system_fingerprints()
    yield
    reset_system_fingerprints()


@pytest.fixture
def providers() -> dict[str, ProviderSpec]:
    return {
        "ollama": ProviderSpec(family=ProviderFamily.OLLAMA, base_url="http://localhost:11434"),
        "openrouter": ProviderSpec(
            family=ProviderFamily.AGGREGATOR,
            base_url="https://openrouter.ai/api/v1",
            api_key_env="TEST_OPENROUTER_KEY",
        ),
    }


@pytest.fixture
def config(providers: dict[str, ProviderSpec]) -> XenoConfig:
    """Mirrors the PRD S9.5 example: local light/medium, API flagship, with a
    declared upward fallback on the medium chain."""
    flagship = ModelSpec(
        provider="openrouter", model="glm-5.2", usd_per_1m_input=1.40, usd_per_1m_output=2.20
    )
    return XenoConfig(
        providers=providers,
        tiers={
            Tier.FLAGSHIP: (flagship,),
            Tier.MEDIUM: (
                ModelSpec(provider="ollama", model="qwen2.5-coder:14b"),
                ModelSpec(
                    provider="openrouter",
                    model="glm-5.2",
                    escalation=True,
                    usd_per_1m_input=1.40,
                    usd_per_1m_output=2.20,
                ),
            ),
            Tier.LIGHT: (ModelSpec(provider="ollama", model="qwen2.5-coder:7b"),),
        },
        nodes={role: NodeSpec(tier=tier) for role, tier in DEFAULT_NODE_TIERS.items()},
    )


@pytest.fixture
def keyring(tmp_path: Path) -> CacheKeyring:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / "main.py").write_text("print('hello')\n")
    ring = CacheKeyring(run_id="test-run", worktree_root=worktree)
    ring.content_hash()  # settle the initial hash so the map reads as fresh
    return ring
