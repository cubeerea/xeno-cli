"""Secret scanning on outbound context (PRD S11.3)."""

from __future__ import annotations

from xeno.core.types import Breakpoint, NodeRole
from xeno.prompt.assembly import PromptBuilder
from xeno.security.outbound import sanitize
from xeno.security.scanner import (
    _SCAN_CACHE_ENTRIES,
    SecretScanner,
    shannon_entropy,
)

scanner = SecretScanner()


def test_known_key_prefixes_are_redacted() -> None:
    cases = {
        "github_token": "ghp_" + "a" * 36,
        "anthropic_api_key": "sk-ant-api03-" + "b" * 40,
        "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "openrouter_api_key": "sk-or-v1-" + "c" * 40,
        "slack_token": "xoxb-123456789012-abcdefghijkl",
    }
    for label, secret in cases.items():
        result = scanner.scan(f"config value: {secret} end")
        assert secret not in result.text, label
        assert any(f.label == label for f in result.findings), label


def test_private_key_block_is_redacted_whole() -> None:
    """Not shredded into fragments by the entropy detector matching inside it."""
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEAx7Vv8kQ2mZpLd9fT3sWqRbNcY6hJ1uPzKgAeD4oXsCvB0nMi\n"
        "-----END RSA PRIVATE KEY-----"
    )
    result = scanner.scan(f"key:\n{pem}\ndone")
    assert "BEGIN RSA PRIVATE KEY" not in result.text
    assert len(result.findings) == 1
    assert result.findings[0].label == "private_key_block"


def test_secret_shaped_assignments_are_caught_without_a_prefix() -> None:
    result = scanner.scan('DATABASE_PASSWORD = "hunter2istoolong"')
    assert "hunter2istoolong" not in result.text
    assert result.findings[0].detector == "assignment"


def test_placeholders_are_not_flagged() -> None:
    """Noisy findings teach users to ignore the log."""
    for value in ('API_KEY = "changeme"', 'token: "your_api_key_here"', 'secret = "TODO"'):
        assert scanner.scan(value).clean, value


def test_hex_digests_and_uuids_survive_by_default() -> None:
    """Git SHAs and lockfile hashes are pervasive in real repos. Redacting them
    wholesale would degrade every codebase map the harness builds."""
    text = (
        "commit 7c3f9a1b2d4e5f60718293a4b5c6d7e8f9012345\n"
        "id = 550e8400-e29b-41d4-a716-446655440000\n"
    )
    assert scanner.scan(text).clean


def test_hex_digests_can_be_redacted_when_asked() -> None:
    strict = SecretScanner(redact_hex_digests=True)
    result = strict.scan("token 7c3f9a1b2d4e5f60718293a4b5c6d7e8f9012345")
    assert not result.clean


def test_high_entropy_strings_are_caught() -> None:
    result = scanner.scan("value=Xk9$mQ2vRt7wNp4LzBc8YfHj3sGd6AeU")
    assert not result.clean


def test_prose_and_identifiers_are_left_alone() -> None:
    text = (
        "def calculate_rate_limit(requests_per_minute: int) -> RateLimiter:\n"
        "    return RateLimiter(requests_per_minute, window_seconds=60)\n"
    )
    assert scanner.scan(text).clean


def test_findings_never_carry_the_secret() -> None:
    secret = "ghp_" + "z" * 36
    finding = scanner.scan(secret).findings[0]
    assert secret not in finding.preview
    assert secret not in str(finding)
    assert "40 chars" in finding.preview


def test_redaction_is_deterministic() -> None:
    """Non-deterministic markers would make identical prompts differ
    byte-for-byte and destroy the static breakpoints (PRD T8)."""
    text = "key = ghp_" + "q" * 36
    assert scanner.scan(text).text == scanner.scan(text).text


def test_markers_do_not_leak_a_hash_of_the_secret() -> None:
    result = scanner.scan("ghp_" + "w" * 36)
    assert "[REDACTED:github_token:#1]" in result.text


def test_entropy_ranks_random_above_prose() -> None:
    assert shannon_entropy("Xk9mQ2vRt7wNp4LzBc8YfHj3") > shannon_entropy("the quick brown fox")


# ---- the prompt boundary ---------------------------------------------------


def test_sanitize_preserves_structure_and_reports_the_layer(keyring) -> None:  # type: ignore[no-untyped-def]
    builder = PromptBuilder(node=NodeRole.CODER, keyring=keyring, system_text="system text")
    builder.set_codebase_map("map with AKIAIOSFODNN7EXAMPLE inside")
    builder.append_turn("user", "earlier turn")
    prompt = builder.build("current turn")

    sanitized = sanitize(prompt, scanner)

    assert not sanitized.clean
    assert sanitized.breakpoints_hit == frozenset({Breakpoint.CODEBASE_MAP})
    assert [b.breakpoint for b in sanitized.prompt.blocks] == [
        b.breakpoint for b in prompt.blocks
    ]
    assert len(sanitized.prompt.history) == 1
    assert "AKIAIOSFODNN7EXAMPLE" not in str(sanitized.prompt.blocks)


def test_sanitize_scans_history_too(keyring) -> None:  # type: ignore[no-untyped-def]
    builder = PromptBuilder(node=NodeRole.DEBUGGER, keyring=keyring, system_text="sys")
    builder.append_turn("assistant", "I found sk-ant-api03-" + "k" * 40)
    sanitized = sanitize(builder.build("next"), scanner)
    assert sanitized.breakpoints_hit == frozenset({Breakpoint.ACCUMULATED_HISTORY})
    assert "sk-ant-" not in sanitized.prompt.history[0].content


def test_clean_prompt_passes_through_unchanged(keyring) -> None:  # type: ignore[no-untyped-def]
    builder = PromptBuilder(node=NodeRole.PLANNER, keyring=keyring, system_text="plan things")
    prompt = builder.build("decompose the goal")
    sanitized = sanitize(prompt, scanner)
    assert sanitized.clean
    assert sanitized.prompt.current_turn == prompt.current_turn


def test_repeated_scans_are_memoized_without_changing_the_result() -> None:
    """`sanitize` re-scans every static block and every accumulated history
    turn on every model call, and those are byte-identical call after call —
    so `scan` memoizes. Redaction must stay deterministic through the cache
    (PRD T8: a drifting SYSTEM or CODEBASE MAP loses its cache hits)."""
    text = "AWS_SECRET=" + "A" * 40 + " and sk-ant-api03-" + "k" * 40
    fresh = SecretScanner(entropy_threshold=4.0)

    first = fresh.scan(text)
    second = fresh.scan(text)

    assert second is first  # served from the memo, not rescanned
    assert second.text == SecretScanner(entropy_threshold=4.0).scan(text).text


def test_the_scan_memo_is_bounded() -> None:
    """The entries are prompt-sized strings, so the cache is a memory ceiling
    as much as a speed-up — it must not grow with the length of a run."""
    bounded = SecretScanner()
    for i in range(_SCAN_CACHE_ENTRIES * 3):
        bounded.scan(f"harmless text number {i}")
    assert len(bounded._cache) <= _SCAN_CACHE_ENTRIES
