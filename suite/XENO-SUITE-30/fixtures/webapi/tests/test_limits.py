"""Baseline behaviour of the token-bucket rate limiter.

The policies below are arbitrary numbers chosen to make the arithmetic easy to
read; they are not, and must not be read as, any deployment's tier.
"""

from __future__ import annotations

import pytest

from webapi.limits import RateLimiter, RateLimitPolicy


class _ScriptedClock:
    """A clock that only advances when the test says so."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_policy_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="requests_per_minute"):
        RateLimitPolicy(requests_per_minute=0, burst=1)
    with pytest.raises(ValueError, match="burst"):
        RateLimitPolicy(requests_per_minute=60, burst=0)


def test_refill_per_second_is_derived_from_the_minute_rate() -> None:
    assert RateLimitPolicy(requests_per_minute=60, burst=1).refill_per_second == 1.0
    assert RateLimitPolicy(requests_per_minute=30, burst=1).refill_per_second == 0.5


def test_bucket_starts_full_and_drains() -> None:
    clock = _ScriptedClock()
    limiter = RateLimiter(RateLimitPolicy(requests_per_minute=60, burst=2), clock)
    assert limiter.allow() is True
    assert limiter.allow() is True
    assert limiter.allow() is False


def test_tokens_refill_over_time() -> None:
    clock = _ScriptedClock()
    limiter = RateLimiter(RateLimitPolicy(requests_per_minute=60, burst=2), clock)
    assert limiter.allow(2) is True
    assert limiter.allow() is False
    clock.advance(1.0)
    assert limiter.allow() is True


def test_tokens_never_exceed_the_burst_capacity() -> None:
    clock = _ScriptedClock()
    limiter = RateLimiter(RateLimitPolicy(requests_per_minute=60, burst=2), clock)
    clock.advance(3600.0)
    assert limiter.tokens == 2.0


def test_a_costly_request_can_be_refused_without_draining_the_bucket() -> None:
    clock = _ScriptedClock()
    limiter = RateLimiter(RateLimitPolicy(requests_per_minute=60, burst=2), clock)
    assert limiter.allow(5) is False
    assert limiter.tokens == 2.0
    assert limiter.allow(2) is True


def test_non_positive_cost_is_rejected() -> None:
    clock = _ScriptedClock()
    limiter = RateLimiter(RateLimitPolicy(requests_per_minute=60, burst=2), clock)
    with pytest.raises(ValueError, match="cost"):
        limiter.allow(0)


def test_policy_is_exposed_for_inspection() -> None:
    policy = RateLimitPolicy(requests_per_minute=60, burst=2)
    limiter = RateLimiter(policy, _ScriptedClock())
    assert limiter.policy is policy
