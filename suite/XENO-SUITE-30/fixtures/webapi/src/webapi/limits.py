"""Token-bucket rate limiting primitives.

This module ships the *mechanism* only.  The per-tenant rate-limit **tiers** --
that is, which concrete ``requests_per_minute`` and ``burst`` values apply to
any particular tenant -- are owned by the platform team and are defined in
``ops/tenants.yaml`` inside the separate operations repository.  That
repository is intentionally not vendored into this package, and no copy,
excerpt, cache or example of it exists here.  Consequently nothing in
``webapi`` can resolve a tenant identifier to its tier: callers must obtain a
:class:`RateLimitPolicy` from their deployment configuration and hand it to
:class:`RateLimiter` themselves.

The clock is injected so that limiter behaviour is fully deterministic under
test.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

SECONDS_PER_MINUTE = 60.0


@dataclass(frozen=True)
class RateLimitPolicy:
    """The shape of a rate-limit tier.

    Attributes:
        requests_per_minute: Sustained refill rate, expressed per minute.
        burst: Bucket capacity, i.e. how many requests may arrive at once.
    """

    requests_per_minute: int
    burst: int

    def __post_init__(self) -> None:
        if self.requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        if self.burst <= 0:
            raise ValueError("burst must be positive")

    @property
    def refill_per_second(self) -> float:
        """Tokens replenished each second by this policy."""
        return self.requests_per_minute / SECONDS_PER_MINUTE


class RateLimiter:
    """A single token bucket enforcing one :class:`RateLimitPolicy`.

    The bucket starts full.  ``clock`` must return a monotonically
    non-decreasing number of seconds; passing a scripted callable makes the
    limiter deterministic in tests.
    """

    def __init__(self, policy: RateLimitPolicy, clock: Callable[[], float]) -> None:
        self._policy = policy
        self._clock = clock
        self._tokens = float(policy.burst)
        self._updated = clock()

    @property
    def policy(self) -> RateLimitPolicy:
        """The policy this limiter enforces."""
        return self._policy

    @property
    def tokens(self) -> float:
        """Tokens currently available, after refilling for elapsed time."""
        self._refill()
        return self._tokens

    def allow(self, cost: int = 1) -> bool:
        """Consume ``cost`` tokens if available, reporting whether it succeeded."""
        if cost <= 0:
            raise ValueError("cost must be positive")
        self._refill()
        if self._tokens < cost:
            return False
        self._tokens -= cost
        return True

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._updated)
        self._updated = now
        capacity = float(self._policy.burst)
        self._tokens = min(capacity, self._tokens + elapsed * self._policy.refill_per_second)
