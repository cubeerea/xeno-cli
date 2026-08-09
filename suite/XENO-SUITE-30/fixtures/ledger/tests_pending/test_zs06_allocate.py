"""ZS-06 acceptance spec. Not collected by the baseline run (see testpaths)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from ledger.money import Money


def usd(text: str) -> Money:
    return Money(Decimal(text), "USD")


def test_allocate_five_cents_over_three_seven() -> None:
    assert usd("0.05").allocate([3, 7]) == [usd("0.02"), usd("0.03")]


def test_allocate_hundred_over_three_equal_shares() -> None:
    assert usd("100.00").allocate([1, 1, 1]) == [usd("33.34"), usd("33.33"), usd("33.33")]


def test_allocate_remainder_goes_to_earliest_buckets() -> None:
    assert usd("0.10").allocate([1, 1, 1, 1]) == [
        usd("0.03"),
        usd("0.03"),
        usd("0.02"),
        usd("0.02"),
    ]


def test_allocate_is_exact_when_it_divides_evenly() -> None:
    assert usd("9.00").allocate([1, 2]) == [usd("3.00"), usd("6.00")]


def test_allocate_loses_no_cents() -> None:
    for ratios in ([3, 7], [1, 1, 1], [5, 2, 9, 1], [1, 1, 1, 1, 1, 1, 1]):
        parts = usd("100.00").allocate(ratios)
        total = parts[0]
        for part in parts[1:]:
            total = total + part
        assert total == usd("100.00")
        assert len(parts) == len(ratios)


def test_allocate_ignores_zero_weighted_buckets() -> None:
    assert usd("1.00").allocate([0, 1]) == [usd("0.00"), usd("1.00")]


def test_allocate_preserves_currency() -> None:
    parts = Money(Decimal("1.00"), "EUR").allocate([1, 1])
    assert [part.currency for part in parts] == ["EUR", "EUR"]


def test_allocate_rejects_empty_ratios() -> None:
    with pytest.raises(ValueError):
        usd("1.00").allocate([])


def test_allocate_rejects_all_zero_ratios() -> None:
    with pytest.raises(ValueError):
        usd("1.00").allocate([0, 0])


def test_allocate_rejects_negative_ratios() -> None:
    with pytest.raises(ValueError):
        usd("1.00").allocate([-1, 3])
