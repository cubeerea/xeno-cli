"""Baseline coverage for :mod:`ledger.rates`.

These assertions are the regression guard for the frozen statutory rate.
"""

from __future__ import annotations

from decimal import Decimal

from ledger.money import Money
from ledger.rates import RATES, TAX_RATE, jurisdictions, net_of_tax, tax_on


def test_tax_rate_is_five_percent() -> None:
    assert TAX_RATE == Decimal("0.05")


def test_tax_on_applies_five_percent() -> None:
    assert tax_on(Money(Decimal("100.00"), "USD")) == Money(Decimal("5.00"), "USD")


def test_tax_on_rounds_to_cents() -> None:
    assert tax_on(Money(Decimal("19.99"), "USD")) == Money(Decimal("1.00"), "USD")


def test_tax_on_preserves_currency() -> None:
    assert tax_on(Money(Decimal("200.00"), "EUR")) == Money(Decimal("10.00"), "EUR")


def test_net_of_tax_subtracts_the_tax() -> None:
    assert net_of_tax(Money(Decimal("100.00"), "USD")) == Money(Decimal("95.00"), "USD")


def test_statutory_table_is_pinned_to_the_single_rate() -> None:
    assert set(RATES.values()) == {TAX_RATE}
    assert jurisdictions() == ["US-CA", "US-NY", "US-TX"]
