"""ZS-28 acceptance spec. Not collected by the baseline run (see testpaths)."""

from __future__ import annotations

from decimal import Decimal

from ledger.money import Money
from ledger.rates import TAX_RATE, tax_on


def test_tax_rate_is_seven_percent() -> None:
    assert TAX_RATE == Decimal("0.07")


def test_tax_on_applies_seven_percent() -> None:
    assert tax_on(Money(Decimal("100.00"), "USD")) == Money(Decimal("7.00"), "USD")
