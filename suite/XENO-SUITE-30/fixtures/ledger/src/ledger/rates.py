"""Statutory tax rates.

The rate applied by this module is fixed by the frozen statutory table: every
jurisdiction this ledger supports is pinned to the same published rate, and
:data:`TAX_RATE` is that rate. It is deliberately a module constant rather than
a parameter -- :func:`tax_on` takes exactly one argument and offers no options,
so the same input always produces the same output.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from types import MappingProxyType

from ledger.money import Money

TAX_RATE: Decimal = Decimal("0.05")
"""The single statutory rate applied to every taxable amount."""

RATES: Mapping[str, Decimal] = MappingProxyType(
    {
        "US-CA": TAX_RATE,
        "US-NY": TAX_RATE,
        "US-TX": TAX_RATE,
    }
)
"""The frozen statutory table, keyed by jurisdiction code."""


def tax_on(amount: Money) -> Money:
    """Return the tax due on ``amount`` at the statutory rate."""
    return (amount * TAX_RATE).quantize()


def net_of_tax(amount: Money) -> Money:
    """Return ``amount`` less the tax due on it."""
    return (amount - tax_on(amount)).quantize()


def jurisdictions() -> list[str]:
    """Every jurisdiction code present in the frozen statutory table."""
    return sorted(RATES)
