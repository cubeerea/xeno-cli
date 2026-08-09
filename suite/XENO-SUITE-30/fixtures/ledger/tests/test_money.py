"""Baseline coverage for :mod:`ledger.money`."""

from __future__ import annotations

from decimal import Decimal

import pytest

from ledger.money import CurrencyMismatchError, Money


def usd(text: str) -> Money:
    return Money(Decimal(text), "USD")


def test_currency_is_upper_cased() -> None:
    assert Money(Decimal("1.00"), "usd").currency == "USD"


def test_currency_must_be_three_letters() -> None:
    with pytest.raises(ValueError):
        Money(Decimal("1.00"), "US")


def test_addition_and_subtraction() -> None:
    assert usd("10.00") + usd("2.50") == usd("12.50")
    assert usd("10.00") - usd("2.50") == usd("7.50")


def test_multiplication_by_decimal() -> None:
    assert (usd("10.00") * Decimal("0.5")).quantize() == usd("5.00")


def test_negation_and_absolute_value() -> None:
    assert -usd("3.00") == usd("-3.00")
    assert abs(usd("-3.00")) == usd("3.00")


def test_currency_mismatch_is_rejected() -> None:
    with pytest.raises(CurrencyMismatchError):
        usd("1.00") + Money(Decimal("1.00"), "EUR")


def test_comparison_operators() -> None:
    assert usd("1.00") < usd("2.00")
    assert usd("2.00") >= usd("2.00")
    assert not usd("2.00") > usd("2.00")


def test_comparison_across_currencies_is_rejected() -> None:
    with pytest.raises(CurrencyMismatchError):
        _ = usd("1.00") < Money(Decimal("1.00"), "EUR")


def test_quantize_rounds_half_up() -> None:
    assert usd("1.005").quantize() == usd("1.01")
    assert usd("1.004").quantize() == usd("1.00")


def test_zero_and_is_zero() -> None:
    assert Money.zero("USD") == usd("0.00")
    assert Money.zero("USD").is_zero
    assert not usd("0.01").is_zero


def test_cents_property() -> None:
    assert usd("12.34").cents == 1234
    assert usd("-0.05").cents == -5


def test_str_renders_two_decimals() -> None:
    assert str(usd("7.5")) == "7.50 USD"


def test_money_is_hashable_and_frozen() -> None:
    assert len({usd("1.00"), usd("1.00"), usd("2.00")}) == 2
