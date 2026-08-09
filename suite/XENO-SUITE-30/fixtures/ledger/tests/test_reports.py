"""Baseline coverage for :mod:`ledger.reports`."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from ledger.accounts import Account, AccountType, ChartOfAccounts, UnknownAccountError
from ledger.journal import Entry, Journal, Line
from ledger.money import Money
from ledger.reports import account_balance, is_balanced, trial_balance


def usd(text: str) -> Money:
    return Money(Decimal(text), "USD")


def chart() -> ChartOfAccounts:
    return ChartOfAccounts(
        [
            Account("1000", "Cash", AccountType.ASSET),
            Account("4000", "Sales", AccountType.INCOME),
            Account("5000", "Rent", AccountType.EXPENSE),
        ]
    )


def journal() -> Journal:
    return Journal(
        [
            Entry(
                id="E1",
                date=date(2024, 1, 5),
                description="Cash sale",
                lines=[Line("1000", debit=usd("300.00")), Line("4000", credit=usd("300.00"))],
            ),
            Entry(
                id="E2",
                date=date(2024, 1, 6),
                description="Pay rent",
                lines=[Line("5000", debit=usd("120.00")), Line("1000", credit=usd("120.00"))],
            ),
        ]
    )


def test_account_balance_is_debits_minus_credits() -> None:
    assert account_balance(journal(), "1000") == usd("180.00")
    assert account_balance(journal(), "4000") == usd("-300.00")
    assert account_balance(journal(), "5000") == usd("120.00")


def test_account_balance_of_untouched_account_is_zero() -> None:
    assert account_balance(journal(), "9999") == Money.zero("USD")


def test_trial_balance_rows_carry_both_sides() -> None:
    rows = {code: (debit, credit) for code, debit, credit in trial_balance(journal(), chart())}
    assert rows["1000"] == (usd("300.00"), usd("120.00"))
    assert rows["4000"] == (Money.zero("USD"), usd("300.00"))
    assert rows["5000"] == (usd("120.00"), Money.zero("USD"))


def test_trial_balance_lists_every_active_account_once() -> None:
    codes = [row[0] for row in trial_balance(journal(), chart())]
    for code in ("1000", "4000", "5000"):
        assert codes.count(code) == 1


def test_trial_balance_rejects_accounts_outside_the_chart() -> None:
    thin = ChartOfAccounts([Account("1000", "Cash", AccountType.ASSET)])
    with pytest.raises(UnknownAccountError):
        trial_balance(journal(), thin)


def test_journal_is_balanced_overall() -> None:
    assert is_balanced(journal(), chart())
    assert is_balanced(Journal(), chart())
