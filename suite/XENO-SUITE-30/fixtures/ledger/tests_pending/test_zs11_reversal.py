"""ZS-11 acceptance spec. Not collected by the baseline run (see testpaths)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from ledger.accounts import Account, AccountType
from ledger.journal import Entry, Journal, Line, UnknownEntryError
from ledger.money import Money


def usd(text: str) -> Money:
    return Money(Decimal(text), "USD")


def sale() -> Entry:
    return Entry(
        id="E1",
        date=date(2024, 1, 5),
        description="Cash sale",
        lines=[Line("1000", debit=usd("300.00")), Line("4000", credit=usd("300.00"))],
    )


def test_account_type_exposes_its_normal_balance() -> None:
    assert AccountType.ASSET.normal_balance == "debit"
    assert AccountType.EXPENSE.normal_balance == "debit"
    assert AccountType.LIABILITY.normal_balance == "credit"
    assert AccountType.EQUITY.normal_balance == "credit"
    assert AccountType.INCOME.normal_balance == "credit"


def test_account_delegates_normal_balance_to_its_type() -> None:
    assert Account("1000", "Cash", AccountType.ASSET).normal_balance == "debit"
    assert Account("4000", "Sales", AccountType.INCOME).normal_balance == "credit"


def test_reverse_returns_a_new_entry_registered_in_the_journal() -> None:
    journal = Journal([sale()])
    reversal = journal.reverse("E1")
    assert reversal.id == "E1-R"
    assert journal.get("E1-R") is reversal
    assert len(journal) == 2
    assert journal.ids() == ["E1", "E1-R"]


def test_reverse_mirrors_every_line() -> None:
    reversal = Journal([sale()]).reverse("E1")
    assert reversal.lines == (
        Line("1000", credit=usd("300.00")),
        Line("4000", debit=usd("300.00")),
    )
    assert reversal.total_debits == usd("300.00")
    assert reversal.total_credits == usd("300.00")


def test_reverse_prefixes_the_description_and_keeps_the_date() -> None:
    reversal = Journal([sale()]).reverse("E1")
    assert reversal.description == "REVERSAL: Cash sale"
    assert reversal.date == date(2024, 1, 5)


def test_reverse_of_unknown_id_raises() -> None:
    with pytest.raises(UnknownEntryError):
        Journal([sale()]).reverse("nope")
