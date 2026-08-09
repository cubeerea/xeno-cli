"""Baseline coverage for :mod:`ledger.journal`."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from ledger.journal import (
    BalanceError,
    DuplicateEntryError,
    Entry,
    Journal,
    Line,
    UnknownEntryError,
)
from ledger.money import CurrencyMismatchError, Money


def usd(text: str) -> Money:
    return Money(Decimal(text), "USD")


def simple_entry(entry_id: str = "E1") -> Entry:
    return Entry(
        id=entry_id,
        date=date(2024, 1, 31),
        description="Opening balance",
        lines=[
            Line("1000", debit=usd("100.00")),
            Line("3000", credit=usd("100.00")),
        ],
    )


def test_line_requires_exactly_one_side() -> None:
    with pytest.raises(ValueError):
        Line("1000")
    with pytest.raises(ValueError):
        Line("1000", debit=usd("1.00"), credit=usd("1.00"))


def test_line_reports_side_and_amount() -> None:
    line = Line("1000", debit=usd("5.00"))
    assert line.is_debit
    assert line.amount == usd("5.00")
    assert str(line) == "DR 1000 5.00 USD"


def test_line_rejects_negative_amounts() -> None:
    with pytest.raises(ValueError):
        Line("1000", debit=usd("-1.00"))


def test_entry_totals_balance() -> None:
    entry = simple_entry()
    assert entry.total_debits == usd("100.00")
    assert entry.total_credits == usd("100.00")
    assert entry.currency == "USD"


def test_entry_normalises_lines_to_a_tuple() -> None:
    assert isinstance(simple_entry().lines, tuple)


def test_entry_lists_touched_accounts() -> None:
    assert simple_entry().accounts() == ["1000", "3000"]


def test_unbalanced_entry_is_rejected() -> None:
    with pytest.raises(BalanceError):
        Entry(
            id="E9",
            date=date(2024, 1, 31),
            description="Bad",
            lines=[Line("1000", debit=usd("100.00")), Line("3000", credit=usd("90.00"))],
        )


def test_entry_needs_at_least_two_lines() -> None:
    with pytest.raises(BalanceError):
        Entry(id="E9", date=date(2024, 1, 31), description="Bad", lines=[])


def test_entry_rejects_mixed_currencies() -> None:
    with pytest.raises(CurrencyMismatchError):
        Entry(
            id="E9",
            date=date(2024, 1, 31),
            description="Bad",
            lines=[
                Line("1000", debit=usd("100.00")),
                Line("3000", credit=Money(Decimal("100.00"), "EUR")),
            ],
        )


def test_journal_add_get_and_iterate() -> None:
    journal = Journal([simple_entry("E1"), simple_entry("E2")])
    assert len(journal) == 2
    assert journal.get("E2").id == "E2"
    assert journal.ids() == ["E1", "E2"]
    assert [e.id for e in journal] == ["E1", "E2"]
    assert "E1" in journal


def test_journal_rejects_duplicate_ids() -> None:
    journal = Journal([simple_entry("E1")])
    with pytest.raises(DuplicateEntryError):
        journal.add(simple_entry("E1"))


def test_journal_unknown_id_raises() -> None:
    with pytest.raises(UnknownEntryError):
        Journal().get("nope")
