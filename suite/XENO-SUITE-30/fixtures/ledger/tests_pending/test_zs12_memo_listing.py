"""ZS-12 acceptance spec. Not collected by the baseline run (see testpaths)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from ledger import journal_listing
from ledger.journal import Entry, Journal, Line
from ledger.money import Money


def usd(text: str) -> Money:
    return Money(Decimal(text), "USD")


def opening() -> Entry:
    return Entry(
        id="E1",
        date=date(2024, 1, 31),
        description="Opening balance",
        lines=[Line("1000", debit=usd("100.00")), Line("3000", credit=usd("100.00"))],
        memo="initial funding",
    )


def coffee() -> Entry:
    return Entry(
        id="E2",
        date=date(2024, 2, 1),
        description="Coffee",
        lines=[Line("5000", debit=usd("4.00")), Line("1000", credit=usd("4.00"))],
    )


def test_memo_defaults_to_empty_string() -> None:
    assert coffee().memo == ""


def test_memo_is_stored_on_the_entry() -> None:
    assert opening().memo == "initial funding"


def test_listing_renders_memo_in_brackets() -> None:
    assert journal_listing(Journal([opening()])) == [
        "E1 2024-01-31 Opening balance [initial funding]"
    ]


def test_listing_omits_brackets_when_memo_is_empty() -> None:
    assert journal_listing(Journal([coffee()])) == ["E2 2024-02-01 Coffee"]


def test_listing_follows_journal_order() -> None:
    assert journal_listing(Journal([opening(), coffee()])) == [
        "E1 2024-01-31 Opening balance [initial funding]",
        "E2 2024-02-01 Coffee",
    ]


def test_listing_is_exported_from_the_package_root() -> None:
    import ledger
    import ledger.reports

    assert "journal_listing" in ledger.__all__
    assert ledger.journal_listing is ledger.reports.journal_listing


def test_listing_of_an_empty_journal_is_empty() -> None:
    assert journal_listing(Journal()) == []
