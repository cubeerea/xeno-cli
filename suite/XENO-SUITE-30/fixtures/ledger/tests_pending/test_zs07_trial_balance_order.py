"""ZS-07 acceptance spec. Not collected by the baseline run (see testpaths)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from ledger.accounts import Account, AccountType, ChartOfAccounts
from ledger.journal import Entry, Journal, Line
from ledger.money import Money
from ledger.reports import trial_balance


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


def sale() -> Entry:
    return Entry(
        id="E1",
        date=date(2024, 1, 5),
        description="Cash sale",
        lines=[Line("4000", credit=usd("300.00")), Line("1000", debit=usd("300.00"))],
    )


def rent() -> Entry:
    return Entry(
        id="E2",
        date=date(2024, 1, 6),
        description="Pay rent",
        lines=[Line("5000", debit=usd("120.00")), Line("1000", credit=usd("120.00"))],
    )


def test_rows_are_sorted_by_account_code() -> None:
    rows = trial_balance(Journal([sale(), rent()]), chart())
    assert [row[0] for row in rows[:-1]] == ["1000", "4000", "5000"]


def test_last_row_is_the_totals_row() -> None:
    rows = trial_balance(Journal([sale(), rent()]), chart())
    assert rows[-1][0] == "TOTAL"


def test_totals_row_debits_equal_credits() -> None:
    rows = trial_balance(Journal([sale(), rent()]), chart())
    _code, debit_total, credit_total = rows[-1]
    assert debit_total == credit_total
    assert debit_total == usd("420.00")


def test_totals_row_sums_the_detail_rows() -> None:
    rows = trial_balance(Journal([sale(), rent()]), chart())
    debit_sum = Money.zero("USD")
    credit_sum = Money.zero("USD")
    for _code, debit, credit in rows[:-1]:
        debit_sum = debit_sum + debit
        credit_sum = credit_sum + credit
    assert rows[-1] == ("TOTAL", debit_sum, credit_sum)


def test_row_order_is_independent_of_journal_order() -> None:
    forward = trial_balance(Journal([sale(), rent()]), chart())
    backward = trial_balance(Journal([rent(), sale()]), chart())
    assert forward == backward
    assert [row[0] for row in backward] == ["1000", "4000", "5000", "TOTAL"]


def test_totals_row_is_the_only_extra_row() -> None:
    rows = trial_balance(Journal([sale(), rent()]), chart())
    assert len(rows) == 4
    assert [row[0] for row in rows] == ["1000", "4000", "5000", "TOTAL"]
