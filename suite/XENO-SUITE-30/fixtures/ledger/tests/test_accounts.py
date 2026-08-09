"""Baseline coverage for :mod:`ledger.accounts`."""

from __future__ import annotations

import pytest

from ledger.accounts import (
    Account,
    AccountType,
    ChartOfAccounts,
    DuplicateAccountError,
    UnknownAccountError,
)


def sample_chart() -> ChartOfAccounts:
    return ChartOfAccounts(
        [
            Account("4000", "Sales", AccountType.INCOME),
            Account("1000", "Cash", AccountType.ASSET),
            Account("5000", "Rent", AccountType.EXPENSE),
        ]
    )


def test_account_type_values() -> None:
    assert AccountType.ASSET.value == "asset"
    assert AccountType("expense") is AccountType.EXPENSE


def test_balance_sheet_classification() -> None:
    assert AccountType.LIABILITY.is_balance_sheet
    assert not AccountType.INCOME.is_balance_sheet


def test_account_strips_and_renders() -> None:
    account = Account(" 1000 ", " Cash ", AccountType.ASSET)
    assert account.code == "1000"
    assert str(account) == "1000 Cash"


def test_blank_code_is_rejected() -> None:
    with pytest.raises(ValueError):
        Account("  ", "Cash", AccountType.ASSET)


def test_lookup_by_code() -> None:
    chart = sample_chart()
    assert chart.get("1000").name == "Cash"
    assert "5000" in chart
    assert len(chart) == 3


def test_unknown_code_raises() -> None:
    with pytest.raises(UnknownAccountError):
        sample_chart().get("9999")


def test_iteration_preserves_insertion_order() -> None:
    assert sample_chart().codes() == ["4000", "1000", "5000"]
    assert [a.code for a in sample_chart()] == ["4000", "1000", "5000"]


def test_duplicate_code_is_rejected() -> None:
    chart = sample_chart()
    with pytest.raises(DuplicateAccountError):
        chart.add(Account("1000", "Petty cash", AccountType.ASSET))


def test_filter_by_type() -> None:
    assert [a.code for a in sample_chart().of_type(AccountType.ASSET)] == ["1000"]
