"""ZS-20 acceptance spec. Not collected by the baseline run (see testpaths)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from ledger import BudgetPlan, budget_variance_report
from ledger.accounts import Account, AccountType, ChartOfAccounts
from ledger.budget import BudgetPlan as SubpackageBudgetPlan
from ledger.budget.plan import BudgetPlan as ModuleBudgetPlan
from ledger.journal import Entry, Journal, Line
from ledger.money import Money


def usd(text: str) -> Money:
    return Money(Decimal(text), "USD")


def chart() -> ChartOfAccounts:
    return ChartOfAccounts(
        [
            Account("1000", "Cash", AccountType.ASSET),
            Account("5000", "Rent", AccountType.EXPENSE),
            Account("5100", "Travel", AccountType.EXPENSE),
            Account("5200", "Software", AccountType.EXPENSE),
        ]
    )


def journal() -> Journal:
    return Journal(
        [
            Entry(
                id="E1",
                date=date(2024, 1, 6),
                description="Pay rent",
                lines=[Line("5000", debit=usd("900.00")), Line("1000", credit=usd("900.00"))],
            ),
            Entry(
                id="E2",
                date=date(2024, 1, 9),
                description="Buy software",
                lines=[Line("5200", debit=usd("50.00")), Line("1000", credit=usd("50.00"))],
            ),
        ]
    )


def plan() -> BudgetPlan:
    return BudgetPlan({"5000": usd("1000.00"), "5100": usd("400.00")})


def test_plan_class_is_the_same_object_everywhere() -> None:
    assert SubpackageBudgetPlan is BudgetPlan
    assert ModuleBudgetPlan is BudgetPlan


def test_budgeted_returns_the_planned_amount() -> None:
    assert plan().budgeted("5000") == usd("1000.00")
    assert plan().budgeted("5100") == usd("400.00")


def test_budgeted_returns_zero_for_unbudgeted_codes() -> None:
    assert plan().budgeted("5200") == Money.zero("USD")
    assert plan().budgeted("9999").is_zero


def test_report_rows_are_sorted_by_code() -> None:
    rows = budget_variance_report(journal(), chart(), plan())
    assert [row[0] for row in rows] == ["1000", "5000", "5100", "5200"]


def test_report_pairs_budget_with_actual_and_variance() -> None:
    rows = {row[0]: row for row in budget_variance_report(journal(), chart(), plan())}
    assert rows["5000"] == ("5000", usd("1000.00"), usd("900.00"), usd("100.00"))
    assert rows["5200"] == ("5200", usd("0.00"), usd("50.00"), usd("-50.00"))


def test_report_includes_budgeted_accounts_with_no_activity() -> None:
    rows = {row[0]: row for row in budget_variance_report(journal(), chart(), plan())}
    assert rows["5100"] == ("5100", usd("400.00"), usd("0.00"), usd("400.00"))


def test_report_variance_is_budget_minus_actual() -> None:
    for _code, budgeted, actual, variance in budget_variance_report(journal(), chart(), plan()):
        assert variance == budgeted - actual


def test_report_is_exported_from_the_package_root() -> None:
    import ledger

    assert "BudgetPlan" in ledger.__all__
    assert "budget_variance_report" in ledger.__all__
    assert budget_variance_report is ledger.budget_variance_report
