"""A minimal double-entry accounting core.

Fixture for XENO-SUITE-30. Deliberately small, dependency-free and green on
ruff / mypy / pytest before any suite task is applied.
"""

from __future__ import annotations

from ledger.accounts import (
    Account,
    AccountType,
    ChartOfAccounts,
    DuplicateAccountError,
    UnknownAccountError,
)
from ledger.journal import (
    BalanceError,
    DuplicateEntryError,
    Entry,
    Journal,
    Line,
    UnknownEntryError,
)
from ledger.money import CurrencyMismatchError, Money
from ledger.rates import RATES, TAX_RATE, net_of_tax, tax_on
from ledger.reports import account_balance, is_balanced, trial_balance

__all__ = [
    "RATES",
    "TAX_RATE",
    "Account",
    "AccountType",
    "BalanceError",
    "ChartOfAccounts",
    "CurrencyMismatchError",
    "DuplicateAccountError",
    "DuplicateEntryError",
    "Entry",
    "Journal",
    "Line",
    "Money",
    "UnknownAccountError",
    "UnknownEntryError",
    "account_balance",
    "is_balanced",
    "net_of_tax",
    "tax_on",
    "trial_balance",
]
__version__ = "0.1.0"
