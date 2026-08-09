"""The chart of accounts: account types, accounts and their container."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum


class UnknownAccountError(LookupError):
    """Raised when a code is not present in the chart of accounts."""


class DuplicateAccountError(ValueError):
    """Raised when the same account code is registered twice."""


class AccountType(StrEnum):
    """The five classical account categories."""

    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"
    INCOME = "income"
    EXPENSE = "expense"

    @property
    def is_balance_sheet(self) -> bool:
        """True for the account types that roll up into the balance sheet."""
        return self in (AccountType.ASSET, AccountType.LIABILITY, AccountType.EQUITY)


@dataclass(frozen=True)
class Account:
    """A single named account identified by its numeric code."""

    code: str
    name: str
    type: AccountType

    def __post_init__(self) -> None:
        code = self.code.strip()
        if not code:
            raise ValueError("account code must not be blank")
        if not self.name.strip():
            raise ValueError("account name must not be blank")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "name", self.name.strip())

    def __str__(self) -> str:
        return f"{self.code} {self.name}"


class ChartOfAccounts:
    """An ordered collection of accounts, looked up by code."""

    def __init__(self, accounts: Iterable[Account] = ()) -> None:
        self._accounts: dict[str, Account] = {}
        for account in accounts:
            self.add(account)

    def add(self, account: Account) -> Account:
        """Register ``account``; raise if its code is already present."""
        if account.code in self._accounts:
            raise DuplicateAccountError(f"account {account.code!r} already registered")
        self._accounts[account.code] = account
        return account

    def get(self, code: str) -> Account:
        """Return the account registered under ``code``."""
        try:
            return self._accounts[code]
        except KeyError as exc:
            raise UnknownAccountError(f"no account with code {code!r}") from exc

    def codes(self) -> list[str]:
        """Every registered code, in insertion order."""
        return list(self._accounts)

    def of_type(self, account_type: AccountType) -> list[Account]:
        """Every account of ``account_type``, in insertion order."""
        return [a for a in self._accounts.values() if a.type is account_type]

    def __iter__(self) -> Iterator[Account]:
        return iter(self._accounts.values())

    def __len__(self) -> int:
        return len(self._accounts)

    def __contains__(self, code: object) -> bool:
        return code in self._accounts
