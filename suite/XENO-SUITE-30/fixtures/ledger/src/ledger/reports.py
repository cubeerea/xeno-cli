"""Read-only views over a journal."""

from __future__ import annotations

from ledger.accounts import ChartOfAccounts, UnknownAccountError
from ledger.journal import Journal
from ledger.money import Money

DEFAULT_CURRENCY = "USD"
"""Currency used for zero results when the journal offers no better answer."""


def account_balance(journal: Journal, code: str, *, currency: str = DEFAULT_CURRENCY) -> Money:
    """Return the net debit-minus-credit balance of one account code.

    A positive result means the account carries a net debit balance. If the
    account is never touched, a zero amount in ``currency`` is returned.
    """
    total: Money | None = None
    for entry in journal:
        for line in entry.lines:
            if line.account != code:
                continue
            delta = line.amount if line.is_debit else -line.amount
            total = delta if total is None else total + delta
    return total if total is not None else Money.zero(currency)


def trial_balance(journal: Journal, chart: ChartOfAccounts) -> list[tuple[str, Money, Money]]:
    """Return ``(code, debit_total, credit_total)`` for every account with activity.

    Rows are emitted in the order the account codes are first encountered while
    walking the journal.
    """
    debits: dict[str, Money] = {}
    credits: dict[str, Money] = {}
    order: list[str] = []

    for entry in journal:
        for line in entry.lines:
            code = line.account
            if code not in chart:
                raise UnknownAccountError(f"entry {entry.id!r} posts to unknown account {code!r}")
            if code not in debits:
                order.append(code)
                zero = Money.zero(line.amount.currency)
                debits[code] = zero
                credits[code] = zero
            if line.is_debit:
                debits[code] = debits[code] + line.amount
            else:
                credits[code] = credits[code] + line.amount

    return [(code, debits[code], credits[code]) for code in order]


def is_balanced(journal: Journal, chart: ChartOfAccounts) -> bool:
    """True when the journal's total debits equal its total credits."""
    rows = trial_balance(journal, chart)
    if not rows:
        return True
    currency = rows[0][1].currency
    debit_total = Money.zero(currency)
    credit_total = Money.zero(currency)
    for _code, debit, credit in rows:
        debit_total = debit_total + debit
        credit_total = credit_total + credit
    return debit_total == credit_total
