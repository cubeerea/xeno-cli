"""Journal entries: the double-entry bookkeeping primitives."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date

from ledger.money import CurrencyMismatchError, Money


class BalanceError(ValueError):
    """Raised when an entry's debits do not equal its credits."""


class UnknownEntryError(LookupError):
    """Raised when a journal is asked for an id it does not hold."""


class DuplicateEntryError(ValueError):
    """Raised when the same entry id is posted twice."""


@dataclass(frozen=True)
class Line:
    """One side of one posting: a debit *or* a credit against an account."""

    account: str
    debit: Money | None = None
    credit: Money | None = None

    def __post_init__(self) -> None:
        if (self.debit is None) == (self.credit is None):
            raise ValueError("a line must carry exactly one of debit or credit")
        if self.amount.amount < 0:
            raise ValueError("line amounts must not be negative")
        if not self.account.strip():
            raise ValueError("a line must name an account code")

    @property
    def is_debit(self) -> bool:
        """True when this line debits its account."""
        return self.debit is not None

    @property
    def amount(self) -> Money:
        """The magnitude of the posting, regardless of side."""
        value = self.debit if self.debit is not None else self.credit
        if value is None:  # pragma: no cover - guarded by __post_init__
            raise ValueError("line has no amount")
        return value

    def __str__(self) -> str:
        side = "DR" if self.is_debit else "CR"
        return f"{side} {self.account} {self.amount}"


@dataclass(frozen=True)
class Entry:
    """A balanced set of lines posted on a single date."""

    id: str
    date: date
    description: str
    lines: Sequence[Line]

    def __post_init__(self) -> None:
        object.__setattr__(self, "lines", tuple(self.lines))
        if len(self.lines) < 2:
            raise BalanceError("an entry needs at least two lines")
        currencies = {line.amount.currency for line in self.lines}
        if len(currencies) > 1:
            raise CurrencyMismatchError(f"entry {self.id!r} mixes currencies {sorted(currencies)}")
        if self.total_debits != self.total_credits:
            raise BalanceError(
                f"entry {self.id!r} is unbalanced: "
                f"{self.total_debits} debit vs {self.total_credits} credit"
            )

    @property
    def currency(self) -> str:
        """The single currency every line of the entry is denominated in."""
        return self.lines[0].amount.currency

    def _side_total(self, *, debit: bool) -> Money:
        total = Money.zero(self.currency)
        for line in self.lines:
            if line.is_debit is debit:
                total = total + line.amount
        return total

    @property
    def total_debits(self) -> Money:
        """Sum of every debit line."""
        return self._side_total(debit=True)

    @property
    def total_credits(self) -> Money:
        """Sum of every credit line."""
        return self._side_total(debit=False)

    def accounts(self) -> list[str]:
        """Every account code touched by this entry, in line order."""
        seen: dict[str, None] = {}
        for line in self.lines:
            seen.setdefault(line.account, None)
        return list(seen)


class Journal:
    """An append-only collection of entries keyed by id."""

    def __init__(self, entries: Iterable[Entry] = ()) -> None:
        self._entries: dict[str, Entry] = {}
        for entry in entries:
            self.add(entry)

    def add(self, entry: Entry) -> Entry:
        """Post ``entry``; raise if its id has already been used."""
        if entry.id in self._entries:
            raise DuplicateEntryError(f"entry {entry.id!r} already posted")
        self._entries[entry.id] = entry
        return entry

    def get(self, entry_id: str) -> Entry:
        """Return the entry posted under ``entry_id``."""
        try:
            return self._entries[entry_id]
        except KeyError as exc:
            raise UnknownEntryError(f"no entry with id {entry_id!r}") from exc

    def ids(self) -> list[str]:
        """Every posted entry id, in posting order."""
        return list(self._entries)

    def __iter__(self) -> Iterator[Entry]:
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, entry_id: object) -> bool:
        return entry_id in self._entries
