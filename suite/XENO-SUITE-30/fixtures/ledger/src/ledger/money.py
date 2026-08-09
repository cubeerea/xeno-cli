"""Currency-safe money values built on :class:`decimal.Decimal`.

Every amount carries an explicit three-letter currency code. Arithmetic between
two different currencies is a programming error, never an implicit conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

CENTS = Decimal("0.01")
"""The smallest unit every reported amount is rounded to."""


class CurrencyMismatchError(ValueError):
    """Raised when two amounts in different currencies are combined."""


def _normalise_currency(currency: str) -> str:
    text = currency.strip().upper()
    if len(text) != 3 or not text.isalpha():
        raise ValueError(f"currency must be three letters, got {currency!r}")
    return text


@dataclass(frozen=True)
class Money:
    """An immutable amount of a single currency."""

    amount: Decimal
    currency: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "currency", _normalise_currency(self.currency))

    @classmethod
    def zero(cls, currency: str) -> Money:
        """Return a zero amount in ``currency``."""
        return cls(Decimal("0.00"), currency)

    def _check(self, other: Money) -> None:
        if self.currency != other.currency:
            raise CurrencyMismatchError(
                f"cannot combine {self.currency} with {other.currency}"
            )

    def quantize(self) -> Money:
        """Return the same amount rounded half-up to two decimal places."""
        return Money(self.amount.quantize(CENTS, rounding=ROUND_HALF_UP), self.currency)

    @property
    def is_zero(self) -> bool:
        """True when the amount is exactly zero."""
        return self.amount == 0

    @property
    def cents(self) -> int:
        """The amount expressed as a whole number of minor units."""
        return int((self.amount * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

    def __add__(self, other: Money) -> Money:
        self._check(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: Money) -> Money:
        self._check(other)
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, factor: int | Decimal) -> Money:
        return Money(self.amount * Decimal(factor), self.currency)

    def __neg__(self) -> Money:
        return Money(-self.amount, self.currency)

    def __abs__(self) -> Money:
        return Money(abs(self.amount), self.currency)

    def __lt__(self, other: Money) -> bool:
        self._check(other)
        return self.amount < other.amount

    def __le__(self, other: Money) -> bool:
        self._check(other)
        return self.amount <= other.amount

    def __gt__(self, other: Money) -> bool:
        self._check(other)
        return self.amount > other.amount

    def __ge__(self, other: Money) -> bool:
        self._check(other)
        return self.amount >= other.amount

    def __str__(self) -> str:
        return f"{self.amount.quantize(CENTS, rounding=ROUND_HALF_UP)} {self.currency}"
