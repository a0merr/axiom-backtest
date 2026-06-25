"""Position sizing strategies.

Sizing is factored out of the portfolio so the *how much* decision is swappable
without touching order/fill plumbing. A ``Sizer`` sees prices each bar
(``observe``) and is asked for a share count when a new position opens
(``size``). Three implementations are provided, increasing in sophistication:

* ``FixedFractionalSizer`` — a fixed fraction of equity per position (the
  original, dependency-free default).
* ``VolatilityTargetSizer`` — scales each position so its standalone risk hits a
  target annualized volatility; low-vol names get bigger, high-vol names smaller.
* ``CorrelationAwareSizer`` — vol targeting plus a haircut for how correlated the
  new name is to what you already hold, so a book of look-alike positions isn't
  silently concentrated into one bet.

These are deliberately transparent heuristics, not a mean-variance optimizer.
The limits are stated where they bite.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Iterable

import numpy as np

from .data import DataHandler


class Sizer(ABC):
    def observe(self, data: DataHandler) -> None:  # noqa: B027 - optional hook
        """Called once per bar with the current data handler. Default: no-op."""

    @abstractmethod
    def size(
        self,
        symbol: str,
        price: float,
        strength: float,
        equity: float,
        held_symbols: Iterable[str],
    ) -> float:
        """Return a non-negative share count for a new position."""


class FixedFractionalSizer(Sizer):
    """Allocate ``fraction`` of current equity to each new position."""

    def __init__(self, fraction: float = 0.1):
        if not 0 <= fraction <= 1:
            raise ValueError("fraction must be in [0, 1]")
        self.fraction = fraction

    def size(
        self,
        symbol: str,
        price: float,
        strength: float,
        equity: float,
        held_symbols: Iterable[str],
    ) -> float:
        budget = equity * self.fraction * max(0.0, min(strength, 1.0))
        return float(int(budget // price))


class _ReturnTracker(Sizer):
    """Mixin: maintain a rolling window of per-symbol returns."""

    def __init__(self, lookback: int = 63, periods_per_year: int = 252):
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        self.lookback = lookback
        self.periods_per_year = periods_per_year
        # store prices (lookback + 1 to yield `lookback` returns)
        self._prices: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=lookback + 1)
        )

    def observe(self, data: DataHandler) -> None:
        for symbol in data.symbols:
            price = data.latest_price(symbol)
            if price is not None and price > 0:
                self._prices[symbol].append(price)

    def _returns(self, symbol: str) -> np.ndarray:
        prices = np.asarray(self._prices.get(symbol, ()), dtype=float)
        if prices.size < 2:
            return np.empty(0)
        return np.diff(prices) / prices[:-1]

    def _annual_vol(self, symbol: str) -> float | None:
        r = self._returns(symbol)
        if r.size < 2:
            return None
        sd = r.std(ddof=1)
        if sd == 0:
            return None
        return float(sd * np.sqrt(self.periods_per_year))


class VolatilityTargetSizer(_ReturnTracker):
    """Size each position to a target annualized volatility.

    ``weight = target_vol / asset_vol`` (capped at ``max_weight``), so a name
    that is half as volatile gets twice the notional. Until enough history has
    accumulated it falls back to ``fallback_fraction`` of equity.

    Limit: this targets *standalone* position vol, not portfolio vol — it does
    not net correlations across holdings. Use ``CorrelationAwareSizer`` for that.
    """

    def __init__(
        self,
        target_vol: float = 0.15,
        lookback: int = 63,
        max_weight: float = 1.0,
        fallback_fraction: float = 0.1,
        periods_per_year: int = 252,
    ):
        super().__init__(lookback, periods_per_year)
        if target_vol <= 0:
            raise ValueError("target_vol must be positive")
        self.target_vol = target_vol
        self.max_weight = max_weight
        self.fallback_fraction = fallback_fraction

    def _weight(self, symbol: str) -> float:
        vol = self._annual_vol(symbol)
        if vol is None:
            return self.fallback_fraction
        return min(self.max_weight, self.target_vol / vol)

    def size(
        self,
        symbol: str,
        price: float,
        strength: float,
        equity: float,
        held_symbols: Iterable[str],
    ) -> float:
        weight = self._weight(symbol) * max(0.0, min(strength, 1.0))
        return float(int((equity * weight) // price))


class CorrelationAwareSizer(VolatilityTargetSizer):
    """Vol targeting with a diversification haircut against the current book.

    The vol-target weight is multiplied by ``(1 - penalty * max(0, avg_corr))``,
    where ``avg_corr`` is the mean correlation of the candidate's recent returns
    to those of the symbols already held. A new position that mostly duplicates
    existing risk is scaled down; an uncorrelated diversifier keeps its full
    vol-target weight.

    Limit: this is a pairwise-correlation heuristic, not a full covariance
    optimization — it discourages obvious concentration without claiming to find
    the minimum-variance book.
    """

    def __init__(
        self,
        target_vol: float = 0.15,
        lookback: int = 63,
        max_weight: float = 1.0,
        fallback_fraction: float = 0.1,
        correlation_penalty: float = 1.0,
        periods_per_year: int = 252,
    ):
        super().__init__(
            target_vol, lookback, max_weight, fallback_fraction, periods_per_year
        )
        if correlation_penalty < 0:
            raise ValueError("correlation_penalty must be non-negative")
        self.correlation_penalty = correlation_penalty

    def _avg_correlation(self, symbol: str, held_symbols: Iterable[str]) -> float:
        r = self._returns(symbol)
        corrs: list[float] = []
        for other in held_symbols:
            if other == symbol:
                continue
            ro = self._returns(other)
            n = min(r.size, ro.size)
            if n < 2:
                continue
            a, b = r[-n:], ro[-n:]
            if a.std(ddof=1) == 0 or b.std(ddof=1) == 0:
                continue
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
        return float(np.mean(corrs)) if corrs else 0.0

    def size(
        self,
        symbol: str,
        price: float,
        strength: float,
        equity: float,
        held_symbols: Iterable[str],
    ) -> float:
        held = list(held_symbols)
        weight = self._weight(symbol)
        avg_corr = self._avg_correlation(symbol, held)
        haircut = max(0.0, 1.0 - self.correlation_penalty * max(0.0, avg_corr))
        weight *= haircut * max(0.0, min(strength, 1.0))
        return float(int((equity * weight) // price))
