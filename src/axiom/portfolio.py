"""Portfolio: position sizing, order generation, and equity tracking.

The portfolio is the only component that knows about cash. It converts signals
into orders, applies fills to its books, and records a mark-to-market equity
point on every bar so the metrics module has a clean return series to work with.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .data import DataHandler
from .event import Direction, FillEvent, MarketEvent, OrderEvent, SignalEvent


@dataclass
class _Position:
    quantity: float = 0.0
    avg_price: float = 0.0


@dataclass
class Portfolio:
    """Fixed-fractional portfolio.

    Each new long allocates ``risk_fraction`` of current equity to the symbol.
    Exits flatten the position. This is deliberately conservative and easy to
    reason about; smarter sizing (vol targeting, Kelly) can replace ``_size``.
    """

    data: DataHandler
    initial_capital: float = 100_000.0
    risk_fraction: float = 0.1

    cash: float = field(init=False)
    positions: dict[str, _Position] = field(init=False)
    equity_curve: list[dict] = field(init=False)
    _total_commission: float = field(init=False, default=0.0)
    _total_slippage: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.cash = self.initial_capital
        self.positions = {s: _Position() for s in self.data.symbols}
        self.equity_curve = []

    # -- signal -> order ---------------------------------------------------
    def on_signal(self, signal: SignalEvent) -> list[OrderEvent]:
        price = self.data.latest_price(signal.symbol)
        if price is None or price <= 0:
            return []
        pos = self.positions[signal.symbol]

        if signal.direction == Direction.EXIT:
            if pos.quantity == 0:
                return []
            direction = Direction.SHORT if pos.quantity > 0 else Direction.LONG
            return [OrderEvent(signal.symbol, signal.timestamp, direction, abs(pos.quantity))]

        if signal.direction == Direction.LONG and pos.quantity <= 0:
            qty = self._size(price, signal.strength)
            if qty <= 0:
                return []
            return [OrderEvent(signal.symbol, signal.timestamp, Direction.LONG, qty)]

        if signal.direction == Direction.SHORT and pos.quantity >= 0:
            qty = self._size(price, signal.strength)
            if qty <= 0:
                return []
            return [OrderEvent(signal.symbol, signal.timestamp, Direction.SHORT, qty)]

        return []

    def _size(self, price: float, strength: float) -> float:
        budget = self.equity() * self.risk_fraction * max(0.0, min(strength, 1.0))
        return float(int(budget // price))

    # -- fill -> books -----------------------------------------------------
    def on_fill(self, fill: FillEvent) -> None:
        pos = self.positions[fill.symbol]
        signed = fill.signed_quantity
        notional = signed * fill.fill_price

        # Cash out for buys, in for sells; costs always reduce cash.
        self.cash -= notional
        self.cash -= fill.commission
        self._total_commission += fill.commission
        self._total_slippage += fill.slippage

        new_qty = pos.quantity + signed
        if pos.quantity == 0 or (pos.quantity > 0) == (signed > 0):
            # opening or adding in the same direction -> blend avg price
            if new_qty != 0:
                pos.avg_price = (
                    pos.avg_price * pos.quantity + fill.fill_price * signed
                ) / new_qty
        elif new_qty == 0:
            pos.avg_price = 0.0
        # reducing/closing leaves avg_price unchanged for the remainder
        pos.quantity = new_qty

    # -- mark to market ----------------------------------------------------
    def on_market(self, event: MarketEvent) -> None:
        self.equity_curve.append(
            {"timestamp": event.timestamp, "equity": self.equity()}
        )

    def market_value(self) -> float:
        total = 0.0
        for symbol, pos in self.positions.items():
            if pos.quantity == 0:
                continue
            price = self.data.latest_price(symbol)
            if price is not None:
                total += pos.quantity * price
        return total

    def equity(self) -> float:
        return self.cash + self.market_value()

    def equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self.equity_curve).set_index("timestamp")
        return df["equity"]

    @property
    def total_commission(self) -> float:
        return self._total_commission

    @property
    def total_slippage(self) -> float:
        return self._total_slippage
