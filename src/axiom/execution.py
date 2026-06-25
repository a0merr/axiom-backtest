"""Execution handlers: turn orders into fills with realistic frictions.

Two costs are modeled explicitly and separately:

* **Slippage** — the gap between the decision price and the actual fill. Modeled
  here as a fraction of price scaled by direction (you buy worse, sell worse).
  A volume/volatility-aware model can subclass ``SlippageModel``.
* **Commission** — explicit transaction cost, per-share plus a floor, the shape
  most retail/prime broker schedules take.

Ignoring these is the most common way a backtest lies: a strategy that trades
often can look brilliant gross and be ruinous net.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .data import DataHandler
from .event import Direction, FillEvent, OrderEvent

if TYPE_CHECKING:
    import pandas as pd


def _apply(reference_price: float, direction: Direction, frac: float) -> float:
    """Move price against the trader by ``frac`` of the reference price."""
    if direction == Direction.LONG:
        return reference_price * (1 + frac)
    if direction == Direction.SHORT:
        return reference_price * (1 - frac)
    return reference_price


class SlippageModel(ABC):
    @abstractmethod
    def fill_price(
        self,
        reference_price: float,
        direction: Direction,
        quantity: float = 0.0,
        bar: pd.Series | None = None,
    ) -> float:
        """Return the realized fill price.

        ``quantity`` and ``bar`` (the current OHLCV row, may carry ``volume``)
        are passed so impact models can scale slippage with order size and
        market conditions. Simple models ignore them.
        """


class PercentageSlippage(SlippageModel):
    """Fill moves against you by a fixed ``rate`` of the reference price."""

    def __init__(self, rate: float = 0.0005):
        if rate < 0:
            raise ValueError("slippage rate must be non-negative")
        self.rate = rate

    def fill_price(
        self,
        reference_price: float,
        direction: Direction,
        quantity: float = 0.0,
        bar: pd.Series | None = None,
    ) -> float:
        return _apply(reference_price, direction, self.rate)


class VolumeShareSlippage(SlippageModel):
    """Square-root market-impact model (Almgren-style).

    Slippage fraction = ``base_rate`` + ``impact_coef * sqrt(participation)``,
    where ``participation = |quantity| / bar_volume``. Bigger orders relative to
    traded volume pay disproportionately more — the realistic penalty that flat
    per-trade slippage misses and that flatters high-turnover strategies. Falls
    back to ``base_rate`` when the bar has no volume.
    """

    def __init__(
        self,
        base_rate: float = 0.0002,
        impact_coef: float = 0.1,
        volume_field: str = "volume",
    ):
        if base_rate < 0 or impact_coef < 0:
            raise ValueError("base_rate and impact_coef must be non-negative")
        self.base_rate = base_rate
        self.impact_coef = impact_coef
        self.volume_field = volume_field

    def fill_price(
        self,
        reference_price: float,
        direction: Direction,
        quantity: float = 0.0,
        bar: pd.Series | None = None,
    ) -> float:
        frac = self.base_rate
        if bar is not None and self.volume_field in bar.index:
            volume = float(bar[self.volume_field])
            if volume > 0:
                participation = abs(quantity) / volume
                frac += self.impact_coef * math.sqrt(participation)
        return _apply(reference_price, direction, frac)


class VolatilitySlippage(SlippageModel):
    """Slippage that scales with a precomputed per-bar volatility column.

    Fraction = ``floor + k * volatility``, where ``volatility`` is read from the
    bar (e.g. an ATR-as-fraction-of-price feature you attach to the data). Wider
    spreads in turbulent regimes cost more. Falls back to ``floor`` if absent.
    """

    def __init__(
        self,
        k: float = 0.5,
        floor: float = 0.0001,
        vol_field: str = "volatility",
    ):
        if k < 0 or floor < 0:
            raise ValueError("k and floor must be non-negative")
        self.k = k
        self.floor = floor
        self.vol_field = vol_field

    def fill_price(
        self,
        reference_price: float,
        direction: Direction,
        quantity: float = 0.0,
        bar: pd.Series | None = None,
    ) -> float:
        frac = self.floor
        if bar is not None and self.vol_field in bar.index:
            frac += self.k * float(bar[self.vol_field])
        return _apply(reference_price, direction, frac)


class CommissionModel(ABC):
    @abstractmethod
    def commission(self, quantity: float, fill_price: float) -> float:
        ...


class PerShareCommission(CommissionModel):
    """Per-share cost with a per-order minimum (e.g. IBKR-style)."""

    def __init__(self, per_share: float = 0.005, minimum: float = 1.0):
        self.per_share = per_share
        self.minimum = minimum

    def commission(self, quantity: float, fill_price: float) -> float:
        return max(self.minimum, abs(quantity) * self.per_share)


class ExecutionHandler(ABC):
    """Turn orders into fills.

    Two timing hooks let the engine support both same-bar and next-bar fills
    without knowing which model is in use:

    * ``execute_order`` is called when an ORDER event is processed.
    * ``on_bar`` is called at the start of every bar, before strategy logic, so
      a deferred handler can fill orders queued on the previous bar using the
      new bar's price.
    """

    def __init__(
        self,
        data: DataHandler,
        slippage: SlippageModel | None = None,
        commission: CommissionModel | None = None,
    ):
        self.data = data
        self.slippage = slippage or PercentageSlippage()
        self.commission_model = commission or PerShareCommission()

    @abstractmethod
    def execute_order(self, order: OrderEvent) -> FillEvent | None:
        ...

    def on_bar(self, event) -> list[FillEvent]:  # noqa: ANN001 - MarketEvent
        return []

    def _build_fill(
        self, order: OrderEvent, reference_price: float
    ) -> FillEvent:
        bar = self.data.latest_bar(order.symbol)
        fill_price = self.slippage.fill_price(
            reference_price, order.direction, order.quantity, bar
        )
        slippage_cost = abs(fill_price - reference_price) * order.quantity
        commission = self.commission_model.commission(order.quantity, fill_price)
        return FillEvent(
            symbol=order.symbol,
            timestamp=order.timestamp,
            direction=order.direction,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage_cost,
        )

    def _reference_price(self, symbol: str, field: str) -> float | None:
        """Latest value of ``field``, falling back to close if absent."""
        bar = self.data.latest_bar(symbol)
        if bar is None:
            return None
        if field in bar.index:
            return float(bar[field])
        return self.data.latest_price(symbol)  # close fallback


class SimulatedExecutionHandler(ExecutionHandler):
    """Fill at the current bar's close, plus frictions.

    Simple and conservative-ish once slippage is applied, but optimistic: a
    signal computed from a bar's close cannot in reality be traded at that same
    close. Use ``NextBarExecutionHandler`` to remove that look-ahead.
    """

    def execute_order(self, order: OrderEvent) -> FillEvent | None:
        ref = self.data.latest_price(order.symbol)
        if ref is None:
            return None
        return self._build_fill(order, ref)


class NextBarExecutionHandler(ExecutionHandler):
    """Fill at the *next* bar's open — the realistic, no-look-ahead model.

    Orders are queued when received and filled on the following bar using its
    open price (falling back to close if the data has no ``open`` column). The
    consequence is honest: a signal on bar *t* cannot be acted on until bar
    *t+1*, and an order generated on the final bar never fills (there is no next
    bar to trade into).
    """

    def __init__(
        self,
        data: DataHandler,
        slippage: SlippageModel | None = None,
        commission: CommissionModel | None = None,
        price_field: str = "open",
    ):
        super().__init__(data, slippage, commission)
        self.price_field = price_field
        self._pending: list[OrderEvent] = []

    def execute_order(self, order: OrderEvent) -> FillEvent | None:
        self._pending.append(order)
        return None  # filled later, in on_bar

    def on_bar(self, event) -> list[FillEvent]:  # noqa: ANN001 - MarketEvent
        if not self._pending:
            return []
        fills: list[FillEvent] = []
        for order in self._pending:
            ref = self._reference_price(order.symbol, self.price_field)
            if ref is not None:
                fills.append(self._build_fill(order, ref))
        self._pending.clear()
        return fills
