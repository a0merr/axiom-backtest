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

from abc import ABC, abstractmethod

from .data import DataHandler
from .event import Direction, FillEvent, OrderEvent


class SlippageModel(ABC):
    @abstractmethod
    def fill_price(self, reference_price: float, direction: Direction) -> float:
        ...


class PercentageSlippage(SlippageModel):
    """Fill moves against you by ``rate`` of the reference price."""

    def __init__(self, rate: float = 0.0005):
        if rate < 0:
            raise ValueError("slippage rate must be non-negative")
        self.rate = rate

    def fill_price(self, reference_price: float, direction: Direction) -> float:
        if direction == Direction.LONG:
            return reference_price * (1 + self.rate)
        if direction == Direction.SHORT:
            return reference_price * (1 - self.rate)
        return reference_price


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


class SimulatedExecutionHandler:
    """Fill market orders at the next available price plus frictions.

    The reference price is the latest close. In a daily backtest, orders
    generated on bar *t* should be assumed to fill on bar *t* (close) or *t+1*
    (open); we use the current close and let slippage absorb the optimism. For
    intraday work, swap in a handler that reads the next bar's open.
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

    def execute_order(self, order: OrderEvent) -> FillEvent | None:
        ref = self.data.latest_price(order.symbol)
        if ref is None:
            return None
        fill_price = self.slippage.fill_price(ref, order.direction)
        slippage_cost = abs(fill_price - ref) * order.quantity
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
