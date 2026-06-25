"""Event types that flow through the backtest queue.

The engine is event-driven: a data handler emits ``MarketEvent``s, strategies
turn those into ``SignalEvent``s, the portfolio sizes them into ``OrderEvent``s,
and the execution handler returns ``FillEvent``s. Keeping these as small, typed
objects (rather than passing tuples around) makes the data flow explicit and the
components independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class EventType(str, Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"


class Event:
    """Base class. ``type`` is set by each subclass."""

    type: EventType


@dataclass
class MarketEvent(Event):
    """A new bar of market data is available for ``timestamp``."""

    timestamp: datetime
    type: EventType = EventType.MARKET


@dataclass
class SignalEvent(Event):
    """A strategy's view on an instrument at a point in time.

    ``strength`` is an optional [0, 1] conviction used by the portfolio for
    position sizing. A plain long/flat strategy can leave it at 1.0.
    """

    symbol: str
    timestamp: datetime
    direction: Direction
    strength: float = 1.0
    type: EventType = EventType.SIGNAL


@dataclass
class OrderEvent(Event):
    """An order the portfolio wants executed."""

    symbol: str
    timestamp: datetime
    direction: Direction
    quantity: float  # always positive; ``direction`` carries the sign
    order_type: str = "MKT"
    type: EventType = EventType.ORDER

    def __post_init__(self) -> None:
        if self.quantity < 0:
            raise ValueError("quantity must be non-negative; use direction for side")


@dataclass
class FillEvent(Event):
    """The result of executing an order, including realized costs.

    ``fill_price`` already incorporates slippage. ``commission`` is the explicit
    transaction cost. Both are surfaced separately so cost attribution is
    auditable rather than baked invisibly into P&L.
    """

    symbol: str
    timestamp: datetime
    direction: Direction
    quantity: float
    fill_price: float
    commission: float
    slippage: float
    type: EventType = EventType.FILL

    @property
    def signed_quantity(self) -> float:
        if self.direction == Direction.SHORT:
            return -self.quantity
        return self.quantity
