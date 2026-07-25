"""Execution handlers: turn orders into fills with realistic frictions.

Two costs are modeled explicitly and separately:

* **Slippage** — the gap between the decision price and the actual fill. Modeled
  here as a fraction of price scaled by direction (you buy worse, sell worse).
  A volume/volatility-aware model can subclass ``SlippageModel``.
* **Commission** — explicit transaction cost, per-share plus a floor, the shape
  most retail/prime broker schedules take.

Ignoring these is the most common way a backtest lies: a strategy that trades
often can look brilliant gross and be ruinous net.

The slippage models here form a deliberate ladder of honesty, cheapest
assumption first:

===========================  ==================================================
``PercentageSlippage``       Flat fraction. Size-blind, regime-blind.
``VolatilitySlippage``       Scales with a volatility column. Regime-aware.
``VolumeShareSlippage``      Square-root impact in participation. Size-aware.
``OrderBookSlippage``        Finite depth. Size- and regime-aware, and convex
                             past the depth the book actually holds.
===========================  ==================================================

Only the last one has a notion of running *out* of liquidity, which is the
failure the first three are structurally unable to show you. Pair it with
``ExecutionHandler``'s ``max_participation`` cap: one makes size expensive, the
other makes size impossible, and real markets do both at once.
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
        if bar is not None and self.volume_field in bar:
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
        if bar is not None and self.vol_field in bar:
            frac += self.k * float(bar[self.vol_field])
        return _apply(reference_price, direction, frac)


class OrderBookSlippage(SlippageModel):
    """Depth-aware slippage: sweep a synthetic limit order book.

    The models above all share an assumption that is comfortable and wrong —
    that the price you get degrades smoothly and boundedly with size. Real fills
    come from a book of finite depth. Inside that depth, cost grows steadily;
    past it, you are lifting a much thinner tail and the cost per share turns
    sharply convex. That elbow is where a strategy that backtests beautifully at
    scale actually dies, and a flat or square-root model has no elbow at all.

    The book is modeled as liquidity of uniform density resting across a price
    range. Let ``D = depth_share * bar_volume`` shares sit uniformly across
    ``depth_range`` (as a fraction of price), and let ``u = |quantity| / D`` be
    how much of that book the order consumes. Sweeping ``u`` of a uniform book
    walks the price linearly, so the *average* concession is::

        u <= 1:  depth_range * u / 2
        u >  1:  depth_range * (1/(2u) + (u-1)/u + tail_factor*(u-1)^2/(2u))

    The second branch is the first branch's book fully consumed, plus a tail
    whose density is ``tail_factor`` times thinner. The two agree at ``u = 1``
    (both give ``depth_range / 2``), so cost is continuous in size, and beyond
    the book it grows quadratically rather than as a square root.

    **Liquidity collapse.** Depth and spread are not constants — they evaporate
    together exactly when you most want them. If the bar carries a volatility
    field and ``reference_vol`` is set, a bar whose volatility is ``r`` times
    normal widens the spread by ``1 + spread_vol_coef * (r - 1)`` and thins the
    book by the same kind of factor. The order then eats a larger share of a
    smaller book at a wider spread, so the modeled cost rises faster than
    volatility does. This is the regime that flat slippage silently underwrites.

    ``max_frac`` caps the total concession. It is a guard rail, not physics: an
    uncapped sweep can exceed 100% of the price and mint negative fills on the
    short side, and a backtest should refuse to print a number that absurd.
    """

    def __init__(
        self,
        half_spread: float = 0.0001,
        depth_share: float = 0.05,
        depth_range: float = 0.002,
        tail_factor: float = 5.0,
        volume_field: str = "volume",
        vol_field: str = "volatility",
        reference_vol: float = 0.0,
        spread_vol_coef: float = 1.0,
        depth_vol_coef: float = 1.0,
        fallback_rate: float = 0.001,
        max_frac: float = 0.5,
    ):
        if half_spread < 0 or depth_range < 0 or fallback_rate < 0:
            raise ValueError("half_spread, depth_range, fallback_rate must be >= 0")
        if depth_share <= 0:
            raise ValueError("depth_share must be positive")
        if tail_factor < 0 or spread_vol_coef < 0 or depth_vol_coef < 0:
            raise ValueError("tail_factor and vol coefficients must be >= 0")
        if reference_vol < 0:
            raise ValueError("reference_vol must be >= 0")
        if max_frac <= 0:
            raise ValueError("max_frac must be positive")
        self.half_spread = half_spread
        self.depth_share = depth_share
        self.depth_range = depth_range
        self.tail_factor = tail_factor
        self.volume_field = volume_field
        self.vol_field = vol_field
        self.reference_vol = reference_vol
        self.spread_vol_coef = spread_vol_coef
        self.depth_vol_coef = depth_vol_coef
        self.fallback_rate = fallback_rate
        self.max_frac = max_frac

    def _stress(self, bar: pd.Series | None) -> float:
        """How much worse than normal this bar's liquidity is. 0.0 = normal."""
        if self.reference_vol <= 0 or bar is None or self.vol_field not in bar:
            return 0.0
        vol = float(bar[self.vol_field])
        if vol <= 0:
            return 0.0
        return max(0.0, vol / self.reference_vol - 1.0)

    def sweep_cost(self, utilization: float) -> float:
        """Average price concession, as a fraction, for consuming ``u`` of the book."""
        u = utilization
        if u <= 0:
            return 0.0
        if u <= 1.0:
            return self.depth_range * u / 2.0
        excess = u - 1.0
        return self.depth_range * (
            1.0 / (2.0 * u)
            + excess / u
            + self.tail_factor * excess * excess / (2.0 * u)
        )

    def fill_price(
        self,
        reference_price: float,
        direction: Direction,
        quantity: float = 0.0,
        bar: pd.Series | None = None,
    ) -> float:
        stress = self._stress(bar)
        half_spread = self.half_spread * (1.0 + self.spread_vol_coef * stress)

        volume = 0.0
        if bar is not None and self.volume_field in bar:
            volume = float(bar[self.volume_field])
        if volume <= 0:
            # No volume to size the book against; fall back to a flat charge
            # rather than pretending the book is infinitely deep.
            return _apply(reference_price, direction, half_spread + self.fallback_rate)

        depth_share = self.depth_share / (1.0 + self.depth_vol_coef * stress)
        depth = depth_share * volume
        frac = half_spread + self.sweep_cost(abs(quantity) / depth)
        return _apply(reference_price, direction, min(frac, self.max_frac))


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

    ``max_participation`` caps the fraction of a bar's volume a single order may
    take. Paying more for size is only half of what illiquidity does to you; the
    other half is not getting filled at all. Without a cap, a backtest will
    happily execute 10x a bar's entire traded volume at *some* price and count
    the P&L, which is the most flattering assumption in the whole simulation.
    With one, an order that exceeds the cap fills partially — and it is up to
    the concrete handler to decide what happens to the remainder. Unfilled
    quantity is accumulated in :attr:`unfilled_quantity` rather than discarded
    quietly, so a strategy that is too big for its market says so out loud.
    :attr:`partial_fills` counts the occasions a fill came up short — bar-by-bar
    events, not distinct orders, since one order may be short on several bars.

    ``max_price_drift`` bounds how far the market may move between the price an
    order was sized at (``OrderEvent.reference_price``) and the price it would
    fill at. Position sizes are computed from a price that is, by necessity, not
    the fill price — under next-bar execution the fill price does not exist yet,
    and using it would be look-ahead. Normally that gap is noise. Across a gap
    open it is not: a position sized for 10% of equity can land at 15%, and the
    backtest reports a risk profile the strategy never chose. Setting a drift
    bound turns those into cancellations (counted in :attr:`cancelled_on_drift`)
    the way a marketable limit order would.
    """

    def __init__(
        self,
        data: DataHandler,
        slippage: SlippageModel | None = None,
        commission: CommissionModel | None = None,
        max_participation: float | None = None,
        volume_field: str = "volume",
        max_price_drift: float | None = None,
    ):
        if max_participation is not None and max_participation <= 0:
            raise ValueError("max_participation must be positive or None")
        if max_price_drift is not None and max_price_drift < 0:
            raise ValueError("max_price_drift must be non-negative or None")
        self.data = data
        self.slippage = slippage or PercentageSlippage()
        self.commission_model = commission or PerShareCommission()
        self.max_participation = max_participation
        self.volume_field = volume_field
        self.max_price_drift = max_price_drift
        self.unfilled_quantity: float = 0.0
        self.partial_fills: int = 0
        self.cancelled_on_drift: int = 0

    def _drifted(self, order: OrderEvent, fill_reference: float) -> bool:
        """True if the market has run too far from where the order was sized.

        Only adverse moves count. Getting a better price than you sized at is
        not a reason to cancel, and treating it as one would bias the backtest
        toward keeping only its bad fills.
        """
        if self.max_price_drift is None or order.reference_price is None:
            return False
        if order.reference_price <= 0:
            return False
        move = fill_reference / order.reference_price - 1.0
        if order.direction == Direction.SHORT:
            move = -move  # a seller is hurt by the price falling
        return move > self.max_price_drift

    @abstractmethod
    def execute_order(self, order: OrderEvent) -> FillEvent | None:
        ...

    def on_bar(self, event) -> list[FillEvent]:  # noqa: ANN001 - MarketEvent
        return []

    def _fillable(self, order: OrderEvent, bar) -> float:  # noqa: ANN001 - Bar
        """How much of ``order`` this bar can absorb under the participation cap."""
        if self.max_participation is None or bar is None:
            return order.quantity
        if self.volume_field not in bar:
            return order.quantity
        volume = float(bar[self.volume_field])
        if volume <= 0:
            return 0.0  # a bar that traded nothing cannot fill anything
        return min(order.quantity, self.max_participation * volume)

    def _build_fill(
        self, order: OrderEvent, reference_price: float, quantity: float | None = None
    ) -> FillEvent:
        bar = self.data.latest_bar(order.symbol)
        qty = order.quantity if quantity is None else quantity
        fill_price = self.slippage.fill_price(
            reference_price, order.direction, qty, bar
        )
        slippage_cost = abs(fill_price - reference_price) * qty
        commission = self.commission_model.commission(qty, fill_price)
        return FillEvent(
            symbol=order.symbol,
            timestamp=order.timestamp,
            direction=order.direction,
            quantity=qty,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage_cost,
        )

    def _reference_price(self, symbol: str, field: str) -> float | None:
        """Latest value of ``field``, falling back to close if absent."""
        bar = self.data.latest_bar(symbol)
        if bar is None:
            return None
        if field in bar:
            return float(bar[field])
        return self.data.latest_price(symbol)  # close fallback


class SimulatedExecutionHandler(ExecutionHandler):
    """Fill at the current bar's close, plus frictions.

    Simple and conservative-ish once slippage is applied, but optimistic: a
    signal computed from a bar's close cannot in reality be traded at that same
    close. Use ``NextBarExecutionHandler`` to remove that look-ahead.

    Under a participation cap this handler fills what the bar can absorb and
    drops the rest: it has no queue to work an order through, so there is
    nowhere for a remainder to go. The dropped quantity lands in
    ``unfilled_quantity``.
    """

    def execute_order(self, order: OrderEvent) -> FillEvent | None:
        ref = self.data.latest_price(order.symbol)
        if ref is None:
            return None
        if self._drifted(order, ref):
            self.cancelled_on_drift += 1
            self.unfilled_quantity += order.quantity
            return None
        qty = self._fillable(order, self.data.latest_bar(order.symbol))
        if qty < order.quantity:
            self.unfilled_quantity += order.quantity - qty
            self.partial_fills += 1
        if qty <= 0:
            return None
        return self._build_fill(order, ref, qty)


class NextBarExecutionHandler(ExecutionHandler):
    """Fill at the *next* bar's open — the realistic, no-look-ahead model.

    Orders are queued when received and filled on the following bar using its
    open price (falling back to close if the data has no ``open`` column). The
    consequence is honest: a signal on bar *t* cannot be acted on until bar
    *t+1*, and an order generated on the final bar never fills (there is no next
    bar to trade into).

    Because it already keeps a queue, this handler can *work* an order that is
    too large for one bar: under a participation cap it fills what the bar
    absorbs and carries the remainder forward, which is what a real execution
    desk does. ``max_working_bars`` bounds that patience — an order still
    unfilled after that many bars is cancelled and its remainder recorded in
    ``unfilled_quantity``. Without a bound, a position too big for its market
    would keep leaking into the book forever and the backtest would report a
    round trip that never actually happened.

    Note that a carried remainder fills at *later* bars' prices, so working a
    large order picks up genuine timing risk on top of impact — which is the
    point.
    """

    def __init__(
        self,
        data: DataHandler,
        slippage: SlippageModel | None = None,
        commission: CommissionModel | None = None,
        price_field: str = "open",
        max_participation: float | None = None,
        volume_field: str = "volume",
        max_working_bars: int = 5,
        max_price_drift: float | None = None,
    ):
        super().__init__(
            data,
            slippage,
            commission,
            max_participation,
            volume_field,
            max_price_drift,
        )
        if max_working_bars < 1:
            raise ValueError("max_working_bars must be >= 1")
        self.price_field = price_field
        self.max_working_bars = max_working_bars
        # (order, bars this order has already been worked for)
        self._pending: list[tuple[OrderEvent, int]] = []

    def execute_order(self, order: OrderEvent) -> FillEvent | None:
        self._pending.append((order, 0))
        return None  # filled later, in on_bar

    def on_bar(self, event) -> list[FillEvent]:  # noqa: ANN001 - MarketEvent
        if not self._pending:
            return []
        fills: list[FillEvent] = []
        carried: list[tuple[OrderEvent, int]] = []
        for order, worked in self._pending:
            ref = self._reference_price(order.symbol, self.price_field)
            if ref is None:
                # No price for this symbol yet, so the order lapses. Book it as
                # unfilled: an order that evaporates is still an order that
                # didn't execute, and silence here would overstate the strategy.
                self.unfilled_quantity += order.quantity
                self.partial_fills += 1
                continue
            if self._drifted(order, ref):
                self.cancelled_on_drift += 1
                self.unfilled_quantity += order.quantity
                continue
            qty = self._fillable(order, self.data.latest_bar(order.symbol))
            if qty > 0:
                fills.append(self._build_fill(order, ref, qty))
            remainder = order.quantity - qty
            if remainder <= 0:
                continue
            self.partial_fills += 1
            if worked + 1 >= self.max_working_bars:
                self.unfilled_quantity += remainder  # cancelled, not filled
                continue
            carried.append(
                (
                    OrderEvent(
                        order.symbol,
                        order.timestamp,
                        order.direction,
                        remainder,
                        order.order_type,
                        # The remainder keeps the *original* sizing price, so
                        # drift accumulates over the bars it is worked across
                        # rather than resetting each time it is re-queued.
                        order.reference_price,
                    ),
                    worked + 1,
                )
            )
        self._pending = carried
        return fills
