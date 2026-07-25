"""Portfolio: position sizing, order generation, and equity tracking.

The portfolio is the only component that knows about cash. It converts signals
into orders, applies fills to its books, and records a mark-to-market equity
point on every bar so the metrics module has a clean return series to work with.

Three constraints keep the books from reporting things that cannot happen:

* **Ruin.** An account whose equity reaches zero is finished. Without a check,
  the engine happily keeps trading a negative balance and every metric computed
  downstream becomes arithmetic on a fiction — returns flip sign across zero,
  and drawdowns exceed 100%.
* **Leverage.** ``risk_fraction`` allocates a share of equity *per symbol*, so
  N symbols can silently sum to N times the account. ``max_gross_leverage``
  bounds the total.
* **Margin.** Buying on credit is not free. Short borrow was already charged;
  a debit cash balance now is too, which removes an asymmetry that quietly
  subsidized long-side leverage.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import DataHandler
from .event import Direction, FillEvent, MarketEvent, OrderEvent, SignalEvent
from .sizing import FixedFractionalSizer, Sizer


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
    annual_borrow_rate: float = 0.0  # financing on short notional, e.g. 0.03 = 3%/yr
    annual_margin_rate: float = 0.0  # interest on a debit cash balance
    periods_per_year: int = 252
    sizer: Sizer | None = None  # defaults to FixedFractionalSizer(risk_fraction)
    # Cap on gross exposure (long + short notional) as a multiple of equity.
    # None means unlimited, which is the old behaviour and is rarely what a
    # multi-symbol strategy actually intends.
    max_gross_leverage: float | None = None
    # Equity at or below which the account is declared dead. Raise it to model
    # a hard stop ("shut me down at a 75% loss") rather than literal ruin.
    ruin_equity: float = 0.0

    cash: float = field(init=False)
    positions: dict[str, _Position] = field(init=False)
    # The equity curve is one point per bar, so its representation is a memory
    # decision, not a cosmetic one: a dict per bar costs ~184 bytes and a boxed
    # timestamp on top, which is gigabytes over a multi-million-bar run and is
    # then copied again to build the DataFrame. Two typed column arrays cost 16
    # bytes per bar and hand their buffers straight to pandas at the end.
    _curve_ns: array = field(init=False)
    _curve_eq: array = field(init=False)
    _total_commission: float = field(init=False, default=0.0)
    _total_slippage: float = field(init=False, default=0.0)
    _total_financing: float = field(init=False, default=0.0)
    _total_margin_interest: float = field(init=False, default=0.0)
    _ruined: bool = field(init=False, default=False)
    _ruin_timestamp: object = field(init=False, default=None)
    # Notional committed per symbol by orders emitted on this bar but not yet
    # filled. Deferred execution means several signals can be sized against the
    # same equity before any settles; without this they each pass the leverage
    # check individually and breach it collectively. Keyed by symbol so a fill
    # can retire its own commitment — once a position is real it is counted by
    # ``gross_exposure`` instead, and counting it twice would under-allocate.
    _pending: dict[str, float] = field(init=False)
    # Symbols currently holding a non-zero position. Marking to market happens
    # on every bar, so scanning the full universe there costs O(bars x symbols)
    # for a book that is usually mostly flat; these keep it O(bars x held).
    # Dicts, not sets: iteration order feeds a float summation, and set order
    # depends on string hash randomization, which would make equity curves
    # differ run to run. Insertion order is a function of the fills alone.
    _open: dict[str, None] = field(init=False)
    _short: dict[str, None] = field(init=False)

    def __post_init__(self) -> None:
        if self.max_gross_leverage is not None and self.max_gross_leverage <= 0:
            raise ValueError("max_gross_leverage must be positive or None")
        if self.annual_margin_rate < 0 or self.annual_borrow_rate < 0:
            raise ValueError("financing rates must be non-negative")
        self.cash = self.initial_capital
        self.positions = {s: _Position() for s in self.data.symbols}
        self._curve_ns = array("q")  # int64 epoch nanoseconds
        self._curve_eq = array("d")  # float64 equity
        self._open = {}
        self._short = {}
        self._pending = {}
        if self.sizer is None:
            self.sizer = FixedFractionalSizer(self.risk_fraction)

    # -- signal -> order ---------------------------------------------------
    def on_signal(self, signal: SignalEvent) -> list[OrderEvent]:
        if self._ruined:
            return []  # a closed account does not trade
        price = self.data.latest_price(signal.symbol)
        if price is None or price <= 0:
            return []
        pos = self.positions[signal.symbol]

        if signal.direction == Direction.EXIT:
            if pos.quantity == 0:
                return []
            direction = Direction.SHORT if pos.quantity > 0 else Direction.LONG
            return [
                OrderEvent(
                    signal.symbol,
                    signal.timestamp,
                    direction,
                    abs(pos.quantity),
                    reference_price=price,
                )
            ]

        opening = (
            signal.direction == Direction.LONG and pos.quantity <= 0
        ) or (signal.direction == Direction.SHORT and pos.quantity >= 0)
        if not opening:
            return []

        # Re-sizing this symbol replaces any commitment it already made.
        self._pending.pop(signal.symbol, None)

        # Reversals are one order: cover what is held, then open the new side.
        target = self._size(signal.symbol, price, signal.strength)
        target = self._cap_to_leverage(signal.symbol, price, target)
        qty = target + abs(pos.quantity)
        if qty <= 0:
            return []
        self._pending[signal.symbol] = target * price
        return [
            OrderEvent(
                signal.symbol,
                signal.timestamp,
                signal.direction,
                qty,
                reference_price=price,
            )
        ]

    def _size(self, symbol: str, price: float, strength: float) -> float:
        assert self.sizer is not None  # set in __post_init__
        # Universe order, not fill order, so sizing is deterministic.
        held = [s for s in self.positions if s in self._open]
        return self.sizer.size(symbol, price, strength, self.equity(), held)

    def _cap_to_leverage(self, symbol: str, price: float, target: float) -> float:
        """Shrink a target position so gross exposure stays within the cap.

        Exposure is measured after the trade: opening ``target`` in a symbol
        replaces whatever is already held there (the order covers it first), so
        only the difference counts against the limit. Orders already emitted on
        this bar are counted too, since they will settle before the next signal
        is sized.
        """
        if self.max_gross_leverage is None or target <= 0:
            return target
        equity = self.equity()
        if equity <= 0:
            return 0.0
        held = abs(self.positions[symbol].quantity) * price
        budget = (
            self.max_gross_leverage * equity
            - (self.gross_exposure() - held)
            - sum(self._pending.values())
        )
        if budget <= 0:
            return 0.0
        # Whole shares, matching what the sizers hand back.
        return min(target, float(int(budget // price)))

    # -- fill -> books -----------------------------------------------------
    def on_fill(self, fill: FillEvent) -> None:
        if self._ruined:
            # The account was closed out. Anything still in flight is cancelled
            # by the broker, not filled into a dead account.
            return
        # This symbol's exposure is about to become real, so its provisional
        # commitment retires; gross_exposure counts it from here on.
        self._pending.pop(fill.symbol, None)
        pos = self.positions[fill.symbol]
        signed = fill.signed_quantity
        notional = signed * fill.fill_price

        # Cash out for buys, in for sells; costs always reduce cash.
        self.cash -= notional
        self.cash -= fill.commission
        self._total_commission += fill.commission
        self._total_slippage += fill.slippage

        new_qty = pos.quantity + signed
        if new_qty == 0:
            # fully closed
            pos.avg_price = 0.0
        elif pos.quantity == 0 or (pos.quantity > 0) == (signed > 0):
            # opening or adding in the same direction -> blend avg price
            pos.avg_price = (
                pos.avg_price * pos.quantity + fill.fill_price * signed
            ) / new_qty
        elif (pos.quantity > 0) != (new_qty > 0):
            # crossed through zero (reversed side) -> new basis is the fill price
            pos.avg_price = fill.fill_price
        # else: reduced but same side -> keep existing avg_price
        pos.quantity = new_qty

        if new_qty == 0:
            self._open.pop(fill.symbol, None)
            self._short.pop(fill.symbol, None)
        else:
            self._open[fill.symbol] = None
            if new_qty < 0:
                self._short[fill.symbol] = None
            else:
                self._short.pop(fill.symbol, None)

    # -- mark to market ----------------------------------------------------
    def on_market(self, event: MarketEvent) -> None:
        self._pending.clear()  # unfilled orders lapse rather than tie up capital
        if not self._ruined:
            if self.sizer is not None:
                self.sizer.observe(self.data)
            self._accrue_borrow()
            self._accrue_margin_interest()
            if self.equity() <= self.ruin_equity:
                self._declare_ruin(event.timestamp)

        ts = event.timestamp
        # pd.Timestamp already carries its epoch-ns integer; anything else has
        # to be converted, which is ~20x slower per bar.
        ns = getattr(ts, "value", None)
        if ns is None:
            ns = pd.Timestamp(ts).value
        self._curve_ns.append(ns)
        self._curve_eq.append(self.equity())

    def _declare_ruin(self, timestamp) -> None:  # noqa: ANN001 - timestamp type varies
        """Close the account: liquidate everything, stop trading, stay flat.

        Liquidation is frictionless — positions are marked out at the current
        price with no exit slippage or commission. That is optimistic, and
        deliberately so: the alternative is inventing an unwind schedule that
        the engine has no way to validate. The number that matters here is the
        *date* the account died, not the last few basis points of it.
        """
        self.cash = self.equity()
        for pos in self.positions.values():
            pos.quantity = 0.0
            pos.avg_price = 0.0
        self._open.clear()
        self._short.clear()
        self._ruined = True
        self._ruin_timestamp = timestamp

    def _accrue_borrow(self) -> None:
        """Charge one period of financing on the notional of short positions."""
        if self.annual_borrow_rate <= 0 or not self._short:
            return
        rate = self.annual_borrow_rate / self.periods_per_year
        for symbol in self._short:
            price = self.data.latest_price(symbol)
            if price is None:
                continue
            cost = abs(self.positions[symbol].quantity) * price * rate
            self.cash -= cost
            self._total_financing += cost

    def _accrue_margin_interest(self) -> None:
        """Charge one period of interest on a debit cash balance.

        Shorting was already financed; buying on credit was not. Leaving that
        asymmetry in place makes long leverage look free relative to short
        leverage, which is backwards from every real brokerage statement.
        """
        if self.annual_margin_rate <= 0 or self.cash >= 0:
            return
        cost = -self.cash * self.annual_margin_rate / self.periods_per_year
        self.cash -= cost
        self._total_margin_interest += cost

    def market_value(self) -> float:
        if not self._open:
            return 0.0
        total = 0.0
        positions = self.positions
        latest_price = self.data.latest_price
        for symbol in self._open:
            price = latest_price(symbol)
            if price is not None:
                total += positions[symbol].quantity * price
        return total

    def gross_exposure(self) -> float:
        """Long + short notional, unsigned. The number leverage is measured on."""
        if not self._open:
            return 0.0
        total = 0.0
        for symbol in self._open:
            price = self.data.latest_price(symbol)
            if price is not None:
                total += abs(self.positions[symbol].quantity) * price
        return total

    def equity(self) -> float:
        return self.cash + self.market_value()

    def leverage(self) -> float:
        """Gross exposure as a multiple of equity. 0.0 for a dead account."""
        equity = self.equity()
        if equity <= 0:
            return 0.0
        return self.gross_exposure() / equity

    @property
    def is_ruined(self) -> bool:
        """True once equity reached ``ruin_equity`` and the account was closed."""
        return self._ruined

    @property
    def ruin_timestamp(self):  # noqa: ANN201 - mirrors the bar timestamp type
        """When the account died, or None if it survived."""
        return self._ruin_timestamp

    def equity_series(self) -> pd.Series:
        if not self._curve_eq:
            return pd.Series(dtype=float)
        index = pd.DatetimeIndex(
            np.array(self._curve_ns, dtype="int64").view("datetime64[ns]"),
            name="timestamp",
        )
        values = np.array(self._curve_eq, dtype=float)
        return pd.Series(values, index=index, name="equity")

    @property
    def equity_curve(self) -> list[dict]:
        """Mark-to-market points as records. Materialized on access.

        Kept for readability at the call site; the run itself never builds this
        list, so asking for it on a multi-million-bar backtest is expensive.
        Prefer :meth:`equity_series`.
        """
        return [
            {"timestamp": ts, "equity": eq}
            for ts, eq in zip(self.equity_series().index, self._curve_eq, strict=True)
        ]

    @property
    def total_commission(self) -> float:
        return self._total_commission

    @property
    def total_slippage(self) -> float:
        return self._total_slippage

    @property
    def total_financing(self) -> float:
        """Borrow cost on short notional only. Margin interest is reported separately."""
        return self._total_financing

    @property
    def total_margin_interest(self) -> float:
        return self._total_margin_interest
