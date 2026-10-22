"""Round-trip trade log: what each position actually earned, and for how long.

An equity curve says *whether* a strategy made money; it cannot say *how*. Two
strategies with the same Sharpe can be one that wins 70% of the time by a little
and one that loses most trades and lives off a few outliers — very different
things to run with real money, and indistinguishable from the curve alone.

A trade opens when a symbol leaves flat and closes when it returns to flat.
Fills in between scale the position in or out; a fill that crosses through zero
closes one trade and opens the next, with its commission and slippage split
between the two in proportion to quantity. Everything a trade costs is charged
to it — commission, short borrow — so the per-trade net P&L of every closed and
open trade sums to the account's change in equity, less only margin interest,
which is a property of the account's cash rather than of any one position.

``fill_price`` already includes slippage, so ``gross_pnl`` is net of it; the
``slippage`` column is attribution, not a second charge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .event import FillEvent


@dataclass
class Trade:
    symbol: str
    side: int  # +1 long, -1 short
    entry_time: object
    entry_bar: int
    quantity: float = 0.0  # currently held (unsigned)
    peak_quantity: float = 0.0
    bought: float = 0.0  # unsigned quantity added on the opening side
    entry_notional: float = 0.0  # sum of opening fill notionals
    closed: float = 0.0  # unsigned quantity taken off
    exit_notional: float = 0.0
    gross_pnl: float = 0.0  # realized, from fill prices
    commission: float = 0.0
    slippage: float = 0.0
    financing: float = 0.0
    exit_time: object = None
    exit_bar: int | None = None
    exit_reason: str | None = None

    @property
    def avg_entry(self) -> float:
        return self.entry_notional / self.bought if self.bought else 0.0

    @property
    def avg_exit(self) -> float:
        return self.exit_notional / self.closed if self.closed else float("nan")

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    def net_pnl(self, mark: float | None = None) -> float:
        """Realized net P&L, plus unrealized at ``mark`` if the trade is open."""
        pnl = self.gross_pnl - self.commission - self.financing
        if self.is_open and mark is not None:
            pnl += self.side * (mark - self.avg_entry) * self.quantity
        return pnl


@dataclass
class TradeLog:
    """Builds round trips from fills. Fed by :class:`~axiom.Portfolio`."""

    _open: dict[str, Trade] = field(default_factory=dict)
    _closed: list[Trade] = field(default_factory=list)

    def on_fill(self, fill: FillEvent, bar: int) -> None:
        signed = fill.signed_quantity
        if signed == 0:
            return
        qty = abs(signed)
        trade = self._open.get(fill.symbol)

        if trade is None or (signed > 0) == (trade.side > 0):
            if trade is None:
                trade = Trade(fill.symbol, 1 if signed > 0 else -1, fill.timestamp, bar)
                self._open[fill.symbol] = trade
            self._add(trade, qty, fill, share=1.0)
            return

        # Opposite side: reduce, close, or reverse.
        closing = min(qty, trade.quantity)
        share = closing / qty
        trade.quantity -= closing
        trade.closed += closing
        trade.exit_notional += closing * fill.fill_price
        trade.gross_pnl += trade.side * (fill.fill_price - trade.avg_entry) * closing
        trade.commission += fill.commission * share
        trade.slippage += fill.slippage * share
        if trade.quantity <= 1e-12:
            trade.quantity = 0.0
            self._close(trade, fill.timestamp, bar, "signal")

        remainder = qty - closing
        if remainder > 1e-12:
            fresh = Trade(fill.symbol, 1 if signed > 0 else -1, fill.timestamp, bar)
            self._open[fill.symbol] = fresh
            self._add(fresh, remainder, fill, share=remainder / qty)

    @staticmethod
    def _add(trade: Trade, qty: float, fill: FillEvent, share: float) -> None:
        trade.quantity += qty
        trade.bought += qty
        trade.entry_notional += qty * fill.fill_price
        trade.peak_quantity = max(trade.peak_quantity, trade.quantity)
        trade.commission += fill.commission * share
        trade.slippage += fill.slippage * share

    def _close(self, trade: Trade, timestamp, bar: int, reason: str) -> None:  # noqa: ANN001
        trade.exit_time = timestamp
        trade.exit_bar = bar
        trade.exit_reason = reason
        self._closed.append(trade)
        del self._open[trade.symbol]

    def charge_financing(self, symbol: str, cost: float) -> None:
        trade = self._open.get(symbol)
        if trade is not None:
            trade.financing += cost

    def liquidate(self, prices: dict[str, float], timestamp, bar: int) -> None:  # noqa: ANN001
        """Close every open trade at ``prices`` without costs (account ruin)."""
        for symbol, trade in list(self._open.items()):
            price = prices[symbol]
            trade.closed += trade.quantity
            trade.exit_notional += trade.quantity * price
            trade.gross_pnl += trade.side * (price - trade.avg_entry) * trade.quantity
            trade.quantity = 0.0
            self._close(trade, timestamp, bar, "ruin")

    @property
    def closed_trades(self) -> list[Trade]:
        return list(self._closed)

    @property
    def open_trades(self) -> list[Trade]:
        return list(self._open.values())

    def to_frame(self, marks: dict[str, float] | None = None) -> pd.DataFrame:
        """One row per round trip, closed first, then open trades marked at ``marks``."""
        marks = marks or {}
        rows = []
        for trade in [*self._closed, *self._open.values()]:
            mark = marks.get(trade.symbol)
            entry_value = trade.avg_entry * trade.peak_quantity
            net = trade.net_pnl(mark)
            rows.append(
                {
                    "symbol": trade.symbol,
                    "side": "long" if trade.side > 0 else "short",
                    "entry_time": trade.entry_time,
                    "exit_time": trade.exit_time,
                    "bars_held": (trade.exit_bar if trade.exit_bar is not None else np.nan)
                    - trade.entry_bar,
                    "peak_quantity": trade.peak_quantity,
                    "avg_entry": trade.avg_entry,
                    "avg_exit": trade.avg_exit if not trade.is_open else mark,
                    "gross_pnl": net + trade.commission + trade.financing,
                    "commission": trade.commission,
                    "slippage": trade.slippage,
                    "financing": trade.financing,
                    "net_pnl": net,
                    "return_pct": net / entry_value if entry_value else np.nan,
                    "is_open": trade.is_open,
                    "exit_reason": trade.exit_reason or "open",
                }
            )
        columns = [
            "symbol", "side", "entry_time", "exit_time", "bars_held", "peak_quantity",
            "avg_entry", "avg_exit", "gross_pnl", "commission", "slippage", "financing",
            "net_pnl", "return_pct", "is_open", "exit_reason",
        ]
        return pd.DataFrame(rows, columns=columns)


@dataclass(frozen=True)
class TradeSummary:
    trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    payoff_ratio: float  # avg win / |avg loss|
    profit_factor: float  # gross wins / |gross losses|
    expectancy: float  # mean net P&L per trade
    avg_bars_held: float
    largest_win: float
    largest_loss: float
    max_consecutive_losses: int

    def __str__(self) -> str:
        return (
            f"trades            {self.trades}\n"
            f"win rate          {self.win_rate:.1%}\n"
            f"avg win / loss    {self.avg_win:,.2f} / {self.avg_loss:,.2f}"
            f"  (payoff {self.payoff_ratio:.2f})\n"
            f"profit factor     {self.profit_factor:.2f}\n"
            f"expectancy        {self.expectancy:,.2f} per trade\n"
            f"avg bars held     {self.avg_bars_held:.1f}\n"
            f"largest win/loss  {self.largest_win:,.2f} / {self.largest_loss:,.2f}\n"
            f"max losing streak {self.max_consecutive_losses}"
        )


def trade_summary(trades: pd.DataFrame) -> TradeSummary:
    """Per-trade statistics over **closed** trades only.

    Open trades are excluded: an unrealized mark is not an outcome, and counting
    it would let a strategy improve its win rate by never exiting losers.
    """
    closed = trades[~trades["is_open"]] if len(trades) else trades
    pnl = closed["net_pnl"].to_numpy(dtype=float) if len(closed) else np.array([])
    if pnl.size == 0:
        nan = float("nan")
        return TradeSummary(0, nan, nan, nan, nan, nan, nan, nan, nan, nan, 0)

    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    avg_win = wins.mean() if wins.size else 0.0
    avg_loss = losses.mean() if losses.size else 0.0

    streak = longest = 0
    for value in pnl:
        streak = streak + 1 if value <= 0 else 0
        longest = max(longest, streak)

    return TradeSummary(
        trades=int(pnl.size),
        win_rate=float(wins.size / pnl.size),
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
        payoff_ratio=float(avg_win / -avg_loss) if avg_loss < 0 else float("inf"),
        profit_factor=float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        expectancy=float(pnl.mean()),
        avg_bars_held=float(closed["bars_held"].mean()),
        largest_win=float(pnl.max()),
        largest_loss=float(pnl.min()),
        max_consecutive_losses=int(longest),
    )
