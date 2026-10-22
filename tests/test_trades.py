"""Round-trip trade log.

The headline invariant: every dollar the account gained or lost is attributed
to some trade (or to margin interest), so per-trade net P&L reconciles to the
equity curve exactly. Unit tests pin the awkward cases — scaling, partial
exits, reversals — where a naive log double-counts or drops costs.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from axiom import (
    BacktestEngine,
    Direction,
    FillEvent,
    HistoricCSVDataHandler,
    NextBarExecutionHandler,
    PercentageSlippage,
    PerShareCommission,
    Portfolio,
    SignalEvent,
    Strategy,
    TradeLog,
    trade_summary,
)

T0 = datetime(2020, 1, 1)


def _fill(direction, qty, price, commission=0.0, slippage=0.0, day=0, symbol="AAA"):
    ts = T0 + timedelta(days=day)
    return FillEvent(symbol, ts, direction, qty, price, commission=commission, slippage=slippage)


def test_simple_long_round_trip():
    log = TradeLog()
    log.on_fill(_fill(Direction.LONG, 10, 100.0, commission=1.0), bar=0)
    log.on_fill(_fill(Direction.SHORT, 10, 110.0, commission=1.0, day=5), bar=5)
    row = log.to_frame().iloc[0]
    assert row["side"] == "long" and not row["is_open"]
    assert row["gross_pnl"] == pytest.approx(100.0)
    assert row["net_pnl"] == pytest.approx(98.0)
    assert row["bars_held"] == 5
    assert row["return_pct"] == pytest.approx(98.0 / 1000.0)


def test_scale_in_and_partial_exits_are_one_trade():
    log = TradeLog()
    log.on_fill(_fill(Direction.LONG, 10, 100.0), bar=0)
    log.on_fill(_fill(Direction.LONG, 10, 120.0), bar=1)  # avg 110
    log.on_fill(_fill(Direction.SHORT, 5, 130.0), bar=2)  # +100
    assert len(log.closed_trades) == 0
    log.on_fill(_fill(Direction.SHORT, 15, 100.0), bar=3)  # -150
    frame = log.to_frame()
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["avg_entry"] == pytest.approx(110.0)
    assert row["avg_exit"] == pytest.approx((5 * 130 + 15 * 100) / 20)
    assert row["peak_quantity"] == 20
    assert row["net_pnl"] == pytest.approx(-50.0)


def test_reversal_splits_one_fill_into_two_trades():
    log = TradeLog()
    log.on_fill(_fill(Direction.LONG, 10, 100.0), bar=0)
    # Sell 30: 10 close the long, 20 open a short. Costs split 1:2.
    log.on_fill(_fill(Direction.SHORT, 30, 90.0, commission=3.0, slippage=6.0), bar=4)
    closed, = log.closed_trades
    short, = log.open_trades
    assert closed.side == 1 and closed.gross_pnl == pytest.approx(-100.0)
    assert closed.commission == pytest.approx(1.0) and closed.slippage == pytest.approx(2.0)
    assert short.side == -1 and short.quantity == 20 and short.avg_entry == 90.0
    assert short.commission == pytest.approx(2.0)
    assert short.net_pnl(mark=80.0) == pytest.approx(200.0 - 2.0)


def test_open_trade_is_marked_not_summarized():
    log = TradeLog()
    log.on_fill(_fill(Direction.LONG, 10, 100.0), bar=0)
    frame = log.to_frame({"AAA": 105.0})
    assert frame.iloc[0]["is_open"] and frame.iloc[0]["net_pnl"] == pytest.approx(50.0)
    assert trade_summary(frame).trades == 0


def test_trade_summary_statistics():
    frame = pd.DataFrame(
        {
            "net_pnl": [100.0, -50.0, -25.0, 200.0, -10.0],
            "bars_held": [1, 2, 3, 4, 5],
            "is_open": [False] * 5,
        }
    )
    s = trade_summary(frame)
    assert s.trades == 5
    assert s.win_rate == pytest.approx(0.4)
    assert s.avg_win == pytest.approx(150.0)
    assert s.avg_loss == pytest.approx(-85.0 / 3)
    assert s.profit_factor == pytest.approx(300.0 / 85.0)
    assert s.expectancy == pytest.approx(43.0)
    assert s.max_consecutive_losses == 2
    assert s.avg_bars_held == 3


# -- Integration ----------------------------------------------------------


class _RandomFlipper(Strategy):
    """Long, short, flat at random — exercises opens, exits and reversals."""

    def __init__(self, data, seed=0):
        self.data = data
        self.rng = np.random.default_rng(seed)

    def on_market(self, event):
        out = []
        for symbol in self.data.symbols:
            roll = self.rng.random()
            if roll < 0.08:
                out.append(SignalEvent(symbol, event.timestamp, Direction.LONG))
            elif roll < 0.16:
                out.append(SignalEvent(symbol, event.timestamp, Direction.SHORT))
            elif roll < 0.22:
                out.append(SignalEvent(symbol, event.timestamp, Direction.EXIT))
        return out


def _frames(n=600, symbols=("AAA", "BBB", "CCC"), seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    frames = {}
    for s in symbols:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
        frames[s] = pd.DataFrame(
            {"open": close * (1 + rng.normal(0, 0.002, n)), "close": close}, index=idx
        )
    return frames


def _run(**portfolio_kwargs):
    data = HistoricCSVDataHandler(_frames())
    portfolio = Portfolio(data, initial_capital=100_000.0, risk_fraction=0.4, **portfolio_kwargs)
    execution = NextBarExecutionHandler(
        data, slippage=PercentageSlippage(0.001), commission=PerShareCommission(0.01, 1.0)
    )
    BacktestEngine(data, _RandomFlipper(data), portfolio, execution).run()
    return portfolio


def test_trade_pnl_reconciles_to_equity():
    pf = _run(annual_borrow_rate=0.05, annual_margin_rate=0.08)
    trades = pf.trades()
    assert (trades["side"] == "short").any() and len(trades) > 20
    change = pf.equity_series().iloc[-1] - pf.initial_capital
    attributed = trades["net_pnl"].sum() - pf.total_margin_interest
    assert attributed == pytest.approx(change, rel=1e-9, abs=1e-6)
    assert trades["commission"].sum() == pytest.approx(pf.total_commission)
    assert trades["slippage"].sum() == pytest.approx(pf.total_slippage)
    assert trades["financing"].sum() == pytest.approx(pf.total_financing)


def test_ruin_closes_every_trade():
    pf = _run(ruin_equity=99_000.0)  # hard stop at a 1% loss
    assert pf.is_ruined
    trades = pf.trades()
    assert not trades["is_open"].any()
    assert (trades["exit_reason"] == "ruin").any()
    change = pf.equity_series().iloc[-1] - pf.initial_capital
    assert trades["net_pnl"].sum() == pytest.approx(change, abs=1e-6)


def test_turnover_counts_both_legs():
    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    data = HistoricCSVDataHandler({"AAA": pd.DataFrame({"close": [100.0] * 252}, index=idx)})
    pf = Portfolio(data, initial_capital=10_000.0)
    first = True
    while (event := data.update_bars()) is not None:
        if first:
            pf.on_fill(_fill(Direction.LONG, 50, 100.0))
            pf.on_fill(_fill(Direction.SHORT, 50, 100.0))
            first = False
        pf.on_market(event)
    # $5k in + $5k out over one year on $10k of equity
    assert pf.turnover() == pytest.approx(1.0)
