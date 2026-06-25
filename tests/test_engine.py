import numpy as np
import pandas as pd
import pytest

from axiom import (
    BacktestEngine,
    Direction,
    HistoricCSVDataHandler,
    MovingAverageCrossover,
    PercentageSlippage,
    PerShareCommission,
    Portfolio,
    SignalEvent,
    SimulatedExecutionHandler,
)
from axiom.strategy import Strategy


def _trend_frame(n=120, start=100.0, drift=0.5):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    prices = start + np.arange(n) * drift
    return {"AAA": pd.DataFrame({"close": prices}, index=idx)}


class _BuyAndHold(Strategy):
    """Long on the first bar it sees, then nothing."""

    def __init__(self, data):
        self.data = data
        self._fired = False

    def on_market(self, event):
        if self._fired:
            return []
        self._fired = True
        return [SignalEvent("AAA", event.timestamp, Direction.LONG)]


def _run(frames, strategy_cls, **kw):
    data = HistoricCSVDataHandler(frames)
    strat = strategy_cls(data, **kw)
    pf = Portfolio(data, initial_capital=100_000.0)
    ex = SimulatedExecutionHandler(
        data, slippage=PercentageSlippage(0.0), commission=PerShareCommission(0, 0)
    )
    BacktestEngine(data, strat, pf, ex).run()
    return pf


def test_buy_and_hold_profits_in_uptrend():
    pf = _run(_trend_frame(drift=1.0), _BuyAndHold)
    eq = pf.equity_series()
    assert eq.iloc[-1] > eq.iloc[0]
    assert len(eq) == 120


def test_no_lookahead_equity_length_matches_bars():
    pf = _run(_trend_frame(), MovingAverageCrossover, fast=5, slow=20)
    assert len(pf.equity_series()) == 120


def test_costs_reduce_terminal_equity():
    frames = _trend_frame(drift=1.0)
    free = _run(frames, _BuyAndHold).equity_series().iloc[-1]

    data = HistoricCSVDataHandler(frames)
    pf = Portfolio(data, initial_capital=100_000.0)
    ex = SimulatedExecutionHandler(
        data,
        slippage=PercentageSlippage(0.005),
        commission=PerShareCommission(0.01, 1.0),
    )
    BacktestEngine(data, _BuyAndHold(data), pf, ex).run()
    costed = pf.equity_series().iloc[-1]
    assert costed < free
    assert pf.total_commission > 0
    assert pf.total_slippage > 0


def test_crossover_fast_must_be_shorter():
    data = HistoricCSVDataHandler(_trend_frame())
    with pytest.raises(ValueError):
        MovingAverageCrossover(data, fast=50, slow=20)


def test_flat_market_no_runaway_positions():
    idx = pd.date_range("2020-01-01", periods=80, freq="D")
    frames = {"AAA": pd.DataFrame({"close": np.full(80, 100.0)}, index=idx)}
    pf = _run(frames, MovingAverageCrossover, fast=5, slow=20)
    # Constant price -> no crossover -> never invested -> equity flat.
    assert pf.equity_series().iloc[-1] == pytest.approx(100_000.0)
