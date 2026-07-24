"""Ruin, leverage, margin interest, and sizing-vs-fill price drift.

These are correctness guards rather than features: without them the engine
reports numbers for situations that cannot occur — trading a negative account,
holding more exposure than the account can carry, and borrowing for free.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from axiom import (
    Direction,
    FillEvent,
    HistoricCSVDataHandler,
    MarketEvent,
    NextBarExecutionHandler,
    OrderEvent,
    PercentageSlippage,
    PerShareCommission,
    Portfolio,
    SignalEvent,
    SimulatedExecutionHandler,
)
from axiom.metrics import max_drawdown, returns_from_equity


def _data(prices=(100.0,), symbols=("AAA",)):
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="D")
    frames = {s: pd.DataFrame({"close": list(prices)}, index=idx) for s in symbols}
    return HistoricCSVDataHandler(frames)


def _fill(symbol, direction, qty, price, ts=datetime(2020, 1, 1)):
    return FillEvent(symbol, ts, direction, qty, price, commission=0.0, slippage=0.0)


def _mark(pf, day=1):
    pf.on_market(MarketEvent(timestamp=datetime(2020, 1, day)))


# -- Ruin ------------------------------------------------------------------


def test_account_through_zero_is_declared_ruined():
    data = _data(prices=(100.0, 1.0))
    data.update_bars()
    pf = Portfolio(data, initial_capital=1_000.0)

    # Borrow heavily: 100 shares at 100 costs 10,000 against 1,000 of capital.
    pf.on_fill(_fill("AAA", Direction.LONG, 100, 100.0))
    _mark(pf, 1)
    assert not pf.is_ruined

    data.update_bars()  # price collapses 100 -> 1
    _mark(pf, 2)
    assert pf.is_ruined
    assert pf.ruin_timestamp == datetime(2020, 1, 2)


def test_ruin_liquidates_and_freezes_the_curve():
    data = _data(prices=(100.0, 1.0, 1.0, 1.0))
    data.update_bars()
    pf = Portfolio(data, initial_capital=1_000.0)
    pf.on_fill(_fill("AAA", Direction.LONG, 100, 100.0))
    _mark(pf, 1)

    for day in (2, 3, 4):
        data.update_bars()
        _mark(pf, day)

    assert pf.positions["AAA"].quantity == 0.0
    assert pf._open == {}
    curve = pf.equity_series().tolist()
    assert curve[1] == curve[2] == curve[3]  # dead accounts do not move


def test_ruined_portfolio_stops_generating_orders():
    data = _data(prices=(100.0, 1.0))
    data.update_bars()
    pf = Portfolio(data, initial_capital=1_000.0)
    pf.on_fill(_fill("AAA", Direction.LONG, 100, 100.0))
    _mark(pf, 1)
    data.update_bars()
    _mark(pf, 2)

    signal = SignalEvent("AAA", datetime(2020, 1, 2), Direction.LONG)
    assert pf.on_signal(signal) == []


def test_ruined_portfolio_ignores_in_flight_fills():
    data = _data(prices=(100.0, 1.0))
    data.update_bars()
    pf = Portfolio(data, initial_capital=1_000.0)
    pf.on_fill(_fill("AAA", Direction.LONG, 100, 100.0))
    _mark(pf, 1)
    data.update_bars()
    _mark(pf, 2)
    equity_at_death = pf.equity()

    pf.on_fill(_fill("AAA", Direction.LONG, 50, 1.0))
    assert pf.positions["AAA"].quantity == 0.0
    assert pf.equity() == pytest.approx(equity_at_death)


def test_ruin_equity_can_model_a_hard_stop():
    """Stop trading at a 60% loss rather than waiting for literal zero."""
    data = _data(prices=(100.0, 30.0))
    data.update_bars()
    pf = Portfolio(data, initial_capital=1_000.0, ruin_equity=400.0)
    pf.on_fill(_fill("AAA", Direction.LONG, 10, 100.0))  # all-in, no leverage
    _mark(pf, 1)
    assert not pf.is_ruined

    data.update_bars()  # 10 shares now worth 300 -> equity 300 < 400
    _mark(pf, 2)
    assert pf.is_ruined
    assert pf.equity() == pytest.approx(300.0)


def test_healthy_account_is_never_ruined():
    data = _data(prices=(100.0, 110.0))
    pf = Portfolio(data, initial_capital=1_000.0)
    while data.update_bars() is not None:
        _mark(pf, 1)
    assert not pf.is_ruined
    assert pf.ruin_timestamp is None


# -- Metrics survive a dead account ---------------------------------------


def test_returns_truncate_at_ruin_instead_of_flipping_sign():
    """Recovering from -10 to -5 is not a -50% return."""
    equity = pd.Series([100.0, 50.0, -10.0, -5.0])
    rets = returns_from_equity(equity)

    assert len(rets) == 2
    assert rets.iloc[0] == pytest.approx(-0.5)
    assert rets.iloc[1] == pytest.approx(-1.2)  # the real step into ruin
    assert (rets > 0).sum() == 0  # no phantom gain from the -10 -> -5 move


def test_returns_unaffected_when_equity_stays_positive():
    equity = pd.Series([100.0, 110.0, 99.0])
    assert len(returns_from_equity(equity)) == 2


def test_drawdown_past_total_loss_is_reported_not_clipped():
    assert max_drawdown(pd.Series([100.0, -40.0])) == pytest.approx(-1.4)


def test_drawdown_ignores_a_curve_that_never_had_equity():
    assert max_drawdown(pd.Series([-1.0, -2.0])) == 0.0


# -- Gross leverage --------------------------------------------------------


def _four_symbol_portfolio(**kwargs):
    data = _data(prices=(100.0,), symbols=("AAA", "BBB", "CCC", "DDD"))
    data.update_bars()
    return data, Portfolio(data, initial_capital=100_000.0, risk_fraction=0.5, **kwargs)


def test_unlimited_leverage_is_the_default():
    """Four symbols at 50% each = 2x gross, and nothing stops it."""
    data, pf = _four_symbol_portfolio()
    for symbol in data.symbols:
        orders = pf.on_signal(SignalEvent(symbol, datetime(2020, 1, 1), Direction.LONG))
        pf.on_fill(_fill(symbol, Direction.LONG, orders[0].quantity, 100.0))

    assert pf.gross_exposure() == pytest.approx(200_000.0)
    assert pf.leverage() == pytest.approx(2.0)


def test_gross_leverage_cap_binds_across_symbols():
    data, pf = _four_symbol_portfolio(max_gross_leverage=1.0)
    for symbol in data.symbols:
        orders = pf.on_signal(SignalEvent(symbol, datetime(2020, 1, 1), Direction.LONG))
        if orders:
            pf.on_fill(_fill(symbol, Direction.LONG, orders[0].quantity, 100.0))

    assert pf.gross_exposure() == pytest.approx(100_000.0)
    assert pf.leverage() == pytest.approx(1.0)


def test_cap_holds_when_signals_arrive_before_any_fill():
    """Deferred execution: all four size against the same untouched equity."""
    data, pf = _four_symbol_portfolio(max_gross_leverage=1.0)
    committed = 0.0
    for symbol in data.symbols:
        orders = pf.on_signal(SignalEvent(symbol, datetime(2020, 1, 1), Direction.LONG))
        committed += sum(o.quantity * 100.0 for o in orders)

    assert committed == pytest.approx(100_000.0)


def test_pending_commitment_resets_each_bar():
    data, pf = _four_symbol_portfolio(max_gross_leverage=1.0)
    pf.on_signal(SignalEvent("AAA", datetime(2020, 1, 1), Direction.LONG))
    assert sum(pf._pending.values()) > 0
    _mark(pf, 1)
    assert pf._pending == {}


def test_exits_are_allowed_at_the_cap():
    data, pf = _four_symbol_portfolio(max_gross_leverage=1.0)
    orders = pf.on_signal(SignalEvent("AAA", datetime(2020, 1, 1), Direction.LONG))
    pf.on_fill(_fill("AAA", Direction.LONG, orders[0].quantity, 100.0))

    exits = pf.on_signal(SignalEvent("AAA", datetime(2020, 1, 1), Direction.EXIT))
    assert exits[0].direction == Direction.SHORT
    assert exits[0].quantity == pytest.approx(orders[0].quantity)


def test_reversal_does_not_double_count_the_covered_side():
    """Flipping long -> short replaces exposure, it does not add to it."""
    data = _data(prices=(100.0,))
    data.update_bars()
    pf = Portfolio(
        data, initial_capital=100_000.0, risk_fraction=1.0, max_gross_leverage=1.0
    )
    long_orders = pf.on_signal(SignalEvent("AAA", datetime(2020, 1, 1), Direction.LONG))
    pf.on_fill(_fill("AAA", Direction.LONG, long_orders[0].quantity, 100.0))
    _mark(pf, 1)

    flip = pf.on_signal(SignalEvent("AAA", datetime(2020, 1, 1), Direction.SHORT))
    # 1,000 to sell out the long + 1,000 to open the short.
    assert flip[0].quantity == pytest.approx(2_000.0)


def test_short_exposure_counts_toward_the_cap():
    data, pf = _four_symbol_portfolio(max_gross_leverage=1.0)
    for symbol in data.symbols:
        orders = pf.on_signal(SignalEvent(symbol, datetime(2020, 1, 1), Direction.SHORT))
        if orders:
            pf.on_fill(_fill(symbol, Direction.SHORT, orders[0].quantity, 100.0))

    assert pf.gross_exposure() == pytest.approx(100_000.0)


def test_leverage_is_zero_for_a_dead_account():
    data = _data(prices=(100.0,))
    data.update_bars()
    pf = Portfolio(data, initial_capital=-5.0)
    assert pf.leverage() == 0.0


def test_invalid_leverage_rejected():
    with pytest.raises(ValueError):
        Portfolio(_data(), max_gross_leverage=0.0)


def test_negative_financing_rates_rejected():
    with pytest.raises(ValueError):
        Portfolio(_data(), annual_margin_rate=-0.01)


# -- Margin interest -------------------------------------------------------


def test_margin_interest_charged_on_a_debit_balance():
    data = _data(prices=(100.0,))
    data.update_bars()
    pf = Portfolio(
        data,
        initial_capital=1_000.0,
        annual_margin_rate=0.252,
        periods_per_year=252,
    )
    pf.on_fill(_fill("AAA", Direction.LONG, 20, 100.0))  # 2,000 spent, cash -1,000
    assert pf.cash == pytest.approx(-1_000.0)

    _mark(pf, 1)
    assert pf.total_margin_interest == pytest.approx(1.0)  # one day at 25.2%/yr
    assert pf.total_financing == 0.0  # borrow is short-side only


def test_no_margin_interest_on_a_credit_balance():
    data = _data(prices=(100.0,))
    data.update_bars()
    pf = Portfolio(data, initial_capital=100_000.0, annual_margin_rate=0.10)
    pf.on_fill(_fill("AAA", Direction.LONG, 10, 100.0))
    _mark(pf, 1)
    assert pf.total_margin_interest == 0.0


def test_long_and_short_leverage_are_both_financed():
    """The old asymmetry: shorts paid, margin longs rode free."""
    data = _data(prices=(100.0,))
    data.update_bars()
    short_pf = Portfolio(data, initial_capital=1_000.0, annual_borrow_rate=0.252)
    short_pf.on_fill(_fill("AAA", Direction.SHORT, 10, 100.0))
    _mark(short_pf, 1)

    long_pf = Portfolio(data, initial_capital=1_000.0, annual_margin_rate=0.252)
    long_pf.on_fill(_fill("AAA", Direction.LONG, 20, 100.0))
    _mark(long_pf, 1)

    assert short_pf.total_financing > 0
    assert long_pf.total_margin_interest > 0


# -- Sizing price vs fill price -------------------------------------------


def _gap_data(open_prices, close_prices):
    idx = pd.date_range("2020-01-01", periods=len(open_prices), freq="D")
    df = pd.DataFrame({"open": open_prices, "close": close_prices}, index=idx)
    return HistoricCSVDataHandler({"AAA": df})


def test_order_cancelled_when_the_market_gaps_away():
    data = _gap_data([100.0, 130.0], [100.0, 130.0])
    handler = NextBarExecutionHandler(
        data, slippage=PercentageSlippage(0.0), max_price_drift=0.05
    )
    data.update_bars()
    handler.execute_order(
        OrderEvent("AAA", datetime(2020, 1, 1), Direction.LONG, 10, reference_price=100.0)
    )
    data.update_bars()  # opens 30% higher than the order was sized at

    assert handler.on_bar(None) == []
    assert handler.cancelled_on_drift == 1
    assert handler.unfilled_quantity == pytest.approx(10)


def test_favorable_moves_are_not_cancelled():
    """Cancelling only the good fills would bias the backtest downward."""
    data = _gap_data([100.0, 70.0], [100.0, 70.0])
    handler = NextBarExecutionHandler(
        data, slippage=PercentageSlippage(0.0), max_price_drift=0.05
    )
    data.update_bars()
    handler.execute_order(
        OrderEvent("AAA", datetime(2020, 1, 1), Direction.LONG, 10, reference_price=100.0)
    )
    data.update_bars()

    fills = handler.on_bar(None)
    assert len(fills) == 1
    assert handler.cancelled_on_drift == 0


def test_drift_direction_is_inverted_for_sells():
    data = _gap_data([100.0, 70.0], [100.0, 70.0])
    handler = NextBarExecutionHandler(
        data, slippage=PercentageSlippage(0.0), max_price_drift=0.05
    )
    data.update_bars()
    handler.execute_order(
        OrderEvent("AAA", datetime(2020, 1, 1), Direction.SHORT, 10, reference_price=100.0)
    )
    data.update_bars()  # a seller is hurt by the drop the buyer welcomed

    assert handler.on_bar(None) == []
    assert handler.cancelled_on_drift == 1


def test_no_drift_guard_without_a_reference_price():
    data = _gap_data([100.0, 300.0], [100.0, 300.0])
    handler = NextBarExecutionHandler(
        data, slippage=PercentageSlippage(0.0), max_price_drift=0.01
    )
    data.update_bars()
    handler.execute_order(OrderEvent("AAA", datetime(2020, 1, 1), Direction.LONG, 10))
    data.update_bars()
    assert len(handler.on_bar(None)) == 1


def test_drift_guard_is_off_by_default():
    data = _gap_data([100.0, 300.0], [100.0, 300.0])
    handler = NextBarExecutionHandler(data, slippage=PercentageSlippage(0.0))
    data.update_bars()
    handler.execute_order(
        OrderEvent("AAA", datetime(2020, 1, 1), Direction.LONG, 10, reference_price=100.0)
    )
    data.update_bars()
    assert len(handler.on_bar(None)) == 1


def test_worked_remainder_accumulates_drift_from_the_original_price():
    """An order worked for days is judged against where it was sized, not re-based."""
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 103.0, 120.0],
            "close": [100.0, 101.0, 103.0, 120.0],
            "volume": [1_000.0] * 4,
        },
        index=idx,
    )
    data = HistoricCSVDataHandler({"AAA": df})
    handler = NextBarExecutionHandler(
        data,
        slippage=PercentageSlippage(0.0),
        commission=PerShareCommission(0.0, 0.0),
        max_participation=0.1,  # 100 shares/bar, so 300 gets worked
        max_price_drift=0.10,
        max_working_bars=5,
    )
    data.update_bars()
    handler.execute_order(
        OrderEvent("AAA", datetime(2020, 1, 1), Direction.LONG, 300, reference_price=100.0)
    )

    filled = 0.0
    for _ in range(3):
        data.update_bars()
        filled += sum(f.quantity for f in handler.on_bar(None))

    assert filled == pytest.approx(200.0)  # bars at 101 and 103 fill
    assert handler.cancelled_on_drift == 1  # the bar at 120 is 20% away
    assert handler.unfilled_quantity == pytest.approx(100.0)


def test_portfolio_stamps_the_sizing_price_on_orders():
    data = _data(prices=(100.0,))
    data.update_bars()
    pf = Portfolio(data, initial_capital=10_000.0, risk_fraction=0.5)
    orders = pf.on_signal(SignalEvent("AAA", datetime(2020, 1, 1), Direction.LONG))
    assert orders[0].reference_price == pytest.approx(100.0)


def test_simulated_handler_also_honours_drift():
    data = _data(prices=(100.0,))
    data.update_bars()
    handler = SimulatedExecutionHandler(
        data, slippage=PercentageSlippage(0.0), max_price_drift=0.01
    )
    fill = handler.execute_order(
        OrderEvent("AAA", datetime(2020, 1, 1), Direction.LONG, 10, reference_price=50.0)
    )
    assert fill is None
    assert handler.cancelled_on_drift == 1


def test_invalid_drift_rejected():
    with pytest.raises(ValueError):
        SimulatedExecutionHandler(_data(), max_price_drift=-0.1)


# -- End to end ------------------------------------------------------------


def test_engine_run_with_all_guards_enabled_stays_consistent():
    from axiom import BacktestEngine, MovingAverageCrossover

    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=400, freq="D")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400)))
    frames = {
        "AAA": pd.DataFrame(
            {"open": close, "close": close, "volume": np.full(400, 50_000.0)}, index=idx
        )
    }
    data = HistoricCSVDataHandler(frames)
    pf = Portfolio(
        data,
        initial_capital=100_000.0,
        risk_fraction=0.5,
        max_gross_leverage=1.0,
        annual_margin_rate=0.05,
    )
    execution = NextBarExecutionHandler(
        data, max_participation=0.1, max_price_drift=0.10
    )
    BacktestEngine(data, MovingAverageCrossover(data), pf, execution).run()

    assert len(pf.equity_series()) == 400
    assert pf.leverage() <= 1.0 + 1e-9
    assert (pf.equity_series() > 0).all()  # guards held; no ruin on this path
