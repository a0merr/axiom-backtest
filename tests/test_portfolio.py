"""Portfolio bookkeeping, including the column-array equity curve.

The equity curve is stored as two typed arrays rather than a list of dicts, and
open/short positions are tracked in side indexes so per-bar marking does not
scan the whole universe. Both are optimizations, so both need tests that pin
the observable behaviour they must not change.
"""

from datetime import datetime

import pandas as pd
import pytest

from axiom import (
    Direction,
    FillEvent,
    HistoricCSVDataHandler,
    MarketEvent,
    Portfolio,
)


def _data(prices=(100.0, 110.0), symbols=("AAA",)):
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="D")
    frames = {s: pd.DataFrame({"close": list(prices)}, index=idx) for s in symbols}
    return HistoricCSVDataHandler(frames)


def _fill(symbol, direction, qty, price, ts=datetime(2020, 1, 1)):
    return FillEvent(symbol, ts, direction, qty, price, commission=0.0, slippage=0.0)


# -- Equity curve representation ------------------------------------------


def test_equity_series_is_indexed_by_timestamp():
    data = _data()
    pf = Portfolio(data, initial_capital=1_000.0)
    while (event := data.update_bars()) is not None:
        pf.on_market(event)

    series = pf.equity_series()
    assert isinstance(series.index, pd.DatetimeIndex)
    assert series.index.name == "timestamp"
    assert series.name == "equity"
    assert list(series.index) == list(pd.date_range("2020-01-01", periods=2, freq="D"))
    assert series.tolist() == [1_000.0, 1_000.0]


def test_equity_curve_records_still_available():
    data = _data()
    pf = Portfolio(data, initial_capital=1_000.0)
    while (event := data.update_bars()) is not None:
        pf.on_market(event)

    records = pf.equity_curve
    assert records[0] == {"timestamp": pd.Timestamp("2020-01-01"), "equity": 1_000.0}
    assert len(records) == 2


def test_empty_equity_series():
    assert Portfolio(_data()).equity_series().empty


def test_equity_curve_tracks_marked_value():
    data = _data(prices=(100.0, 110.0))
    pf = Portfolio(data, initial_capital=1_000.0)

    data.update_bars()
    pf.on_fill(_fill("AAA", Direction.LONG, 10, 100.0))
    pf.on_market(MarketEvent(timestamp=datetime(2020, 1, 1)))
    data.update_bars()
    pf.on_market(MarketEvent(timestamp=datetime(2020, 1, 2)))

    # 10 shares bought at 100 from 1,000 cash, then marked at 110.
    assert pf.equity_series().tolist() == [1_000.0, 1_100.0]


# -- Open / short position indexes ----------------------------------------


def test_market_value_follows_positions_open_and_closed():
    data = _data(prices=(100.0,), symbols=("AAA", "BBB"))
    data.update_bars()
    pf = Portfolio(data)

    assert pf.market_value() == 0.0
    pf.on_fill(_fill("AAA", Direction.LONG, 10, 100.0))
    pf.on_fill(_fill("BBB", Direction.LONG, 5, 100.0))
    assert pf.market_value() == pytest.approx(1_500.0)

    pf.on_fill(_fill("AAA", Direction.SHORT, 10, 100.0))  # flatten AAA
    assert pf.market_value() == pytest.approx(500.0)
    assert "AAA" not in pf._open


def test_short_position_marks_negative_and_accrues_financing():
    data = _data(prices=(100.0,))
    data.update_bars()
    pf = Portfolio(data, initial_capital=1_000.0, annual_borrow_rate=0.252, periods_per_year=252)

    pf.on_fill(_fill("AAA", Direction.SHORT, 10, 100.0))
    assert pf.market_value() == pytest.approx(-1_000.0)

    pf.on_market(MarketEvent(timestamp=datetime(2020, 1, 1)))
    # one day of 25.2%/yr on 1,000 notional = 1.00
    assert pf.total_financing == pytest.approx(1.0)


def test_reversing_through_zero_updates_both_indexes():
    data = _data(prices=(100.0,))
    data.update_bars()
    pf = Portfolio(data)

    pf.on_fill(_fill("AAA", Direction.LONG, 10, 100.0))
    assert "AAA" in pf._open and "AAA" not in pf._short

    pf.on_fill(_fill("AAA", Direction.SHORT, 25, 100.0))  # now net short 15
    assert pf.positions["AAA"].quantity == pytest.approx(-15)
    assert "AAA" in pf._open and "AAA" in pf._short

    pf.on_fill(_fill("AAA", Direction.LONG, 15, 100.0))  # flat
    assert "AAA" not in pf._open and "AAA" not in pf._short


def test_no_financing_charged_on_flat_or_long_book():
    data = _data(prices=(100.0,))
    data.update_bars()
    pf = Portfolio(data, annual_borrow_rate=0.10)

    pf.on_fill(_fill("AAA", Direction.LONG, 10, 100.0))
    pf.on_market(MarketEvent(timestamp=datetime(2020, 1, 1)))
    assert pf.total_financing == 0.0
