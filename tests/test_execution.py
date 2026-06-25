from datetime import datetime

import pandas as pd
import pytest

from axiom import (
    Direction,
    HistoricCSVDataHandler,
    NextBarExecutionHandler,
    OrderEvent,
    PercentageSlippage,
    PerShareCommission,
    SimulatedExecutionHandler,
    VolatilitySlippage,
    VolumeShareSlippage,
)


def _data_at(price: float):
    df = pd.DataFrame({"close": [price]}, index=pd.to_datetime(["2020-01-01"]))
    data = HistoricCSVDataHandler({"AAA": df})
    data.update_bars()  # expose the bar
    return data


def test_next_bar_defers_fill_to_following_open():
    df = pd.DataFrame(
        {"open": [100.0, 110.0], "close": [101.0, 111.0]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
    )
    data = HistoricCSVDataHandler({"AAA": df})
    handler = NextBarExecutionHandler(
        data, slippage=PercentageSlippage(0.0), commission=PerShareCommission(0, 0)
    )

    # Bar 1: engine calls on_bar first (nothing pending), then a signal queues
    # an order during the bar. The order does NOT fill on bar 1.
    data.update_bars()
    assert handler.on_bar(None) == []
    order = OrderEvent("AAA", datetime(2020, 1, 1), Direction.LONG, 10)
    assert handler.execute_order(order) is None

    # Bar 2: on_bar (called before strategy) fills the queued order at bar 2's
    # OPEN (110), not bar 1's close (101).
    data.update_bars()
    fills = handler.on_bar(None)
    assert len(fills) == 1
    assert fills[0].fill_price == pytest.approx(110.0)


def test_next_bar_falls_back_to_close_without_open_column():
    df = pd.DataFrame(
        {"close": [100.0, 120.0]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
    )
    data = HistoricCSVDataHandler({"AAA": df})
    handler = NextBarExecutionHandler(
        data, slippage=PercentageSlippage(0.0), commission=PerShareCommission(0, 0)
    )
    data.update_bars()
    handler.execute_order(OrderEvent("AAA", datetime(2020, 1, 1), Direction.LONG, 5))
    data.update_bars()
    fills = handler.on_bar(None)
    assert fills[0].fill_price == pytest.approx(120.0)  # next bar's close


def _bar(**fields):
    return pd.Series(fields)


def test_volume_impact_scales_with_order_size():
    model = VolumeShareSlippage(base_rate=0.0, impact_coef=0.1)
    small = model.fill_price(100.0, Direction.LONG, 100, _bar(volume=10_000))
    large = model.fill_price(100.0, Direction.LONG, 400, _bar(volume=10_000))
    # participation 0.01 -> 0.04; sqrt scaling -> frac 0.01 -> 0.02
    assert small == pytest.approx(101.0)
    assert large == pytest.approx(102.0)
    assert large > small  # bigger order pays disproportionately more


def test_volume_impact_falls_back_without_volume():
    model = VolumeShareSlippage(base_rate=0.001, impact_coef=0.1)
    price = model.fill_price(100.0, Direction.LONG, 999, _bar(close=100.0))
    assert price == pytest.approx(100.0 * 1.001)  # base_rate only


def test_volume_impact_rejects_negative_params():
    with pytest.raises(ValueError):
        VolumeShareSlippage(base_rate=-0.1)


def test_volatility_slippage_scales_with_vol_column():
    model = VolatilitySlippage(k=0.5, floor=0.0)
    quiet = model.fill_price(100.0, Direction.SHORT, 10, _bar(volatility=0.01))
    wild = model.fill_price(100.0, Direction.SHORT, 10, _bar(volatility=0.05))
    assert quiet == pytest.approx(100.0 * (1 - 0.005))
    assert wild < quiet  # short fills worse (lower) in the turbulent regime


def test_long_slippage_pays_up():
    data = _data_at(100.0)
    handler = SimulatedExecutionHandler(data, slippage=PercentageSlippage(0.01))
    order = OrderEvent("AAA", datetime(2020, 1, 1), Direction.LONG, 10)
    fill = handler.execute_order(order)
    assert fill.fill_price == pytest.approx(101.0)  # bought worse
    assert fill.slippage == pytest.approx(1.0 * 10)


def test_short_slippage_receives_less():
    data = _data_at(100.0)
    handler = SimulatedExecutionHandler(data, slippage=PercentageSlippage(0.01))
    order = OrderEvent("AAA", datetime(2020, 1, 1), Direction.SHORT, 10)
    fill = handler.execute_order(order)
    assert fill.fill_price == pytest.approx(99.0)  # sold worse


def test_commission_minimum_applies():
    comm = PerShareCommission(per_share=0.005, minimum=1.0)
    assert comm.commission(10, 100.0) == 1.0  # 10*0.005=0.05 -> floored to 1.0
    assert comm.commission(1000, 100.0) == pytest.approx(5.0)


def test_negative_quantity_rejected():
    with pytest.raises(ValueError):
        OrderEvent("AAA", datetime(2020, 1, 1), Direction.LONG, -5)


def test_negative_slippage_rejected():
    with pytest.raises(ValueError):
        PercentageSlippage(-0.01)
