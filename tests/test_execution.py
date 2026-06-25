from datetime import datetime

import pandas as pd
import pytest

from axiom import (
    Direction,
    HistoricCSVDataHandler,
    OrderEvent,
    PercentageSlippage,
    PerShareCommission,
    SimulatedExecutionHandler,
)


def _data_at(price: float):
    df = pd.DataFrame({"close": [price]}, index=pd.to_datetime(["2020-01-01"]))
    data = HistoricCSVDataHandler({"AAA": df})
    data.update_bars()  # expose the bar
    return data


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
