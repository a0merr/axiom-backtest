"""Tests for depth-aware slippage and participation-capped fills."""

from datetime import datetime

import pandas as pd
import pytest

from axiom import (
    Direction,
    HistoricCSVDataHandler,
    NextBarExecutionHandler,
    OrderBookSlippage,
    OrderEvent,
    PercentageSlippage,
    PerShareCommission,
    SimulatedExecutionHandler,
    VolumeShareSlippage,
)


def _bar(**fields):
    return pd.Series(fields)


def _free_commission():
    return PerShareCommission(per_share=0.0, minimum=0.0)


# -- OrderBookSlippage: shape of the cost curve ----------------------------


def test_sweep_cost_is_continuous_at_book_exhaustion():
    m = OrderBookSlippage(depth_range=0.002, tail_factor=5.0)
    just_under = m.sweep_cost(0.999999)
    just_over = m.sweep_cost(1.000001)
    assert just_under == pytest.approx(0.001, rel=1e-4)  # depth_range / 2
    assert just_over == pytest.approx(just_under, rel=1e-4)


def test_sweep_cost_is_convex_past_the_book():
    """Doubling size inside the book doubles cost; past it, cost grows faster."""
    m = OrderBookSlippage(depth_range=0.002, tail_factor=5.0)
    # Inside the book the walk is linear, so average cost is linear in size.
    assert m.sweep_cost(0.4) == pytest.approx(2 * m.sweep_cost(0.2))
    # Past it, the thin tail makes the average cost grow superlinearly.
    assert m.sweep_cost(4.0) > 2 * m.sweep_cost(2.0)


def test_order_book_costs_more_than_sqrt_impact_for_oversized_orders():
    """The elbow is the whole point: sqrt models flatter big orders."""
    book = OrderBookSlippage(
        half_spread=0.0, depth_share=0.05, depth_range=0.002, tail_factor=5.0
    )
    sqrt = VolumeShareSlippage(base_rate=0.0, impact_coef=0.1)
    bar = _bar(volume=100_000.0)

    # 2% of volume: inside a book holding 5% of volume. Both models are mild.
    small = 2_000
    # 20% of volume: four times the modeled book. The sqrt model barely notices.
    huge = 20_000

    book_small = book.fill_price(100.0, Direction.LONG, small, bar) - 100.0
    sqrt_small = sqrt.fill_price(100.0, Direction.LONG, small, bar) - 100.0
    book_huge = book.fill_price(100.0, Direction.LONG, huge, bar) - 100.0
    sqrt_huge = sqrt.fill_price(100.0, Direction.LONG, huge, bar) - 100.0

    assert book_huge / book_small > sqrt_huge / sqrt_small


def test_direction_still_works_against_the_trader():
    m = OrderBookSlippage(half_spread=0.001, depth_share=1.0, depth_range=0.0)
    bar = _bar(volume=10_000.0)
    assert m.fill_price(100.0, Direction.LONG, 100, bar) == pytest.approx(100.1)
    assert m.fill_price(100.0, Direction.SHORT, 100, bar) == pytest.approx(99.9)


def test_fallback_rate_used_when_bar_has_no_volume():
    m = OrderBookSlippage(half_spread=0.0002, fallback_rate=0.001)
    price = m.fill_price(100.0, Direction.LONG, 500, _bar(close=100.0))
    assert price == pytest.approx(100.0 * 1.0012)


def test_zero_volume_bar_falls_back_rather_than_dividing_by_zero():
    m = OrderBookSlippage(half_spread=0.0, fallback_rate=0.01)
    price = m.fill_price(100.0, Direction.LONG, 500, _bar(volume=0.0))
    assert price == pytest.approx(101.0)


def test_concession_is_capped():
    m = OrderBookSlippage(depth_share=0.001, depth_range=0.05, max_frac=0.25)
    # Absurd size: uncapped this would exceed 100% of the price.
    price = m.fill_price(100.0, Direction.SHORT, 10_000_000, _bar(volume=1_000.0))
    assert price == pytest.approx(75.0)  # floored by max_frac, not negative


# -- OrderBookSlippage: liquidity collapse --------------------------------


def test_stress_widens_spread_and_thins_the_book():
    m = OrderBookSlippage(
        half_spread=0.0002,
        depth_share=0.05,
        depth_range=0.002,
        reference_vol=0.01,
        spread_vol_coef=1.0,
        depth_vol_coef=1.0,
    )
    calm = m.fill_price(100.0, Direction.LONG, 2_000, _bar(volume=100_000.0, volatility=0.01))
    stressed = m.fill_price(
        100.0, Direction.LONG, 2_000, _bar(volume=100_000.0, volatility=0.03)
    )
    assert stressed > calm


def test_stress_response_is_worse_than_linear_in_volatility():
    """Cost must rise faster than vol: depth and spread collapse together."""
    m = OrderBookSlippage(
        half_spread=0.0002,
        depth_share=0.05,
        depth_range=0.002,
        reference_vol=0.01,
        spread_vol_coef=1.0,
        depth_vol_coef=1.0,
    )
    ref, qty, vol = 100.0, 4_000, 100_000.0
    calm = m.fill_price(ref, Direction.LONG, qty, _bar(volume=vol, volatility=0.01)) - ref
    x3 = m.fill_price(ref, Direction.LONG, qty, _bar(volume=vol, volatility=0.03)) - ref
    assert x3 > 3 * calm


def test_stress_is_inert_without_a_reference_vol():
    m = OrderBookSlippage(reference_vol=0.0)
    bar_calm = _bar(volume=100_000.0, volatility=0.01)
    bar_wild = _bar(volume=100_000.0, volatility=0.50)
    assert m.fill_price(100.0, Direction.LONG, 500, bar_calm) == pytest.approx(
        m.fill_price(100.0, Direction.LONG, 500, bar_wild)
    )


def test_order_book_rejects_bad_params():
    with pytest.raises(ValueError):
        OrderBookSlippage(depth_share=0.0)
    with pytest.raises(ValueError):
        OrderBookSlippage(half_spread=-0.1)
    with pytest.raises(ValueError):
        OrderBookSlippage(tail_factor=-1.0)
    with pytest.raises(ValueError):
        OrderBookSlippage(max_frac=0.0)


# -- Participation caps and partial fills ---------------------------------


def _one_bar_data(volume: float | None = None, price: float = 100.0):
    cols = {"close": [price]}
    if volume is not None:
        cols["volume"] = [volume]
    df = pd.DataFrame(cols, index=pd.to_datetime(["2020-01-01"]))
    data = HistoricCSVDataHandler({"AAA": df})
    data.update_bars()
    return data


def test_participation_cap_truncates_the_fill():
    data = _one_bar_data(volume=10_000.0)
    handler = SimulatedExecutionHandler(
        data,
        slippage=PercentageSlippage(0.0),
        commission=_free_commission(),
        max_participation=0.1,
    )
    fill = handler.execute_order(
        OrderEvent("AAA", datetime(2020, 1, 1), Direction.LONG, 5_000)
    )
    assert fill.quantity == pytest.approx(1_000)  # 10% of 10,000
    assert handler.unfilled_quantity == pytest.approx(4_000)
    assert handler.partial_fills == 1


def test_no_cap_means_unlimited_size():
    data = _one_bar_data(volume=10_000.0)
    handler = SimulatedExecutionHandler(data, slippage=PercentageSlippage(0.0))
    fill = handler.execute_order(
        OrderEvent("AAA", datetime(2020, 1, 1), Direction.LONG, 5_000_000)
    )
    assert fill.quantity == pytest.approx(5_000_000)
    assert handler.unfilled_quantity == 0.0


def test_cap_is_inert_without_a_volume_column():
    data = _one_bar_data(volume=None)
    handler = SimulatedExecutionHandler(
        data, slippage=PercentageSlippage(0.0), max_participation=0.1
    )
    fill = handler.execute_order(
        OrderEvent("AAA", datetime(2020, 1, 1), Direction.LONG, 5_000)
    )
    assert fill.quantity == pytest.approx(5_000)


def test_zero_volume_bar_fills_nothing():
    data = _one_bar_data(volume=0.0)
    handler = SimulatedExecutionHandler(
        data, slippage=PercentageSlippage(0.0), max_participation=0.1
    )
    fill = handler.execute_order(
        OrderEvent("AAA", datetime(2020, 1, 1), Direction.LONG, 500)
    )
    assert fill is None
    assert handler.unfilled_quantity == pytest.approx(500)


def test_participation_rejects_non_positive():
    with pytest.raises(ValueError):
        SimulatedExecutionHandler(_one_bar_data(volume=1.0), max_participation=0.0)


# -- Working an order across bars -----------------------------------------


def _multi_bar_data(n=6, volume=10_000.0):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {"open": [100.0] * n, "close": [100.0] * n, "volume": [volume] * n}, index=idx
    )
    return HistoricCSVDataHandler({"AAA": df})


def test_large_order_is_worked_across_bars():
    data = _multi_bar_data()
    handler = NextBarExecutionHandler(
        data,
        slippage=PercentageSlippage(0.0),
        commission=_free_commission(),
        max_participation=0.1,  # 1,000 shares per bar
        max_working_bars=5,
    )
    data.update_bars()
    handler.execute_order(OrderEvent("AAA", datetime(2020, 1, 1), Direction.LONG, 2_500))

    filled = []
    for _ in range(4):
        data.update_bars()
        filled += [f.quantity for f in handler.on_bar(None)]

    assert filled == [
        pytest.approx(1_000),
        pytest.approx(1_000),
        pytest.approx(500),
    ]
    assert handler.unfilled_quantity == 0.0


def test_order_too_big_for_the_market_is_cancelled_and_reported():
    data = _multi_bar_data(n=8)
    handler = NextBarExecutionHandler(
        data,
        slippage=PercentageSlippage(0.0),
        commission=_free_commission(),
        max_participation=0.1,  # 1,000 per bar
        max_working_bars=3,
    )
    data.update_bars()
    handler.execute_order(OrderEvent("AAA", datetime(2020, 1, 1), Direction.LONG, 10_000))

    total = 0.0
    for _ in range(6):
        data.update_bars()
        total += sum(f.quantity for f in handler.on_bar(None))

    assert total == pytest.approx(3_000)  # 3 bars x 1,000
    assert handler.unfilled_quantity == pytest.approx(7_000)  # rest cancelled
    assert handler._pending == []


def test_worked_remainder_fills_at_later_prices():
    """Carrying an order forward must pick up real timing risk, not a free ride."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 200.0],
            "close": [100.0, 100.0, 200.0],
            "volume": [10_000.0] * 3,
        },
        index=idx,
    )
    data = HistoricCSVDataHandler({"AAA": df})
    handler = NextBarExecutionHandler(
        data,
        slippage=PercentageSlippage(0.0),
        commission=_free_commission(),
        max_participation=0.1,
    )
    data.update_bars()
    handler.execute_order(OrderEvent("AAA", datetime(2020, 1, 1), Direction.LONG, 1_500))

    data.update_bars()
    first = handler.on_bar(None)
    data.update_bars()
    second = handler.on_bar(None)

    assert first[0].fill_price == pytest.approx(100.0)
    assert second[0].fill_price == pytest.approx(200.0)  # gapped away mid-order


def test_order_for_an_unpriced_symbol_is_reported_not_dropped():
    """A lapsed order must show up as unfilled, not vanish."""
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    frames = {
        "AAA": pd.DataFrame({"open": [1.0, 1.0], "close": [1.0, 1.0]}, index=idx),
        "BBB": pd.DataFrame({"open": [5.0], "close": [5.0]}, index=idx[1:]),
    }
    data = HistoricCSVDataHandler(frames)
    handler = NextBarExecutionHandler(data, slippage=PercentageSlippage(0.0))

    data.update_bars()  # BBB has not listed yet
    handler.execute_order(OrderEvent("BBB", datetime(2020, 1, 1), Direction.LONG, 100))
    assert handler.on_bar(None) == []
    assert handler.unfilled_quantity == pytest.approx(100)
