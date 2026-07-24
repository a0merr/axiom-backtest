"""Tests for the numpy-backed data layer.

These pin the behaviour that the fast path must not quietly change: what a bar
exposes, when a symbol becomes visible, and that carrying a sparse symbol
forward still never reveals a future row.
"""

import numpy as np
import pandas as pd
import pytest

from axiom import Bar, HistoricCSVDataHandler


def _frame(prices, start="2020-01-01", **extra):
    idx = pd.date_range(start, periods=len(prices), freq="D")
    return pd.DataFrame({"close": prices, **extra}, index=idx)


# -- Bar ------------------------------------------------------------------


def test_bar_reads_fields_as_plain_floats():
    data = HistoricCSVDataHandler({"AAA": _frame([100.0], volume=[5_000.0])})
    data.update_bars()
    bar = data.latest_bar("AAA")

    assert isinstance(bar, Bar)
    assert bar["close"] == 100.0
    assert type(bar["close"]) is float  # not np.float64
    assert "volume" in bar
    assert "spread" not in bar
    assert bar.get("spread") is None
    assert bar.get("spread", 0.5) == 0.5
    assert set(bar.keys()) == {"close", "volume"}
    assert len(bar) == 2
    assert bar.symbol == "AAA"


def test_bar_supports_pandas_style_index_membership():
    """Slippage models written against pd.Series must keep working."""
    data = HistoricCSVDataHandler({"AAA": _frame([100.0], volume=[1.0])})
    data.update_bars()
    bar = data.latest_bar("AAA")
    assert "volume" in bar.index
    assert "nope" not in bar.index


def test_bar_round_trips_to_series():
    data = HistoricCSVDataHandler({"AAA": _frame([100.0], volume=[5_000.0])})
    data.update_bars()
    series = data.latest_bar("AAA").to_series()

    assert isinstance(series, pd.Series)
    assert series["close"] == 100.0
    assert series["volume"] == 5_000.0


def test_bar_raises_on_unknown_field():
    data = HistoricCSVDataHandler({"AAA": _frame([100.0])})
    data.update_bars()
    with pytest.raises(KeyError):
        data.latest_bar("AAA")["nope"]


# -- Visibility and alignment ---------------------------------------------


def test_nothing_is_visible_before_the_first_update():
    data = HistoricCSVDataHandler({"AAA": _frame([100.0, 101.0])})
    assert data.latest_bar("AAA") is None
    assert data.latest_price("AAA") is None


def test_bars_are_exposed_one_at_a_time_and_never_ahead():
    prices = [100.0, 101.0, 102.0]
    data = HistoricCSVDataHandler({"AAA": _frame(prices)})
    seen = []
    while (event := data.update_bars()) is not None:
        seen.append((event.timestamp, data.latest_price("AAA")))

    assert [p for _, p in seen] == prices
    assert data.update_bars() is None  # exhausted, and stays exhausted


def test_sparse_symbol_carries_forward_without_look_ahead():
    """BBB only trades on days 1 and 3; day 2 must show day 1's bar."""
    idx_a = pd.date_range("2020-01-01", periods=3, freq="D")
    frame_a = pd.DataFrame({"close": [10.0, 11.0, 12.0]}, index=idx_a)
    frame_b = pd.DataFrame(
        {"close": [50.0, 70.0]}, index=pd.to_datetime(["2020-01-01", "2020-01-03"])
    )
    data = HistoricCSVDataHandler({"AAA": frame_a, "BBB": frame_b})

    observed = []
    while data.update_bars() is not None:
        observed.append(data.latest_price("BBB"))

    assert observed == [50.0, 50.0, 70.0]  # day 2 repeats day 1, never peeks at 70


def test_symbol_with_no_bar_yet_is_invisible():
    frame_a = pd.DataFrame(
        {"close": [10.0, 11.0]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"])
    )
    frame_b = pd.DataFrame({"close": [99.0]}, index=pd.to_datetime(["2020-01-02"]))
    data = HistoricCSVDataHandler({"AAA": frame_a, "BBB": frame_b})

    data.update_bars()  # 2020-01-01: BBB has not listed yet
    assert data.latest_bar("BBB") is None
    assert data.latest_price("BBB") is None

    data.update_bars()  # 2020-01-02
    assert data.latest_price("BBB") == 99.0


def test_unsorted_input_is_sorted_not_trusted():
    idx = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"])
    data = HistoricCSVDataHandler({"AAA": pd.DataFrame({"close": [3.0, 1.0, 2.0]}, index=idx)})
    seen = []
    while data.update_bars() is not None:
        seen.append(data.latest_price("AAA"))
    assert seen == [1.0, 2.0, 3.0]


# -- Validation -----------------------------------------------------------


def test_duplicate_timestamps_rejected():
    idx = pd.to_datetime(["2020-01-01", "2020-01-01"])
    with pytest.raises(ValueError, match="duplicate"):
        HistoricCSVDataHandler({"AAA": pd.DataFrame({"close": [1.0, 2.0]}, index=idx)})


def test_missing_close_rejected():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    with pytest.raises(ValueError, match="close"):
        HistoricCSVDataHandler({"AAA": pd.DataFrame({"open": [1.0, 2.0]}, index=idx)})


def test_non_datetime_index_rejected():
    with pytest.raises(ValueError, match="DatetimeIndex"):
        HistoricCSVDataHandler({"AAA": pd.DataFrame({"close": [1.0, 2.0]})})


def test_non_numeric_close_rejected():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    with pytest.raises(ValueError, match="numeric"):
        HistoricCSVDataHandler({"AAA": pd.DataFrame({"close": ["a", "b"]}, index=idx)})


def test_non_numeric_columns_are_dropped_not_fatal():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    df = pd.DataFrame({"close": [1.0, 2.0], "note": ["x", "y"]}, index=idx)
    data = HistoricCSVDataHandler({"AAA": df})
    data.update_bars()
    assert "note" not in data.latest_bar("AAA")
    assert data.latest_price("AAA") == 1.0


def test_empty_frames_rejected():
    with pytest.raises(ValueError):
        HistoricCSVDataHandler({})


def test_latest_price_field_selection():
    data = HistoricCSVDataHandler({"AAA": _frame([100.0], open=[98.0])})
    data.update_bars()
    assert data.latest_price("AAA", "open") == 98.0
    with pytest.raises(KeyError):
        data.latest_price("AAA", "nope")


def test_integer_columns_are_readable_as_floats():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    df = pd.DataFrame({"close": [100.0, 101.0], "volume": np.array([5, 6])}, index=idx)
    data = HistoricCSVDataHandler({"AAA": df})
    data.update_bars()
    assert data.latest_bar("AAA")["volume"] == 5.0
