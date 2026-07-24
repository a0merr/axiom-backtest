"""The incremental moving averages must equal the naive ones, exactly.

An O(1) rolling sum is only worth having if it produces the same signals as the
O(window) recompute it replaced. These tests compare against a deliberately
dumb reference implementation over a long random walk, including across the
periodic resync boundary where drift correction kicks in.
"""

import numpy as np
import pandas as pd
import pytest

from axiom import Direction, HistoricCSVDataHandler, MovingAverageCrossover


def _walk(n: int, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range("2000-01-03", periods=n, freq="min")
    return pd.DataFrame({"close": close}, index=idx)


def _reference_signals(prices, fast: int, slow: int):
    """Recompute both averages from scratch on every bar."""
    out = []
    invested = False
    for i in range(len(prices)):
        if i + 1 < slow:
            continue
        window = prices[i + 1 - slow : i + 1]
        fast_ma = window[-fast:].mean()
        slow_ma = window.mean()
        if fast_ma > slow_ma and not invested:
            invested = True
            out.append((i, Direction.LONG))
        elif fast_ma < slow_ma and invested:
            invested = False
            out.append((i, Direction.EXIT))
    return out


def _engine_signals(df, fast: int, slow: int):
    data = HistoricCSVDataHandler({"AAA": df})
    strat = MovingAverageCrossover(data, fast=fast, slow=slow)
    out = []
    i = -1
    while (event := data.update_bars()) is not None:
        i += 1
        for signal in strat.on_market(event):
            out.append((i, signal.direction))
    return out


@pytest.mark.parametrize("fast,slow", [(2, 3), (5, 20), (20, 50)])
def test_incremental_averages_match_naive_recompute(fast, slow):
    df = _walk(3_000)
    assert _engine_signals(df, fast, slow) == _reference_signals(
        df["close"].to_numpy(), fast, slow
    )


def test_signals_still_match_across_the_resync_boundary():
    """_RESYNC_EVERY bars in, the sums are rebuilt; signals must not jump."""
    n = MovingAverageCrossover._RESYNC_EVERY * 2 + 250
    df = _walk(n, seed=5)
    assert _engine_signals(df, 20, 50) == _reference_signals(
        df["close"].to_numpy(), 20, 50
    )


def test_running_sums_stay_accurate_over_a_long_run():
    """Drift in the running sum would show up as a wrong average."""
    n = MovingAverageCrossover._RESYNC_EVERY * 3
    df = _walk(n, seed=3)
    data = HistoricCSVDataHandler({"AAA": df})
    strat = MovingAverageCrossover(data, fast=20, slow=50)
    while (event := data.update_bars()) is not None:
        strat.on_market(event)

    window = np.array(strat._prices["AAA"])
    assert strat._slow_sum["AAA"] == pytest.approx(window.sum(), rel=1e-12)
    assert strat._fast_sum["AAA"] == pytest.approx(window[-20:].sum(), rel=1e-12)


def test_sparse_symbol_does_not_desynchronize_its_window():
    """A symbol with no bar yet must not consume a slot in another's window."""
    idx = pd.date_range("2020-01-01", periods=60, freq="D")
    a = pd.DataFrame({"close": np.linspace(100, 160, 60)}, index=idx)
    b = pd.DataFrame({"close": np.linspace(50, 20, 20)}, index=idx[40:])
    data = HistoricCSVDataHandler({"AAA": a, "BBB": b})
    strat = MovingAverageCrossover(data, fast=2, slow=5)
    while (event := data.update_bars()) is not None:
        strat.on_market(event)

    assert len(strat._prices["AAA"]) == 5
    assert len(strat._prices["BBB"]) == 5
    assert strat._slow_sum["BBB"] == pytest.approx(sum(strat._prices["BBB"]))


def test_fast_must_be_shorter_than_slow():
    data = HistoricCSVDataHandler({"AAA": _walk(10)})
    with pytest.raises(ValueError):
        MovingAverageCrossover(data, fast=50, slow=20)
