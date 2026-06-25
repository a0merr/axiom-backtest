"""End-to-end example: SMA crossover with costs, then walk-forward validation.

Run from the repo root:

    pip install -e .
    python examples/moving_average_crossover.py

It uses synthetic geometric-Brownian-motion prices so the example is
self-contained (no data download). Swap ``synthetic_prices`` for a real OHLCV
loader to evaluate an actual strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from axiom import (
    BacktestEngine,
    HistoricCSVDataHandler,
    MovingAverageCrossover,
    NextBarExecutionHandler,
    PercentageSlippage,
    PerShareCommission,
    Portfolio,
    analyze,
    walk_forward,
)
from axiom.validation import stitch_oos_equity


def synthetic_prices(n: int = 1500, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2017-01-01", periods=n, freq="B")
    # GBM with small positive drift and regime noise.
    rets = rng.normal(0.0003, 0.012, n)
    close = 100 * np.exp(np.cumsum(rets))
    # Next bar's open gaps slightly from the prior close (overnight move).
    open_ = np.concatenate([[close[0]], close[:-1] * (1 + rng.normal(0, 0.002, n - 1))])
    return {"SYN": pd.DataFrame({"open": open_, "close": close}, index=idx)}


def single_backtest(frames: dict) -> None:
    data = HistoricCSVDataHandler(frames)
    strategy = MovingAverageCrossover(data, fast=20, slow=50)
    portfolio = Portfolio(data, initial_capital=100_000.0, risk_fraction=0.5)
    # Realistic timing: a signal on bar t fills at bar t+1's open, not t's close.
    execution = NextBarExecutionHandler(
        data,
        slippage=PercentageSlippage(0.0005),
        commission=PerShareCommission(0.005, 1.0),
    )
    BacktestEngine(data, strategy, portfolio, execution).run()

    report = analyze(portfolio.equity_series())
    print("=== Full-sample backtest (net of costs) ===")
    print(report)
    print(f"Commission paid : ${portfolio.total_commission:,.2f}")
    print(f"Slippage paid   : ${portfolio.total_slippage:,.2f}")


def walk_forward_validation(frames: dict) -> None:
    def factory(data, params):
        return MovingAverageCrossover(data, fast=params["fast"], slow=params["slow"])

    windows = walk_forward(
        frames,
        factory,
        n_splits=8,
        train_size=378,
        test_size=63,
        base_params={"fast": 20, "slow": 50},
    )
    table = stitch_oos_equity(windows)
    print("\n=== Walk-forward out-of-sample folds ===")
    cols = ["fold", "test_start", "test_end", "total_return", "sharpe", "max_drawdown"]
    print(table[cols].to_string(index=False))
    print(
        f"\nMean OOS Sharpe : {table['sharpe'].mean():.2f}"
        f"  | OOS Sharpe std : {table['sharpe'].std():.2f}"
    )
    print(
        "Read this honestly: a high full-sample Sharpe with a low/negative mean "
        "OOS Sharpe is the signature of overfitting."
    )


if __name__ == "__main__":
    frames = synthetic_prices()
    single_backtest(frames)
    walk_forward_validation(frames)
