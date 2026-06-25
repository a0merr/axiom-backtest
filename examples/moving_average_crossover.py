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
    PerShareCommission,
    Portfolio,
    VolumeShareSlippage,
    analyze,
    bootstrap_sharpe,
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
    volume = rng.lognormal(mean=13.8, sigma=0.4, size=n)  # ~1M shares/bar
    return {
        "SYN": pd.DataFrame(
            {"open": open_, "close": close, "volume": volume}, index=idx
        )
    }


def single_backtest(frames: dict) -> None:
    data = HistoricCSVDataHandler(frames)
    strategy = MovingAverageCrossover(data, fast=20, slow=50)
    portfolio = Portfolio(data, initial_capital=100_000.0, risk_fraction=0.5)
    # Realistic timing (next-bar open) + size-aware square-root market impact.
    execution = NextBarExecutionHandler(
        data,
        slippage=VolumeShareSlippage(base_rate=0.0002, impact_coef=0.1),
        commission=PerShareCommission(0.005, 1.0),
    )
    BacktestEngine(data, strategy, portfolio, execution).run()

    equity = portfolio.equity_series()
    report = analyze(equity)
    print("=== Full-sample backtest (net of costs) ===")
    print(report)
    print(f"Commission paid : ${portfolio.total_commission:,.2f}")
    print(f"Slippage paid   : ${portfolio.total_slippage:,.2f}")

    ci = bootstrap_sharpe(equity.pct_change().dropna(), n_boot=2000, seed=0)
    verdict = "significant" if ci.significant else "NOT significant (CI includes 0)"
    print(
        f"Sharpe 95% CI : [{ci.lower:.2f}, {ci.upper:.2f}] "
        f"(point {ci.point:.2f}, p={ci.p_value:.2f}) -> {verdict}"
    )


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
