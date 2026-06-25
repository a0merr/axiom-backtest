"""Real-data case study: trend following across a diversified ETF basket.

Loads the CSVs written by ``fetch_data.py`` and runs a 50/200 moving-average
crossover across five ETFs with the full realism stack — next-bar-open fills,
square-root volume impact, per-share commission, and correlation-aware position
sizing — then validates out of sample and tests Sharpe significance.

    pip install -e ".[data]"
    python examples/fetch_data.py
    python examples/etf_case_study.py
"""

from __future__ import annotations

import pathlib

import pandas as pd

from axiom import (
    BacktestEngine,
    CorrelationAwareSizer,
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
from axiom.validation import oos_summary

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
TICKERS = ["SPY", "TLT", "GLD", "QQQ", "EFA"]
FAST, SLOW = 50, 200


def load_frames() -> dict:
    frames = {}
    for ticker in TICKERS:
        path = DATA_DIR / f"{ticker}.csv"
        if not path.exists():
            raise SystemExit(f"missing {path}; run examples/fetch_data.py first")
        df = pd.read_csv(path, index_col="date", parse_dates=True)
        frames[ticker] = df[["open", "close", "volume"]]
    return frames


def buy_and_hold_spy(frames: dict) -> None:
    spy = frames["SPY"]["close"]
    report = analyze(spy / spy.iloc[0] * 100_000.0)
    print("=== Benchmark: buy & hold SPY ===")
    print(report)


def strategy_backtest(frames: dict) -> None:
    data = HistoricCSVDataHandler(frames)
    strategy = MovingAverageCrossover(data, fast=FAST, slow=SLOW)
    portfolio = Portfolio(
        data,
        initial_capital=100_000.0,
        sizer=CorrelationAwareSizer(
            target_vol=0.10, lookback=63, correlation_penalty=1.0
        ),
    )
    execution = NextBarExecutionHandler(
        data,
        slippage=VolumeShareSlippage(base_rate=0.0002, impact_coef=0.1),
        commission=PerShareCommission(0.005, 1.0),
    )
    BacktestEngine(data, strategy, portfolio, execution).run()

    equity = portfolio.equity_series()
    print(f"\n=== {FAST}/{SLOW} MA crossover, 5-ETF basket (net of costs) ===")
    print(analyze(equity))
    print(f"Commission paid : ${portfolio.total_commission:,.2f}")
    print(f"Slippage paid   : ${portfolio.total_slippage:,.2f}")

    ci = bootstrap_sharpe(equity.pct_change().dropna(), n_boot=5000, seed=0)
    verdict = "significant" if ci.significant else "NOT significant (CI includes 0)"
    print(
        f"Sharpe 95% CI : [{ci.lower:.2f}, {ci.upper:.2f}] "
        f"(point {ci.point:.2f}, p={ci.p_value:.3f}) -> {verdict}"
    )


def walk_forward_validation(frames: dict) -> None:
    def factory(data, params):
        return MovingAverageCrossover(data, fast=params["fast"], slow=params["slow"])

    windows = walk_forward(
        frames,
        factory,
        n_splits=10,
        train_size=504,  # ~2y
        test_size=126,  # ~6m
        base_params={"fast": FAST, "slow": SLOW},
    )
    table = oos_summary(windows)
    cols = ["fold", "test_start", "test_end", "total_return", "sharpe", "max_drawdown"]
    print("\n=== Walk-forward out-of-sample folds ===")
    print(table[cols].to_string(index=False))
    print(
        f"\nMean OOS Sharpe : {table['sharpe'].mean():.2f}"
        f"  | OOS Sharpe std : {table['sharpe'].std():.2f}"
    )


if __name__ == "__main__":
    frames = load_frames()
    buy_and_hold_spy(frames)
    strategy_backtest(frames)
    walk_forward_validation(frames)
