"""Walk-forward / out-of-sample validation.

A single backtest over all history is the easiest way to fool yourself: you
tune parameters until the one curve looks good. Walk-forward analysis splits the
timeline into consecutive (in-sample, out-of-sample) windows, optionally
re-fitting parameters on each in-sample block, and reports performance on the
*concatenated out-of-sample* segments only. That OOS curve is the number worth
trusting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from .data import HistoricCSVDataHandler
from .engine import BacktestEngine
from .execution import SimulatedExecutionHandler
from .metrics import PerformanceReport, analyze
from .portfolio import Portfolio
from .strategy import Strategy

# A factory builds a fresh strategy bound to a data handler, given parameters.
StrategyFactory = Callable[[HistoricCSVDataHandler, dict], Strategy]


@dataclass
class WalkForwardWindow:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    params: dict
    report: PerformanceReport


def _slice(frames: dict[str, pd.DataFrame], start, end) -> dict[str, pd.DataFrame]:
    return {s: df.loc[start:end] for s, df in frames.items()}


def walk_forward(
    frames: dict[str, pd.DataFrame],
    strategy_factory: StrategyFactory,
    *,
    n_splits: int = 5,
    train_size: int = 252,
    test_size: int = 63,
    optimizer: Callable[[dict[str, pd.DataFrame]], dict] | None = None,
    base_params: dict | None = None,
    initial_capital: float = 100_000.0,
    periods_per_year: int = 252,
) -> list[WalkForwardWindow]:
    """Run rolling-origin walk-forward analysis.

    ``optimizer`` (if given) receives the in-sample frames and returns the best
    parameters for the next out-of-sample block — this is where overfitting
    would show up as a train/test performance gap. Without it, ``base_params``
    are used unchanged (pure OOS robustness check).
    """
    base_params = base_params or {}
    # Union timeline drives the window boundaries.
    index = None
    for df in frames.values():
        index = df.index if index is None else index.union(df.index)
    assert index is not None  # guaranteed: frames is non-empty
    timeline = list(index)

    windows: list[WalkForwardWindow] = []
    step = test_size
    start = 0
    fold = 0
    while fold < n_splits:
        train_lo = start
        train_hi = start + train_size
        test_hi = train_hi + test_size
        if test_hi > len(timeline):
            break

        train_frames = _slice(frames, timeline[train_lo], timeline[train_hi - 1])
        test_frames = _slice(frames, timeline[train_hi], timeline[test_hi - 1])

        params = dict(base_params)
        if optimizer is not None:
            params.update(optimizer(train_frames))

        data = HistoricCSVDataHandler(test_frames)
        strategy = strategy_factory(data, params)
        portfolio = Portfolio(data, initial_capital=initial_capital)
        execution = SimulatedExecutionHandler(data)
        BacktestEngine(data, strategy, portfolio, execution).run()

        report = analyze(
            portfolio.equity_series(), periods_per_year=periods_per_year
        )
        windows.append(
            WalkForwardWindow(
                fold=fold,
                train_start=timeline[train_lo],
                train_end=timeline[train_hi - 1],
                test_start=timeline[train_hi],
                test_end=timeline[test_hi - 1],
                params=params,
                report=report,
            )
        )
        fold += 1
        start += step
    return windows


def stitch_oos_equity(windows: list[WalkForwardWindow]) -> pd.DataFrame:
    """Summarize per-fold OOS results into a comparison table."""
    rows = []
    for w in windows:
        rows.append(
            {
                "fold": w.fold,
                "test_start": w.test_start,
                "test_end": w.test_end,
                **w.report.as_dict(),
                "params": w.params,
            }
        )
    return pd.DataFrame(rows)
