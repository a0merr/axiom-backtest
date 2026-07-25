"""Throughput benchmark for the event loop.

The engine processes bars one at a time on purpose — that is what makes
look-ahead bias structurally impossible rather than merely discouraged. The
cost of that choice is that per-bar overhead is multiplied by the number of
bars, so anything allocated or re-derived inside the loop shows up directly in
how long a parameter sweep takes.

Run::

    python examples/benchmark.py               # default sizes
    python examples/benchmark.py 1000000 10    # 1M bars x 10 symbols

Reported rows/sec is bars x symbols per wall-clock second, including strategy,
sizing, execution and portfolio marking — not just data iteration.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from axiom import (  # noqa: E402
    BacktestEngine,
    HistoricCSVDataHandler,
    MovingAverageCrossover,
    NextBarExecutionHandler,
    Portfolio,
    VolumeShareSlippage,
)


def make_frames(n_bars: int, n_symbols: int, seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-03", periods=n_bars, freq="min")
    frames = {}
    for i in range(n_symbols):
        # Mean-zero, clipped: a drifting random walk over millions of steps
        # overflows float64 long before it says anything about throughput.
        walk = np.cumsum(rng.normal(0.0, 0.01, n_bars))
        close = 100 * np.exp(np.clip(walk - walk.mean(), -3, 3))
        frames[f"S{i}"] = pd.DataFrame(
            {
                "open": close * (1 + rng.normal(0, 0.0005, n_bars)),
                "close": close,
                "volume": rng.integers(50_000, 200_000, n_bars).astype(float),
            },
            index=idx,
        )
    return frames


def run_once(frames: dict[str, pd.DataFrame]) -> Portfolio:
    data = HistoricCSVDataHandler(frames)
    strategy = MovingAverageCrossover(data, fast=20, slow=50)
    portfolio = Portfolio(data, initial_capital=100_000, risk_fraction=0.1)
    execution = NextBarExecutionHandler(data, slippage=VolumeShareSlippage())
    BacktestEngine(data, strategy, portfolio, execution).run()
    return portfolio


def main() -> None:
    n_bars = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    n_symbols = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"building {n_bars:,} bars x {n_symbols} symbols ...")
    frames = make_frames(n_bars, n_symbols)

    start = time.perf_counter()
    portfolio = run_once(frames)
    elapsed = time.perf_counter() - start

    rows = n_bars * n_symbols
    print(f"rows           : {rows:,}")
    print(f"elapsed        : {elapsed:.2f}s")
    print(f"throughput     : {rows / elapsed:,.0f} rows/sec")
    print(f"final equity   : ${portfolio.equity():,.2f}")


if __name__ == "__main__":
    main()
