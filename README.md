# axiom-backtest

[![ci](https://github.com/a0merr/axiom-backtest/actions/workflows/ci.yml/badge.svg)](https://github.com/a0merr/axiom-backtest/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![license: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An event-driven backtesting framework for systematic trading strategies — built
to answer one question honestly: **would this strategy have survived out of
sample, after costs?**

Anyone can curve-fit a profitable equity curve. The harder, more valuable skill
is knowing *when a strategy is overfit and why*. This library is engineered
around that idea: realistic frictions, strict no-look-ahead data flow, and
walk-forward validation as a first-class feature rather than an afterthought.

> Status: `v0.1.0`. Core engine, costs/slippage, metrics, and walk-forward are
> implemented and tested. See [Roadmap](#roadmap).

---

## Why event-driven?

A vectorized backtest computes signals over the whole price series at once,
which makes look-ahead bias easy to introduce and hard to spot. This engine
processes one bar at a time through a queue of typed events:

```
MarketEvent ──> Strategy ──> SignalEvent ──> Portfolio ──> OrderEvent ──> Execution ──> FillEvent ──> Portfolio
```

A strategy can only ever see data up to "now" because future bars have not been
yielded yet. The same loop runs identically in a backtest and (with a live data
handler and broker execution handler) in production — the path to deployment is
swapping two components, not rewriting the strategy.

## What's modeled honestly

| Concern | How |
| --- | --- |
| **Transaction costs** | Per-share commission with a per-order minimum (`PerShareCommission`). |
| **Short financing** | `Portfolio(annual_borrow_rate=…)` accrues per-bar borrow cost on short notional — shorting isn't free. |
| **Slippage** | Three models: fixed (`PercentageSlippage`), size-aware square-root impact (`VolumeShareSlippage`), and regime-aware (`VolatilitySlippage`). Bigger orders relative to volume pay disproportionately more. |
| **Look-ahead bias** | Data handler exposes only `asof`-resolved bars. `NextBarExecutionHandler` fills a signal from bar *t* at bar *t+1*'s open — you can't trade on a price you used to decide. |
| **Execution timing** | Two models: `SimulatedExecutionHandler` (same-bar close, simple) and `NextBarExecutionHandler` (next-bar open, realistic). |
| **Out-of-sample validation** | Rolling-origin `walk_forward` reports performance on concatenated OOS windows only. |
| **Risk metrics** | Sharpe, Sortino, max drawdown, Calmar, CAGR, hit rate — annualized with an explicit risk-free rate. |
| **Statistical significance** | `bootstrap_sharpe` block-resamples returns for a Sharpe confidence interval + p-value. A high Sharpe whose CI includes zero is not evidence of edge. |

Costs are surfaced separately (`portfolio.total_commission`,
`portfolio.total_slippage`) so P&L attribution is auditable, not buried.

## Install

```bash
git clone https://github.com/a0merr/axiom-backtest.git
cd axiom-backtest
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Quickstart

```python
import numpy as np, pandas as pd
from axiom import (
    HistoricCSVDataHandler, MovingAverageCrossover, Portfolio,
    SimulatedExecutionHandler, PercentageSlippage, PerShareCommission,
    BacktestEngine, analyze,
)

idx = pd.date_range("2018-01-01", periods=1000, freq="B")
prices = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(3e-4, 0.012, 1000)))
frames = {"SYN": pd.DataFrame({"close": prices}, index=idx)}

data = HistoricCSVDataHandler(frames)
strategy = MovingAverageCrossover(data, fast=20, slow=50)
portfolio = Portfolio(data, initial_capital=100_000, risk_fraction=0.5)
execution = SimulatedExecutionHandler(
    data, slippage=PercentageSlippage(0.0005), commission=PerShareCommission(0.005, 1.0)
)

BacktestEngine(data, strategy, portfolio, execution).run()
print(analyze(portfolio.equity_series()))
```

Full runnable demo (single backtest **and** walk-forward):

```bash
python examples/moving_average_crossover.py
```

## Walk-forward validation

```python
from axiom import walk_forward, MovingAverageCrossover
from axiom.validation import stitch_oos_equity

def factory(data, params):
    return MovingAverageCrossover(data, fast=params["fast"], slow=params["slow"])

windows = walk_forward(
    frames, factory,
    n_splits=8, train_size=378, test_size=63,
    base_params={"fast": 20, "slow": 50},
    # optimizer=my_param_search,   # re-fit on each in-sample block
)
print(stitch_oos_equity(windows))
```

Pass an `optimizer` to re-fit parameters on each in-sample block. The gap
between in-sample and out-of-sample performance *is* the overfitting measurement
— that gap is the headline number, not the full-sample Sharpe.

## Architecture

```
src/axiom/
  event.py        # typed events: Market/Signal/Order/Fill
  data.py         # DataHandler: streams bars, no look-ahead
  strategy.py     # Strategy base + MovingAverageCrossover reference
  portfolio.py    # sizing, order generation, equity tracking
  execution.py    # slippage + commission models -> fills
  engine.py       # the event loop
  metrics.py      # Sharpe, Sortino, drawdown, Calmar, hit rate
  validation.py   # walk-forward / OOS analysis
```

Each component is an abstract base with a reference implementation, so swapping
in a live data feed, a real broker, or a new sizing rule is a subclass — not a
fork.

## Tests

```bash
pytest                      # unit + integration
pytest --cov=axiom          # with coverage
ruff check src tests        # lint
mypy src/axiom              # type check
```

The suite covers metric correctness against closed-form formulas, cost/slippage
direction, no-look-ahead invariants, and walk-forward fold construction.

## Case study

[**docs/CASE_STUDY.md**](docs/CASE_STUDY.md) walks a full evaluation of the SMA
crossover end to end — costs → walk-forward → bootstrap significance — and reads
the result honestly, including the **where it breaks** part: most OOS folds
never trade, the active folds disagree (mean OOS Sharpe −0.37, std 2.05), and
the Sharpe 95% CI [−1.53, 0.28] crosses zero, so there is no statistically real
edge. It uses synthetic data as a methodology template; the same three checks
apply unchanged once you plug in a real strategy and dataset.

## Roadmap

- [x] Next-bar-open execution handler (remove current-close optimism)
- [x] Volume/volatility-aware slippage model
- [ ] Multi-asset portfolio with correlation-aware sizing
- [x] Borrow costs and financing for shorts
- [x] Strategy case-study writeup ([docs/CASE_STUDY.md](docs/CASE_STUDY.md); synthetic demo — real-data version pending)
- [x] Bootstrapped/Monte-Carlo confidence intervals on Sharpe

## License

MIT — see [LICENSE](LICENSE).
