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
| **Margin interest** | `annual_margin_rate` charges a debit cash balance. Shorts were already financed; long leverage no longer rides free. |
| **Ruin** | `ruin_equity` closes the account when equity hits zero (or your hard-stop level): positions liquidated, trading stopped, curve flat. Reported as a *date*, not a return past −100%. |
| **Leverage** | `max_gross_leverage` bounds gross exposure. `risk_fraction` allocates per symbol, so N symbols silently sum to N× the account without it. |
| **Slippage** | Four models, in increasing order of honesty: fixed (`PercentageSlippage`), regime-aware (`VolatilitySlippage`), square-root impact (`VolumeShareSlippage`), and depth-aware (`OrderBookSlippage`). Only the last one models running *out* of liquidity — see [Market microstructure](#market-microstructure). |
| **Liquidity limits** | `max_participation` caps an order at a fraction of the bar's volume. `NextBarExecutionHandler` works the remainder across later bars and cancels what it can't fill, reporting it in `unfilled_quantity` — because size you can't execute isn't a position. |
| **Look-ahead bias** | Data handler exposes only `asof`-resolved bars. `NextBarExecutionHandler` fills a signal from bar *t* at bar *t+1*'s open — you can't trade on a price you used to decide. |
| **Execution timing** | Two models: `SimulatedExecutionHandler` (same-bar close, simple) and `NextBarExecutionHandler` (next-bar open, realistic). |
| **Out-of-sample validation** | Rolling-origin `walk_forward` reports performance on concatenated OOS windows only. |
| **Risk metrics** | Sharpe, Sortino, max drawdown, Calmar, CAGR, hit rate — annualized with an explicit risk-free rate. |
| **Throughput** | ~1M rows/sec on a single core, so a 10M-row sweep is seconds rather than half an hour — see [Performance](#performance). |
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
from axiom.validation import oos_summary, stitch_oos_equity

def factory(data, params):
    return MovingAverageCrossover(data, fast=params["fast"], slow=params["slow"])

windows = walk_forward(
    frames, factory,
    n_splits=8, train_size=378, test_size=63,
    base_params={"fast": 20, "slow": 50},
    # optimizer=my_param_search,   # re-fit on each in-sample block
)
print(oos_summary(windows))           # per-fold OOS metrics table
print(stitch_oos_equity(windows))     # one continuous out-of-sample equity curve
```

Pass an `optimizer` to re-fit parameters on each in-sample block. The gap
between in-sample and out-of-sample performance *is* the overfitting measurement
— that gap is the headline number, not the full-sample Sharpe.

## Market microstructure

Slippage models are ranked by what they are *able* to tell you. Calibrate them
to charge the same on a small order and they still disagree violently on a
large one, because they disagree about shape:

| Order size (% of bar volume) | `flat` | `volatility` | `sqrt impact` | `order book` |
| --- | --- | --- | --- | --- |
| 1% (calibration point) | 5bp | 5bp | 5bp | 5bp |
| 10% | 5bp | 5bp | 14bp | **81bp** |
| 50% | 5bp | 5bp | 29bp | **849bp** |
| 50%, liquidity collapse | 5bp | 17bp | 29bp | **3,846bp** |

`OrderBookSlippage` treats liquidity as a book of finite depth: `depth_share` of
each bar's volume rests across a `depth_range` of price. Consuming part of that
book walks the price linearly; consuming *more than* the book means lifting a
tail `tail_factor` times thinner, and cost turns convex. That elbow — the size
at which a strategy stops being executable — is exactly what fixed and
square-root models are structurally unable to represent.

Set `reference_vol` and the book responds to stress: a bar at `r` times normal
volatility widens its spread and thins its depth together, so cost rises faster
than volatility does. This is the regime where flat slippage is most flattering
and most wrong.

```python
from axiom import OrderBookSlippage, NextBarExecutionHandler

execution = NextBarExecutionHandler(
    data,
    slippage=OrderBookSlippage(
        half_spread=0.0001,
        depth_share=0.05,      # 5% of bar volume rests in the book
        depth_range=0.004,     # spread across 40bp of price
        tail_factor=5.0,       # past the book, liquidity is 5x thinner
        reference_vol=0.01,    # enables spread/depth collapse under stress
    ),
    max_participation=0.10,    # never take more than 10% of a bar
    max_working_bars=5,        # work the remainder, then cancel it
)
```

Paying more for size is only half of illiquidity; the other half is not getting
filled. `max_participation` caps each fill, `NextBarExecutionHandler` works the
remainder across subsequent bars (picking up genuine timing risk), and anything
still unfilled after `max_working_bars` is cancelled and reported in
`execution.unfilled_quantity` rather than dropped silently.

```bash
python examples/liquidity_stress.py
```

Same strategy, same data, same trades — at \$50M the flat model reports a 0.47
Sharpe and the order-book model reports −0.89.

## Risk guards

Three constraints stop the engine reporting results for situations that cannot
occur. All are opt-in except ruin, which defaults to the only threshold that is
never wrong: zero.

```python
portfolio = Portfolio(
    data,
    initial_capital=100_000,
    risk_fraction=0.5,
    max_gross_leverage=1.0,    # long + short notional, as a multiple of equity
    annual_margin_rate=0.06,   # interest on a debit cash balance
    ruin_equity=0.0,           # raise it to model a hard stop, e.g. 25_000
)
```

**Ruin.** An account that reaches zero is closed: positions liquidated, trading
stopped, equity curve flat from that bar. Without it the engine keeps trading a
negative balance, and every metric downstream inherits the fiction — returns
invert sign across zero (recovering from −10 to −5 reads as a 50% *loss*), so
Sharpe is computed on noise. `returns_from_equity` now truncates at the first
non-positive point regardless of where the curve came from.

The output changes from a number to a date, which is the honest answer:

```
  account |         flat |   volatility |  sqrt impact |   order book
    $50M  |        40.9% |        39.9% |        24.0% |       RUINED
   $200M  |        40.9% |        39.9% |         8.5% |       RUINED

    $50M / order book   account closed 2021-07-19
   $200M / order book   account closed 2018-05-21
```

**Leverage.** `risk_fraction` allocates a share of equity *per symbol*, so ten
symbols at 0.5 is 5x gross and nothing said so. `max_gross_leverage` caps it,
scaling positions down to fit rather than rejecting outright. Commitments made
earlier on the same bar count against the limit — under deferred execution
several signals are sized against the same untouched equity, and each would
otherwise pass the check individually while breaching it collectively.

**Sizing drift.** Positions are sized at the last price; with next-bar
execution they fill at a price that does not exist yet. That gap is unavoidable
(using the fill price would be look-ahead) but it should be *bounded*: across a
gap open, a position sized for 10% of equity can land at 15%, and the backtest
reports a risk profile the strategy never chose. `OrderEvent.reference_price`
carries the sizing price, and `max_price_drift` cancels fills that have run too
far from it — a marketable limit rather than a naked market order. Only adverse
moves cancel; discarding favorable ones would keep only the bad fills.

```python
execution = NextBarExecutionHandler(data, max_price_drift=0.05)
# ... execution.cancelled_on_drift
```

## Performance

The engine is event-driven by design: one bar at a time, so look-ahead bias is
structurally impossible rather than merely discouraged. That makes per-bar
overhead the whole performance story, since it is multiplied by every bar.

Alignment is therefore computed **once**, vectorized, at construction — each
symbol's row index for every point on the timeline — and the loop then reads
from a contiguous `float64` matrix with no pandas calls and no per-bar
allocation. Indicators keep running sums instead of recomputing windows, and the
equity curve is stored as typed column arrays rather than a dict per bar.

Measured on identical workloads (50k bars x 10 symbols, Python 3.14, single core):

| | rows/sec | 500k rows |
| --- | --- | --- |
| Per-bar pandas lookups | 6,877 | 72.7s |
| Vectorized alignment | 864,332 | **0.58s** |

**126x**, with a bit-identical equity curve. At scale: 10M rows (1M bars x 10
symbols) runs in **8.5s** using ~1.1GB. A single-symbol 10M-bar series takes 23s
and ~2.7GB — the timeline's boxed timestamps dominate memory when bars, rather
than symbols, are what you have a lot of.

```bash
python examples/benchmark.py 1000000 10
```

## Architecture

```
src/axiom/
  event.py        # typed events: Market/Signal/Order/Fill
  data.py         # DataHandler: streams bars, no look-ahead, numpy-backed
  strategy.py     # Strategy base + MovingAverageCrossover reference
  portfolio.py    # order generation, cash/positions, equity, borrow costs
  sizing.py       # Sizer: fixed / vol-target / correlation-aware
  execution.py    # slippage + commission models, participation caps -> fills
  engine.py       # the event loop
  metrics.py      # Sharpe/Sortino/drawdown/Calmar + bootstrap significance
  validation.py   # walk-forward / OOS analysis
```

Position sizing is pluggable via `Portfolio(sizer=...)`:
`FixedFractionalSizer` (default), `VolatilityTargetSizer` (size to a target
annualized vol), and `CorrelationAwareSizer` (vol targeting plus a
diversification haircut against correlated holdings).

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
direction, no-look-ahead invariants, and walk-forward fold construction. The
optimized paths are pinned by equivalence tests: the incremental moving averages
are checked against a naive recompute bar for bar, and the data layer's
carry-forward and visibility rules are tested directly.

## Case study

[**docs/CASE_STUDY.md**](docs/CASE_STUDY.md) evaluates a 50/200 trend strategy
across five real ETFs (SPY/QQQ/EFA/TLT/GLD, 2010-2024) with the full stack —
next-bar fills, volume-impact slippage, correlation-aware sizing, walk-forward,
and bootstrap significance — benchmarked against buy-and-hold SPY.

The honest result: the strategy's Sharpe is **statistically significant** (0.78,
95% CI [0.30, 1.27], p≈0.000) and **OOS-robust** (mean OOS Sharpe 0.81 over 10
folds, no in-sample collapse) — yet it **still loses to simply holding SPY**
(0.84 Sharpe, shallower drawdown) after paying \$24.7k in slippage. Real money,
real significance, and *still not worth running*: beta dressed up as alpha. Being
able to prove that distinction is the point.

Reproduce with `pip install -e ".[data]"`, then `python examples/fetch_data.py`
and `python examples/etf_case_study.py`.

## Roadmap

- [x] Next-bar-open execution handler (remove current-close optimism)
- [x] Volume/volatility-aware slippage model
- [x] Multi-asset portfolio with correlation-aware sizing
- [x] Borrow costs and financing for shorts
- [x] Strategy case-study writeup on real data ([docs/CASE_STUDY.md](docs/CASE_STUDY.md))
- [x] Bootstrapped/Monte-Carlo confidence intervals on Sharpe
- [x] Depth-aware order-book slippage with liquidity-collapse response
- [x] Participation caps, partial fills, and worked orders
- [x] Vectorized bar alignment (126x throughput, ~1M rows/sec)
- [x] Ruin check, gross leverage cap, and margin interest on debit balances
- [ ] Per-trade round-trip log (turnover, holding period, per-trade P&L)
- [ ] Staleness bound on carried-forward bars (halted/delisted symbols)
- [ ] Survivorship-bias handling via a listing/delisting calendar
- [ ] Limit and stop orders (`OrderEvent.order_type` is currently unused)
- [ ] Parallel walk-forward folds
- [ ] Intraday and tick-level data handlers

## License

MIT — see [LICENSE](LICENSE).
