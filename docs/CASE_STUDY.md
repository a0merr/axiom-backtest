# Case study: SMA crossover — a worked example of an honest evaluation

> **What this is.** An end-to-end demonstration of how `axiom-backtest` is meant
> to be used to *judge* a strategy, not to sell one. The data here is
> **synthetic** (geometric Brownian motion, see `examples/moving_average_crossover.py`),
> so the numbers prove nothing about real markets — they exist to show the
> method. Swap `synthetic_prices()` for a real OHLCV/odds loader and the same
> three checks (costs → walk-forward → significance) carry over unchanged.
>
> Reproduce: `python examples/moving_average_crossover.py`.

## The strategy

A 20/50 simple-moving-average crossover, long/flat, on a single instrument.
Sizing is fixed-fractional (50% of equity per entry). It is the canonical
trend-follower and the canonical thing people overfit. The point is not whether
*this* strategy works; it's to walk the evidence a quant desk would actually ask
for.

## Step 1 — does it survive costs?

Frictions modeled: next-bar-open execution (no fill-at-decision-price),
square-root volume impact, per-share commission with a \$1 floor.

| Metric | Value |
| --- | --- |
| Total return | **−21.22%** |
| CAGR | −3.93% |
| Annualized volatility | 6.41% |
| Sharpe | **−0.59** |
| Sortino | −0.68 |
| Max drawdown | −21.93% |
| Calmar | −0.18 |
| Hit rate | 21.0% |
| Commission paid | \$127 |
| Slippage paid | **\$4,190** |

Two things to notice. First, slippage (\$4,190) dwarfs commission (\$127) by ~33×
— the cost that flatters high-turnover backtests is market impact, not the
broker fee, and a flat per-trade slippage assumption would have hidden most of
it. Second, on this synthetic series the strategy simply loses: negative Sharpe,
deep drawdown, low hit rate.

## Step 2 — is the in-sample result even stable out of sample?

Eight-fold walk-forward (378-bar train, 63-bar test, rolling origin). Parameters
are held fixed; this is a pure out-of-sample robustness check.

| Fold | Test window | OOS return | OOS Sharpe |
| --- | --- | --- | --- |
| 0 | 2018-06 → 2018-09 | −0.64% | −4.29 |
| 1 | 2018-09 → 2018-12 | 0.00% | 0.00 |
| 2 | 2018-12 → 2019-03 | 0.00% | 0.00 |
| 3 | 2019-03 → 2019-05 | 0.00% | 0.00 |
| 4 | 2019-06 → 2019-08 | −0.27% | −1.68 |
| 5 | 2019-08 → 2019-11 | 0.00% | 0.00 |
| 6 | 2019-11 → 2020-02 | 0.00% | 0.00 |
| 7 | 2020-02 → 2020-05 | +0.37% | +3.05 |

**Mean OOS Sharpe −0.37, std 2.05.**

This is the more important picture, and it shows the failure mode plainly:

- **Most folds do nothing.** Five of eight windows never trigger a trade — the
  fast/slow MAs don't cross inside a 63-bar window often enough. The strategy is
  *inactive*, not edge-generating, most of the time.
- **The active folds disagree wildly.** Fold 7 posts a +3.05 Sharpe; fold 0
  posts −4.29. A std of 2.05 around a mean of −0.37 means any single fold's
  result is noise. If you had tuned parameters to one good window, you'd have
  "discovered" an edge that the next window erases.

## Step 3 — is the headline Sharpe statistically real?

A point Sharpe is a single draw from a distribution. Block-bootstrap (2,000
resamples, circular blocks preserving autocorrelation):

```
Sharpe point estimate : -0.59
95% confidence interval : [-1.53, 0.28]
one-sided p(Sharpe <= 0): 0.90
Verdict                 : NOT significant — the interval includes zero
```

Even though the point estimate is negative, the interval *crosses zero*: we
cannot reject "no edge" in either direction. The mirror-image trap is the one
that matters for live money — a strategy with a **positive** point Sharpe whose
CI also includes zero is indistinguishable from luck, and shipping it is how
backtested edges evaporate in production.

## What would change the verdict

This is the honest "where it breaks / what I'd need" section a reviewer wants:

1. **Real data with regime variety.** GBM has no momentum, so a trend follower
   *can't* win here by construction — a fair test needs instruments that
   actually trend (and periods where they don't).
2. **More trades per fold.** Five dead folds means the OOS estimate is starved.
   Either shorter MAs, more instruments (see roadmap: multi-asset sizing), or
   longer test windows.
3. **Parameter search inside the walk-forward.** Pass an `optimizer` to
   `walk_forward` so each in-sample block picks its own MA lengths; the
   train-vs-test Sharpe *gap* then directly measures overfitting, rather than
   assuming fixed params.
4. **A significant, positive bootstrap CI.** The bar to clear before risking
   capital: lower bound of the Sharpe CI above zero, on out-of-sample data,
   after costs.

## How to run this on your own strategy

```python
# 1. Replace the data source
frames = your_loader()          # {symbol: DataFrame[open, close, volume, ...]}

# 2. Implement Strategy.on_market to emit SignalEvents from your signal
class YourStrategy(Strategy): ...

# 3. Reuse the exact three checks above:
#    - single_backtest()        -> net-of-cost metrics
#    - walk_forward()           -> OOS stability
#    - bootstrap_sharpe()       -> significance
```

The framework's job is done when it can *kill* a bad strategy cheaply. On this
synthetic example it did: costs, walk-forward, and the bootstrap all point the
same direction, and none of them needed the strategy's author to be honest —
the method is.
