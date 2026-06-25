# Case study: trend following on real ETFs — significant, but not better than SPY

> **Data.** Split/dividend-adjusted daily bars for five ETFs — SPY (US equity),
> QQQ (US tech), EFA (developed intl), TLT (long Treasuries), GLD (gold) —
> 2010-2024, pulled from Yahoo Finance. Reproduce end to end:
>
> ```bash
> pip install -e ".[data]"
> python examples/fetch_data.py        # writes data/*.csv (gitignored)
> python examples/etf_case_study.py
> ```
>
> A synthetic-data smoke test of the same pipeline lives in
> `examples/moving_average_crossover.py`.

## The strategy

A 50/200 simple-moving-average crossover ("golden cross"), long/flat, run across
all five ETFs simultaneously. Position sizing is **correlation-aware**
(`CorrelationAwareSizer`, 10% vol target): each new position is scaled to a
target volatility and then cut for how correlated it is to what's already held,
so the book doesn't collapse into one equity bet. Execution is realistic —
**next-bar-open fills**, square-root **volume impact** slippage, and per-share
commission.

The question is not "does it make money." It's "does it earn its complexity and
costs versus the dumbest possible alternative — buying and holding SPY?"

## The benchmark

| Metric | Buy & hold SPY |
| --- | --- |
| Total return | 583.8% |
| CAGR | 13.70% |
| Volatility | 17.05% |
| **Sharpe** | **0.84** |
| Sortino | 1.03 |
| Max drawdown | −33.7% |
| Calmar | 0.41 |
| Hit rate | 55.3% |

## The strategy, net of costs

| Metric | 50/200 MA, 5-ETF basket |
| --- | --- |
| Total return | 606.0% |
| CAGR | 13.95% |
| Volatility | 18.97% |
| **Sharpe** | **0.78** |
| Sortino | 0.91 |
| Max drawdown | **−39.5%** |
| Calmar | 0.35 |
| Hit rate | 50.5% |
| Commission paid | \$840 |
| Slippage paid | **\$24,699** |

## What the numbers say — read honestly

**1. The edge is statistically real.** Block-bootstrap (5,000 resamples) on the
strategy's returns:

```
Sharpe point estimate : 0.78
95% confidence interval : [0.30, 1.27]
one-sided p(Sharpe <= 0): 0.000
Verdict                 : significant — interval excludes zero
```

This is the opposite of the synthetic example: the lower bound is comfortably
above zero, so "no edge" is rejected. The strategy genuinely produces a positive
risk-adjusted return stream.

**2. It holds up out of sample.** Ten-fold walk-forward (504-bar train / 126-bar
test, rolling origin, strategy warmed up on each in-sample block so the 200-day
average is live before the OOS window starts):

| Fold | Test window | OOS Sharpe |
| --- | --- | --- |
| 0 | 2012 H1 | 1.03 |
| 1 | 2012 H2 | 0.64 |
| 2 | 2013 H1 | 1.04 |
| 3 | 2013 H2 | 2.86 |
| 4 | 2014 H1 | 1.78 |
| 5 | 2014 H2 | 0.96 |
| 6 | 2015 H1 | 0.06 |
| 7 | 2015 H2 | **−0.76** |
| 8 | 2016 H1 | 0.68 |
| 9 | 2016 H2 | −0.16 |

**Mean OOS Sharpe 0.81, std 1.02.** Mostly positive, and the mean OOS matches
the full-sample Sharpe — no in-sample/out-of-sample collapse. That is what a
non-overfit strategy looks like.

**3. And yet — it does not beat SPY. This is where it breaks.** Despite being
real, significant, and OOS-robust, the strategy is *worse than the one-line
passive benchmark on every risk-adjusted measure*:

- Lower Sharpe (0.78 vs **0.84**) and Sortino (0.91 vs 1.03).
- Deeper max drawdown (−39.5% vs −33.7%) and worse Calmar (0.35 vs 0.41).
- Its 22 percentage-point total-return edge (606% vs 584%) **vanishes** once you
  adjust for the extra volatility it took to get there.

It pays **\$24,699 in slippage** — 29× its commission, and the dominant cost — to
deliver, after all that machinery, slightly *less* risk-adjusted return than
doing nothing. That is the honest verdict: this is **beta dressed up as alpha**.
The trend overlay mostly reconstructs equity-market exposure with extra steps,
extra turnover, and extra cost.

**4. Regime dependence.** The two negative OOS folds are 2015 H2 and 2016 H2 —
choppy, trendless, whipsaw markets where a crossover system bleeds on false
signals. The strong folds (2013-2014) are clean trends. The strategy is a bet on
*trend persistence*, and it underperforms precisely when trends don't persist.

## What would actually justify it

This is the "what I'd need to ship it" section a desk wants:

1. **A real Sharpe edge over the benchmark**, not just over zero. The bar isn't
   `Sharpe > 0` (cleared) — it's `Sharpe > buy-and-hold`, after costs (not
   cleared).
2. **Lower correlation to SPY.** The value of a trend overlay is supposed to be
   crisis convexity / diversification. Measure the strategy's beta to SPY and
   its performance in equity drawdowns; if it just tracks SPY, it has no reason
   to exist alongside it.
3. **Turnover reduction.** \$25k of impact says the signal trades too much for
   the edge it captures. Slower signals, trade-size caps, or a no-trade band
   around the crossover would test whether the edge survives once you stop
   paying the spread so often.
4. **Parameter robustness.** Run `walk_forward` with an `optimizer` that re-fits
   the MA lengths on each in-sample block; if OOS performance degrades versus
   this fixed-parameter run, the 50/200 choice was itself a mild overfit.

## The point

The framework did its job. It didn't just report a profitable-looking equity
curve (606% total return looks great in isolation) — it surfaced, with a
benchmark, honest costs, OOS folds, and a significance test, that the strategy
**makes real money and is still not worth running** over a passive alternative.
Knowing *that* distinction — and being able to prove it — is the actual skill.
