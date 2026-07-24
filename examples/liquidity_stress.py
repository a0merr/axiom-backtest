"""What your slippage model is hiding from you.

Four slippage models, **calibrated to agree** at a small order size, then asked
what a large one costs. Calibration matters: comparing uncalibrated models only
shows that they have different constants. Pinning them together at 1% of volume
isolates the thing that actually differs, which is their *shape*:

* ``PercentageSlippage``  — flat. Cost per share never changes. Size is free.
* ``VolatilitySlippage``  — tracks the regime, ignores your size.
* ``VolumeShareSlippage`` — square-root impact: cost grows with size, but ever
  more slowly per share. There is no size it refuses to price.
* ``OrderBookSlippage``   — finite depth. Inside the book, cost is linear in
  size; past it, convex. This is the only one with an elbow.

The data contains a liquidity collapse — a stretch where volume drops 80% and
volatility quadruples, the way it does in every real crisis. Three of these
four models price that window exactly as they price a calm one.

Run::

    python examples/liquidity_stress.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from axiom import (  # noqa: E402
    BacktestEngine,
    Direction,
    HistoricCSVDataHandler,
    MovingAverageCrossover,
    NextBarExecutionHandler,
    OrderBookSlippage,
    PercentageSlippage,
    PerShareCommission,
    Portfolio,
    SlippageModel,
    VolatilitySlippage,
    VolumeShareSlippage,
    analyze,
)

N_BARS = 1_500
CRISIS = slice(900, 1_000)
BASE_VOLUME = 400_000.0
NORMAL_VOL = 0.010
CRISIS_VOL_MULT = 4.0
CRISIS_VOLUME_MULT = 0.2

# All four models are tuned to charge ~5bp on an order worth 1% of a bar's
# volume in a normal regime, so the tables below compare curvature, not levels.
CALIBRATION_PARTICIPATION = 0.01


def build_frames(seed: int = 42) -> dict[str, pd.DataFrame]:
    """A tradeable uptrend interrupted by a liquidity collapse."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=N_BARS, freq="B")

    returns = rng.normal(1.2e-3, NORMAL_VOL, N_BARS)
    volume = rng.normal(BASE_VOLUME, BASE_VOLUME * 0.15, N_BARS).clip(min=50_000)
    vol_feature = np.full(N_BARS, NORMAL_VOL)

    n_crisis = CRISIS.stop - CRISIS.start
    returns[CRISIS] = rng.normal(-3e-3, NORMAL_VOL * CRISIS_VOL_MULT, n_crisis)
    volume[CRISIS] *= CRISIS_VOLUME_MULT
    vol_feature[CRISIS] = NORMAL_VOL * CRISIS_VOL_MULT

    close = 100 * np.exp(np.cumsum(returns))
    frame = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 3e-4, N_BARS)),
            "close": close,
            "volume": volume,
            "volatility": vol_feature,
        },
        index=idx,
    )
    return {"SYN": frame}


def models() -> dict[str, SlippageModel]:
    return {
        "flat": PercentageSlippage(0.0005),
        "volatility": VolatilitySlippage(k=0.04, floor=0.0001),
        "sqrt impact": VolumeShareSlippage(base_rate=0.0001, impact_coef=0.004),
        "order book": OrderBookSlippage(
            half_spread=0.0001,
            depth_share=0.05,   # 5% of a bar's volume rests in the book
            depth_range=0.004,  # spread across 40bp of price
            tail_factor=5.0,    # liquidity past the book is 5x thinner
            reference_vol=NORMAL_VOL,
            spread_vol_coef=1.0,
            depth_vol_coef=1.0,
        ),
    }


def cost_bps(model: SlippageModel, participation: float, stressed: bool) -> float:
    """Slippage in basis points for an order worth ``participation`` of a bar."""
    volume = BASE_VOLUME * (CRISIS_VOLUME_MULT if stressed else 1.0)
    vol = NORMAL_VOL * (CRISIS_VOL_MULT if stressed else 1.0)
    bar = pd.Series({"close": 100.0, "volume": volume, "volatility": vol})
    quantity = participation * volume
    return (model.fill_price(100.0, Direction.LONG, quantity, bar) / 100.0 - 1.0) * 1e4


def print_cost_curve(stressed: bool) -> None:
    regime = "CRISIS (volume -80%, volatility 4x)" if stressed else "NORMAL"
    print(f"--- cost in bps, {regime} ---\n")
    header = f"{'% of bar volume':>16} | " + " | ".join(f"{n:>12}" for n in models())
    print(header)
    print("-" * len(header))
    for participation in (0.001, 0.01, 0.05, 0.10, 0.25, 0.50):
        cells = [
            f"{cost_bps(m, participation, stressed):>12,.0f}"
            for m in models().values()
        ]
        print(f"{participation:>15.1%} | " + " | ".join(cells))
    print()


def run(frames, slippage: SlippageModel, capital: float, cap: float | None):
    data = HistoricCSVDataHandler(frames)
    strategy = MovingAverageCrossover(data, fast=20, slow=50)
    portfolio = Portfolio(data, initial_capital=capital, risk_fraction=1.0)
    execution = NextBarExecutionHandler(
        data,
        slippage=slippage,
        commission=PerShareCommission(0.005, 1.0),
        max_participation=cap,
    )
    BacktestEngine(data, strategy, portfolio, execution).run()
    return portfolio, execution


def gross_return(frames, capital: float) -> float:
    portfolio, _ = run(frames, PercentageSlippage(0.0), capital, cap=None)
    return portfolio.equity_series().iloc[-1] / capital - 1.0


def main() -> None:
    frames = build_frames()

    print("Strategy: 20/50 MA crossover, fully invested, next-bar-open fills.")
    print(f"Market:   ~{BASE_VOLUME:,.0f} shares/bar around $100, with a")
    print("          100-bar liquidity collapse mid-sample.")
    print(f"\nAll models charge ~5bp at {CALIBRATION_PARTICIPATION:.0%} of volume.")
    print("Only their behaviour away from that point differs.\n")

    print("=" * 76)
    print("1. THE COST CURVES")
    print("=" * 76 + "\n")
    print_cost_curve(stressed=False)
    print_cost_curve(stressed=True)
    print("Read the last two rows. A flat model says a half-volume order in a")
    print("collapsing market costs 5bp — the same as a tiny one on a calm day.")
    print("The order-book model prices the book being gone.\n")

    print("=" * 76)
    print("2. WHAT THAT DOES TO A STRATEGY")
    print("=" * 76 + "\n")

    # $100k is ~0.25% of a normal bar — below the calibration point, where all
    # four models should still agree. Divergence starts when size does.
    sizes = {"$100k": 1e5, "$1M": 1e6, "$10M": 1e7, "$50M": 5e7, "$200M": 2e8}
    print(f"Gross return, before any costs: {gross_return(frames, 1e6):>6.1%}\n")

    header = f"{'account':>9} | " + " | ".join(f"{n:>12}" for n in models())
    print(header)
    print("-" * len(header))
    deaths: list[str] = []
    for label, capital in sizes.items():
        cells = []
        for name, slippage in models().items():
            portfolio, _ = run(frames, slippage, capital, cap=None)
            total = portfolio.equity_series().iloc[-1] / capital - 1.0
            if portfolio.is_ruined:
                cells.append(f"{'RUINED':>11}")
                deaths.append(
                    f"  {label:>6} / {name:<12} account closed "
                    f"{portfolio.ruin_timestamp:%Y-%m-%d}"
                )
            else:
                cells.append(f"{total:>11.1%}")
        print(f"{label:>9} | " + " | ".join(f"{c:>12}" for c in cells))

    print("\nAt $100k every model agrees — that is the calibration working. From")
    print("there the flat and volatility models never move again: they are")
    print("structurally incapable of telling you that you are too big. The")
    print("square-root model degrades gracefully. The order-book model finds the")
    print("size at which this strategy stops existing.")

    if deaths:
        print("\nRUINED means equity reached zero and the portfolio closed the")
        print("account: positions liquidated, trading stopped, curve flat from")
        print("there. The date is the answer that matters — a percentage return")
        print("on an account that no longer exists is not a number worth quoting.\n")
        for line in deaths:
            print(line)
    print()

    print("=" * 76)
    print("3. THE OTHER HALF: FILLS YOU DON'T GET")
    print("=" * 76 + "\n")

    book = models()["order book"]
    uncapped, _ = run(frames, book, 5e7, cap=None)
    capped, execution = run(frames, book, 5e7, cap=0.10)

    print("Paying more for size is only half of illiquidity. The other half is")
    print("that the fill simply is not there. With max_participation=10%:\n")
    uncapped_state = (
        f"ruined {uncapped.ruin_timestamp:%Y-%m-%d}"
        if uncapped.is_ruined
        else f"${uncapped.equity_series().iloc[-1]:,.0f}"
    )
    print(f"  uncapped outcome      : {uncapped_state:>16}")
    print(f"  capped final equity   : ${capped.equity_series().iloc[-1]:>15,.0f}")
    print(f"  partial fill events   : {execution.partial_fills:>15,}")
    print(f"  shares never filled   : {execution.unfilled_quantity:>15,.0f}")
    print(f"  slippage paid, capped : ${capped.total_slippage:>15,.0f}")
    print(f"  slippage paid, uncap'd: ${uncapped.total_slippage:>15,.0f}")
    print("\nThe cap is not a discount — it is a different backtest. Size you")
    print("cannot execute is not a position, and an engine that fills it anyway")
    print("is reporting a strategy nobody could have run.\n")

    print("=" * 76)
    print("4. RISK METRICS AT $50M")
    print("=" * 76 + "\n")
    for name, slippage in models().items():
        portfolio, _ = run(frames, slippage, 5e7, cap=None)
        report = analyze(portfolio.equity_series())
        print(
            f"{name:>12}: Sharpe {report.sharpe:>6.2f}   "
            f"max drawdown {report.max_drawdown:>7.1%}"
        )
    print("\nSame strategy. Same data. Same trades. Only the microstructure")
    print("assumption changed.")


if __name__ == "__main__":
    main()
