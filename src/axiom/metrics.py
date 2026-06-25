"""Performance and risk metrics computed from an equity curve.

All metrics derive from a single equity ``pd.Series`` so they stay consistent
with what the portfolio actually recorded. Annualization uses ``periods_per_year``
(252 for daily bars) and is applied honestly — Sharpe is reported with the
risk-free rate the caller supplies, not silently assumed to be zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceReport:
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    n_periods: int

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"Total return : {self.total_return:>8.2%}\n"
            f"CAGR         : {self.cagr:>8.2%}\n"
            f"Volatility   : {self.annual_volatility:>8.2%}\n"
            f"Sharpe       : {self.sharpe:>8.2f}\n"
            f"Sortino      : {self.sortino:>8.2f}\n"
            f"Max drawdown : {self.max_drawdown:>8.2%}\n"
            f"Calmar       : {self.calmar:>8.2f}\n"
            f"Hit rate     : {self.hit_rate:>8.2%}\n"
            f"Periods      : {self.n_periods:>8d}"
        )


def returns_from_equity(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline as a negative fraction."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def sharpe_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252
) -> float:
    if returns.empty or returns.std(ddof=1) == 0:
        return 0.0
    excess = returns - risk_free / periods_per_year
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=1))


def sortino_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252
) -> float:
    if returns.empty:
        return 0.0
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    if downside.empty or downside.std(ddof=1) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / downside.std(ddof=1))


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    total_growth = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / periods_per_year
    if years <= 0 or total_growth <= 0:
        return 0.0
    return float(total_growth ** (1 / years) - 1)


def analyze(
    equity: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252
) -> PerformanceReport:
    """Build a full report from an equity curve."""
    rets = returns_from_equity(equity)
    mdd = max_drawdown(equity)
    annualized = cagr(equity, periods_per_year)
    total_return = (
        float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) >= 2 else 0.0
    )
    vol = float(rets.std(ddof=1) * np.sqrt(periods_per_year)) if not rets.empty else 0.0
    hit = float((rets > 0).mean()) if not rets.empty else 0.0
    calmar = annualized / abs(mdd) if mdd < 0 else 0.0

    return PerformanceReport(
        total_return=total_return,
        cagr=annualized,
        annual_volatility=vol,
        sharpe=sharpe_ratio(rets, risk_free, periods_per_year),
        sortino=sortino_ratio(rets, risk_free, periods_per_year),
        max_drawdown=mdd,
        calmar=calmar,
        hit_rate=hit,
        n_periods=int(len(rets)),
    )
