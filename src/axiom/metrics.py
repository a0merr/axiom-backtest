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


@dataclass
class BootstrapResult:
    """Sampling distribution summary for an annualized Sharpe ratio."""

    point: float
    lower: float
    upper: float
    p_value: float  # one-sided P(Sharpe <= 0) under the resampling distribution
    confidence: float
    n_boot: int
    block_size: int

    @property
    def significant(self) -> bool:
        """True if the confidence interval excludes zero."""
        return self.lower > 0 or self.upper < 0

    def as_dict(self) -> dict:
        return asdict(self)


def _sharpe_array(r: np.ndarray, rf_per_period: float, ppy: int) -> float:
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(np.sqrt(ppy) * (r.mean() - rf_per_period) / sd)


def bootstrap_sharpe(
    returns: pd.Series,
    *,
    n_boot: int = 2000,
    block_size: int | None = None,
    confidence: float = 0.95,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
    seed: int = 0,
) -> BootstrapResult:
    """Circular block-bootstrap confidence interval for the Sharpe ratio.

    A point Sharpe says nothing about whether the edge is real. This resamples
    the return series in *blocks* (preserving short-run autocorrelation that an
    i.i.d. bootstrap would destroy), recomputes the annualized Sharpe on each
    resample, and reports a percentile interval plus a one-sided p-value for
    "Sharpe ≤ 0". A high point Sharpe whose interval includes zero is not
    evidence of skill — exactly the overfitting trap this library exists to
    expose.

    ``block_size`` defaults to the n^(1/3) rule of thumb. Determinism is via
    ``seed`` so results are reproducible in CI.
    """
    r = np.asarray(returns.dropna(), dtype=float)
    n = r.size
    rf = risk_free / periods_per_year
    point = _sharpe_array(r, rf, periods_per_year)
    if n < 2:
        return BootstrapResult(point, 0.0, 0.0, 1.0, confidence, n_boot, 0)

    if block_size is None:
        block_size = max(1, int(round(n ** (1 / 3))))
    block_size = min(block_size, n)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    offsets = np.arange(block_size)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        # circular blocks: wrap indices past the end back to the start
        idx = (starts[:, None] + offsets).ravel() % n
        boots[i] = _sharpe_array(r[idx[:n]], rf, periods_per_year)

    alpha = (1 - confidence) / 2
    lower = float(np.quantile(boots, alpha))
    upper = float(np.quantile(boots, 1 - alpha))
    p_value = float(np.mean(boots <= 0))
    return BootstrapResult(
        point=point,
        lower=lower,
        upper=upper,
        p_value=p_value,
        confidence=confidence,
        n_boot=n_boot,
        block_size=block_size,
    )


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    total_growth = equity.iloc[-1] / equity.iloc[0]
    # N points span N-1 return periods; using N would overstate elapsed time.
    years = (len(equity) - 1) / periods_per_year
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
    # No drawdown with a positive return is an unbounded Calmar, not zero.
    if mdd < 0:
        calmar = annualized / abs(mdd)
    elif annualized > 0:
        calmar = float("inf")
    else:
        calmar = 0.0

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
