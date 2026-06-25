import numpy as np
import pandas as pd
import pytest

from axiom import metrics


def test_max_drawdown_known_curve():
    equity = pd.Series([100, 120, 90, 110, 80])
    # Peak 120 -> trough 80 = -33.3%
    assert metrics.max_drawdown(equity) == pytest.approx(-1 / 3)


def test_max_drawdown_monotonic_is_zero():
    equity = pd.Series([100, 101, 102, 103])
    assert metrics.max_drawdown(equity) == 0.0


def test_sharpe_zero_when_no_variance():
    rets = pd.Series([0.01, 0.01, 0.01])
    assert metrics.sharpe_ratio(rets) == 0.0


def test_sharpe_matches_manual_formula():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.001, 0.01, 500))
    expected = np.sqrt(252) * rets.mean() / rets.std(ddof=1)
    assert metrics.sharpe_ratio(rets) == pytest.approx(expected)


def test_sortino_only_penalizes_downside():
    rets = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    s = metrics.sortino_ratio(rets)
    # Sortino >= Sharpe here because upside vol is excluded from the denominator.
    assert s >= metrics.sharpe_ratio(rets)


def test_cagr_doubling_in_one_year():
    equity = pd.Series(np.linspace(100, 200, 252))
    assert metrics.cagr(equity) == pytest.approx(1.0, rel=0.05)


def test_analyze_report_fields():
    equity = pd.Series(np.linspace(100, 130, 252))
    report = metrics.analyze(equity)
    assert report.total_return == pytest.approx(0.30)
    assert 0.0 <= report.hit_rate <= 1.0
    assert report.n_periods == 251


def test_bootstrap_strong_signal_is_significant():
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0.001, 0.005, 1000))  # high, stable Sharpe
    res = metrics.bootstrap_sharpe(rets, n_boot=500, seed=0)
    assert res.lower > 0  # CI excludes zero
    assert res.significant
    assert res.p_value < 0.05


def test_bootstrap_noise_straddles_zero():
    rng = np.random.default_rng(2)
    rets = pd.Series(rng.normal(0.0, 0.01, 1000))  # zero mean -> no edge
    res = metrics.bootstrap_sharpe(rets, n_boot=500, seed=0)
    assert res.lower < 0 < res.upper  # interval contains zero
    assert not res.significant
    assert 0.2 < res.p_value < 0.8


def test_bootstrap_is_deterministic_with_seed():
    rng = np.random.default_rng(3)
    rets = pd.Series(rng.normal(0.0005, 0.01, 500))
    a = metrics.bootstrap_sharpe(rets, n_boot=300, seed=42)
    b = metrics.bootstrap_sharpe(rets, n_boot=300, seed=42)
    assert (a.lower, a.upper, a.p_value) == (b.lower, b.upper, b.p_value)


def test_bootstrap_default_block_size_rule():
    rng = np.random.default_rng(4)
    rets = pd.Series(rng.normal(0.0, 0.01, 1000))
    res = metrics.bootstrap_sharpe(rets, n_boot=100, seed=0)
    assert res.block_size == 10  # round(1000 ** (1/3)) == 10
