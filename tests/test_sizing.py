import numpy as np
import pandas as pd
import pytest

from axiom import (
    CorrelationAwareSizer,
    FixedFractionalSizer,
    HistoricCSVDataHandler,
    VolatilityTargetSizer,
)


def _feed(sizer, frames, bars=None):
    """Drive a sizer's observe() over the frames, return the data handler."""
    data = HistoricCSVDataHandler(frames)
    n = bars or len(next(iter(frames.values())))
    for _ in range(n):
        if data.update_bars() is None:
            break
        sizer.observe(data)
    return data


def _series(prices):
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="D")
    return pd.DataFrame({"close": prices}, index=idx)


def test_fixed_fractional_matches_formula():
    s = FixedFractionalSizer(0.25)
    qty = s.size("AAA", price=100.0, strength=1.0, equity=100_000.0, held_symbols=[])
    assert qty == 250  # 100k * 0.25 / 100


def test_fixed_fractional_rejects_bad_fraction():
    with pytest.raises(ValueError):
        FixedFractionalSizer(1.5)


def test_vol_target_gives_low_vol_asset_more_size():
    rng = np.random.default_rng(0)
    n = 200
    calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))  # low vol
    wild = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))  # high vol
    frames = {"CALM": _series(calm), "WILD": _series(wild)}

    sizer = VolatilityTargetSizer(target_vol=0.15, lookback=60)
    _feed(sizer, frames)

    q_calm = sizer.size("CALM", 100.0, 1.0, 100_000.0, [])
    q_wild = sizer.size("WILD", 100.0, 1.0, 100_000.0, [])
    assert q_calm > q_wild  # less volatile name carries more notional


def test_vol_target_falls_back_before_history():
    sizer = VolatilityTargetSizer(target_vol=0.15, lookback=60, fallback_fraction=0.1)
    # no observe() yet -> fallback fraction
    qty = sizer.size("AAA", 100.0, 1.0, 100_000.0, [])
    assert qty == 100  # 100k * 0.1 / 100


def test_correlation_haircut_shrinks_redundant_position():
    rng = np.random.default_rng(1)
    n = 200
    base = rng.normal(0, 0.01, n)
    a = 100 * np.exp(np.cumsum(base))
    a_clone = 100 * np.exp(np.cumsum(base + rng.normal(0, 0.0005, n)))  # ~ corr 1
    indep = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))  # uncorrelated
    frames = {"A": _series(a), "CLONE": _series(a_clone), "INDEP": _series(indep)}

    sizer = CorrelationAwareSizer(target_vol=0.15, lookback=120, correlation_penalty=1.0)
    _feed(sizer, frames)

    # Holding A: a near-clone should be cut hard; an independent name barely.
    q_clone = sizer.size("CLONE", 100.0, 1.0, 100_000.0, ["A"])
    q_indep = sizer.size("INDEP", 100.0, 1.0, 100_000.0, ["A"])
    assert q_clone < q_indep
    # With nothing held, no haircut applies -> clone gets its full vol weight.
    q_clone_solo = sizer.size("CLONE", 100.0, 1.0, 100_000.0, [])
    assert q_clone_solo > q_clone


def test_correlation_penalty_rejects_negative():
    with pytest.raises(ValueError):
        CorrelationAwareSizer(correlation_penalty=-0.5)
