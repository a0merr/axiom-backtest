import numpy as np
import pandas as pd

from axiom import MovingAverageCrossover, walk_forward
from axiom.validation import oos_summary, stitch_oos_equity


def _frames(n=900):
    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    # mild upward drift + noise
    prices = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    return {"AAA": pd.DataFrame({"close": prices}, index=idx)}


def _factory(data, params):
    return MovingAverageCrossover(data, fast=params["fast"], slow=params["slow"])


def test_walk_forward_produces_expected_folds():
    windows = walk_forward(
        _frames(),
        _factory,
        n_splits=4,
        train_size=252,
        test_size=63,
        base_params={"fast": 10, "slow": 30},
    )
    assert len(windows) == 4
    # Test windows must not overlap and must follow their train window.
    for w in windows:
        assert w.test_start > w.train_end
    for a, b in zip(windows, windows[1:], strict=False):
        assert b.test_start > a.test_start


def test_summary_table_has_one_row_per_fold():
    windows = walk_forward(
        _frames(),
        _factory,
        n_splits=3,
        base_params={"fast": 10, "slow": 30},
    )
    table = oos_summary(windows)
    assert len(table) == len(windows)
    assert {"fold", "sharpe", "max_drawdown", "params"} <= set(table.columns)


def test_stitch_concatenates_oos_curves_continuously():
    windows = walk_forward(
        _frames(),
        _factory,
        n_splits=3,
        base_params={"fast": 10, "slow": 30},
    )
    curve = stitch_oos_equity(windows)
    # One continuous series spanning every fold's OOS bars, in order.
    assert isinstance(curve, pd.Series)
    assert len(curve) == sum(len(w.equity) for w in windows)
    assert curve.index.is_monotonic_increasing


def test_optimizer_can_change_params_per_fold():
    def optimizer(train_frames):
        # toy: pick fast window by sign of recent drift
        df = next(iter(train_frames.values()))
        return {"fast": 5 if df["close"].iloc[-1] > df["close"].iloc[0] else 20}

    windows = walk_forward(
        _frames(),
        _factory,
        n_splits=3,
        optimizer=optimizer,
        base_params={"fast": 10, "slow": 40},
    )
    assert all("fast" in w.params for w in windows)
