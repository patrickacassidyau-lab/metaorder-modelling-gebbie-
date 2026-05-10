import numpy as np
import pandas as pd

from metaorder_signal.empirical.benchmarks import (
    buy_and_hold_cumulative_return,
    equity_drawdown_series,
)


def test_buy_hold_matches_simple_return():
    mid = pd.Series([100.0, 102.0, 101.0])
    r = buy_and_hold_cumulative_return(mid)
    np.testing.assert_allclose(r, np.array([0.0, 0.02, 0.01]))


def test_drawdown_non_negative():
    eq = np.array([0.0, 1.0, 0.5, 1.2, 0.3])
    dd = equity_drawdown_series(eq)
    assert np.all(dd >= -1e-9)
