import numpy as np

from metaorder_signal.volatility import parkinson_volatility, range_volatility


def test_range_volatility_basic():
    v = range_volatility(100.0, 101.0, 99.0)
    assert np.isclose(v, 0.02)


def test_parkinson_positive():
    v = parkinson_volatility(102.0, 98.0)
    assert v > 0
