"""Simple tape benchmarks for empirical comparison (not production signals)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold_cumulative_return(mid: pd.Series) -> np.ndarray:
    """Fractional return vs first tick: mid/m0 - 1 (same length as ``mid``)."""
    m = mid.to_numpy(dtype=float)
    if m.size == 0:
        return m
    m0 = float(m[0])
    if not np.isfinite(m0) or m0 == 0:
        return np.zeros_like(m)
    return m / m0 - 1.0


def equity_drawdown_series(equity: np.ndarray) -> np.ndarray:
    """Peak-to-trough drop relative to running peak (underwater, 0..1 scale per step)."""
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return eq
    peak = np.maximum.accumulate(eq)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (peak - eq) / (np.maximum(np.abs(peak), 1e-12))
    return np.nan_to_num(dd, nan=0.0, posinf=0.0)
