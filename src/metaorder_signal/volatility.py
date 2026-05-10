"""Intraday volatility estimators (range, Parkinson, Yang–Zhang)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def range_volatility(mid_open: float, mid_high: float, mid_low: float) -> float:
    """Equation (range_vol): (m_max - m_min) / m_open."""
    if mid_open <= 0:
        return np.nan
    return float((mid_high - mid_low) / mid_open)


def parkinson_volatility(mid_high: float, mid_low: float) -> float:
    """Equation (park_vol): (1 / (4 ln 2)) * (ln(H/L))^2."""
    if mid_high <= 0 or mid_low <= 0 or mid_high < mid_low:
        return np.nan
    ratio = np.log(mid_high / mid_low)
    return float(ratio**2 / (4.0 * np.log(2.0)))


def yang_zhang_volatility(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    n: int | None = None,
) -> float:
    """
    Yang–Zhang drift-independent volatility (equation yz_vol), annualisation-free variance scale.

    Uses the closed-form k in Yang & Zhang (2000) for optimal combination weights when
    sampling frequency is available; falls back to k=0.34 when `n` (observations per day)
    is omitted (common literature default).
    """
    o, h, l, c = map(np.asarray, (open_, high, low, close))
    if len(o) < 2:
        return float("nan")

    c_prev = np.roll(c, 1)
    c_prev[0] = o[0]
    u = np.log(o / c_prev)
    u[0] = 0.0
    v = np.log(c / o)

    rogers_satchell = np.mean((np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)))

    sigma_o2 = np.var(u, ddof=1)
    sigma_c2 = np.var(v, ddof=1)

    if n is None:
        k = 0.34
    else:
        n = float(n)
        k = 0.34 / (1.34 + (n + 1) / (n - 1))

    yz_var = float(k * sigma_o2 + (1.0 - k) * sigma_c2 + (1.0 - k) * rogers_satchell)
    return float(np.sqrt(max(yz_var, 0.0)))


def daily_bars_to_vol_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expect columns: open, high, low, close (mid prices per bar).
    Adds columns range_vol, park_vol, yz_vol per row (YZ uses expanding window min length 2).
    """
    out = df.copy()
    o = out["open"].to_numpy(dtype=float)
    h = out["high"].to_numpy(dtype=float)
    lo = out["low"].to_numpy(dtype=float)
    c = out["close"].to_numpy(dtype=float)

    out["range_vol"] = [
        range_volatility(float(o[i]), float(h[i]), float(lo[i])) for i in range(len(out))
    ]
    out["park_vol"] = [parkinson_volatility(h[i], lo[i]) for i in range(len(out))]

    yz = np.full(len(out), np.nan)
    for i in range(1, len(out)):
        yz[i] = yang_zhang_volatility(o[: i + 1], h[: i + 1], lo[: i + 1], c[: i + 1])
    out["yz_vol"] = yz
    return out
