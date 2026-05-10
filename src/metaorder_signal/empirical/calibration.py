"""Estimate structural scalars from a calibration trade window (no PnL tuning)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metaorder_signal.empirical.runs import extract_run_lengths
from metaorder_signal.powerlaw_fit import fit_powerlaw_lengths
from metaorder_signal.signal import MetaorderSignalParams


@dataclass
class CalibratedParams:
    """Point estimates used for the test-segment signal."""

    alpha: float
    sigma_d: float
    vd: float
    gamma1: float = 0.85
    gamma2: float = 0.77
    beta: float = 0.241
    n_run_lengths: int = 0
    xmin_fit: float = float("nan")

    def to_signal_params(self) -> MetaorderSignalParams:
        return MetaorderSignalParams(
            alpha=float(self.alpha),
            gamma1=float(self.gamma1),
            gamma2=float(self.gamma2),
            beta=float(self.beta),
            sigma_d=float(self.sigma_d),
            vd=float(self.vd),
        )


def _log_return_vol_mid(trades: pd.DataFrame) -> float:
    """Scale-free intraday volatility proxy: std of consecutive log-mid returns."""
    df = trades.sort_values("timestamp").reset_index(drop=True)
    m = df["mid"].to_numpy(dtype=float)
    if len(m) < 10:
        return float("nan")
    lr = np.diff(np.log(np.clip(m, 1e-12, None)))
    lr = lr[np.isfinite(lr)]
    if lr.size < 5:
        return float("nan")
    return float(np.std(lr, ddof=1))


def calibrate_from_trades(
    trades: pd.DataFrame,
    *,
    gamma1: float = 0.85,
    gamma2: float = 0.77,
    beta: float = 0.241,
    min_runs: int = 40,
) -> CalibratedParams:
    """
    Fit tail exponent on empirical same-sign run lengths; set V_D to total share volume
    in the window and sigma_d to std of log-mid returns (dimensionless impact scaling).
    """
    lengths = extract_run_lengths(trades)
    if lengths.size < min_runs:
        fit = {"alpha": float("nan"), "xmin": float("nan")}
    else:
        fit = fit_powerlaw_lengths(lengths, discrete=True)

    alpha = float(fit["alpha"]) if np.isfinite(fit["alpha"]) else 1.8
    xmin_fit = float(fit.get("xmin", float("nan")))

    vd = float(trades["quantity"].sum())
    if not np.isfinite(vd) or vd <= 0:
        vd = float(len(trades))

    sigma_d = _log_return_vol_mid(trades)
    if not np.isfinite(sigma_d) or sigma_d <= 0:
        sigma_d = 0.02

    return CalibratedParams(
        alpha=alpha,
        sigma_d=sigma_d,
        vd=vd,
        gamma1=gamma1,
        gamma2=gamma2,
        beta=beta,
        n_run_lengths=int(lengths.size),
        xmin_fit=xmin_fit,
    )
