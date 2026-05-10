"""Event-driven tick backtest (equation costs, capacity)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metaorder_signal.signal import MetaorderSignalParams, SignalConfig, process_trade_stream


@dataclass
class BacktestConfig:
    """Execution and capacity constraints."""

    half_spread_frac: float = 0.0002
    capacity_frac: float = 1e-4


def _cost(
    q_pos: float,
    *,
    sigma_d: float,
    vd: float,
    half_spread: float,
    kappa: float,
) -> float:
    """Equation (costs): 0.5*spread + kappa * sigma_d * sqrt(q/V_D)."""
    q = abs(float(q_pos))
    if vd <= 0 or sigma_d <= 0:
        impact = 0.0
    else:
        impact = float(kappa * sigma_d * np.sqrt(q / vd))
    return float(0.5 * half_spread + impact)


def run_event_backtest(
    trades: pd.DataFrame,
    params: MetaorderSignalParams,
    signal_cfg: SignalConfig | None = None,
    bt_cfg: BacktestConfig | None = None,
) -> pd.DataFrame:
    """
    Apply capacity clipping then simulate entries/exits with proportional costs.

    Realised PnL hits on exit using mid-to-mid from entry to exit as in equation (pnl).
    """
    signal_cfg = signal_cfg or SignalConfig()
    bt_cfg = bt_cfg or BacktestConfig()

    sig = process_trade_stream(trades, params, signal_cfg)
    q_cap = float(bt_cfg.capacity_frac * params.vd)

    n = len(sig)
    positions = np.zeros(n)
    equity = np.zeros(n)

    pos = 0.0
    entry_mid = np.nan
    cash = 0.0

    mids = sig["mid"].to_numpy(dtype=float)
    entry_l = sig["entry_long"].to_numpy(dtype=bool)
    entry_s = sig["entry_short"].to_numpy(dtype=bool)
    exit_x = sig["exit_signal"].to_numpy(dtype=bool)
    sizes = sig["signal_size"].to_numpy(dtype=float)

    for i in range(n):
        if entry_l[i] or entry_s[i]:
            raw = float(min(max(sizes[i], 0.0), q_cap))
            pos = raw if entry_l[i] else -raw
            entry_mid = mids[i]
            cash -= _cost(
                pos,
                sigma_d=params.sigma_d,
                vd=params.vd,
                half_spread=bt_cfg.half_spread_frac,
                kappa=signal_cfg.kappa_cost,
            )
        elif exit_x[i] and pos != 0.0:
            if np.isfinite(entry_mid):
                cash += float(pos) * (mids[i] - entry_mid)
            cash -= _cost(
                pos,
                sigma_d=params.sigma_d,
                vd=params.vd,
                half_spread=bt_cfg.half_spread_frac,
                kappa=signal_cfg.kappa_cost,
            )
            pos = 0.0
            entry_mid = np.nan

        positions[i] = pos
        equity[i] = cash

    out = sig.copy()
    out["position"] = positions
    out["cash"] = equity
    out["equity"] = equity
    out["pnl_tick"] = np.diff(np.r_[0.0, equity])
    return out
