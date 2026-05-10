#!/usr/bin/env python3
"""Generate synthetic tick data and run the signal + backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from metaorder_signal.backtest import BacktestConfig, run_event_backtest
from metaorder_signal.signal import MetaorderSignalParams, SignalConfig


def main() -> None:
    rng = np.random.default_rng(42)
    n = 5000
    t0 = pd.Timestamp("2024-01-02T09:00:00Z")
    ts = t0 + pd.to_timedelta(np.arange(n), unit="s")

    mid = 100.0 + np.cumsum(rng.normal(0, 0.02, size=n))
    sign = rng.choice([-1, 1], size=n, p=[0.5, 0.5])
    qty = rng.lognormal(mean=2.0, sigma=0.5, size=n)

    trades = pd.DataFrame({"timestamp": ts, "mid": mid, "quantity": qty, "sign": sign})

    params = MetaorderSignalParams(
        alpha=1.8,
        gamma1=0.85,
        gamma2=0.77,
        beta=0.241,
        sigma_d=0.02,
        vd=5e6,
    )

    sig_cfg = SignalConfig(s_max=1000.0)
    bt_cfg = BacktestConfig(half_spread_frac=0.00015)

    out = run_event_backtest(trades, params, sig_cfg, bt_cfg)
    print(out[["mid", "survival_p", "phi", "rho", "entry_long", "exit_signal", "equity"]].tail(10))
    print("final equity:", float(out["equity"].iloc[-1]))


if __name__ == "__main__":
    main()
