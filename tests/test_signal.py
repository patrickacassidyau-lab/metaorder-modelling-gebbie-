import numpy as np
import pandas as pd

from metaorder_signal.signal import MetaorderSignalParams, process_trade_stream


def test_monotone_survival_trend():
    t0 = pd.Timestamp("2024-01-02T09:00:00Z")
    n = 50
    ts = t0 + pd.to_timedelta(np.arange(n), unit="s")
    mid = np.linspace(100.0, 100.5, n)
    qty = np.ones(n)
    sign = np.ones(n, dtype=int)
    trades = pd.DataFrame({"timestamp": ts, "mid": mid, "quantity": qty, "sign": sign})

    params = MetaorderSignalParams(
        alpha=1.5,
        gamma1=0.8,
        gamma2=0.7,
        beta=0.24,
        sigma_d=0.02,
        vd=1e6,
    )

    out = process_trade_stream(trades, params)
    assert out["survival_p"].notna().sum() > 10
    assert np.isfinite(out["phi"].iloc[-1])
