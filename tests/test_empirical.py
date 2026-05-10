import numpy as np
import pandas as pd

from metaorder_signal.empirical.calibration import calibrate_from_trades
from metaorder_signal.empirical.metrics import compute_metrics
from metaorder_signal.empirical.runs import extract_run_lengths
from metaorder_signal.backtest import BacktestConfig, run_event_backtest
from metaorder_signal.signal import MetaorderSignalParams, SignalConfig


def _toy_tape(n: int = 800):
    rng = np.random.default_rng(0)
    t0 = pd.Timestamp("2025-01-01T00:00:00Z")
    ts = t0 + pd.to_timedelta(np.sort(rng.uniform(0, 3600, n)), unit="s")
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    q = rng.uniform(0.1, 2.0, n)
    s = rng.choice([-1, 1], n)
    return pd.DataFrame({"timestamp": ts, "mid": mid, "quantity": q, "sign": s})


def test_run_lengths_positive():
    tape = _toy_tape(300)
    L = extract_run_lengths(tape)
    assert L.size > 0 and np.all(L >= 1)


def test_calibrate_finite():
    tape = _toy_tape(500)
    cp = calibrate_from_trades(tape, min_runs=10)
    assert np.isfinite(cp.alpha)
    assert cp.vd > 0


def test_compute_metrics_pipeline():
    tape = _toy_tape(600)
    params = MetaorderSignalParams(
        alpha=1.7,
        gamma1=0.85,
        gamma2=0.77,
        beta=0.24,
        sigma_d=0.02,
        vd=float(tape["quantity"].sum()),
    )
    out = run_event_backtest(tape, params, SignalConfig(s_max=100.0), BacktestConfig())
    m = compute_metrics(out, tape)
    assert m.n_trades_signal == len(tape)
