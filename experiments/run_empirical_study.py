#!/usr/bin/env python3
"""
Serious empirical backbone: calibrate structural parameters on an early time window,
run the signal + backtest on a later window (no overlap leakage), report metrics.

Data sources:
  • ``--fetch-binance SYMBOL`` — public agg trades (needs network).
  • ``--csv path`` — your own tape (timestamp, mid, quantity, sign).

Writes JSON report under ``--report-dir`` (default ``results/empirical``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metaorder_signal.backtest import BacktestConfig, run_event_backtest
from metaorder_signal.empirical.binance_public import fetch_agg_trades
from metaorder_signal.empirical.calibration import calibrate_from_trades
from metaorder_signal.empirical.metrics import baseline_random_signs, compute_metrics, metrics_to_dict
from metaorder_signal.io_taq import load_trades_csv
from metaorder_signal.signal import SignalConfig


def time_split(df: pd.DataFrame, calib_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    if len(df) < 100:
        raise ValueError("Need at least 100 trades for a meaningful split")
    cut = df["timestamp"].quantile(float(calib_frac))
    cal = df.loc[df["timestamp"] <= cut].copy()
    tst = df.loc[df["timestamp"] > cut].copy()
    if len(cal) < 50 or len(tst) < 50:
        raise ValueError("Calibration or test segment too small after split — increase data or adjust calib-frac")
    return cal, tst


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibration → hold-out backtest (empirical study)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="TAQ-like CSV")
    src.add_argument("--fetch-binance", type=str, metavar="SYMBOL", help="e.g. BTCUSDT")
    ap.add_argument("--max-trades", type=int, default=80_000, help="Cap when fetching from Binance")
    ap.add_argument("--calib-frac", type=float, default=0.55, help="Fraction of time axis for calibration")
    ap.add_argument("--report-dir", type=Path, default=Path("results/empirical"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--s-max", type=float, default=1000.0)
    args = ap.parse_args()

    if args.fetch_binance:
        print(f"Fetching up to {args.max_trades} agg trades for {args.fetch_binance} …")
        df = fetch_agg_trades(args.fetch_binance.upper(), max_trades=args.max_trades)
        if df.empty:
            raise SystemExit("No rows returned from Binance")
        tape = df[["timestamp", "mid", "quantity", "sign"]].copy()
    else:
        tape = load_trades_csv(str(args.csv))

    cal, tst = time_split(tape, args.calib_frac)
    print(f"Calibration rows: {len(cal):,} | Hold-out rows: {len(tst):,}")

    cp = calibrate_from_trades(cal)
    params = cp.to_signal_params()
    print(
        f"Calibrated alpha={params.alpha:.4f} sigma_d={params.sigma_d:.6f} "
        f"V_D={params.vd:.2f} (run-length samples={cp.n_run_lengths})"
    )

    sig_cfg = SignalConfig(s_max=args.s_max)
    result = run_event_backtest(tst, params, sig_cfg, BacktestConfig())
    metrics = compute_metrics(result, tst, ic_horizon=5)
    metrics.baseline_final_equity = baseline_random_signs(tst, seed=args.seed)

    report = {
        "disclaimer": (
            "Research diagnostics only: simplified microstructure model and costs; "
            "not validated for live trading. Crypto spot trades ≠ equities TAQ."
        ),
        "calibration": {
            "rows": len(cal),
            "alpha": cp.alpha,
            "sigma_d": cp.sigma_d,
            "vd": cp.vd,
            "n_run_lengths": cp.n_run_lengths,
            "xmin_fit": cp.xmin_fit,
        },
        "hold_out": {
            "rows": len(tst),
            "metrics": metrics_to_dict(metrics),
        },
        "split": {"calib_frac": args.calib_frac},
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    tag = args.fetch_binance or args.csv.stem
    out_path = args.report_dir / f"report_{tag}.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report["hold_out"]["metrics"], indent=2))
    print(f"Wrote {out_path.resolve()}")

    sharpe_txt = f"{metrics.sharpe_like:.4f}" if metrics.sharpe_like == metrics.sharpe_like else "nan"
    print(
        f"Summary: final_equity={metrics.final_equity:.6f} | sharpe_like(steps)={sharpe_txt} | "
        f"entries={metrics.n_entries} | IC(surv,fwd)={metrics.ic_survival_fwd_ret}"
    )


if __name__ == "__main__":
    main()
