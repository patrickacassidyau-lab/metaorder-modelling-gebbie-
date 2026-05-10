#!/usr/bin/env python3
"""
Plot mid + signal diagnostics (+ optional backtest equity) for one symbol.

Reads CSV with columns timestamp, mid, quantity, sign (symbol optional if single-ticker file).
See visualisation/README.md for index of plotting scripts.

With no ``--csv``, looks for ``data/synthetic_panel.csv`` or similar; use ``--demo`` for a
built-in smoke plot without preparing files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from metaorder_signal.backtest import BacktestConfig, run_event_backtest
from metaorder_signal.io_taq import load_trades_csv
from metaorder_signal.signal import MetaorderSignalParams, SignalConfig


def _default_csv_candidates() -> list[Path]:
    return [
        Path("data/synthetic_panel.csv"),
        Path("data/panel_small.csv"),
    ]


def _resolve_csv(arg: Path | None) -> Path | None:
    if arg is not None:
        if arg.is_file():
            return arg
        print(f"CSV not found: {arg}", file=sys.stderr)
        sys.exit(2)
    seen: set[Path] = set()
    for p in _default_csv_candidates():
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if p.is_file():
            return p
    data = Path("data")
    if data.is_dir():
        for p in sorted(data.glob("*.csv")):
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualise synthetic or real TAQ run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --demo
  %(prog)s --csv data/synthetic_panel.csv --symbol SYM0001 --out plots/SYM0001.png
  %(prog)s --csv data/large_panel/SYM0001.csv --out plots/run.png

Generate CSV first:
  python examples/generate_synthetic_market.py --symbols 5 --out data/synthetic_panel.csv
""",
    )
    ap.add_argument("--csv", type=Path, default=None, help="Trade CSV (optional if --demo or a default file exists)")
    ap.add_argument(
        "--demo",
        action="store_true",
        help="Ignore CSV; plot a small in-memory synthetic stream (smoke test)",
    )
    ap.add_argument("--symbol", type=str, default=None, help="Filter symbol from panel CSV")
    ap.add_argument("--out", type=Path, default=Path("plots/run_overview.png"))
    ap.add_argument("--alpha", type=float, default=1.8)
    ap.add_argument("--gamma1", type=float, default=0.85)
    ap.add_argument("--gamma2", type=float, default=0.77)
    ap.add_argument("--beta", type=float, default=0.241)
    ap.add_argument("--sigma-d", type=float, default=0.018)
    ap.add_argument("--vd", type=float, default=5e6)
    ap.add_argument("--s-max", type=float, default=1000.0)
    ap.add_argument("--no-backtest", action="store_true")
    args = ap.parse_args()

    csv_path: Path | None = None
    if args.demo:
        from metaorder_signal.synthetic_market import SyntheticMarketConfig, generate_symbol_trades

        import numpy as np

        rng = np.random.default_rng(12345)
        cfg = SyntheticMarketConfig(n_metaorders=120, n_sessions=1, length_cap=80)
        df = generate_symbol_trades("DEMO", rng, cfg)
        args.symbol = None
        print("Using in-memory --demo stream (no CSV file read).", file=sys.stderr)
    else:
        csv_path = _resolve_csv(args.csv)
        if csv_path is None:
            print(
                "No input CSV. Choose one:\n"
                "  •  python visualisation/visualise_run.py --demo\n"
                "  •  python visualisation/visualise_run.py --csv path/to/trades.csv\n"
                "  •  python examples/generate_synthetic_market.py --symbols 5 --out data/synthetic_panel.csv\n",
                file=sys.stderr,
            )
            sys.exit(2)
        if args.csv is None:
            print(f"Using default CSV: {csv_path}", file=sys.stderr)
        df = load_trades_csv(str(csv_path))

    if args.symbol is not None:
        if "symbol" not in df.columns:
            raise SystemExit("CSV has no symbol column; omit --symbol")
        df = df.loc[df["symbol"] == args.symbol].copy()
    if df.empty:
        raise SystemExit("No rows after filter")

    params = MetaorderSignalParams(
        alpha=args.alpha,
        gamma1=args.gamma1,
        gamma2=args.gamma2,
        beta=args.beta,
        sigma_d=args.sigma_d,
        vd=args.vd,
    )
    # Default tex thresholds often yield zero entries on short synthetic tapes → flat equity at 0.
    # Relax slightly for --demo so the cash/equity panel is illustrative.
    if args.demo:
        sig_cfg = SignalConfig(
            s_max=args.s_max,
            p_min=0.45,
            phi_entry=0.52,
            n_min=2,
            rho_max=2.5,
        )
    else:
        sig_cfg = SignalConfig(s_max=args.s_max)

    if args.no_backtest:
        from metaorder_signal.signal import process_trade_stream

        out = process_trade_stream(df, params, sig_cfg)
        out["equity"] = 0.0
    else:
        out = run_event_backtest(df, params, sig_cfg, BacktestConfig())

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    t = pd.to_datetime(out["timestamp"], utc=True)

    axes[0].plot(t, out["mid"], color="black", lw=0.8)
    el = out["entry_long"] | out["entry_short"]
    axes[0].scatter(t[el], out.loc[el, "mid"], color="limegreen", s=8, label="entry")
    axes[0].scatter(t[out["exit_signal"]], out.loc[out["exit_signal"], "mid"], color="tomato", s=8, label="exit")
    axes[0].set_ylabel("mid")
    axes[0].legend(loc="upper left")

    axes[1].plot(t, out["survival_p"], color="tab:blue", lw=0.8)
    axes[1].set_ylabel("survival_p")

    axes[2].plot(t, out["phi"], color="tab:purple", lw=0.8)
    axes[2].set_ylabel("phi")
    if float(out["phi"].std(skipna=True) or 0.0) < 1e-12:
        axes[2].annotate(
            "φ nearly constant (run/q̂ ratio)",
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            fontsize=8,
            color="tab:purple",
        )

    axes[3].plot(t, out["equity"], color="tab:green", lw=0.9)
    axes[3].set_ylabel("equity")
    axes[3].set_xlabel("time")
    if float(out["equity"].abs().max()) < 1e-12:
        axes[3].annotate(
            "no fills (entry filters never passed)",
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            fontsize=8,
            color="tab:green",
        )

    plot_title = args.symbol or ("DEMO" if args.demo else Path(csv_path).stem)
    fig.suptitle(f"{plot_title} — signal diagnostics")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
