#!/usr/bin/env python3
"""Generate multi-symbol synthetic TAQ-like data for signal/backtest experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metaorder_signal.synthetic_market import (
    SyntheticMarketConfig,
    generate_panel,
    write_synthetic_dataset,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Synthetic metaorder market — large panels via --output-dir (recommended)",
    )
    p.add_argument("--symbols", type=int, default=500, help="Number of independent symbols")
    p.add_argument("--prefix", type=str, default="SYM", help="Symbol prefix")
    p.add_argument("--metaorders", type=int, default=1200, help="Metaorders per symbol per session")
    p.add_argument("--sessions", type=int, default=1, help="Tape repetitions per symbol")
    p.add_argument("--session-gap-days", type=float, default=1.0, help="Days between sessions")
    p.add_argument("--no-reset-mid", action="store_true", help="Carry mid across sessions")
    p.add_argument("--traders", type=int, default=100, help="Synthetic trader count N")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mean-inter-trade",
        type=float,
        default=None,
        help="Mean seconds between child trades (default: config 0.22)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write one CSV/Parquet per symbol + manifest.json (best for large runs)",
    )
    p.add_argument(
        "--format",
        choices=("csv", "parquet"),
        default="csv",
        help="File format when using --output-dir",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Single combined CSV (small/medium panels only; loads full panel in RAM)",
    )
    p.add_argument("--meta-json", type=Path, default=None, help="Write run metadata JSON")
    args = p.parse_args()

    if args.output_dir is None and args.out is None:
        args.out = Path("data/synthetic_panel.csv")

    symbols = [f"{args.prefix}{i:04d}" for i in range(args.symbols)]
    cfg = SyntheticMarketConfig(
        n_metaorders=args.metaorders,
        n_synthetic_traders=args.traders,
        n_sessions=max(1, args.sessions),
        days_between_sessions=float(args.session_gap_days),
        reset_mid_per_session=not args.no_reset_mid,
    )
    if args.mean_inter_trade is not None:
        cfg.mean_inter_trade_s = float(args.mean_inter_trade)

    if args.output_dir is not None:
        manifest = write_synthetic_dataset(
            symbols,
            args.output_dir,
            seed=args.seed,
            cfg=cfg,
            fmt=args.format,
        )
        meta = {
            "mode": "per_symbol_dir",
            "output_dir": str(args.output_dir.resolve()),
            "manifest": manifest["manifest_path"],
            "total_rows": manifest["total_rows"],
            "symbols_requested": args.symbols,
            "seed": args.seed,
            **manifest["config"],
        }
    else:
        df = generate_panel(symbols, seed=args.seed, cfg=cfg)
        assert args.out is not None
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        meta = {
            "mode": "single_csv",
            "path": str(args.out.resolve()),
            "symbols": args.symbols,
            "metaorders_per_symbol": args.metaorders,
            "sessions": cfg.n_sessions,
            "n_synthetic_traders": args.traders,
            "seed": args.seed,
            "alpha_ccdf": cfg.alpha_ccdf,
            "vd": cfg.vd,
            "sigma_d": cfg.sigma_d,
            "gamma1_impact": cfg.gamma1_impact,
            "rows": int(len(df)),
        }

    if args.meta_json:
        args.meta_json.parent.mkdir(parents=True, exist_ok=True)
        args.meta_json.write_text(json.dumps(meta, indent=2))

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
