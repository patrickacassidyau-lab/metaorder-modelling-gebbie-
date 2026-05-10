"""
Synthetic trade streams with metaorder structure (LMF-style heavy tails + square-root impact).

Inspired by public-data metaorder reconstruction: metaorders are sequences of same-sign child
trades; lengths follow a discrete power law; mids update with a reduced-form SQL impact plus
diffusive noise. Multiple symbols share only global RNG discipline for reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SyntheticMarketConfig:
    """Controls metaorder flow and mid dynamics."""

    n_metaorders: int = 1200
    """Number of metaorders to simulate per symbol (per session)."""

    n_sessions: int = 1
    """Repeat the tape this many times per symbol with timestamp gaps."""

    days_between_sessions: float = 1.0
    """Calendar spacing between sessions when ``n_sessions`` > 1."""

    reset_mid_per_session: bool = True
    """If True, re-draw opening mid each session; else continue price path."""

    n_synthetic_traders: int = 100
    """Used as in thesis-style sensitivity: metaorder id assigns to trader `mo_id % N`."""

    alpha_ccdf: float = 1.8
    """Target CCDF tail for metaorder lengths (maps to discrete sampling exponent)."""

    length_xmin: int = 1
    length_cap: int = 500

    mean_inter_trade_s: float = 0.22
    """Lower ⇒ denser tape (more child trades per clock time)."""
    """Mean gap between consecutive child trades within a metaorder (exponential)."""

    qty_mu: float = 1.8
    qty_sigma: float = 0.55
    """Lognormal parameters for child sizes (before clipping)."""

    vd: float = 5e6
    """Scale volume for square-root impact."""

    sigma_d: float = 0.018
    gamma1_impact: float = 0.85
    """Prefactor on sqrt(q/V_D) for log-impact per trade."""

    diffusion_sigma: float = 3e-5
    """Gaussian shock on log-mid each trade (idiosyncratic noise)."""

    start_mid: float = 100.0
    mid_jitter_frac: float = 5e-4
    """Per-symbol multiplicative jitter applied once at session open."""


def _sample_metaorder_length(rng: np.random.Generator, cfg: SyntheticMarketConfig) -> int:
    """Discrete power-law lengths on [xmin, cap] with tail exponent ~ alpha_ccdf + 1."""
    s = float(cfg.alpha_ccdf + 1.0)
    ks = np.arange(cfg.length_xmin, cfg.length_cap + 1, dtype=float)
    weights = ks ** (-s)
    weights /= weights.sum()
    return int(rng.choice(ks.astype(int), p=weights))


def generate_symbol_trades(
    symbol: str,
    rng: np.random.Generator,
    cfg: SyntheticMarketConfig,
) -> pd.DataFrame:
    """
    Emit trades for one symbol: metaorders as same-sign runs, power-law lengths,
    lognormal sizes, exponential spacing, SQL-style impact on mid.
    """
    rows: list[dict] = []
    mid_global = float(cfg.start_mid * (1.0 + rng.normal(0.0, cfg.mid_jitter_frac)))
    session_gap_s = float(max(cfg.days_between_sessions, 1e-9)) * 86400.0

    for sess in range(max(cfg.n_sessions, 1)):
        t_offset = sess * session_gap_s
        mid = (
            float(cfg.start_mid * (1.0 + rng.normal(0.0, cfg.mid_jitter_frac)))
            if cfg.reset_mid_per_session or sess == 0
            else mid_global
        )
        t_s = t_offset

        for mo_id in range(cfg.n_metaorders):
            trader = mo_id % max(cfg.n_synthetic_traders, 1)
            mo_uid = sess * cfg.n_metaorders + mo_id
            direction = int(rng.choice([-1, 1]))
            n_child = _sample_metaorder_length(rng, cfg)

            for _k in range(n_child):
                q = float(np.clip(rng.lognormal(mean=cfg.qty_mu, sigma=cfg.qty_sigma), 1e-6, 1e9))
                dt = float(rng.exponential(cfg.mean_inter_trade_s))
                t_s += dt

                rt_impact = cfg.gamma1_impact * cfg.sigma_d * np.sqrt(q / max(cfg.vd, 1.0))
                diffusion = rng.normal(0.0, cfg.diffusion_sigma)
                mid *= float(np.exp(direction * rt_impact + diffusion))

                rows.append(
                    {
                        "timestamp": pd.Timestamp(t_s, unit="s", tz="UTC"),
                        "mid": mid,
                        "quantity": q,
                        "sign": direction,
                        "symbol": symbol,
                        "metaorder_id": mo_uid,
                        "session": sess,
                        "synthetic_trader": trader,
                    }
                )

        mid_global = mid

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def generate_panel(
    symbols: list[str],
    *,
    seed: int = 42,
    cfg: SyntheticMarketConfig | None = None,
) -> pd.DataFrame:
    """
    Concatenate independent synthetic streams with deterministic seeds per symbol.

    Timestamps are offset per symbol by whole days so sorting a mixed panel preserves
    rough separation unless you collapse timestamps externally.
    """
    cfg = cfg or SyntheticMarketConfig()
    parts: list[pd.DataFrame] = []

    for i, sym in enumerate(symbols):
        child_seed = int(seed) + i * 1_000_003
        rng = np.random.default_rng(child_seed)
        df = generate_symbol_trades(sym, rng, cfg)
        if df.empty:
            continue
        offset_ns = i * np.timedelta64(1, "D")
        df = df.copy()
        df["timestamp"] = df["timestamp"] + offset_ns
        parts.append(df)

    if not parts:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "mid",
                "quantity",
                "sign",
                "symbol",
                "metaorder_id",
                "session",
                "synthetic_trader",
            ]
        )

    out = pd.concat(parts, ignore_index=True)
    return out.sort_values("timestamp").reset_index(drop=True)


def trades_for_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Strip helper columns; keep contract expected by `process_trade_stream`."""
    cols = ["timestamp", "mid", "quantity", "sign"]
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for signal pipeline: {missing}")
    return df[cols].copy()


def write_synthetic_dataset(
    symbols: list[str],
    out_dir: str | Path,
    *,
    seed: int = 42,
    cfg: SyntheticMarketConfig | None = None,
    fmt: str = "csv",
) -> dict:
    """
    Write one file per symbol under ``out_dir`` (memory-friendly for huge panels).

    Filenames: ``{symbol}.csv`` or ``{symbol}.parquet``. No cross-symbol timestamp
    offset (each tape is self-contained). Returns manifest dict with paths and row counts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg or SyntheticMarketConfig()
    fmt = fmt.lower().strip()
    if fmt not in {"csv", "parquet"}:
        raise ValueError("fmt must be 'csv' or 'parquet'")

    manifest: dict = {
        "format": fmt,
        "symbols_requested": len(symbols),
        "files": [],
        "total_rows": 0,
        "config": {
            "n_metaorders": cfg.n_metaorders,
            "n_sessions": cfg.n_sessions,
            "days_between_sessions": cfg.days_between_sessions,
            "alpha_ccdf": cfg.alpha_ccdf,
            "vd": cfg.vd,
            "mean_inter_trade_s": cfg.mean_inter_trade_s,
        },
    }

    for i, sym in enumerate(symbols):
        rng = np.random.default_rng(int(seed) + i * 1_000_003)
        df = generate_symbol_trades(sym, rng, cfg)
        if df.empty:
            continue
        path = out_dir / f"{sym}.{fmt}"
        if fmt == "csv":
            df.to_csv(path, index=False)
        else:
            try:
                df.to_parquet(path, index=False)
            except ImportError as exc:  # pragma: no cover
                raise ImportError("Parquet output requires pyarrow: pip install pyarrow") from exc

        n = int(len(df))
        manifest["files"].append({"symbol": sym, "path": str(path.resolve()), "rows": n})
        manifest["total_rows"] += n

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest
