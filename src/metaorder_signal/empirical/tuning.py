"""Train / validation / test splits and signal-parameter grid search (validation objective only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Literal

import numpy as np
import pandas as pd

from metaorder_signal.backtest import BacktestConfig, run_event_backtest
from metaorder_signal.empirical.metrics import compute_metrics, metrics_to_dict
from metaorder_signal.signal import MetaorderSignalParams, SignalConfig

GridPreset = Literal["fast", "fine"]

    # Grids: "fast" = 48 configs (CI / quick); "fine" = 128 configs (finer 2D heatmap + search)
_PRESETS: dict[str, dict[str, list[Any]]] = {
    "fast": {
        "p_min_opts": [0.38, 0.48, 0.58],
        "phi_ent": [0.50, 0.62],
        "rho_max_opts": [2.4, 3.2],
        "n_min_opts": [2, 3],
        "phi_exit_opts": [0.88],
        "spread_opts": [0.00011, 0.00022],
        "kappa_opts": [0.008],  # single level keeps ``fast`` at 48 combos
    },
    "fine": {
        "p_min_opts": [0.40, 0.48, 0.56, 0.64],
        "phi_ent": [0.48, 0.56, 0.64, 0.72],
        "rho_max_opts": [2.2, 3.0],
        "n_min_opts": [2],  # single level keeps ``fine`` at 128 combos (4×4 heatmap slice)
        "phi_exit_opts": [0.88],
        "spread_opts": [0.00010, 0.00020],
        "kappa_opts": [0.006, 0.012],
    },
}

_CONFIG_COLS = (
    "p_min",
    "phi_entry",
    "rho_max",
    "n_min",
    "phi_exit",
    "half_spread",
    "kappa",
)


def grid_preset_size(preset: GridPreset) -> int:
    """Number of backtests in the Cartesian product for a preset (for tests / UI)."""
    s = _PRESETS[preset]
    n = 1
    for v in s.values():
        n *= len(v)
    return n


def three_way_time_split(
    df: pd.DataFrame,
    *,
    ts_col: str = "timestamp",
    calib_frac: float = 0.40,
    val_frac: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Non-overlapping chronological segments (test = remainder)."""
    d = df.sort_values(ts_col).reset_index(drop=True)
    n = len(d)
    if n < 300:
        raise ValueError("Need enough rows for three-way split")

    t0 = d[ts_col].quantile(calib_frac)
    t1 = d[ts_col].quantile(calib_frac + val_frac)

    cal = d.loc[d[ts_col] <= t0].copy()
    val = d.loc[(d[ts_col] > t0) & (d[ts_col] <= t1)].copy()
    tst = d.loc[d[ts_col] > t1].copy()

    if len(cal) < 80 or len(val) < 80 or len(tst) < 80:
        raise ValueError("Segments too small — fetch more trades or adjust fractions")

    return cal, val, tst


@dataclass
class TuningResult:
    best_config: dict[str, Any]
    best_validation_equity: float
    validation_metrics: dict
    test_metrics: dict
    test_equity_curve: pd.Series
    grid_rows: pd.DataFrame
    leaderboard: list[dict[str, Any]] = field(default_factory=list)
    grid_preset: str = "fast"


def _top_unique_by_val(
    grid_df: pd.DataFrame,
    k: int,
) -> list[dict[str, Any]]:
    if k <= 0 or grid_df.empty:
        return []
    g = grid_df.sort_values("val_final_equity", ascending=False)
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for _, row in g.iterrows():
        key = tuple(row[c] for c in _CONFIG_COLS)
        if key in seen:
            continue
        seen.add(key)
        out.append({c: row[c] for c in _CONFIG_COLS} | {"val_final_equity": float(row["val_final_equity"])})
        if len(out) >= k:
            break
    return out


def grid_search_signal_backtest(
    structural: MetaorderSignalParams,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    s_max: float = 5000.0,
    maximize: str = "final_equity",
    grid_preset: GridPreset = "fine",
    leaderboard_k: int = 10,
) -> TuningResult:
    """
    Search signal + execution hyperparameters on **validation** only; evaluate winner once on ``test_df``.

    ``maximize``: ``final_equity`` or ``sharpe_like`` (from ``compute_metrics``).

    ``grid_preset``: ``fast`` (~48) or ``fine`` (128).

    ``leaderboard_k``: number of **distinct** top validation configs (by score) also evaluated on **test**
    (useful for spotting validation–test gaps / overfitting).
    """
    if grid_preset not in _PRESETS:
        raise ValueError(f"Unknown grid_preset {grid_preset!r}; expected one of {tuple(_PRESETS)}")

    spec = _PRESETS[grid_preset]

    rows: list[dict] = []
    best_score = float("-inf")
    best: dict[str, Any] = {}

    for pm, pe, rm, nm, px, sp, kap in product(
        spec["p_min_opts"],
        spec["phi_ent"],
        spec["rho_max_opts"],
        spec["n_min_opts"],
        spec["phi_exit_opts"],
        spec["spread_opts"],
        spec["kappa_opts"],
    ):
        sig_cfg = SignalConfig(
            p_min=pm,
            phi_entry=pe,
            rho_max=rm,
            n_min=nm,
            phi_exit=px,
            s_max=s_max,
            kappa_cost=kap,
        )
        bt_cfg = BacktestConfig(half_spread_frac=sp)
        out = run_event_backtest(val_df, structural, sig_cfg, bt_cfg)
        m = compute_metrics(out, val_df, ic_horizon=5)

        score = m.final_equity if maximize == "final_equity" else (
            m.sharpe_like if m.sharpe_like == m.sharpe_like else -1e9
        )

        row = {
            "p_min": pm,
            "phi_entry": pe,
            "rho_max": rm,
            "n_min": nm,
            "phi_exit": px,
            "half_spread": sp,
            "kappa": kap,
            "val_final_equity": m.final_equity,
            "val_sharpe_like": m.sharpe_like,
            "val_entries": m.n_entries,
            "val_max_dd": m.max_drawdown,
        }
        rows.append(row)

        if score > best_score:
            best_score = score
            best = {
                "p_min": pm,
                "phi_entry": pe,
                "rho_max": rm,
                "n_min": nm,
                "phi_exit": px,
                "half_spread": sp,
                "kappa": kap,
                "s_max": s_max,
            }

    grid_df = pd.DataFrame(rows)

    sig_best = SignalConfig(
        p_min=best["p_min"],
        phi_entry=best["phi_entry"],
        rho_max=best["rho_max"],
        n_min=int(best["n_min"]),
        phi_exit=best["phi_exit"],
        s_max=s_max,
        kappa_cost=best["kappa"],
    )
    bt_best = BacktestConfig(half_spread_frac=best["half_spread"])

    out_val = run_event_backtest(val_df, structural, sig_best, bt_best)
    val_m_obj = compute_metrics(out_val, val_df, ic_horizon=5)

    out_test = run_event_backtest(test_df, structural, sig_best, bt_best)
    test_m = compute_metrics(out_test, test_df, ic_horizon=5)

    leaderboard: list[dict[str, Any]] = []
    rank_configs = _top_unique_by_val(grid_df, leaderboard_k)
    for rank, cfg in enumerate(rank_configs, start=1):
        sc = SignalConfig(
            p_min=cfg["p_min"],
            phi_entry=cfg["phi_entry"],
            rho_max=cfg["rho_max"],
            n_min=int(cfg["n_min"]),
            phi_exit=cfg["phi_exit"],
            s_max=s_max,
            kappa_cost=cfg["kappa"],
        )
        bc = BacktestConfig(half_spread_frac=cfg["half_spread"])
        o_te = run_event_backtest(test_df, structural, sc, bc)
        te_m = compute_metrics(o_te, test_df, ic_horizon=5)
        leaderboard.append(
            {
                "rank": rank,
                "p_min": cfg["p_min"],
                "phi_entry": cfg["phi_entry"],
                "rho_max": cfg["rho_max"],
                "n_min": cfg["n_min"],
                "phi_exit": cfg["phi_exit"],
                "half_spread": cfg["half_spread"],
                "kappa": cfg["kappa"],
                "val_final_equity": cfg["val_final_equity"],
                "test_final_equity": float(o_te["equity"].iloc[-1]),
                "test_sharpe_like": te_m.sharpe_like,
                "test_max_dd": te_m.max_drawdown,
            }
        )

    return TuningResult(
        best_config=best,
        best_validation_equity=float(val_m_obj.final_equity),
        validation_metrics=metrics_to_dict(val_m_obj),
        test_metrics={
            "final_equity": float(out_test["equity"].iloc[-1]),
            "sharpe_like": test_m.sharpe_like,
            "n_entries": test_m.n_entries,
            "max_drawdown": test_m.max_drawdown,
            "ic": test_m.ic_survival_fwd_ret,
        },
        test_equity_curve=out_test["equity"].copy(),
        grid_rows=grid_df,
        leaderboard=leaderboard,
        grid_preset=grid_preset,
    )
