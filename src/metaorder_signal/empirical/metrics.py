"""Performance metrics for empirical studies (no claim of live trading validity)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except ImportError:  # pragma: no cover
    spearmanr = None


@dataclass
class StudyMetrics:
    final_equity: float
    mean_equity_step: float
    vol_equity_step: float
    sharpe_like: float
    max_drawdown: float
    n_trades_signal: int
    n_entries: int
    ic_survival_fwd_ret: float | None
    baseline_final_equity: float | None


def _max_drawdown(equity: np.ndarray) -> float:
    """Peak-to-trough drop relative to the series scale (robust when equity crosses zero)."""
    equity = np.asarray(equity, dtype=float)
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    underwater = peak - equity
    scale = float(np.max(np.abs(equity)) + 1e-12)
    return float(np.max(underwater) / scale)


def _sharpe_like(x: np.ndarray, *, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd < eps:
        return float("nan")
    return mu / sd * np.sqrt(float(x.size))


def forward_mid_returns(trades: pd.DataFrame, horizons: int = 5) -> pd.Series:
    """Log return from current mid to mid ``horizons`` trades ahead."""
    df = trades.sort_values("timestamp").reset_index(drop=True)
    m = df["mid"].to_numpy(dtype=float)
    out = np.full(len(df), np.nan)
    for i in range(len(df) - horizons):
        if m[i] > 0 and m[i + horizons] > 0:
            out[i] = np.log(m[i + horizons]) - np.log(m[i])
    return pd.Series(out, index=df.index)


def ic_survival_forward(
    sig_out: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    horizon: int = 5,
) -> float | None:
    """Spearman IC between survival_p and forward log mid return."""
    if spearmanr is None:
        return None
    fw = forward_mid_returns(trades, horizons=horizon).to_numpy()
    surv = sig_out["survival_p"].to_numpy()
    m = np.isfinite(surv) & np.isfinite(fw)
    if m.sum() < 50:
        return None
    r, _ = spearmanr(surv[m], fw[m])
    return float(r) if np.isfinite(r) else None


def baseline_random_signs(
    trades: pd.DataFrame,
    *,
    seed: int = 0,
    cost_frac: float = 0.0002,
) -> float:
    """
    Strawman: flip coin for direction each trade, accumulate proportional costs only.

    Returns approximate cumulative cash impact of random churn (negative).
    """
    rng = np.random.default_rng(seed)
    cash = 0.0
    for _ in range(len(trades)):
        cash -= cost_frac
    return float(cash)


def compute_metrics(
    result_df: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    ic_horizon: int = 5,
) -> StudyMetrics:
    """``result_df`` must contain equity column from ``run_event_backtest``."""
    eq = result_df["equity"].to_numpy(dtype=float)
    steps = np.diff(np.r_[0.0, eq])

    surv_ic = None
    try:
        surv_ic = ic_survival_forward(result_df, trades, horizon=ic_horizon)
    except Exception:
        surv_ic = None

    ent = int(result_df["entry_long"].sum() + result_df["entry_short"].sum())
    return StudyMetrics(
        final_equity=float(eq[-1]) if len(eq) else 0.0,
        mean_equity_step=float(np.mean(steps)),
        vol_equity_step=float(np.std(steps, ddof=1)) if len(steps) > 2 else float("nan"),
        sharpe_like=float(_sharpe_like(steps)),
        max_drawdown=_max_drawdown(eq),
        n_trades_signal=int(len(result_df)),
        n_entries=ent,
        ic_survival_fwd_ret=surv_ic,
        baseline_final_equity=None,
    )


def metrics_to_dict(m: StudyMetrics) -> dict:
    return {k: v for k, v in asdict(m).items()}
