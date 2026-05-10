"""Composite stylised-fact loss for data-driven N selection (equation loss / nhat)."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def loss_sql_r2(log_x: np.ndarray, log_y: np.ndarray) -> float:
    """
    L1: 1 - R² from OLS log-log (lower is better); clamp to [0, 2].

    Square-root law in logs: log I ~ const + 0.5 log(Q/V) ideally; we penalise lack of linear fit.
    """
    x = np.asarray(log_x, dtype=float)
    y = np.asarray(log_y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 5:
        return 1.0
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = np.sum(x0**2)
    if denom <= 0:
        return 1.0
    beta = np.sum(x0 * y0) / denom
    y_hat = y.mean() + beta * x0
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return 1.0
    r2 = 1.0 - ss_res / ss_tot
    return float(np.clip(1.0 - r2, 0.0, 2.0))


def loss_time_independence(impacts_by_bin: np.ndarray) -> float:
    """
    L2: Normalised variance of mean impacts across duration bins (lower is better).

    Impacts should not systematically depend on execution duration under idealisation.
    """
    v = np.asarray(impacts_by_bin, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return 0.0
    m = float(np.mean(np.abs(v)))
    if m <= 0:
        return float(np.var(v, ddof=1))
    return float(np.var(v, ddof=1) / m**2)


def loss_concavity_deviation(gamma2: float, *, target: float = 0.5) -> float:
    """L3: |gamma2 - 0.5|."""
    if not np.isfinite(gamma2):
        return 1.0
    return float(abs(float(gamma2) - target))


def decay_profile(z: np.ndarray, beta: float) -> np.ndarray:
    """Bracket from equation (decay) for z >= 1 (dimensionless post-completion shape)."""
    z = np.asarray(z, dtype=float)
    return np.power(np.maximum(z, 1e-9), 1.0 - beta) - np.power(
        np.maximum(z - 1.0, 1e-9), 1.0 - beta
    )


def loss_decay_rmse(
    z: np.ndarray,
    y_scaled: np.ndarray,
    *,
    beta: float,
    gamma0: float,
) -> float:
    """
    L4: RMSE of scaled impact vs gamma0 * decay_profile(z).

    y_scaled should align with I(Q,z) / (sigma_D sqrt(Q)).
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y_scaled, dtype=float)
    m = np.isfinite(z) & np.isfinite(y) & (z >= 1.0)
    z, y = z[m], y[m]
    if z.size < 5:
        return 1.0
    y_hat = float(gamma0) * decay_profile(z, float(beta))
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def composite_loss(
    l1: float,
    l2: float,
    l3: float,
    l4: float,
    weights: Iterable[float] | None = None,
) -> float:
    """Equation (loss): weighted sum; default equal 0.25 each."""
    w = np.asarray(list(weights) if weights is not None else [0.25, 0.25, 0.25, 0.25], dtype=float)
    w = w / w.sum()
    parts = np.array([l1, l2, l3, l4], dtype=float)
    parts[~np.isfinite(parts)] = 1.0
    return float(np.dot(w, parts))


def select_n_argmin(grid_loss: dict[int, float]) -> int:
    """Equation (nhat): argmin over candidate N."""
    if not grid_loss:
        raise ValueError("empty grid")
    return int(min(grid_loss.keys(), key=lambda k: grid_loss[k]))
