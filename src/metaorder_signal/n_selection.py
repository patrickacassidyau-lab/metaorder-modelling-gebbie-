"""Grid search for synthetic trader count N (equation nhat) using composite loss."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from metaorder_signal.losses import composite_loss, select_n_argmin
from metaorder_signal.reconstruction import assign_synthetic_traders, build_metaorders_from_trader_streams
from metaorder_signal.thesis_reconstruction import build_metaorder_records_uct, reconstruct_metaorders_uct


def default_n_grid() -> list[int]:
    return [25, 50, 75, 100, 150, 200]


def evaluate_n(
    trades: pd.DataFrame,
    n_traders: int,
    loss_fn: Callable[[pd.DataFrame, list], tuple[float, float, float, float]],
    *,
    sign_col: str = "sign",
    qty_col: str = "quantity",
) -> float:
    """Assign traders, build metaorders, return composite loss."""
    a = assign_synthetic_traders(trades, n_traders)
    enriched = trades.copy()
    enriched["synthetic_trader"] = a.to_numpy()
    _, records = build_metaorders_from_trader_streams(enriched, sign_col=sign_col, qty_col=qty_col)
    l1, l2, l3, l4 = loss_fn(enriched, records)
    return composite_loss(l1, l2, l3, l4)


def evaluate_n_uct(
    trades: pd.DataFrame,
    n_traders: int,
    loss_fn: Callable[[pd.DataFrame, list], tuple[float, float, float, float]],
    *,
    sign_col: str = "sign",
    qty_col: str = "quantity",
    participation_method: str = "homogenous",
    alpha: float = 2.0,
    seed: int = 1,
) -> float:
    """Same as ``evaluate_n`` but uses UCT thesis-style inverse-CDF trader routing."""
    enriched = reconstruct_metaorders_uct(
        trades,
        n_traders,
        sign_col=sign_col,
        qty_col=qty_col,
        participation_method=participation_method,
        alpha=alpha,
        seed=seed,
    )
    records = build_metaorder_records_uct(enriched, sign_col=sign_col, qty_col=qty_col)
    l1, l2, l3, l4 = loss_fn(enriched, records)
    return composite_loss(l1, l2, l3, l4)


def grid_search_n(
    trades: pd.DataFrame,
    loss_fn: Callable[[pd.DataFrame, list], tuple[float, float, float, float]],
    *,
    grid: list[int] | None = None,
) -> tuple[int, dict[int, float]]:
    """Equation (nhat): argmin_N L(N)."""
    grid = grid or default_n_grid()
    losses = {int(n): evaluate_n(trades, int(n), loss_fn) for n in grid}
    return select_n_argmin(losses), losses


def grid_search_n_uct(
    trades: pd.DataFrame,
    loss_fn: Callable[[pd.DataFrame, list], tuple[float, float, float, float]],
    *,
    grid: list[int] | None = None,
    participation_method: str = "homogenous",
    alpha: float = 2.0,
    seed: int = 1,
) -> tuple[int, dict[int, float]]:
    """UCT thesis reconstruction variant of ``grid_search_n``."""
    grid = grid or default_n_grid()
    losses = {
        int(n): evaluate_n_uct(
            trades,
            int(n),
            loss_fn,
            participation_method=participation_method,
            alpha=alpha,
            seed=seed,
        )
        for n in grid
    }
    return select_n_argmin(losses), losses
