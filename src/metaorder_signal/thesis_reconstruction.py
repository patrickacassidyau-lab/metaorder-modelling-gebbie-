"""
UCT thesis-style synthetic trader assignment + metaorder segmentation.

Uses participation weights + inverse-CDF trade routing (see ``uct_auxiliary``), then splits each
trader stream into metaorders at sign changes (length ≥ 2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metaorder_signal.reconstruction import MetaorderRecord
from metaorder_signal.uct_auxiliary import (
    cumulative_probs,
    metaorders_segment_same_sign,
    orders,
    trader_participation,
)


def reconstruct_metaorders_uct(
    trades: pd.DataFrame,
    n_traders: int,
    *,
    sign_col: str = "sign",
    qty_col: str = "quantity",
    ts_col: str = "timestamp",
    participation_method: str = "homogenous",
    alpha: float = 2.0,
    f_min: int = 1,
    f_max: int = 1000,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Return ``trades`` plus ``synthetic_trader`` (0..N-1) and ``metaorder_id`` (global).

    Rows belonging to thesis metaorders of length 1 do **not** receive a ``metaorder_id``
    (NaN) — matching the thesis notebooks which analyse only multi-child metaorders.
    Every row receives ``synthetic_trader``.
    """
    if n_traders < 1:
        raise ValueError("n_traders must be >= 1")

    df = trades.sort_values(ts_col).reset_index(drop=True)
    df["Row"] = np.arange(len(df), dtype=int)
    df["Trade Sign"] = np.sign(df[sign_col].astype(float))

    participation = trader_participation(
        n_traders,
        alpha=alpha,
        f_min=f_min,
        f_max=f_max,
        method=participation_method,
        seed=seed,
    )
    c = cumulative_probs(participation)
    assignments = orders(n_traders, df, c)

    trader_ids = np.full(len(df), -1, dtype=int)
    for k in range(n_traders):
        for idx in assignments[k]:
            trader_ids[idx] = k

    mo_ids = np.full(len(df), np.nan)
    next_mo = 0

    for k in range(n_traders):
        idxs = assignments[k]
        if not idxs:
            continue
        sub = df.loc[sorted(idxs)].copy()
        mos = metaorders_segment_same_sign(sub)
        for mo in mos:
            if mo.empty:
                continue
            for _, row in mo.iterrows():
                mo_ids[int(row["Row"])] = next_mo
            next_mo += 1

    out = trades.sort_values(ts_col).reset_index(drop=True).copy()
    out["synthetic_trader"] = trader_ids
    out["metaorder_id"] = mo_ids
    out["thesis_metaorder_eligible"] = np.isfinite(mo_ids)
    return out


def build_metaorder_records_uct(
    enriched: pd.DataFrame,
    *,
    sign_col: str = "sign",
    qty_col: str = "quantity",
) -> list[MetaorderRecord]:
    """``MetaorderRecord`` rows derived from ``metaorder_id`` groups (for ``loss_fn`` hooks)."""
    records: list[MetaorderRecord] = []
    work = enriched.reset_index(drop=True)
    if "metaorder_id" not in work.columns:
        return records

    next_id = 0
    for mo_id, grp in work.groupby("metaorder_id"):
        if not np.isfinite(mo_id):
            continue
        idx = grp.index.to_numpy()
        start_idx = int(idx.min())
        end_idx = int(idx.max())
        tid = int(grp["synthetic_trader"].iloc[0])
        s = int(np.sign(grp.iloc[0][sign_col]))
        vol = float(grp[qty_col].sum())
        n_ch = int(len(grp))
        records.append(
            MetaorderRecord(
                metaorder_id=next_id,
                trader_id=tid,
                sign=s,
                n_children=n_ch,
                volume=vol,
                start_idx=start_idx,
                end_idx=end_idx,
            )
        )
        next_id += 1

    return records


def metaorder_lengths_from_labels(df: pd.DataFrame) -> np.ndarray:
    """Lengths of trades sharing the same finite metaorder_id (for power-law fitting)."""
    sub = df[np.isfinite(df["metaorder_id"])].copy()
    if sub.empty:
        return np.array([])
    lengths = sub.groupby("metaorder_id").size().to_numpy(dtype=float)
    return lengths
