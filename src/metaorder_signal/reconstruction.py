"""
Synthetic metaorder reconstruction for public TAQ-style streams.

Full Maitrier et al. generative assignment is not replicated here. For a **baseline**,
trades are assigned to N synthetic traders via modular indexing, and each trader's
stream is split into metaorders at sign changes.

For **UCT thesis–consistent** inverse-CDF routing and metaorder segmentation, use
``metaorder_signal.thesis_reconstruction.reconstruct_metaorders_uct`` (Ezra Goliath MSc code).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MetaorderRecord:
    metaorder_id: int
    trader_id: int
    sign: int
    n_children: int
    volume: float
    start_idx: int
    end_idx: int


def assign_synthetic_traders(trades: pd.DataFrame, n_traders: int) -> pd.Series:
    """Round-robin trader assignment by chronological trade index."""
    if n_traders < 1:
        raise ValueError("n_traders must be >= 1")
    idx = np.arange(len(trades), dtype=int)
    return pd.Series(idx % n_traders, index=trades.index, name="synthetic_trader")


def build_metaorders_from_trader_streams(
    trades: pd.DataFrame,
    *,
    sign_col: str = "sign",
    qty_col: str = "quantity",
    trader_col: str = "synthetic_trader",
) -> tuple[pd.DataFrame, list[MetaorderRecord]]:
    """
    Within each synthetic trader, consecutive trades with constant sign form one metaorder.
    """
    if trader_col not in trades.columns:
        raise KeyError(trader_col)

    work = trades.reset_index(drop=True).copy()
    rows: list[dict] = []
    records: list[MetaorderRecord] = []

    next_mo_id = 0
    for tid in sorted(work[trader_col].unique()):
        sub = work.loc[work[trader_col] == tid]
        if sub.empty:
            continue
        prev_sign = None
        buf_idx: list[int] = []

        def flush():
            nonlocal next_mo_id, buf_idx
            if not buf_idx:
                return
            w = work.loc[buf_idx]
            s = int(w.iloc[0][sign_col])
            vol = float(w[qty_col].sum())
            rec = MetaorderRecord(
                metaorder_id=next_mo_id,
                trader_id=int(tid),
                sign=s,
                n_children=len(buf_idx),
                volume=vol,
                start_idx=int(buf_idx[0]),
                end_idx=int(buf_idx[-1]),
            )
            records.append(rec)
            for j in buf_idx:
                rows.append({"trade_row": j, "metaorder_id": next_mo_id})
            next_mo_id += 1
            buf_idx = []

        for i, r in sub.iterrows():
            sgn = int(np.sign(r[sign_col]))
            if sgn == 0:
                continue
            if prev_sign is None or sgn == prev_sign:
                buf_idx.append(int(i))
                prev_sign = sgn
            else:
                flush()
                buf_idx = [int(i)]
                prev_sign = sgn
        flush()

    mo_map = pd.DataFrame(rows)
    if mo_map.empty:
        out = work.copy()
        out["metaorder_id"] = np.nan
        return out, []

    mo_map = mo_map.sort_values("trade_row")
    out = work.merge(mo_map, left_index=True, right_on="trade_row", how="left")
    out = out.drop(columns=["trade_row"])
    return out, records
