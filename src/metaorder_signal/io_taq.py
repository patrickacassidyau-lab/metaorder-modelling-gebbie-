"""Load normalised tick/trade tables."""

from __future__ import annotations

import pandas as pd


def load_trades_csv(
    path: str,
    *,
    ts_col: str = "timestamp",
    mid_col: str = "mid",
    qty_col: str = "quantity",
    sign_col: str = "sign",
) -> pd.DataFrame:
    """Read CSV; requires mid, quantity, sign (+1 buy / -1 sell), timestamp."""
    df = pd.read_csv(path)
    missing = {ts_col, mid_col, qty_col, sign_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    return df
