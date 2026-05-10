"""Extract same-sign run lengths from Lee–Ready-style signed trades."""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_run_lengths(trades: pd.DataFrame, *, sign_col: str = "sign") -> np.ndarray:
    """
    Lengths of maximal contiguous segments of constant non-zero sign (chronological).
    """
    df = trades.sort_values("timestamp").reset_index(drop=True)
    sgn = np.sign(df[sign_col].to_numpy(dtype=float)).astype(int)
    lengths: list[int] = []
    cur = 0
    prev = 0

    for s in sgn:
        if s == 0:
            continue
        if cur == 0:
            cur = 1
            prev = s
        elif s == prev:
            cur += 1
        else:
            lengths.append(cur)
            cur = 1
            prev = s

    if cur > 0:
        lengths.append(cur)

    return np.asarray(lengths, dtype=float)
