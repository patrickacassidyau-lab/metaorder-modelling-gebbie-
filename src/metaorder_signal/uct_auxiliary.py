"""
Algorithms adapted from Ezra Goliath's UCT MSc thesis implementation.

Original module: ``modules/auxiliary_functions.py`` in
https://github.com/EzraGoliath/Metaorder-modelling-and-identification-Msc-thesis-

Licensed under the upstream repository terms (no separate LICENSE file in source snapshot).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trader_participation(
    N: int,
    alpha: float = 2,
    f_min: int = 1,
    f_max: int = 1000,
    method: str = "homogenous",
    seed: int = 1,
) -> np.ndarray:
    """Participation weights for N synthetic traders (thesis ``trader_participation``)."""
    if method == "power":
        np.random.seed(seed)
        u = np.random.uniform(0, 1, N)
        samples = f_min * (1 - u) ** (-1 / (alpha - 1))
        samples = np.clip(samples, f_min, f_max)
        samples = np.round(samples).astype(int)
        return samples

    if method == "uniform":
        np.random.seed(seed)
        samples = np.random.uniform(f_min, f_max, N)
        samples = np.round(samples).astype(int)
        return samples

    if method == "homogenous":
        samples = np.linspace(f_min, f_max, N + 1)
        samples = samples[samples > f_min]
        return samples

    raise ValueError(f"unknown participation method: {method}")


def cumulative_probs(participation: np.ndarray) -> np.ndarray:
    """Cumulative sampling thresholds (thesis ``cumulative_probs``)."""
    p_i = participation / sum(participation)
    c_0 = [0]
    c_i = np.cumsum(p_i)
    c = np.concatenate((c_0, c_i))
    return c


def orders(N: int, trades: pd.DataFrame, cumulative_probs_arr: np.ndarray) -> list[list[int]]:
    """
    Assign each chronological trade index to a synthetic trader (thesis ``orders``).

    Uses inverse-CDF sampling with ``np.random.seed(i)`` per trade index ``i`` as in the thesis.
    """
    number_trades = trades.shape[0]
    assignments: list[list[int]] = [[] for _ in range(N)]

    for i in range(number_trades):
        np.random.seed(i)
        u = np.random.uniform(0 + 1e-10, 1 - 1e-10)
        trader_index = int(np.searchsorted(cumulative_probs_arr, u, side="right"))
        assignments[trader_index - 1].append(i)

    return assignments


def metaorders_segment_same_sign(traders_trades: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Metaorders = maximal segments of ≥2 consecutive trades with fixed Trade Sign for one trader.

    Matches thesis intent; fixes upstream behaviour when **all** trades share one sign (thesis
    ``metaorders`` returned an empty frame in that case — clearly unintentional for |stream|≥2).
    Segments of length 1 are omitted (thesis convention for stylised-fact tables).
    """
    if traders_trades.shape[0] <= 1:
        return []

    ts = traders_trades["Trade Sign"].to_numpy()
    diffs = np.diff(ts)
    indices = np.where(diffs != 0)[0] + 1

    bounds = [0] + list(indices) + [traders_trades.shape[0]]
    traders_metaorders: list[pd.DataFrame] = []

    for i in range(len(bounds) - 1):
        seg = traders_trades.iloc[bounds[i] : bounds[i + 1]]
        n = seg.shape[0]
        if n >= 2:
            traders_metaorders.append(seg)

    return traders_metaorders
