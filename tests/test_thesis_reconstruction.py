import numpy as np
import pandas as pd

from metaorder_signal.thesis_reconstruction import (
    metaorder_lengths_from_labels,
    reconstruct_metaorders_uct,
)


def _tape(n: int = 200):
    rng = np.random.default_rng(0)
    t0 = pd.Timestamp("2025-06-01T12:00:00Z")
    ts = t0 + pd.to_timedelta(np.sort(rng.uniform(0, 600, n)), unit="s")
    mid = 50 + np.cumsum(rng.normal(0, 0.01, n))
    q = rng.uniform(0.5, 3.0, n)
    s = rng.choice([-1, 1], n)
    return pd.DataFrame({"timestamp": ts, "mid": mid, "quantity": q, "sign": s})


def test_uct_assigns_all_traders():
    tape = _tape(120)
    out = reconstruct_metaorders_uct(tape, n_traders=25)
    assert (out["synthetic_trader"] >= 0).all()
    assert out["synthetic_trader"].max() < 25


def test_metaorder_lengths_positive():
    tape = _tape(300)
    out = reconstruct_metaorders_uct(tape, n_traders=40, participation_method="homogenous")
    L = metaorder_lengths_from_labels(out)
    assert L.size > 0
    assert np.all(L >= 2)


def test_homogenous_participation_length_matches_n():
    from metaorder_signal.uct_auxiliary import trader_participation

    for n in (10, 50, 100):
        p = trader_participation(n, method="homogenous", seed=1)
        assert len(p) == n
