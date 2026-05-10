import numpy as np
import pandas as pd

from metaorder_signal.empirical.tuning import three_way_time_split


def _tape(n: int = 800):
    rng = np.random.default_rng(1)
    t0 = pd.Timestamp("2025-01-01T00:00:00Z")
    ts = t0 + pd.to_timedelta(np.sort(rng.uniform(0, 1e6, n)), unit="s")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "mid": 100 + np.cumsum(rng.normal(0, 0.01, n)),
            "quantity": rng.uniform(0.1, 2.0, n),
            "sign": rng.choice([-1, 1], n),
        }
    )


def test_three_way_split_lengths():
    tape = _tape(1000)
    a, b, c = three_way_time_split(tape, calib_frac=0.4, val_frac=0.3)
    assert len(a) > 100 and len(b) > 100 and len(c) > 100
