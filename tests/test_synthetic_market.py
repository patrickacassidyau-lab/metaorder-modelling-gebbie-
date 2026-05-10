import numpy as np

from metaorder_signal.synthetic_market import (
    SyntheticMarketConfig,
    _sample_metaorder_length,
    generate_panel,
    generate_symbol_trades,
    trades_for_signal,
)


def test_symbol_stream_contract():
    rng = np.random.default_rng(7)
    cfg = SyntheticMarketConfig(n_metaorders=30, length_cap=80)
    df = generate_symbol_trades("TEST", rng, cfg)
    assert not df.empty
    assert {"timestamp", "mid", "quantity", "sign"}.issubset(df.columns)
    ts = trades_for_signal(df)
    assert len(ts) == len(df)


def test_panel_multiple_symbols_balanced():
    syms = ["A", "B", "C"]
    cfg = SyntheticMarketConfig(n_metaorders=20)
    panel = generate_panel(syms, seed=11, cfg=cfg)
    assert panel["symbol"].nunique() == 3
    counts = panel.groupby("symbol").size()
    assert counts.min() >= 10


def test_lengths_have_heavy_tail():
    rng = np.random.default_rng(0)
    cfg = SyntheticMarketConfig(length_cap=200)
    lengths = [_sample_metaorder_length(rng, cfg) for _ in range(800)]
    assert float(np.mean(lengths)) > 1.2
    assert int(np.max(lengths)) >= 15
