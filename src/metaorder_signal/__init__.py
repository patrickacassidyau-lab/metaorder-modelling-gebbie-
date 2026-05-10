"""Metaorder detection signal from structural LMF-style estimation (public TAQ)."""

from metaorder_signal.signal import MetaorderSignalParams, SignalConfig, process_trade_stream
from metaorder_signal.backtest import BacktestConfig, run_event_backtest
from metaorder_signal.synthetic_market import SyntheticMarketConfig, generate_panel, write_synthetic_dataset

__all__ = [
    "MetaorderSignalParams",
    "SignalConfig",
    "process_trade_stream",
    "BacktestConfig",
    "run_event_backtest",
    "SyntheticMarketConfig",
    "generate_panel",
    "write_synthetic_dataset",
]
