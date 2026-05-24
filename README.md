# Metaorder signal (LMF-structured)

Python implementation of the structural metaorder pipeline described in `docs/metaorder_signal.tex`: volatility estimators (range, Parkinson, Yang-Zhang), Clauset-style power-law fitting via `powerlaw`, synthetic-trader sensitivity for \(N\), composite stylised-fact loss, tick-level signal algebra, and an event-driven backtest with square-root impact costs and theory-style capacity \(q < 10^{-4} V_D\) implemented as `BacktestConfig.capacity_frac` (default \(10^{-4}\)) times calibrated \(V_D\).

Notation follows Goliath & Gebbie, *Metaorder modelling and identification from public data* ([arXiv:2602.19590](https://arxiv.org/abs/2602.19590)).

uses public binance trades
