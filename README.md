# Metaorder signal (LMF-structured)

Python implementation of the structural metaorder detection pipeline described in `docs/metaorder_signal.tex`: volatility estimators (range, Parkinson, Yang–Zhang), Clauset-style power-law fitting via `powerlaw`, synthetic-trader sensitivity grid for \(N\), composite stylised-fact loss, tick-level signal algebra, and an event-driven backtest with square-root impact costs and theory-style capacity \(q < 10^{-4} V_D\).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

- Example end-to-end run on synthetic ticks: `python examples/synthetic_demo.py`
- **Synthetic data (large runs):** use **`--output-dir`** so each symbol is written as its own file plus **`manifest.json`** (avoids one giant CSV in RAM):  
  `python examples/generate_synthetic_market.py --symbols 2000 --metaorders 1500 --sessions 2 --output-dir data/large_panel --format csv --meta-json data/large_meta.json`  
  Defaults are already large (**500 symbols**, **1200 metaorders/symbol/session**, denser inter-trade spacing). Use `--mean-inter-trade 0.12` for even more ticks.
- **Visualisers** are in **`visualisation/`** (not under `examples/`): see **`visualisation/README.md`**. Quick start:  
  `python visualisation/visualise_run.py --csv data/large_panel/SYM0001.csv --out plots/SYM0001.png`  
  `python visualisation/visualise_panel_overview.py --manifest data/large_panel/manifest.json --out plots/overview.png`  
  (`examples/visualise_run.py` still forwards to the canonical script.)
- Optional Parquet shards: `pip install -e ".[parquet]"` then `--output-dir … --format parquet`.
- Load trades with `metaorder_signal.io_taq.load_trades_csv` (expects `timestamp`, `mid`, `quantity`, `sign`; extra columns like `symbol` are allowed).
- Core API: `process_trade_stream`, `run_event_backtest`, `write_synthetic_dataset`, `generate_panel` from `metaorder_signal`.

## UCT thesis reconstruction (Ezra Goliath)

The inverse-CDF synthetic trader assignment + same-sign metaorder segmentation from the UCT MSc thesis are ported as:

- ``metaorder_signal.uct_auxiliary`` — ``trader_participation``, ``cumulative_probs``, ``orders``, ``metaorders_segment_same_sign``
- ``metaorder_signal.thesis_reconstruction.reconstruct_metaorders_uct`` — adds ``synthetic_trader`` and ``metaorder_id`` columns
- ``metaorder_signal.n_selection.evaluate_n_uct`` / ``grid_search_n_uct`` — \( \hat N \) search using thesis routing

Source attribution: [EzraGoliath/Metaorder-modelling-and-identification-Msc-thesis-](https://github.com/EzraGoliath/Metaorder-modelling-and-identification-Msc-thesis-) (`modules/auxiliary_functions.py`). One upstream edge case is patched (when a trader’s stream has a single sign, it now forms one metaorder when length ≥ 2). Singleton segments remain excluded from ``metaorder_id``, matching thesis stylised-fact tables.

Optional reference clone (ignored by git): ``git clone … vendor/uct_msc_thesis``.

## Empirical backbone

Hold-out evaluation **without peeking** at the test segment when fitting structure:

1. **Time split** — earlier trades → **calibration**, later trades → **evaluation** (`experiments/run_empirical_study.py`).
2. **Calibration** — same-sign **run lengths** → Clauset-style **`powerlaw`** tail \(\hat{\alpha}\); **\(V_D\)** = total traded qty in the calibration window; **\(\sigma_D\)** = std of consecutive **log-mid** returns (dimensionless scale for impact).
3. **Hold-out** — freeze **`MetaorderSignalParams`**, run **`run_event_backtest`**, report JSON metrics (final equity, Sharpe-like on equity steps, scale-normalized max drawdown, entry count, Spearman **IC** between survival and forward log-mid returns, random-churn strawman).
4. **Data** — bring your own CSV, or **`--fetch-binance BTCUSDT`** for **public** aggregate trades (spot, no API key; **not** equity L2 data).

```bash
python experiments/run_empirical_study.py --fetch-binance BTCUSDT --max-trades 50000 --report-dir results/empirical
```

Programmatic API: `metaorder_signal.empirical` (`calibrate_from_trades`, `compute_metrics`, `extract_run_lengths`). **Caveat:** live microstructure differs from crypto; use this as a **reproducible pipeline**, not proof of edge.

## Source PDF

Empirical targets and notation follow Goliath & Gebbie, *Metaorder modelling and identification from public data* ([arXiv:2602.19590](https://arxiv.org/abs/2602.19590)).

## Tuning notebook (validation grid → held-out test plots)

```bash
pip install -e ".[notebook]"
jupyter nbconvert --execute notebooks/metaorder_tuning_report.ipynb --to notebook \
  --output metaorder_tuning_report_executed.ipynb
```

Generates Matplotlib PNGs under `notebooks/` (`fig_heatmap_val_equity.png`, `fig_grid_distributions.png`, `fig_equity_val_vs_test.png`). First run fetches Binance trades into `data/binance_BTCUSDT_tuning.csv` (cached).

Hyperparameters are tuned on the **validation** segment only; **test** metrics are out-of-sample but still vulnerable to overfitting and stylised costs — **not** a profitability promise.

## Tests

```bash
pytest
```
