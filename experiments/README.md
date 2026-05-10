# Empirical studies

| Script | Purpose |
|--------|---------|
| **`run_empirical_study.py`** | Time-split calibration → hold-out backtest; writes **`results/empirical/report_*.json`**. |

### Example (offline CSV)

```bash
python experiments/run_empirical_study.py --csv path/to/trades.csv --report-dir results/empirical
```

### Example (public Binance agg trades — requires network)

```bash
python experiments/run_empirical_study.py --fetch-binance BTCUSDT --max-trades 60000
```

**Interpretation:** metrics are research diagnostics (scaled toy costs), not live-trading forecasts. See repo README “Empirical backbone”.
