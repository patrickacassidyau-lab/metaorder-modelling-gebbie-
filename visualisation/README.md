# Visualisation scripts

Runnable helpers (not installed as importable package modules). From repo root with venv activated:

| Script | Purpose |
|--------|---------|
| **`visualise_run.py`** | One symbol: mid, entries/exits, `survival_p`, `phi`, backtest **equity**. Needs `timestamp`, `mid`, `quantity`, `sign`; optional `symbol` column to filter. |
| **`visualise_panel_overview.py`** | Dataset-wide bar chart of trade counts per symbol from **`manifest.json`** produced by `examples/generate_synthetic_market.py --output-dir …`. |

### Examples

```bash
# Smoke test (no CSV on disk) — uses slightly relaxed entry thresholds so equity isn’t a flat zero line
python visualisation/visualise_run.py --demo

# If you already created data/synthetic_panel.csv, you can omit --csv
python visualisation/visualise_run.py --out plots/run.png

# Single combined CSV (or one symbol file from per-symbol export)
python visualisation/visualise_run.py --csv data/synthetic_panel.csv --symbol SYM0001 --out plots/SYM0001.png

# After generating with --output-dir data/large_panel/
python visualisation/visualise_run.py --csv data/large_panel/SYM0001.csv --out plots/SYM0001.png

python visualisation/visualise_panel_overview.py --manifest data/large_panel/manifest.json --out plots/overview.png
# Or omit --manifest if a manifest exists under data/**/
python visualisation/visualise_panel_overview.py --out plots/overview.png
```

**Compatibility:** `examples/visualise_run.py` forwards to `visualisation/visualise_run.py` so older docs/commands still work.

Requires **matplotlib** (installed as a dependency of `powerlaw` in this project).
