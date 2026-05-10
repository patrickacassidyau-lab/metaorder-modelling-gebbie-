"""
Fetch public aggregate trades from Binance (no API key).

Maps to our tape schema: timestamp, mid≈trade price, quantity, sign from aggressor side.

Sign convention: +1 buyer-initiated (buyer is taker), -1 seller-initiated.
See Binance ``isBuyerMaker``: True → buyer was maker → seller aggressed → sign -1.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

import pandas as pd


BASE_URL = "https://api.binance.com/api/v3/aggTrades"


def _get_json(url: str, *, timeout: float = 30.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "metaorder-signal-empirical/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_agg_trades(
    symbol: str = "BTCUSDT",
    *,
    max_trades: int = 50_000,
    pause_s: float = 0.08,
) -> pd.DataFrame:
    """
    Paginate aggTrades until ``max_trades`` or API exhaustion.

    Columns: timestamp (UTC), mid, quantity, sign, raw_trade_id, is_buyer_maker.
    """
    symbol = symbol.upper().strip()
    rows: list[dict] = []
    last_id: int | None = None
    page = 1000

    while len(rows) < max_trades:
        take = min(page, max_trades - len(rows))
        q = f"symbol={symbol}&limit={take}"
        if last_id is not None:
            q += f"&fromId={last_id + 1}"
        url = f"{BASE_URL}?{q}"
        try:
            chunk = _get_json(url)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Binance HTTP error: {exc}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network error (offline?): {exc}") from exc

        if not chunk:
            break

        for a in chunk:
            if len(rows) >= max_trades:
                break
            price = float(a["p"])
            qty = float(a["q"])
            is_maker = bool(a["m"])
            sign = -1 if is_maker else 1
            ts = pd.to_datetime(int(a["T"]), unit="ms", utc=True)
            rows.append(
                {
                    "timestamp": ts,
                    "mid": price,
                    "quantity": qty,
                    "sign": sign,
                    "raw_trade_id": int(a["a"]),
                    "is_buyer_maker": is_maker,
                }
            )

        last_id = int(chunk[-1]["a"])
        if len(chunk) < take or len(rows) >= max_trades:
            break
        time.sleep(pause_s)

    df = pd.DataFrame(rows[:max_trades])
    if df.empty:
        return df
    return df.sort_values("timestamp").reset_index(drop=True)
