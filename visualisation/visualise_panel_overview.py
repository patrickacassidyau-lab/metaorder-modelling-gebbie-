#!/usr/bin/env python3
"""
Summary chart for a synthetic dataset written with ``write_synthetic_dataset`` (manifest.json).

Plots trade counts per symbol (horizontal bars) and prints totals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _resolve_manifest(arg: Path | None) -> Path | None:
    if arg is not None:
        if arg.is_file():
            return arg
        print(f"Manifest not found: {arg}", file=sys.stderr)
        sys.exit(2)
    candidates = [
        Path("data/large_panel/manifest.json"),
        Path("data/synthetic_panel/manifest.json"),
        Path("manifest.json"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    data = Path("data")
    if data.is_dir():
        found = sorted(data.rglob("manifest.json"))
        if found:
            return found[0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overview plot from synthetic manifest.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Looks for manifest.json under ./data/... if --manifest is omitted.

Examples:
  %(prog)s --manifest data/large_panel/manifest.json
  %(prog)s   # auto-discover first data/**/manifest.json
""",
    )
    ap.add_argument("--manifest", type=Path, default=None, help="manifest.json (optional if discoverable)")
    ap.add_argument("--out", type=Path, default=Path("plots/panel_overview.png"))
    ap.add_argument("--top", type=int, default=60, help="Max symbols to show (by row count)")
    args = ap.parse_args()

    mpath = _resolve_manifest(args.manifest)
    if mpath is None:
        print(
            "No manifest.json found. Pass --manifest path/to/manifest.json\n"
            "or generate data with:\n"
            "  python examples/generate_synthetic_market.py --symbols 10 --output-dir data/my_panel\n",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.manifest is None:
        print(f"Using manifest: {mpath}", file=sys.stderr)

    data = json.loads(mpath.read_text())
    rows = [(f["symbol"], f["rows"]) for f in data.get("files", [])]
    rows.sort(key=lambda x: x[1], reverse=True)
    rows = rows[: max(1, args.top)]

    if not rows:
        raise SystemExit("No files in manifest")

    syms, counts = zip(*rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, max(5.0, 0.18 * len(rows))))
    ax.barh(list(syms)[::-1], list(counts)[::-1], color="steelblue")
    ax.set_xlabel("trade rows")
    ax.set_title(
        f"Synthetic panel — {data.get('symbols_requested', len(rows))} symbols requested, "
        f"{data.get('total_rows', sum(counts)):,} total rows"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out.resolve()}")
    print(f"total_rows={data.get('total_rows')}")


if __name__ == "__main__":
    main()
