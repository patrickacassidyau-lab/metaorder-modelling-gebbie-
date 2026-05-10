#!/usr/bin/env python3
"""Deprecated entrypoint — calls ``visualisation/visualise_run.py``. Prefer that path."""

from __future__ import annotations

import runpy
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    runpy.run_path(str(_REPO_ROOT / "visualisation" / "visualise_run.py"), run_name="__main__")
