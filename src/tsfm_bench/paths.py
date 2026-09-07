"""Canonical project paths.

Defined once, absolutely, and derived from this file's location, so every driver
works regardless of the current working directory.
"""
from __future__ import annotations

from pathlib import Path

# src/tsfm_bench/paths.py -> repo root is two levels up from the package dir
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"

for _d in (DATA, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

__all__ = ["ROOT", "DATA", "RESULTS", "PAPER", "FIGURES"]
