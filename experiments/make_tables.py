#!/usr/bin/env python3
"""Regenerate the paper's LaTeX tables from result CSVs (no model re-runs)."""
from tsfm_bench.provenance import record
from tsfm_bench.tables import TABLE_DIR, write_all

if __name__ == "__main__":
    written = write_all()
    print("wrote:", ", ".join(f"paper/tables/{w}.tex" for w in written) or "(nothing)")
    record("make_tables", outputs=[f"paper/tables/{w}.tex" for w in written])
