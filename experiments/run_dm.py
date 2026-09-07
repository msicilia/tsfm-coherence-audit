#!/usr/bin/env python3
"""Per-horizon Diebold-Mariano test for the accuracy comparison.

The Wilcoxon signed-rank test over per-series MASE is a valid paired test, but MASE
is a per-series scalar, so it cannot account for autocorrelation in the loss
differential across the forecast horizon, which is what Diebold-Mariano is for.
This driver re-runs the frontier on a subsample with per-step losses retained and
applies DM with the Harvey-Leybourne-Newbold small-sample correction, per series,
then aggregates.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from tsfm_bench.frontier import DATASETS, run_dataset
from tsfm_bench.models import Chronos2, ChronosBolt, Moirai, TimesFM
from tsfm_bench.paths import RESULTS
from tsfm_bench.provenance import record


def dm_test(d: np.ndarray, h: int) -> tuple[float, float]:
    """Diebold-Mariano on a loss-differential series, HLN-corrected.

    d = loss(model A) - loss(model B) over the horizon. Negative mean favours A.
    Uses h-1 autocovariance lags, as the h-step-ahead differential is MA(h-1).
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 8:
        return np.nan, np.nan
    dbar = d.mean()
    dc = d - dbar
    gamma0 = float(dc @ dc) / n
    lags = min(h - 1, n - 1)
    var = gamma0 + 2.0 * sum(float(dc[k:] @ dc[:-k]) / n for k in range(1, lags + 1))
    if var <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var / n)
    # Harvey-Leybourne-Newbold small-sample correction
    corr = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    stat *= corr
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=300, help="series per dataset")
    ap.add_argument("--n-windows", type=int, default=8)
    ap.add_argument("--tag", default="_dm")
    args = ap.parse_args()

    tsfms = {
        "Chronos-Bolt": ChronosBolt("amazon/chronos-bolt-small", device="cuda",
                                    name="Chronos-Bolt"),
        "Chronos-Bolt-Base": ChronosBolt("amazon/chronos-bolt-base", device="cuda",
                                         name="Chronos-Bolt-Base"),
        "Chronos-2": Chronos2(device="cuda"),
        "Moirai": Moirai(size="small", device="cuda"),
        "TimesFM": TimesFM(device="cuda"),
    }

    parts = []
    for cfg in DATASETS:
        cfg = dict(cfg); cfg["cap"] = args.cap
        print(f"[run] {cfg['name']} (cap {args.cap})", flush=True)
        _mase, steps = run_dataset(cfg, tsfms, n_windows=args.n_windows, per_step=True)
        steps["horizon"] = cfg["horizon"]
        parts.append(steps)
        print(f"   {len(steps):,} per-step rows", flush=True)
    allsteps = pd.concat(parts, ignore_index=True)
    allsteps.to_parquet(RESULTS / f"frontier_steps{args.tag}.parquet", index=False)

    # --- DM per (dataset, T, series): AutoETS vs each foundation model ---
    small = "AutoETS"
    fms = [m for m in tsfms if m != small]
    rows = []
    for (ds, T), sub in allsteps.groupby(["dataset", "T"]):
        h = int(sub["horizon"].iloc[0])
        piv = sub.pivot_table(index=["uid", "window", "step"], columns="model",
                              values="ae_scaled")
        if small not in piv.columns:
            continue
        for fm in fms:
            if fm not in piv.columns:
                continue
            stats_, ps = [], []
            for uid, g in piv.groupby(level="uid"):
                s_, p_ = dm_test((g[small] - g[fm]).to_numpy(), h)
                if np.isfinite(p_):
                    stats_.append(s_); ps.append(p_)
            if not ps:
                continue
            ps = np.array(ps); stats_ = np.array(stats_)
            rows.append(dict(dataset=ds, T=int(T), small=small, tsfm=fm, n_series=len(ps),
                             pct_small_better_sig=100 * np.mean((ps < 0.05) & (stats_ < 0)),
                             pct_tsfm_better_sig=100 * np.mean((ps < 0.05) & (stats_ > 0)),
                             pct_indistinguishable=100 * np.mean(ps >= 0.05),
                             mean_dm_stat=float(stats_.mean())))
    res = pd.DataFrame(rows).sort_values(["dataset", "tsfm", "T"])
    res.to_csv(RESULTS / f"dm_tests{args.tag}.csv", index=False)
    print("\n=== Diebold-Mariano (per series, HLN-corrected), AutoETS vs each model ===")
    print(res.round(1).to_string(index=False))
    record("dm", cap=args.cap, n_windows=args.n_windows)
    print(f"\nResults -> results/dm_tests{args.tag}.csv")


if __name__ == "__main__":
    main()
