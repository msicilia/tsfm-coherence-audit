#!/usr/bin/env python3
"""Phase E: stronger significance testing for the accuracy-heterogeneity section.

The paper's win-rates use a binomial sign test only. This upgrades to:
  * a paired Wilcoxon signed-rank test (uses the magnitude of per-series MASE
    differences, not just the sign);
  * Benjamini-Hochberg FDR control across all (dataset x T x small-vs-TSFM)
    comparisons, so the many simultaneous tests do not inflate false positives.

Operates on the cached results/study2_raw.csv (no model re-run). A per-horizon
Diebold-Mariano would need per-step losses; noted as a separate re-run.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats
from tsfm_bench.paths import RESULTS


SMALLS = ("AutoETS", "CAA")
TSFMS = ("Chronos-Bolt", "TimesFM", "Chronos-Bolt-Base", "Chronos-2", "Moirai")


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05):
    """Return (rejected_bool, qvalues) via BH step-up."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]          # enforce monotonicity
    qvals = np.empty(n); qvals[order] = np.clip(q, 0, 1)
    rejected = qvals <= alpha
    return rejected, qvals


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--in-tag", default="_w8")
    args = ap.parse_args()
    path = os.path.join(RESULTS, f"study2_raw{args.in_tag}.csv")
    if not os.path.exists(path):
        path = os.path.join(RESULTS, "study2_raw.csv")
    df = pd.read_csv(path)
    piv = df.pivot_table(index=["dataset", "window", "uid", "T"],
                         columns="model", values="mase")
    present_tsfms = [t for t in TSFMS if t in piv.columns]
    print(f"models present: {[c for c in piv.columns]}")

    rows = []
    for (dataset, T), sub in piv.groupby(level=["dataset", "T"]):
        for small in SMALLS:
            if small not in piv.columns:
                continue
            for tsfm in present_tsfms:
                paired = sub[[small, tsfm]].dropna()
                n = len(paired)
                if n < 10:
                    continue
                d = paired[small].to_numpy() - paired[tsfm].to_numpy()
                wins = int((d < 0).sum())                      # small beats tsfm
                # Wilcoxon signed-rank (paired, two-sided); guard all-zero diffs
                try:
                    w_stat, w_p = stats.wilcoxon(d, zero_method="wilcox",
                                                 alternative="two-sided")
                except ValueError:
                    w_stat, w_p = np.nan, 1.0
                rows.append(dict(dataset=dataset, T=int(T), small=small, tsfm=tsfm,
                                 n_pairs=n, winrate=round(wins / n, 3),
                                 mean_mase_diff=round(float(d.mean()), 4),
                                 median_mase_diff=round(float(np.median(d)), 4),
                                 wilcoxon_p=float(w_p)))
    res = pd.DataFrame(rows)
    if res.empty:
        print("no comparisons produced"); return
    rej, q = benjamini_hochberg(res["wilcoxon_p"].to_numpy(), alpha=0.05)
    res["bh_qvalue"] = np.round(q, 4)
    res["bh_signif_5pct"] = rej
    res["wilcoxon_p"] = res["wilcoxon_p"].round(4)
    res = res.sort_values(["dataset", "small", "tsfm", "T"]).reset_index(drop=True)
    res.to_csv(os.path.join(RESULTS, "accuracy_stats.csv"), index=False)

    print("\n=== Paired Wilcoxon + BH-FDR: small model vs foundation model ===")
    print("(mean_mase_diff<0 => small model better; bh_signif_5pct after FDR)")
    print(res.to_string(index=False))
    n_sig = int(res["bh_signif_5pct"].sum())
    print(f"\n{n_sig}/{len(res)} comparisons significant after BH-FDR (alpha=0.05)")
    print("Results -> results/accuracy_stats.csv")


if __name__ == "__main__":
    main()
