#!/usr/bin/env python3
"""Regress the CRPS cost of distributional coherence on cross-series dependence.

Reads results/prob_coherence.csv (which must carry `dependence` and `crps_gauss`)
and asks whether the cost of reconciling the predictive distributions scales with
how strongly the bottom series move together. It does not for independent
summation; it does for the copula couplings, in the opposite direction. Emits the
per-point CSV, the regression at both aggregation levels, and the scatter figure.
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats
from tsfm_bench.paths import RESULTS, FIGURES as FIGS
from tsfm_bench.provenance import record



def main():
    ap = argparse.ArgumentParser()
    # Default to the untagged CSVs, which are the ones the current paper reports.
    # The tagged (_w8 / _multimodel / _all) files, if present, are earlier runs.
    ap.add_argument("--in-tags", nargs="*", default=[""],
                    help="suffixes of results/prob_coherence<tag>.csv to pool")
    args = ap.parse_args()
    parts = []
    for tag in args.in_tags:
        p = os.path.join(RESULTS, f"prob_coherence{tag}.csv")
        if os.path.exists(p):
            parts.append(pd.read_csv(p))
            print(f"reading {os.path.basename(p)} ({len(parts[-1])} rows)")
        else:
            print(f"  [missing] {os.path.basename(p)}")
    if not parts:
        parts = [pd.read_csv(os.path.join(RESULTS, "prob_coherence.csv"))]
    df = pd.concat(parts, ignore_index=True)
    agg = df[~df["is_bottom"]].copy()

    # per (model, hierarchy): mean CRPS by variant + the hierarchy dependence
    g = (agg.groupby(["model", "group"])
         .agg(dependence=("dependence", "mean"),
              raw=("crps_raw", "mean"),
              indep=("crps_indep", "mean"),
              schaake=("crps_schaake", "mean"),
              gauss=("crps_gauss", "mean"),
              n=("uid", "count"))
         .reset_index())
    # cost of restoring distributional coherence, relative to the raw (incoherent) dist
    g["infl_indep"] = g["indep"] - g["raw"]
    g["infl_schaake"] = g["schaake"] - g["raw"]
    g["infl_gauss"] = g["gauss"] - g["raw"]
    g["best_coherent"] = g[["indep", "schaake", "gauss"]].min(axis=1)
    g["infl_best"] = g["best_coherent"] - g["raw"]
    g.to_csv(os.path.join(RESULTS, "prob_law_points.csv"), index=False)
    print("=== per (model, hierarchy) CRPS + dependence ===")
    print(g.round(3).to_string(index=False))

    # Regressions: does the reconciliation cost scale with dependence?
    # Reported at TWO levels. The `pooled` level puts one point per
    # (model, hierarchy); its n is inflated, because the models sharing a
    # hierarchy also share its dependence and are not independent replicates.
    # The `hierarchy` level averages the models within each hierarchy first, so
    # n is the number of hierarchies. The design is balanced (same model count
    # per hierarchy), so the two slopes coincide and only the standard errors
    # differ. The hierarchy level is the one the paper quotes.
    print("\n=== REGRESSION: CRPS inflation ~ cross-series dependence ===")
    per_hier = (g.groupby("group")
                .agg({"dependence": "mean",
                      **{c: "mean" for c in ["infl_indep", "infl_schaake",
                                             "infl_gauss", "infl_best"]}})
                .reset_index())
    reg_rows = []
    for level, frame in [("hierarchy", per_hier), ("pooled", g)]:
        for col in ["infl_indep", "infl_schaake", "infl_gauss", "infl_best"]:
            sub = frame[["dependence", col]].dropna()
            sub = sub[np.isfinite(sub["dependence"]) & np.isfinite(sub[col])]
            if len(sub) < 3:
                print(f"  [{level}] {col}: too few points ({len(sub)})"); continue
            lr = stats.linregress(sub["dependence"], sub[col])
            reg_rows.append(dict(level=level, variant=col, slope=lr.slope,
                                 intercept=lr.intercept, r2=lr.rvalue**2,
                                 p=lr.pvalue, n=len(sub)))
            print(f"  [{level:9s}] {col:12s}: slope={lr.slope:+.3f}  "
                  f"R^2={lr.rvalue**2:.3f}  p={lr.pvalue:.4f}  n={len(sub)}")
    pd.DataFrame(reg_rows).to_csv(os.path.join(RESULTS, "prob_law_regression.csv"), index=False)

    _figure(g)
    print(f"\nResults -> results/prob_law_*.csv ; figure -> paper/figures/prob_coherence_law.*")
    record("prob_law", in_tags=args.in_tags)


def _figure(g: pd.DataFrame):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [no figure] matplotlib unavailable: {e}"); return
    os.makedirs(FIGS, exist_ok=True)
    from tsfm_bench import figstyle
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(figstyle.TEXT_WIDTH_IN, 3.4))
    sub = g.dropna(subset=["dependence"])
    # fixed categorical slots (blue, orange, aqua); the marker shape also carries identity
    for variant, mk, col, lab in [("infl_indep", "o", figstyle.SERIES[0], "independent"),
                                  ("infl_schaake", "s", figstyle.SERIES[1], "Schaake/ECC"),
                                  ("infl_gauss", "^", figstyle.SERIES[2], "Gaussian copula")]:
        s = sub[np.isfinite(sub[variant])]
        ax.scatter(s["dependence"], s[variant], marker=mk, s=28, alpha=0.8,
                   color=col, edgecolor="white", linewidth=0.5, label=lab)
        if len(s) >= 3:
            lr = stats.linregress(s["dependence"], s[variant])
            xs = np.linspace(s["dependence"].min(), s["dependence"].max(), 50)
            ax.plot(xs, lr.intercept + lr.slope * xs, lw=1.2, alpha=0.7, color=col)
    ax.axhline(0, color="#52514e", lw=0.6, ls=":")
    ax.set_xlabel("cross-series dependence (mean |corr| of bottom series)")
    ax.set_ylabel("CRPS cost of coherence (reconciled − raw)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "prob_coherence_law.pdf"))
    fig.savefig(os.path.join(FIGS, "prob_coherence_law.png"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
