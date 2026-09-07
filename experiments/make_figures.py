#!/usr/bin/env python3
"""Regenerate submission-grade figures (vector PDF + 600-dpi PNG) from cached
result CSVs, with no model re-runs. ESWA prefers vector (PDF/EPS) for line art."""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tsfm_bench.paths import RESULTS, FIGURES as FIGS

os.makedirs(FIGS, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})



def save(fig, name, tight=True):
    if tight:                      # skip where the figure sets its own spacing
        fig.tight_layout()
    fig.savefig(os.path.join(FIGS, name + ".pdf"))            # vector
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=600)   # raster fallback
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf (+ .png 600dpi)")


def frontier_figure(src="study2_summary.csv"):
    path = os.path.join(RESULTS, src)
    if not os.path.exists(path):
        path = os.path.join(RESULTS, "study2_summary.csv")
    s = pd.read_csv(path)
    s = s[s["model"] != "model"].copy()                       # drop stray header rows
    for c in ["T", "mean_mase", "ci_lo", "ci_hi"]:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    datasets = ["M4-Monthly", "M4-Quarterly", "M5-daily"]
    models = ["Chronos-Bolt", "Chronos-Bolt-Base", "Chronos-2", "Moirai",
              "TimesFM", "AutoETS"]
    fig, axes = plt.subplots(len(datasets), 1, figsize=(7.4, 2.7 * len(datasets)),
                             squeeze=False)
    handles, labels = [], []
    for ax, ds in zip(axes[:, 0], datasets):
        d = s[s["dataset"] == ds]
        for m in models:
            dm = d[d["model"] == m].sort_values("T")
            if dm.empty:
                continue
            (ln,) = ax.plot(dm["T"], dm["mean_mase"], marker="o", lw=1.8, label=m)
            ax.fill_between(dm["T"], dm["ci_lo"], dm["ci_hi"], alpha=0.15)
            if m not in labels:
                handles.append(ln); labels.append(m)
        ax.set_xscale("log")
        # tick at the history lengths actually evaluated, not decades
        ticks = sorted(d["T"].dropna().unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(t)}" for t in ticks])
        ax.minorticks_off()
        ax.set_title(ds, fontsize=11)
        ax.set_ylabel("mean MASE")
    axes[-1, 0].set_xlabel("per-series history length $T$ (log scale)")
    # one shared legend rather than repeating it in every panel
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("Accuracy vs. per-series history with 95% bootstrap CIs "
                 "(eight rolling origins)", y=0.99)
    fig.subplots_adjust(top=0.885, bottom=0.06, left=0.10, right=0.98, hspace=0.45)
    save(fig, "frontier_study", tight=False)


def coherence_figure(src="coherence_incoherence.csv"):
    """Incoherence as an annotated heatmap.

    A grouped bar chart of nine models x eight hierarchies is 72 bars, most of them
    pinned at 100%, which reads as a wall of full-height bars and communicates nothing.
    A heatmap with the values printed in-cell shows both the near-universal saturation
    and the handful of genuine exceptions (Labour and Electricity, dominated by a common
    smooth component, and TinyTimeMixer where it is too flat to be useful) at a glance.
    The two trained global models (LightGBM, TCN) sit to the right of the dividing rule.
    """
    path = os.path.join(RESULTS, src)
    if not os.path.exists(path):
        path = os.path.join(RESULTS, "coherence_incoherence.csv")
    df = pd.read_csv(path)
    piv = df.pivot_table(index="group", columns="model", values="pct_nodes_incoherent")

    order = [m for m in ["TinyTimeMixer", "Chronos-Bolt", "Chronos-Bolt-Base",
                         "Chronos-2", "Moirai", "TimesFM", "AutoETS",
                         "LightGBM", "TCN"] if m in piv.columns]
    hier = [h for h in ["Wiki2", "TourismLarge", "Labour", "TourismSmall", "M5",
                        "Traffic", "Electricity", "Solar"] if h in piv.index]
    M = piv.reindex(index=hier, columns=order).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([m.replace("Chronos-Bolt-Base", "C-Bolt-B")
                          .replace("Chronos-Bolt", "C-Bolt")
                          .replace("Chronos-2", "C-2")
                          .replace("TinyTimeMixer", "TTM")
                          .replace("LightGBM", "LGBM") for m in order],
                       rotation=30, ha="right")
    # rule separating zero-shot models from the two trained global baselines
    n_global = sum(m in ("LightGBM", "TCN") for m in order)
    if n_global:
        ax.axvline(len(order) - n_global - 0.5, color="k", lw=1.4)
    ax.set_yticks(range(len(hier)))
    ax.set_yticklabels([h.replace("M5", "M5 retail") for h in hier])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if (v > 70 or v < 12) else "black")
    ax.set_title("% of aggregate nodes incoherent (independent per-series forecasts)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("% incoherent")
    save(fig, "coherence_incoherence")


if __name__ == "__main__":
    print("Regenerating figures ...")
    frontier_figure()
    coherence_figure()
