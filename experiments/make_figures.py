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
from tsfm_bench import figstyle
from tsfm_bench.figstyle import TEXT_WIDTH_IN, SERIES, SEQ_RAMP, INK, INK_MUTED

os.makedirs(FIGS, exist_ok=True)
figstyle.apply()



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
    fig, axes = plt.subplots(len(datasets), 1,
                             figsize=(TEXT_WIDTH_IN, 1.95 * len(datasets)),
                             squeeze=False)
    handles, labels = [], []
    for ax, ds in zip(axes[:, 0], datasets):
        d = s[s["dataset"] == ds]
        for m, col in zip(models, SERIES):
            dm = d[d["model"] == m].sort_values("T")
            if dm.empty:
                continue
            (ln,) = ax.plot(dm["T"], dm["mean_mase"], marker="o", ms=4, lw=1.6,
                            color=col, label=m)
            ax.fill_between(dm["T"], dm["ci_lo"], dm["ci_hi"], color=col, alpha=0.15,
                            lw=0)
            if m not in labels:
                handles.append(ln); labels.append(m)
        ax.set_xscale("log")
        # tick at the history lengths actually evaluated, not decades
        ticks = sorted(d["T"].dropna().unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(t)}" for t in ticks])
        ax.minorticks_off()
        ax.set_title(ds, loc="left", fontweight="bold")
        ax.set_ylabel("mean MASE")
        ax.tick_params(length=2)
    axes[-1, 0].set_xlabel("per-series history length $T$ (log scale)")
    # one shared legend above the panels rather than repeating it in every panel
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.4)
    fig.subplots_adjust(top=0.90, bottom=0.07, left=0.12, right=0.98, hspace=0.45)
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

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.3))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([m.replace("Chronos-Bolt-Base", "C-Bolt-B")
                          .replace("Chronos-Bolt", "C-Bolt")
                          .replace("Chronos-2", "C-2")
                          .replace("TinyTimeMixer", "TTM")
                          .replace("LightGBM", "LGBM") for m in order],
                       rotation=35, ha="right", rotation_mode="anchor")
    # rule separating zero-shot models from the two trained global baselines
    n_global = sum(m in ("LightGBM", "TCN") for m in order)
    if n_global:
        ax.axvline(len(order) - n_global - 0.5, color=INK, lw=1.4)
    ax.set_yticks(range(len(hier)))
    ax.set_yticklabels([h.replace("M5", "M5 retail") for h in hier])
    ax.tick_params(length=0)
    # Every cell carries its value, so no colour bar is needed; the ink flips to
    # white once the fill is dark enough to need it.
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v >= 50 else INK)
    ax.set_xlabel("% of aggregate nodes incoherent", color=INK_MUTED)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, "coherence_incoherence")


if __name__ == "__main__":
    print("Regenerating figures ...")
    frontier_figure()
    coherence_figure()
