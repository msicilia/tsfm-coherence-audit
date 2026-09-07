#!/usr/bin/env python3
"""Phase F: does constraint-aware SELECTION matter, or is it all projection?

Contribution (4) claims CAA conditions model *selection* on declared hard
constraints. This ablation isolates that claim on a capacity-constrained task
(the branch where selection actually depends on the constraint):

  v1 feature-only, NO projection   -> selection constraint-blind, output may violate
  v2 feature-only + projection     -> feasible, but selection was constraint-blind
  v3 constraint-aware CAA          -> feasible AND selection constraint-informed

If v3 ~ v2 in accuracy, constraint-conditioned *selection* adds nothing beyond
projection (an honest negative). If v3 < v2, selection matters. We also report
how often the selected model actually differs between v2 and v3.
Capacity is imposed as a per-series ceiling the series can plausibly approach.
CPU only (statsforecast).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from tsfm_bench._frontier_metrics import _mase, _series_scale
from tsfm_bench.loaders import load_m4
from tsfm_bench.models import CAA
from tsfm_bench.paths import RESULTS



def main(cap=300, T=96, horizon=18, period=12):
    items = load_m4("Monthly", max_series=cap, min_len=T + horizon + 4)
    rows = []
    sel_changed = 0
    for it in items:
        y = it.y
        train, test = y[-(T + horizon):-horizon], y[-horizon:]
        if len(train) < T:
            continue
        scale = _series_scale(y, period)
        # capacity = a ceiling just above the training range (binding for up-trends)
        ub = float(1.02 * np.max(train))

        caaa_full = CAA(period=period, non_negative=True, upper_bound=ub,
                             constraint_aware_selection=True)
        caaa_feat = CAA(period=period, non_negative=True, upper_bound=ub,
                             constraint_aware_selection=False)

        d_full = caaa_full._diagnose(train)
        kind_full, _ = caaa_full._choose(d_full)
        kind_feat, _ = caaa_feat._choose(caaa_feat._diagnose(train))
        if kind_full != kind_feat:
            sel_changed += 1

        f_full = caaa_full.predict(train, horizon)          # v3: aware + project
        f_feat = caaa_feat.predict(train, horizon)          # v2: blind + project
        # v1: feature-only selection, raw (no projection) -> refit chosen kind raw
        from tsfm_bench.models import Classical
        raw = Classical(kind_feat, period).predict(train, horizon).median

        for tag, pred, projected in [("v1_feat_raw", raw, False),
                                     ("v2_feat_proj", f_feat.median, True),
                                     ("v3_aware_proj", f_full.median, True)]:
            viol = float(np.mean(pred > ub + 1e-9))         # capacity violations
            rows.append(dict(uid=it.uid, variant=tag,
                             mase=_mase(test, pred, scale),
                             cap_violation=viol,
                             kind=kind_feat if tag != "v3_aware_proj" else kind_full))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "caaa_ablation_raw.csv"), index=False)
    summ = (df.groupby("variant")
            .agg(mean_mase=("mase", "mean"),
                 median_mase=("mase", "median"),
                 cap_violation_rate=("cap_violation", "mean"),
                 n=("uid", "count"))
            .reindex(["v1_feat_raw", "v2_feat_proj", "v3_aware_proj"]))
    summ.to_csv(os.path.join(RESULTS, "caaa_ablation_summary.csv"))
    n_series = df["uid"].nunique()

    print("=== CAA ablation: constraint-aware selection vs feature-only ===")
    print(summ.round(4).to_string())
    print(f"\nSelection changed by constraint-awareness on "
          f"{sel_changed}/{n_series} series ({100*sel_changed/max(n_series,1):.1f}%)")
    # paired accuracy test v2 vs v3
    piv = df.pivot_table(index="uid", columns="variant", values="mase")
    if {"v2_feat_proj", "v3_aware_proj"}.issubset(piv.columns):
        from scipy import stats
        d = piv["v3_aware_proj"] - piv["v2_feat_proj"]
        d = d.dropna()
        try:
            w_p = stats.wilcoxon(d, alternative="two-sided").pvalue
        except ValueError:
            w_p = float("nan")
        verdict = ("selection helps" if d.mean() < 0 and w_p < 0.05
                   else "no selection benefit beyond projection")
        print(f"v3-v2 mean MASE diff = {d.mean():+.4f} (Wilcoxon p={w_p:.4f}) -> {verdict}")
    print("Results -> results/caaa_ablation_*.csv")


if __name__ == "__main__":
    main()
