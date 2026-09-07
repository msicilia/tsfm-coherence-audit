#!/usr/bin/env python3
"""Single orchestrator for every experiment in the paper.

One runner over the model and hierarchy registries. It flushes progress and writes
each study's CSVs as soon as that study finishes, so a long run can be
detached and resumed study-by-study.

Studies (default: all):
    coherence  hierarchical incoherence + reconciliation accuracy (raw/BottomUp/
               MinT-ols/ERM), over all 8 hierarchies x {6 FMs, TinyTimeMixer,
               global LightGBM, global TCN} + AutoETS/CAA. The Electricity accuracy
               rows are the energy-domain ML-baseline comparison.
    prob       probabilistic-coherence CRPS over all 8 hierarchies x 5 quantile FMs.
    temporal   temporal (cross-time) coherence on M4-Monthly.
    nonneg     non-negativity violations on M3 (far from 0) and M5 (near 0).
    frontier   accuracy vs per-series history on M4/M5.
    dm         per-series Diebold-Mariano (small vs FM) from per-step errors.
    ablation   CAA projection-vs-selection ablation.
    postproc   derive prob-law regression from the prob CSV.

Usage:
    python run_all.py                       # all studies
    python run_all.py --quick               # tiny smoke run
    python run_all.py --studies coherence prob
    python run_all.py --n-windows 8
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from tsfm_bench.hierarchies import ALL_HIERARCHIES, load
from tsfm_bench.paths import RESULTS
from tsfm_bench.provenance import record

RESULTS.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def out(name: str) -> str:
    return str(RESULTS / name)


# --------------------------------------------------------------------------- #
# Model rosters (FMs loaded once; global models are per-hierarchy, needing period)
# --------------------------------------------------------------------------- #
def load_quantile_fms(device: str = "cuda") -> dict:
    from tsfm_bench.models import Chronos2, ChronosBolt, Moirai, TimesFM
    log("loading quantile foundation models ...")
    fms = {
        "Chronos-Bolt": ChronosBolt("amazon/chronos-bolt-small", device, "Chronos-Bolt"),
        "Chronos-Bolt-Base": ChronosBolt("amazon/chronos-bolt-base", device, "Chronos-Bolt-Base"),
        "Chronos-2": Chronos2(device),
        "TimesFM": TimesFM(device),
    }
    try:                                            # GPU unblocked on this card via cu130
        fms["Moirai"] = Moirai("small", device)
    except Exception as e:
        log(f"  Moirai on {device} failed ({type(e).__name__}); falling back to CPU")
        fms["Moirai"] = Moirai("small", "cpu")
    return fms


def load_ttm(device: str = "cuda"):
    from tsfm_bench.models import TinyTimeMixer
    return TinyTimeMixer(device)


def globals_for(period: int) -> dict:
    from tsfm_bench.global_models import GlobalLightGBM, GlobalTCN
    return {"LightGBM": GlobalLightGBM(period), "TCN": GlobalTCN(period)}


# --------------------------------------------------------------------------- #
# Studies
# --------------------------------------------------------------------------- #
def study_coherence(fms, ttm, groups, n_windows):
    from tsfm_bench.coherence import run_coherence
    inc_parts, acc_parts = [], []
    for g in groups:
        _, _, _, meta = load(g)
        roster = dict(fms)
        roster["TinyTimeMixer"] = ttm
        roster.update(globals_for(meta.seasonality))
        log(f"coherence: {g} ({len(roster)} models, h={meta.horizon}) ...")
        try:
            r = run_coherence(g, roster, n_windows=n_windows, use_caaa=True)
        except Exception as e:
            log(f"  [skip] {g}: {type(e).__name__}: {e}")
            continue
        inc_parts.append(r["incoherence"])
        acc_parts.append(r["accuracy"])
        print(r["incoherence"][["group", "model", "pct_nodes_incoherent"]].to_string(index=False),
              flush=True)
    if inc_parts:
        pd.concat(inc_parts, ignore_index=True).to_csv(out("coherence_incoherence.csv"), index=False)
        pd.concat(acc_parts, ignore_index=True).to_csv(out("coherence_accuracy.csv"), index=False)
        log("wrote coherence_incoherence.csv + coherence_accuracy.csv")


def study_prob(fms, groups, n_windows):
    from tsfm_bench.prob_coherence import run_prob_coherence
    parts = []
    for name, model in fms.items():
        for g in groups:
            log(f"prob-coherence: {name} / {g} ...")
            try:
                df = run_prob_coherence(g, model, n_windows=n_windows)
            except Exception as e:
                log(f"  [skip] {name}/{g}: {type(e).__name__}: {e}")
                continue
            df.insert(0, "model", name)
            parts.append(df)
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(out("prob_coherence.csv"), index=False)
        log("wrote prob_coherence.csv")


def study_temporal(fms, ttm, cap, n_windows):
    from tsfm_bench.loaders import load_m4
    from tsfm_bench.temporal import run_temporal_coherence
    items = load_m4("Monthly", max_series=cap, min_len=300)
    roster = dict(fms)
    roster["TinyTimeMixer"] = ttm
    parts = []
    for name, model in roster.items():
        log(f"temporal: {name} ...")
        try:
            df = run_temporal_coherence(items, model, n_windows=n_windows)
        except Exception as e:
            log(f"  [skip] {name}: {type(e).__name__}: {e}")
            continue
        df.insert(0, "model", name)
        parts.append(df)
    if parts:
        allr = pd.concat(parts, ignore_index=True)
        allr.to_csv(out("temporal_coherence_raw.csv"), index=False)
        summ = (allr.groupby(["model", "k"])
                .agg(pct_nodes_incoherent=("pct_nodes_incoherent", "mean"),
                     mean_rel_incoherence=("mean_rel_incoherence", "mean"),
                     mase_raw=("mase_raw", "mean"), mase_bottomup=("mase_bottomup", "mean"),
                     n=("pct_nodes_incoherent", "size")).reset_index())
        summ.to_csv(out("temporal_coherence_summary.csv"), index=False)
        log("wrote temporal_coherence_*.csv")


def study_nonneg(fms, m3_cap, m5_cap):
    from tsfm_bench.loaders import load_m3, load_m5_sample
    datasets = {"M3-Monthly": dict(items=load_m3("Monthly", max_series=m3_cap), horizon=12),
                "M5-daily": dict(items=load_m5_sample(n_series=m5_cap, min_len=200), horizon=28)}
    rows = []
    for name, model in fms.items():
        for ds, cfg in datasets.items():
            h = cfg["horizon"]
            items = [it for it in cfg["items"] if len(it.y) > h + 4]
            hist = [it.y[:-h] for it in items]
            log(f"nonneg: {name} / {ds} ({len(hist)} series) ...")
            try:
                fc = []
                for i in range(0, len(hist), 256):
                    fc.extend(model.predict_batch(hist[i:i + 256], h))
            except Exception as e:
                log(f"  [skip] {name}/{ds}: {type(e).__name__}: {e}")
                continue
            med = np.concatenate([f.median for f in fc])
            low = np.concatenate([f.q10 for f in fc])
            rows.append(dict(model=name, dataset=ds,
                             pct_median_below0=float(100 * np.mean(med < 0)),
                             pct_lower_below0=float(100 * np.mean(low < 0))))
    if rows:
        pd.DataFrame(rows).to_csv(out("nonneg_summary.csv"), index=False)
        log("wrote nonneg_summary.csv")


def study_frontier(fms, n_windows, quick):
    from tsfm_bench.frontier import DATASETS, run_dataset, summarize, winrates
    parts = []
    for cfg in DATASETS:
        log(f"frontier: {cfg['name']} ...")
        try:
            parts.append(run_dataset(cfg, fms, n_windows=n_windows, quick=quick))
        except Exception as e:
            log(f"  [skip] {cfg['name']}: {type(e).__name__}: {e}")
    if parts:
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(out("study2_raw.csv"), index=False)
        summarize(df).to_csv(out("study2_summary.csv"), index=False)
        winrates(df).to_csv(out("study2_winrates.csv"), index=False)
        log("wrote study2_*.csv")


def study_dm(fms, cap, n_windows):
    from tsfm_bench.frontier import DATASETS, run_dataset
    from run_dm import dm_test           # HLN-corrected DM (h-1 lags, real horizon)

    step_parts = []
    horizon = {}
    for cfg in DATASETS:
        cfg = dict(cfg); cfg["cap"] = cap
        horizon[cfg["name"]] = cfg["horizon"]
        log(f"dm: {cfg['name']} (per-step) ...")
        try:
            _m, steps = run_dataset(cfg, fms, n_windows=n_windows, per_step=True)
            steps["horizon"] = cfg["horizon"]
            step_parts.append(steps)
        except Exception as e:
            log(f"  [skip] {cfg['name']}: {type(e).__name__}: {e}")
    if not step_parts:
        return
    allsteps = pd.concat(step_parts, ignore_index=True)
    allsteps.to_parquet(out("frontier_steps_dm.parquet"), index=False)
    small = "AutoETS"
    fm_names = [m for m in allsteps["model"].unique() if m in fms]
    res = []
    for (ds, T), sub in allsteps.groupby(["dataset", "T"]):
        h = int(sub["horizon"].iloc[0])
        piv = sub.pivot_table(index=["uid", "window", "step"], columns="model", values="ae_scaled")
        if small not in piv.columns:
            continue
        for fm in fm_names:
            if fm not in piv.columns:
                continue
            st, ps = [], []
            for uid, gg in piv.groupby(level="uid"):
                s_, p_ = dm_test((gg[small] - gg[fm]).to_numpy(), h)
                if np.isfinite(p_):
                    st.append(s_); ps.append(p_)
            if not ps:
                continue
            ps = np.array(ps); st = np.array(st)
            res.append(dict(dataset=ds, T=int(T), small=small, tsfm=fm, n_series=len(ps),
                            pct_small_better_sig=round(100 * np.mean((ps < 0.05) & (st < 0)), 2),
                            pct_tsfm_better_sig=round(100 * np.mean((ps < 0.05) & (st > 0)), 2),
                            pct_indistinguishable=round(100 * np.mean(ps >= 0.05), 2),
                            mean_dm_stat=float(st.mean())))
    pd.DataFrame(res).to_csv(out("dm_tests_dm.csv"), index=False)
    log("wrote dm_tests_dm.csv")


def study_ablation():
    import importlib
    log("ablation: CAA projection-vs-selection ...")
    mod = importlib.import_module("run_caa_ablation")
    mod.main()
    log("wrote caaa_ablation_*.csv")


def study_postproc():
    """Derive the probabilistic-coherence dependence law from prob_coherence.csv,
    reporting per-hierarchy (one point per model x hierarchy) with the inflation of
    each reconciliation over the raw distribution."""
    path = out("prob_coherence.csv")
    try:
        d = pd.read_csv(path)
    except FileNotFoundError:
        log("postproc: prob_coherence.csv absent; skipping")
        return
    from scipy import stats
    d = d[~d["is_bottom"]].copy()          # inflation is 0 on bottom nodes; aggregates only
    g = (d.groupby(["model", "group"])
         .agg(dependence=("dependence", "mean"), raw=("crps_raw", "mean"),
              indep=("crps_indep", "mean"), schaake=("crps_schaake", "mean"),
              gauss=("crps_gauss", "mean"), n=("crps_raw", "size")).reset_index())
    for col in ["indep", "schaake", "gauss"]:
        g[f"infl_{col}"] = g[col] - g["raw"]
    g.to_csv(out("prob_law_points.csv"), index=False)

    reg = []
    for variant in ["infl_indep", "infl_schaake", "infl_gauss"]:
        sub = g.dropna(subset=["dependence", variant])
        if len(sub) >= 3:
            r = stats.linregress(sub["dependence"], sub[variant])
            reg.append(dict(variant=variant, slope=r.slope, intercept=r.intercept,
                            r2=r.rvalue ** 2, p=r.pvalue, n=len(sub)))
    pd.DataFrame(reg).to_csv(out("prob_law_regression.csv"), index=False)
    log("wrote prob_law_points.csv + prob_law_regression.csv")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
ALL_STUDIES = ["coherence", "prob", "temporal", "nonneg", "frontier", "dm", "ablation", "postproc"]
NEEDS_FMS = {"coherence", "prob", "temporal", "nonneg", "frontier", "dm"}
NEEDS_TTM = {"coherence", "temporal"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="*", default=ALL_STUDIES,
                    help=f"subset of {ALL_STUDIES}")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    ap.add_argument("--n-windows", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    studies = [s for s in args.studies if s in ALL_STUDIES]
    nw = 2 if args.quick else args.n_windows
    groups = (["Labour", "Electricity"] if args.quick else ALL_HIERARCHIES)
    cap = 60 if args.quick else 400
    t0 = time.time()
    log(f"run start: studies={studies} quick={args.quick} n_windows={nw}")

    fms = ttm = None
    if set(studies) & NEEDS_FMS:
        fms = load_quantile_fms(args.device)
    if set(studies) & NEEDS_TTM:
        ttm = load_ttm(args.device)

    for s in studies:
        t = time.time()
        log(f"===== STUDY: {s} =====")
        try:
            if s == "coherence":
                study_coherence(fms, ttm, groups, nw)
            elif s == "prob":
                study_prob(fms, groups, nw)
            elif s == "temporal":
                study_temporal(fms, ttm, cap, max(2, nw // 2))
            elif s == "nonneg":
                study_nonneg(fms, cap * 3, cap * 3)
            elif s == "frontier":
                study_frontier(fms, nw, args.quick)
            elif s == "dm":
                study_dm(fms, cap, nw)
            elif s == "ablation":
                study_ablation()
            elif s == "postproc":
                study_postproc()
        except Exception as e:
            log(f"STUDY {s} FAILED: {type(e).__name__}: {e}")
        log(f"----- {s} done in {time.time() - t:.0f}s -----")

    record("run_all", studies=studies, quick=args.quick, n_windows=nw)
    log(f"run complete in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
