#!/usr/bin/env python3
"""Wall-clock cost of each step of the pipeline, one rolling origin per hierarchy.

For every hierarchy we time, on the most recent origin and over all nodes:
  * zero-shot point inference of each foundation model (GPU), after a warm-up call;
  * for one base model (Chronos-2, the one Table recon reports): the in-sample
    forecast passes that ERM needs, and each reconciler (BottomUp, MinT-ols,
    MinT-shrink, ERM) through hierarchicalforecast, timed separately;
  * the probabilistic steps for the same base: quantile inference plus sampling,
    independent summation, the Schaake shuffle and the Gaussian copula.
Peak GPU memory of the inference call is recorded as well. Output: results/timing.csv
(long format: group, step, model, seconds, gpu_peak_mb, n_nodes, n_bottom, h).

Usage:
    python run_timing.py              # all hierarchies
    python run_timing.py --groups Labour Electricity
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from tsfm_bench.coherence import (ERM_INSAMPLE_TARGET, ERM_LAMBDA, ERM_MAX_ORIGINS,
                                  _mase_scale, load_hierarchy)
from tsfm_bench.hierarchies import ALL_HIERARCHIES
from tsfm_bench.paths import RESULTS
from tsfm_bench.prob_coherence import (LEVELS, gaussian_copula_shuffle,
                                       samples_from_quantiles, schaake_shuffle)
from tsfm_bench.provenance import record
from run_all import load_quantile_fms, load_ttm, log

BASE = "Chronos-2"
N_SAMPLES = 200
BATCH = 256


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _peak_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / 2 ** 20)
    except Exception:
        pass
    return float("nan")


def _reset_peak():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _timed(fn):
    _sync()
    t0 = time.perf_counter()
    out = fn()
    _sync()
    return out, time.perf_counter() - t0


def _batched(model, hs, h):
    preds = []
    for i in range(0, len(hs), BATCH):
        preds.extend(model.predict_batch(hs[i:i + BATCH], h))
    return [f.median for f in preds]


def time_group(group: str, fms: dict, ttm) -> list[dict]:
    from hierarchicalforecast.core import HierarchicalReconciliation
    from hierarchicalforecast.methods import ERM, BottomUp, MinTrace

    Y, S, tags, meta = load_hierarchy(group)
    h, period = meta.horizon, meta.seasonality
    all_ids, bottom_ids = list(S.index), list(S.columns)
    series = {}
    for uid, g in Y.groupby("unique_id", sort=False):
        yv = g.sort_values("ds")["y"].to_numpy(dtype=float)
        if len(yv) > h + 4:
            series[uid] = yv
    train = {u: yv[:-h] for u, yv in series.items()}          # most recent origin (w=0)
    nodes = [u for u in all_ids if u in train]
    hist = [train[u] for u in nodes]
    common = dict(group=group, n_nodes=len(nodes), n_bottom=len(bottom_ids), h=h)
    rows = []

    def add(step, model, seconds, mem=float("nan"), note=""):
        rows.append(dict(**common, step=step, model=model, seconds=seconds,
                         gpu_peak_mb=mem, note=note))
        log(f"  {group:13s} {model:18s} {step:18s} {seconds:9.3f}s")

    # 1. zero-shot inference per model (warm-up on a few series first)
    roster = dict(fms)
    roster["TinyTimeMixer"] = ttm
    base_pred = None
    for name, model in roster.items():
        try:
            model.predict_batch(hist[:4], h)
            _reset_peak()
            preds, sec = _timed(lambda: _batched(model, hist, h))
            add("inference", name, sec, _peak_mb())
            if name == BASE:
                base_pred = dict(zip(nodes, preds))
        except Exception as e:
            add("inference", name, float("nan"), note=f"{type(e).__name__}: {e}")
    if base_pred is None:
        return rows

    # 2. point reconciliation for the base model, as in coherence._one_window
    model = roster[BASE]
    Lt = len(train[nodes[0]])
    min_ctx = max(2 * period, 16)
    max_k = max(1, (Lt - min_ctx) // h)
    K = int(min(max_k, ERM_MAX_ORIGINS, max(1, -(-ERM_INSAMPLE_TARGET // h))))
    Ki = K * h

    def _insample():
        blocks = {u: np.empty(Ki) for u in nodes}
        for j in range(1, K + 1):
            ctx = [train[u][:Lt - j * h] for u in nodes]
            for u, arr in zip(nodes, _batched(model, ctx, h)):
                blocks[u][(K - j) * h:(K - j) * h + h] = np.asarray(arr, dtype=float)[:h]
        return blocks
    blocks, sec_in = _timed(_insample)
    add("ERM in-sample passes", BASE, sec_in, note=f"K={K}")

    Yhat = pd.DataFrame([{"unique_id": u, "ds": len(train[u]) + k, BASE: base_pred[u][k]}
                         for u in nodes for k in range(h)])
    Ydf = pd.DataFrame([{"unique_id": u, "ds": len(train[u]) - Ki + k,
                         "y": float(train[u][len(train[u]) - Ki + k]), BASE: blocks[u][k]}
                        for u in nodes for k in range(Ki)])
    S_df = S.loc[all_ids].reset_index().rename(columns={"index": "unique_id"})

    def _rec(reconciler):
        return HierarchicalReconciliation(reconcilers=[reconciler]).reconcile(
            Y_hat_df=Yhat, S_df=S_df, tags=tags, Y_df=Ydf)
    for step, rec in [("BottomUp", BottomUp()), ("MinT-ols", MinTrace(method="ols")),
                      ("MinT-shrink", MinTrace(method="mint_shrink")),
                      ("ERM fit", ERM(method="reg_bu", lambda_reg=ERM_LAMBDA))]:
        try:
            _, sec = _timed(lambda: _rec(rec))
            add(step, BASE, sec)
        except Exception as e:
            add(step, BASE, float("nan"), note=f"{type(e).__name__}: {e}")
    # ERM as the paper counts it: the fit plus the in-sample passes it requires.
    erm_fit = next((r["seconds"] for r in rows if r["step"] == "ERM fit"), float("nan"))
    add("ERM", BASE, erm_fit + sec_in, note="fit + in-sample passes")

    # 3. probabilistic steps for the base model, as in prob_coherence
    rng = np.random.default_rng(0)
    idx = {u: i for i, u in enumerate(all_ids)}

    def _quant():
        qmap = {}
        for i in range(0, len(hist), BATCH):
            q = model.predict_quantiles_batch(hist[i:i + BATCH], h, LEVELS)
            for j, u in enumerate(nodes[i:i + BATCH]):
                qmap[u] = q[j]
        return {u: samples_from_quantiles(qmap[u], LEVELS, N_SAMPLES, rng) for u in nodes}
    raw, sec = _timed(_quant)
    add("quantiles+samples", BASE, sec)
    present = [b for b in bottom_ids if b in raw]
    Sb = S[present].to_numpy(dtype=float)

    def reconcile(bottom):
        B = np.stack([bottom[b] for b in present], axis=0)
        return np.einsum("ab,bnh->anh", Sb, B)
    _, sec = _timed(lambda: reconcile(raw))
    add("indep", BASE, sec)
    _, sec = _timed(lambda: reconcile(schaake_shuffle(raw, train, bottom_ids, h, N_SAMPLES, rng)))
    add("Schaake", BASE, sec)
    _, sec = _timed(lambda: reconcile(gaussian_copula_shuffle(raw, train, bottom_ids, h,
                                                              N_SAMPLES, rng)))
    add("Gaussian", BASE, sec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=ALL_HIERARCHIES)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    t0 = time.time()
    fms = load_quantile_fms(args.device)
    ttm = load_ttm(args.device)
    rows = []
    for g in args.groups:
        log(f"timing: {g} ...")
        try:
            rows.extend(time_group(g, fms, ttm))
        except Exception as e:
            log(f"  [skip] {g}: {type(e).__name__}: {e}")
        pd.DataFrame(rows).to_csv(RESULTS / "timing.csv", index=False)   # partial saves
    log(f"wrote timing.csv ({len(rows)} rows) in {(time.time() - t0) / 60:.1f} min")
    record("timing", groups=list(args.groups))


if __name__ == "__main__":
    main()
