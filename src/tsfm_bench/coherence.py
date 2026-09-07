"""Coherence-hardening study (the paper's headline).

For several real hierarchies and several foundation models we measure:
  * raw incoherence of independent per-series forecasts (do bottom levels sum to
    the aggregates?);
  * the accuracy (MASE) of the raw forecasts vs the same forecasts after
    reconciliation (bottom-up and MinTrace-OLS), to show that restoring coherence
    is not an accuracy sacrifice;
  * CAA (a small interpretable base + reconciliation) which is coherent by
    construction.

Reconciliation uses Nixtla's hierarchicalforecast. TSFM base forecasts come from
Chronos-Bolt and TimesFM (GPU) and Moirai (CPU; its GPU path is blocked on this
Blackwell card by an sm_121/nvrtc incompatibility).
"""
from __future__ import annotations

import numpy as np

from .paths import DATA
import pandas as pd

# ERM (Ben Taieb & Koo 2019), regularized toward bottom-up. G is learned from K rolling
# in-sample origins (see _one_window); ERM_INSAMPLE_TARGET sets how many in-sample
# samples to aim for and ERM_MAX_ORIGINS caps the extra forecasting cost. Even so, on the
# small hierarchies G stays under-determined; run_coherence flags ERM 'unstable' there
# rather than reporting the exploded MASE.
ERM_LAMBDA = 1.0
ERM_INSAMPLE_TARGET = 60
ERM_MAX_ORIGINS = 12
ERM_STABILITY_FACTOR = 3.0     # ERM flagged unstable if its MASE > factor x the raw MASE

from .hierarchies import ALL_HIERARCHIES as HIERARCHIES
from .hierarchies import load as load_hierarchy


def _mase_scale(y_full, period):
    y = np.asarray(y_full, dtype=float)
    m = period if (period > 1 and len(y) > period) else 1
    d = np.mean(np.abs(y[m:] - y[:-m])) if len(y) > m else np.nan
    if not np.isfinite(d) or d < 1e-8:
        d = np.mean(np.abs(np.diff(y))) if len(y) > 1 else 1.0
    return float(max(d, 1e-8))


def _bootstrap_ci(values, n_boot=1000, seed=0):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _one_window(group_data, tsfms, h, period, w, batch_size, use_caaa):
    """Base-forecast + reconcile for a single rolling origin; returns
    (incoherence_rows, scaled_error_records). Test = h points ending at L-w*h."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS
    from hierarchicalforecast.core import HierarchicalReconciliation
    from hierarchicalforecast.methods import BottomUp, MinTrace, ERM

    lam = ERM_LAMBDA
    ERM_COL = f"ERM_method-reg_bu_lambda_reg-{lam}"   # hierarchicalforecast's column name

    Y, S, tags, all_ids, bottom_ids, Smat, idx, scale, series = group_data
    train, test = {}, {}
    for uid, yv in series.items():
        end = len(yv) - w * h
        if end - h - 4 < 1:
            continue
        train[uid] = yv[:end - h]
        test[uid] = yv[end - h:end]
    nodes = [u for u in all_ids if u in train]
    if not nodes:
        return [], []

    def _fc_block(model, hs):
        """h-step forecast per history. Trained global models (exposing ``fit``) are
        trained once on the given histories then predicted; zero-shot models are
        batched. Returns a list of median arrays aligned to ``hs``."""
        if hasattr(model, "fit"):
            model.fit(hs, h)
            preds = model.predict_batch(hs, h)
        else:
            preds = []
            for i in range(0, len(hs), batch_size):
                preds.extend(model.predict_batch(hs[i:i + batch_size], h))
        return [f.median for f in preds]

    # ERM (Ben Taieb & Koo) learns G from in-sample (forecast, actual) pairs. A single
    # held-out block gives only h samples for an (n_bottom x n_total) G, badly
    # under-determined, so ERM explodes on small hierarchies. We give it K rolling
    # in-sample origins (origin j forecasts the block ending K-j blocks back), scaled to
    # what the series length supports and targeting ~ERM_INSAMPLE_TARGET samples, spanning
    # the common in-sample range ds in [Lt-Ki, Lt-1].
    Lt = len(train[nodes[0]])                               # common training length
    min_ctx = max(2 * period, 16)
    max_k = max(1, (Lt - min_ctx) // h)
    K = int(min(max_k, ERM_MAX_ORIGINS, max(1, -(-ERM_INSAMPLE_TARGET // h))))
    Ki = K * h

    base, base_in = {}, {}
    for name, model in tsfms.items():                       # FMs + global ML models
        try:
            base[name] = dict(zip(nodes, _fc_block(model, [train[u] for u in nodes])))
            # ERM in-sample origins only for zero-shot models. Producing them for a
            # *trained* global model would need K full retrains per window, and using
            # the full-data model to predict its own training period would leak. Trained
            # models (exposing ``fit``) therefore get structural reconcilers only.
            if not hasattr(model, "fit"):
                blocks = {u: np.empty(Ki) for u in nodes}   # concatenated K in-sample blocks
                for j in range(1, K + 1):
                    ctx = [train[u][:Lt - j * h] for u in nodes]
                    for u, arr in zip(nodes, _fc_block(model, ctx)):
                        blocks[u][(K - j) * h:(K - j) * h + h] = np.asarray(arr, dtype=float)[:h]
                base_in[name] = blocks
        except Exception as e:
            raise RuntimeError(
                f"model {name!r} failed on group "
                f"{group_data[0].attrs.get('group','?')!r} window {w}: "
                f"{type(e).__name__}: {e}"
            ) from e

    if use_caaa:
        # AutoETS out-of-sample forecast AND in-sample fitted values in one fit; the last
        # Ki fitted values align to the same ds range as the foundation-model origins.
        recs = [(u, i, float(v)) for u in nodes for i, v in enumerate(train[u])]
        dft = pd.DataFrame(recs, columns=["unique_id", "ds", "y"])
        sf = StatsForecast(models=[AutoETS(season_length=period)], freq=1, n_jobs=-1,
                           fallback_model=AutoETS(season_length=1))
        fc = sf.forecast(df=dft, h=h, fitted=True).set_index("unique_id")
        fitv = sf.forecast_fitted_values().set_index("unique_id")
        base["AutoETS"] = {u: fc.loc[u].sort_values("ds")["AutoETS"].to_numpy(dtype=float)
                           for u in nodes}
        base_in["AutoETS"] = {u: fitv.loc[u].sort_values("ds")["AutoETS"]
                              .to_numpy(dtype=float)[-Ki:] for u in nodes}

    # incoherence per model (aggregate nodes)
    bottom_pos = [idx[b] for b in bottom_ids]
    agg_mask = np.array([u not in set(bottom_ids) for u in all_ids])
    inc_rows = []
    for name, preds in base.items():
        F = np.full((len(all_ids), h), np.nan)
        for u in nodes:
            F[idx[u]] = preds[u]
        present = agg_mask & ~np.isnan(F).any(axis=1)
        implied = Smat @ F[bottom_pos]
        rel = np.abs(F[present] - implied[present]).sum(axis=1) / (np.abs(F[present]).sum(axis=1) + 1e-9)
        inc_rows.append(dict(window=w, model=name,
                             mean_incoherence_pct=float(100 * np.mean(rel)),
                             pct_nodes_incoherent=float(100 * np.mean(rel > 0.01))))

    # reconcile. Y_hat_df = out-of-sample forecasts; Y_df carries the held-out
    # in-sample block (actual y + each model's in-sample forecast) that ERM needs to
    # learn G. BottomUp and MinTrace(ols) are structural and ignore the model columns.
    Yhat = pd.DataFrame([{"unique_id": u, "ds": len(train[u]) + k,
                          **{n: base[n][u][k] for n in base}}
                         for u in nodes for k in range(h)])
    Ydf = pd.DataFrame([{"unique_id": u, "ds": len(train[u]) - Ki + k,
                         "y": float(train[u][len(train[u]) - Ki + k]),
                         **{n: base_in[n][u][k] for n in base_in}}
                        for u in nodes for k in range(Ki)])
    S_df = S.loc[all_ids].reset_index().rename(columns={"index": "unique_id"})
    id_cols = ["unique_id", "ds"]
    erm_models = [n for n in base if n in base_in]          # zero-shot FMs + AutoETS
    struct_models = [n for n in base if n not in base_in]   # trained global models

    def _reconcile(cols, reconcilers, y_df):
        try:
            return HierarchicalReconciliation(reconcilers=reconcilers).reconcile(
                Y_hat_df=Yhat[id_cols + cols], S_df=S_df, tags=tags,
                Y_df=y_df).reset_index(drop=True)
        except Exception:
            return None

    parts = []
    if erm_models:                          # bottom-up + MinT-ols + ERM (needs Y_df)
        r = _reconcile(erm_models, [BottomUp(), MinTrace(method="ols"),
                                    ERM(method="reg_bu", lambda_reg=lam)], Ydf)
        if r is None:                       # ERM ill-conditioned -> structural only
            r = _reconcile(erm_models, [BottomUp(), MinTrace(method="ols")], Ydf)
        if r is not None:
            parts.append(r)
    if struct_models:                       # structural reconcilers only (no in-sample)
        y_min = Ydf[id_cols + ["y"]] if "y" in Ydf.columns else None
        r = _reconcile(struct_models, [BottomUp(), MinTrace(method="ols")], y_min)
        if r is not None:
            parts.append(r)

    if parts:
        merged = parts[0]
        for r in parts[1:]:
            merged = merged.merge(r, on=id_cols, how="outer")
        Yrec = merged.set_index("unique_id")
    else:
        Yrec = Yhat.set_index("unique_id")

    # per-(node) scaled error for each (model, method)
    variants = {}
    for n in [m for m in tsfms if m in base]:
        variants[(n, "raw")] = n
        variants[(n, "BottomUp")] = f"{n}/BottomUp"
        variants[(n, "MinT-ols")] = f"{n}/MinTrace_method-ols"
        if n in base_in:
            variants[(n, "ERM")] = f"{n}/{ERM_COL}"
    if use_caaa:
        variants[("CAA", "reconciled")] = "AutoETS/BottomUp"
        variants[("AutoETS", "raw")] = "AutoETS"
        variants[("AutoETS", "ERM")] = f"AutoETS/{ERM_COL}"
    err_records = []
    for u in nodes:
        sub = Yrec.loc[u]
        for (model, method), col in variants.items():
            if col not in getattr(sub, "columns", sub.index):
                continue
            pred = np.atleast_1d(sub.sort_values("ds")[col].to_numpy(dtype=float))[:h]
            err_records.append(dict(window=w, model=model, method=method,
                                    sae=float(np.mean(np.abs(test[u] - pred)) / scale[u])))
    return inc_rows, err_records


def run_coherence(group: str, tsfms: dict, n_windows: int = 3, batch_size: int = 256,
                  use_caaa: bool = True) -> dict:
    """Rolling-origin coherence + reconciliation-accuracy for one hierarchy.

    Returns {'incoherence': df (mean over windows), 'accuracy': df (mean + 95% CI
    over node-window pairs)}.
    """
    Y, S, tags, meta = load_hierarchy(group)
    h, period = meta.horizon, meta.seasonality
    all_ids = list(S.index)
    bottom_ids = list(S.columns)
    Smat = S.to_numpy(dtype=float)
    idx = {u: i for i, u in enumerate(all_ids)}
    series, scale = {}, {}
    for uid, g in Y.groupby("unique_id", sort=False):
        yv = g.sort_values("ds")["y"].to_numpy(dtype=float)
        if len(yv) > h + 4:                       # support at least one origin
            series[uid] = yv
            scale[uid] = _mase_scale(yv, period)
    if not series:
        raise ValueError(f"{group}: no series long enough for horizon {h}")
    # adaptively cap rolling origins to what the SHORTEST series supports, so the
    # node set stays consistent across windows (short hierarchies can't do 8).
    min_len = min(len(v) for v in series.values())
    n_eff = int(max(1, min(n_windows, (min_len - 5) // h)))
    S.attrs["group"] = group
    gdata = (S, S, tags, all_ids, bottom_ids, Smat, idx, scale, series)

    inc_all, err_all = [], []
    for w in range(n_eff):
        inc_rows, err_records = _one_window(gdata, tsfms, h, period, w, batch_size, use_caaa)
        inc_all.extend(inc_rows)
        err_all.extend(err_records)

    inc = pd.DataFrame(inc_all)
    if inc.empty:
        raise RuntimeError(
            f"{group}: no incoherence rows produced. Every (series, window) pair was "
            f"dropped; usually n_windows is too large for the shortest series "
            f"(h={h}, requested n_windows={n_windows}, effective={n_eff})."
        )
    inc_summary = (inc.groupby("model")
                   .agg(mean_incoherence_pct=("mean_incoherence_pct", "mean"),
                        pct_nodes_incoherent=("pct_nodes_incoherent", "mean"))
                   .reset_index())
    inc_summary.insert(0, "group", group)
    inc_summary["n_windows"] = n_eff

    err = pd.DataFrame(err_all)
    acc_rows = []
    for (model, method), gm in err.groupby(["model", "method"]):
        mean, lo, hi = _bootstrap_ci(gm["sae"].to_numpy())
        acc_rows.append(dict(group=group, model=model, method=method,
                             mase=round(mean, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                             n=len(gm)))
    acc = pd.DataFrame(acc_rows)
    # Stability flag: a learned reconciler (ERM/MinT-ols) is 'unstable' on a hierarchy
    # when its under-determined fit blows the error far past the raw forecast. We report
    # the flag rather than silently dropping or printing the exploded MASE.
    raw_mase = {m: v for m, v in
                acc[acc.method == "raw"].set_index("model")["mase"].items()}
    def _stable(r):
        if r["method"] in ("raw", "BottomUp", "reconciled"):
            return True
        base = raw_mase.get(r["model"], np.nan)
        return bool(np.isfinite(base) and r["mase"] <= ERM_STABILITY_FACTOR * base)
    acc["stable"] = acc.apply(_stable, axis=1)
    return {"incoherence": inc_summary, "accuracy": acc}
