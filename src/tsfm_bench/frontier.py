"""Experiment A: full, significance-tested sample-efficiency study.

Extends the single-window frontier with:
  * multiple datasets / frequencies (M4-Monthly, M4-Quarterly, M5-daily);
  * rolling-origin evaluation (W windows per series);
  * bootstrap 95% confidence intervals on mean MASE per (dataset, T, model);
  * win-rate of each small model versus the foundation model, with a binomial
    test of H0: win-rate = 0.5.

Classical models are fit with statsforecast's parallel engine (n_jobs=-1) in one
call per (dataset, window, T). Chronos runs batched on the GPU. Per-series MASE
scale is fixed from the full series so it is comparable across T and windows.
Results are cached per dataset (resumable).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ._frontier_metrics import _mase, _series_scale
from .loaders import load_m4, load_m5_sample
from .models import ChronosBolt


# --------------------------------------------------------------------------- #
# Dataset configurations
# --------------------------------------------------------------------------- #
def _cfg_m4_monthly(cap):
    return load_m4("Monthly", max_series=cap, min_len=438)


def _cfg_m4_quarterly(cap):
    return load_m4("Quarterly", max_series=cap, min_len=152)


def _cfg_m5(cap):
    return load_m5_sample(n_series=cap or 1000, min_len=266)


DATASETS = [
    dict(name="M4-Monthly",   make=_cfg_m4_monthly,   period=12, horizon=18,
         history_lengths=(12, 24, 48, 96, 192, 384), cap=1500, non_negative=True),
    dict(name="M4-Quarterly", make=_cfg_m4_quarterly, period=4,  horizon=8,
         history_lengths=(8, 16, 32, 64, 128),        cap=1500, non_negative=True),
    dict(name="M5-daily",     make=_cfg_m5,           period=7,  horizon=14,
         history_lengths=(28, 56, 112, 224),          cap=1000, non_negative=True),
]


def _caaa_choose(train: np.ndarray) -> str:
    """Constraint/regime-aware selector over the computed candidates.

    Intermittency is itself a structural signal: zero-inflated demand calls for an
    intermittent-demand model (Croston), which is small, interpretable and
    non-negative by construction, a far better match than seasonal naive or ETS.
    Smooth series default to ETS.
    """
    if np.mean(np.abs(train) < 1e-9) > 0.3:   # intermittent -> Croston
        return "CrostonOptimized"
    return "AutoETS"


# --------------------------------------------------------------------------- #
# Core run (one dataset) with rolling origins
# --------------------------------------------------------------------------- #
def run_dataset(cfg: dict, tsfms, n_windows: int = 3,
                batch_size: int = 512, quick: bool = False,
                per_step: bool = False):
    """Returns the per-series MASE frame; if `per_step`, also a long frame of
    per-horizon-step scaled absolute errors, which a Diebold--Mariano test needs
    (MASE alone is a per-series scalar and cannot support it)."""
    """`tsfms`: dict name -> model with predict_batch(histories, h). A single
    model object is accepted for backward compatibility (named 'Chronos-Bolt')."""
    if not isinstance(tsfms, dict):
        tsfms = {"Chronos-Bolt": tsfms}
    from statsforecast import StatsForecast
    from statsforecast.models import (AutoETS, AutoTheta, CrostonOptimized,
                                      SeasonalNaive)

    period, horizon = cfg["period"], cfg["horizon"]
    history_lengths = cfg["history_lengths"]
    if quick:
        history_lengths = history_lengths[:3]
        n_windows = 2
    max_T = max(history_lengths)
    cap = 120 if quick else cfg["cap"]

    items = cfg["make"](cap)
    need = max_T + horizon * n_windows
    items = [it for it in items if len(it.y) >= need]
    if not items:
        raise ValueError(f"{cfg['name']}: no series >= {need}")
    scale = {it.uid: _series_scale(it.y, period) for it in items}

    rows = []
    step_rows = []
    for w in range(n_windows):
        for T in history_lengths:
            recs, trains, tests = [], {}, {}
            for it in items:
                L = len(it.y)
                test_end = L - w * horizon
                test = it.y[test_end - horizon:test_end]
                tr = it.y[test_end - horizon - T:test_end - horizon]
                if len(tr) < T or len(test) < horizon:
                    continue
                trains[it.uid] = tr
                tests[it.uid] = test
                for i, v in enumerate(tr):
                    recs.append((it.uid, i, float(v)))
            if not trains:
                continue
            df = pd.DataFrame(recs, columns=["unique_id", "ds", "y"])
            sf = StatsForecast(
                models=[SeasonalNaive(season_length=period),
                        AutoETS(season_length=period),
                        AutoTheta(season_length=period),
                        CrostonOptimized()],
                freq=1, n_jobs=-1,
                fallback_model=SeasonalNaive(season_length=period))
            fc = sf.forecast(df=df, h=horizon).set_index("unique_id")

            uids = list(trains)
            hist_list = [trains[u] for u in uids]
            # each TSFM: batched inference -> uid -> median forecast
            tsfm_pred = {name: {} for name in tsfms}
            for name, model in tsfms.items():
                preds_list = []
                for i in range(0, len(hist_list), batch_size):
                    preds_list.extend(model.predict_batch(hist_list[i:i + batch_size], horizon))
                tsfm_pred[name] = dict(zip(uids, preds_list))

            for uid in uids:
                g = fc.loc[uid].sort_values("ds")
                cand = {"SeasonalNaive": g["SeasonalNaive"].to_numpy(dtype=float),
                        "AutoETS": g["AutoETS"].to_numpy(dtype=float),
                        "AutoTheta": g["AutoTheta"].to_numpy(dtype=float),
                        "CrostonOptimized": g["CrostonOptimized"].to_numpy(dtype=float)}
                caaa = cand[_caaa_choose(trains[uid])].copy()
                if cfg["non_negative"]:
                    caaa = np.maximum(caaa, 0.0)
                preds = {"SeasonalNaive": cand["SeasonalNaive"],
                         "AutoETS": cand["AutoETS"],
                         "CAA": caaa}
                for name in tsfms:
                    preds[name] = tsfm_pred[name][uid].median
                for model, pred in preds.items():
                    rows.append(dict(dataset=cfg["name"], window=w, uid=uid, T=T,
                                     model=model,
                                     mase=_mase(tests[uid], pred, scale[uid])))
                    if per_step:
                        ae = np.abs(np.asarray(tests[uid], dtype=float)
                                    - np.asarray(pred, dtype=float)) / scale[uid]
                        for step, v in enumerate(ae):
                            step_rows.append((cfg["name"], w, uid, T, model, step, float(v)))
    df = pd.DataFrame(rows)
    if not per_step:
        return df
    steps = pd.DataFrame(step_rows, columns=["dataset", "window", "uid", "T",
                                             "model", "step", "ae_scaled"])
    return df, steps


# --------------------------------------------------------------------------- #
# Significance summaries
# --------------------------------------------------------------------------- #
def _bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0):
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _binom_p_two_sided(k: int, n: int) -> float:
    """Two-sided binomial test p-value against p=0.5 (normal approx for large n)."""
    if n == 0:
        return np.nan
    z = (k - n / 2) / np.sqrt(n / 4)
    # two-sided normal tail
    from math import erfc, sqrt
    return float(erfc(abs(z) / sqrt(2)))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean MASE with 95% bootstrap CI per (dataset, T, model)."""
    out = []
    for (dataset, T), g in df.groupby(["dataset", "T"]):
        for model, gm in g.groupby("model"):
            mean, lo, hi = _bootstrap_ci(gm["mase"].to_numpy())
            out.append(dict(dataset=dataset, T=T, model=model,
                            mean_mase=round(mean, 4), ci_lo=round(lo, 4),
                            ci_hi=round(hi, 4)))
    return pd.DataFrame(out).sort_values(["dataset", "T", "model"]).reset_index(drop=True)


def winrates(df: pd.DataFrame, smalls=("AutoETS", "CAA"),
             tsfms=("Chronos-Bolt", "Chronos-Bolt-Base", "Chronos-2", "Moirai",
                    "TimesFM")) -> pd.DataFrame:
    """Win-rate of each small model vs each TSFM per (dataset, T), with a
    two-sided binomial test against 0.5."""
    piv = df.pivot_table(index=["dataset", "window", "uid", "T"],
                         columns="model", values="mase")
    present_tsfms = [t for t in tsfms if t in piv.columns]
    out = []
    for (dataset, T), sub in piv.groupby(level=["dataset", "T"]):
        for small in smalls:
            if small not in piv.columns:
                continue
            for tsfm in present_tsfms:
                paired = sub[[small, tsfm]].dropna()
                n = len(paired)
                wins = int((paired[small] < paired[tsfm]).sum())
                out.append(dict(dataset=dataset, T=T, small=small, tsfm=tsfm,
                                n_pairs=n,
                                winrate=round(wins / n, 3) if n else np.nan,
                                binom_p=round(_binom_p_two_sided(wins, n), 4) if n else np.nan))
    return pd.DataFrame(out).sort_values(["dataset", "T", "small", "tsfm"]).reset_index(drop=True)
