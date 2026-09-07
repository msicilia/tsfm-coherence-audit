"""Temporal-aggregation coherence: the same structural failure on a different axis.

Cross-sectional coherence couples *different series*. A reader can object that one
should simply model the hierarchy jointly. Temporal coherence removes that escape: it
couples a *single* series to itself across time aggregations. If a model forecasts a
monthly series and, independently, the annual aggregate of that same series, the twelve
monthly forecasts must sum to the annual one. There is no second series to model
jointly, yet a per-series forecaster still has no mechanism to enforce it.

For monthly data with a 12-step bottom horizon the temporal hierarchy uses the
divisors of 12:

    k = 1 : 12 monthly nodes   (bottom)
    k = 2 :  6 bi-monthly
    k = 3 :  4 quarterly
    k = 4 :  3 four-monthly
    k = 6 :  2 semi-annual
    k = 12:  1 annual
                              -> 28 nodes, 12 bottom, 16 aggregates

Each level is forecast independently from its OWN aggregated history, exactly as a
practitioner would if they wanted monthly and annual numbers from the same model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DIVISORS_12 = (1, 2, 3, 4, 6, 12)


def aggregate_from_end(y: np.ndarray, k: int) -> np.ndarray:
    """Non-overlapping sums of k, aligned to the END of the array.

    End-alignment matters: the last training block must be immediately adjacent to
    the first forecast block, otherwise the levels are offset and the coherence
    check is meaningless. The oldest remainder is dropped.
    """
    y = np.asarray(y, dtype=float)
    n = len(y) // k
    if n == 0:
        return np.empty(0)
    return y[len(y) - n * k:].reshape(n, k).sum(axis=1)


def build_temporal_S(h_bottom: int = 12, ks=DIVISORS_12):
    """Summing matrix for the temporal hierarchy: (n_nodes, h_bottom)."""
    labels, rows = [], []
    for k in ks:
        for j in range(h_bottom // k):
            r = np.zeros(h_bottom)
            r[j * k:(j + 1) * k] = 1.0
            rows.append(r)
            labels.append(f"k{k}_{j}")
    return np.array(rows), labels


def _mase_scale(y: np.ndarray, season: int) -> float:
    y = np.asarray(y, dtype=float)
    m = season if (season > 1 and len(y) > season) else 1
    d = np.mean(np.abs(y[m:] - y[:-m])) if len(y) > m else np.nan
    if not np.isfinite(d) or d < 1e-8:
        d = np.mean(np.abs(np.diff(y))) if len(y) > 1 else 1.0
    return float(max(d, 1e-8))


def run_temporal_coherence(items, tsfm, h_bottom: int = 12, ks=DIVISORS_12,
                           n_windows: int = 4, batch_size: int = 256,
                           season_bottom: int = 12) -> pd.DataFrame:
    """Per-series, per-window temporal incoherence and per-level accuracy.

    `items` are SeriesItem objects (monthly). Returns one row per
    (series, window, aggregation level).
    """
    S, labels = build_temporal_S(h_bottom, ks)
    usable = [it for it in items if len(it.y) > h_bottom * n_windows + h_bottom * 12]
    if not usable:
        raise ValueError("no series long enough for the temporal hierarchy")

    rows = []
    for w in range(n_windows):
        # split every series at this origin
        trains, tests = [], []
        for it in usable:
            end = len(it.y) - w * h_bottom
            trains.append(it.y[:end - h_bottom])
            tests.append(it.y[end - h_bottom:end])

        # forecast each aggregation level independently (batched per level)
        fc = {}
        for k in ks:
            hk = h_bottom // k
            hists = [aggregate_from_end(tr, k) for tr in trains]
            preds = []
            for i in range(0, len(hists), batch_size):
                preds.extend(tsfm.predict_batch(hists[i:i + batch_size], hk))
            fc[k] = [np.asarray(p.median, dtype=float)[:hk] for p in preds]

        for si, it in enumerate(usable):
            bottom = fc[1][si]                       # (h_bottom,)
            for k in ks:
                if k == 1:
                    continue
                direct = fc[k][si]                   # (h_bottom//k,)
                implied = aggregate_from_end(bottom, k)   # sum of bottom in blocks
                actual = aggregate_from_end(tests[si], k)
                scale = _mase_scale(aggregate_from_end(it.y, k),
                                    max(1, season_bottom // k))
                rel = np.abs(direct - implied) / (np.abs(direct) + 1e-9)
                rows.append(dict(
                    uid=it.uid, window=w, k=k, n_nodes=len(direct),
                    mean_rel_incoherence=float(np.mean(rel)),
                    pct_nodes_incoherent=float(100 * np.mean(rel > 0.01)),
                    mase_raw=float(np.mean(np.abs(actual - direct)) / scale),
                    mase_bottomup=float(np.mean(np.abs(actual - implied)) / scale),
                ))
    return pd.DataFrame(rows)
