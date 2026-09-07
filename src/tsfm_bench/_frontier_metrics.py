"""Experiment A: the sample-efficiency frontier on real data with a real TSFM.

For each series we hold the test window fixed and truncate the available history
to T in {12, 24, ...}. Only the input history changes; the forecast target is
identical across T. This isolates the axis no TSFM benchmark varies: accuracy as a
function of per-series history length.

Speed: classical models are fit with statsforecast's parallel `StatsForecast`
engine (n_jobs=-1, all cores) in ONE call per history length, not a per-series
Python loop. CAA selects among the already-computed candidates and projects, so
it adds no extra fitting. Chronos runs batched on the GPU.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .loaders import SeriesItem
from .models import ChronosBolt


def _series_scale(y_full: np.ndarray, period: int) -> float:
    """MASE scale = in-sample seasonal-naive MAE on the FULL series.

    Defined once per series (not per truncated window) so MASE is comparable
    across history lengths T and is not degenerate when a short window happens to
    be near-constant. Falls back to first-difference MAE, then to the series level.
    """
    y = np.asarray(y_full, dtype=float)
    m = period if (period > 1 and len(y) > period) else 1
    denom = np.mean(np.abs(y[m:] - y[:-m])) if len(y) > m else np.nan
    if not np.isfinite(denom) or denom < 1e-8:
        denom = np.mean(np.abs(np.diff(y))) if len(y) > 1 else np.nan
    if not np.isfinite(denom) or denom < 1e-8:
        denom = max(np.mean(np.abs(y)), 1e-8)   # last resort: relative to level
    return float(denom)


def _mase(y_true, y_pred, scale):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / scale)


def _caaa_choose(train: np.ndarray, period: int) -> str:
    """Lightweight constraint/feature chooser over the 3 computed candidates."""
    n = len(train)
    if n < 2 * period:                       # too short for seasonality
        return "AutoTheta"                   # low-variance workhorse
    # seasonal strength
    sm = np.array([train[i::period].mean() for i in range(period)])
    sfull = np.array([sm[i % period] for i in range(n)])
    seas = max(0.0, 1.0 - np.sum((train - sfull) ** 2) /
               (np.sum((train - train.mean()) ** 2) + 1e-9))
    if n < 96:
        return "AutoTheta"                   # short history -> avoid high variance
    if seas > 0.5:
        return "AutoETS"                     # clear seasonality
    return "AutoETS"


def run_frontier(items: list[SeriesItem], chronos: ChronosBolt,
                 history_lengths=(12, 24, 48, 96, 192, 384),
                 horizon: int = 18, batch_size: int = 512) -> pd.DataFrame:
    """Returns tidy rows [uid, T, model, mase]."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, AutoTheta, SeasonalNaive

    longest = max((len(it.y) for it in items), default=0)
    history_lengths = tuple(T for T in history_lengths if T + horizon + 1 <= longest)
    if not history_lengths:
        raise ValueError(f"longest series {longest} too short for any history length")
    max_T = max(history_lengths)
    usable = [it for it in items if len(it.y) >= max_T + horizon + 1]
    if not usable:
        raise ValueError(f"no series >= max_T({max_T}) + horizon({horizon}) + 1")
    period = usable[0].period                # assumes one frequency per call
    # fixed per-series MASE scale from the full series (comparable across T)
    scale = {it.uid: _series_scale(it.y, period) for it in usable}
    rows = []

    for T in history_lengths:
        # truncate every series to its last T training points (fixed test window)
        recs, trains, tests = [], {}, {}
        for it in usable:
            split = len(it.y) - horizon
            tr = it.y[split - T:split]
            trains[it.uid] = tr
            tests[it.uid] = it.y[split:split + horizon]
            for i, v in enumerate(tr):
                recs.append((it.uid, i, float(v)))
        df = pd.DataFrame(recs, columns=["unique_id", "ds", "y"])

        # one parallel fit/forecast across ALL series and 3 candidate models
        sf = StatsForecast(
            models=[SeasonalNaive(season_length=period),
                    AutoETS(season_length=period),
                    AutoTheta(season_length=period)],
            freq=1, n_jobs=-1, fallback_model=SeasonalNaive(season_length=period))
        fc = sf.forecast(df=df, h=horizon)
        fc = fc.set_index("unique_id")

        # Chronos: batched GPU inference on the same truncated histories
        uids = [it.uid for it in usable]
        cf = []
        hist_list = [trains[u] for u in uids]
        for i in range(0, len(hist_list), batch_size):
            cf.extend(chronos.predict_batch(hist_list[i:i + batch_size], horizon))
        cf_map = dict(zip(uids, cf))

        for it in usable:
            uid = it.uid
            g = fc.loc[uid].sort_values("ds")
            cand = {"SeasonalNaive": g["SeasonalNaive"].to_numpy(dtype=float),
                    "AutoETS": g["AutoETS"].to_numpy(dtype=float),
                    "AutoTheta": g["AutoTheta"].to_numpy(dtype=float)}
            tr, te = trains[uid], tests[uid]

            caaa_kind = _caaa_choose(tr, period)
            caaa_pred = cand[caaa_kind].copy()
            if it.non_negative:                # constraint projection
                caaa_pred = np.maximum(caaa_pred, 0.0)

            chronos_pred = cf_map[uid].median
            for model, pred in [("Chronos-Bolt", chronos_pred),
                                ("SeasonalNaive", cand["SeasonalNaive"]),
                                ("AutoETS", cand["AutoETS"]),
                                ("CAA", caaa_pred)]:
                rows.append(dict(uid=uid, T=T, model=model,
                                 mase=_mase(te, pred, scale[uid])))
    return pd.DataFrame(rows)


def crossover(df: pd.DataFrame, base="AutoETS", challenger="Chronos-Bolt"):
    """Smallest T at which challenger's mean MASE drops below base's."""
    agg = df.groupby(["T", "model"])["mase"].mean().unstack("model")
    for T in sorted(agg.index):
        if agg.loc[T, challenger] < agg.loc[T, base]:
            return int(T)
    return None
