"""Probabilistic coherence: do foundation models' *predictive distributions*
satisfy hierarchical coherence, and does sample-space reconciliation fix the whole
distribution at no probabilistic-accuracy cost?

A per-series foundation model produces an independent predictive distribution for
every node. These are incoherent in a stronger sense than the point forecasts: the
aggregate node's distribution is fitted independently and is not the convolution of
its children's. We:
  1. draw samples from each node's predictive quantiles (inverse-CDF sampling);
  2. reconcile bottom-up in sample space, so that aggregate samples = S @ bottom
     samples, which yields a coherent joint distribution by construction;
  3. score raw vs reconciled distributions with CRPS (a proper scoring rule),
     scaled per series for comparability.

This extends the point-coherence result to the probabilistic setting foundation
models are increasingly deployed in, and is beyond standard point reconciliation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .coherence import _mase_scale, load_hierarchy

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def samples_from_quantiles(qvals: np.ndarray, levels, n_samples: int, rng) -> np.ndarray:
    """qvals: (h, n_levels) increasing in level -> (n_samples, h) by inverse-CDF
    interpolation of the quantile function, with flat tails beyond the extremes."""
    levels = np.asarray(levels, dtype=float)
    h = qvals.shape[0]
    u = rng.random((n_samples, h))
    out = np.empty((n_samples, h), dtype=float)
    for t in range(h):
        out[:, t] = np.interp(u[:, t], levels, qvals[t])   # flat extrapolation at tails
    return out


def crps_sample(samples: np.ndarray, actual: np.ndarray) -> float:
    """Mean CRPS over the horizon from samples (n_samples, h) and actual (h,).
    Energy form with the O(N log N) estimator for E|X-X'|."""
    n = samples.shape[0]
    term1 = np.mean(np.abs(samples - actual[None, :]), axis=0)       # (h,)
    xs = np.sort(samples, axis=0)
    w = (2 * np.arange(1, n + 1) - n - 1)[:, None]                   # (n,1)
    exx = (2.0 / n ** 2) * np.sum(w * xs, axis=0)                    # E|X-X'| per h
    return float(np.mean(term1 - 0.5 * exx))


def schaake_shuffle(raw_samples: dict, train: dict, bottom_ids: list, h: int,
                    n_samples: int, rng) -> dict:
    """Reorder each bottom series' marginal samples to follow the historical
    rank-dependence (ensemble copula coupling / Schaake shuffle).

    The same N historical dates are used for every series, so member i refers to
    the same calendar date across all bottom series, which is what imposes the
    cross-series dependence that naive (independent) summation ignores.
    """
    present = [b for b in bottom_ids if b in raw_samples]
    Lmin = min(len(train[b]) for b in present)
    if Lmin - h < 2:                                   # not enough history -> no shuffle
        return raw_samples
    starts = rng.integers(0, Lmin - h, size=n_samples)  # SHARED across series
    shuffled = {}
    for b in present:
        tr = train[b]
        hist = np.stack([tr[s:s + h] for s in starts], axis=0)   # (N, h) templates
        sm = raw_samples[b]                                       # (N, h) marginals
        out = np.empty_like(sm)
        for t in range(h):
            ranks = np.argsort(np.argsort(hist[:, t]))            # member -> rank
            out[:, t] = np.sort(sm[:, t])[ranks]
        shuffled[b] = out
    return shuffled


def _bottom_corr(train: dict, bottom_ids: list):
    """Correlation matrix of bottom-series first differences (detrended), on a
    common-length tail. Returns (present_ids, R) or (present_ids, None)."""
    present = [b for b in bottom_ids if b in train]
    if len(present) < 2:
        return present, None
    Lmin = min(len(train[b]) for b in present)
    if Lmin < 8:
        return present, None
    M = np.stack([np.asarray(train[b][-Lmin:], dtype=float) for b in present], axis=1)
    D = np.diff(M, axis=0)                       # detrend -> contemporaneous dependence
    if D.shape[0] < 3:
        return present, None
    R = np.nan_to_num(np.corrcoef(D, rowvar=False), nan=0.0)
    np.fill_diagonal(R, 1.0)
    return present, R


def bottom_dependence(train: dict, bottom_ids: list) -> float:
    """Scalar cross-series dependence of a hierarchy: mean |off-diagonal
    correlation| of the bottom series (first differences). This is the axis we
    regress the distributional-reconciliation CRPS cost against."""
    present, R = _bottom_corr(train, bottom_ids)
    if R is None:
        return float("nan")
    n = R.shape[0]
    return float(np.mean(np.abs(R[~np.eye(n, dtype=bool)])))


def gaussian_copula_shuffle(raw_samples: dict, train: dict, bottom_ids: list, h: int,
                            n_samples: int, rng) -> dict:
    """Parametric alternative to the Schaake shuffle: impose a Gaussian copula
    fitted to the historical bottom-series correlation. Draw correlated normals,
    rank them, and reorder each series' marginal samples by those ranks so the
    reconciled joint carries the fitted cross-series dependence."""
    present = [b for b in bottom_ids if b in raw_samples]
    _, R = _bottom_corr({b: train[b] for b in present}, present)
    if R is None:
        return raw_samples
    vals, vecs = np.linalg.eigh(R)               # nearest-PSD via eigenvalue clip
    L = vecs @ np.diag(np.sqrt(np.clip(vals, 1e-6, None)))
    out = {b: np.empty_like(raw_samples[b]) for b in present}
    for t in range(h):
        Z = rng.standard_normal((n_samples, len(present))) @ L.T   # (N, nb) correlated
        for j, b in enumerate(present):
            ranks = np.argsort(np.argsort(Z[:, j]))
            out[b][:, t] = np.sort(raw_samples[b][:, t])[ranks]
    return out


def run_prob_coherence(group: str, tsfm, n_samples: int = 200, seed: int = 0,
                       batch_size: int = 256, n_windows: int = 3) -> pd.DataFrame:
    """Returns per-node, per-window CRPS for one hierarchy: raw (incoherent) vs two
    coherent sample-space reconciliations (independent marginals; Schaake/ECC)."""
    Y, S, tags, meta = load_hierarchy(group)
    h, period = meta.horizon, meta.seasonality
    all_ids = list(S.index)
    bottom_ids = list(S.columns)
    idx = {u: i for i, u in enumerate(all_ids)}
    rng = np.random.default_rng(seed)

    series, scale = {}, {}
    for uid, g in Y.groupby("unique_id", sort=False):
        yv = g.sort_values("ds")["y"].to_numpy(dtype=float)
        if len(yv) > h + 4:                       # support at least one origin
            series[uid] = yv; scale[uid] = _mase_scale(yv, period)
    if not series:
        raise ValueError(f"{group}: no series long enough for horizon {h}")
    min_len = min(len(v) for v in series.values())
    n_eff = int(max(1, min(n_windows, (min_len - 5) // h)))
    rows = []

    for w in range(n_eff):
        train, test = {}, {}
        for uid, yv in series.items():
            end = len(yv) - w * h
            train[uid] = yv[:end - h]; test[uid] = yv[end - h:end]
        nodes = [u for u in all_ids if u in train]

        qmap = {}
        hist = [train[u] for u in nodes]
        for i in range(0, len(hist), batch_size):
            q = tsfm.predict_quantiles_batch(hist[i:i + batch_size], h, LEVELS)
            for j, u in enumerate(nodes[i:i + batch_size]):
                qmap[u] = q[j]

        raw_samples = {u: samples_from_quantiles(qmap[u], LEVELS, n_samples, rng) for u in nodes}
        present_bottom = [b for b in bottom_ids if b in raw_samples]
        Sb = S[present_bottom].to_numpy(dtype=float)

        def reconcile(bottom):
            B = np.stack([bottom[b] for b in present_bottom], axis=0)
            return np.einsum("ab,bnh->anh", Sb, B)

        rec_indep = reconcile(raw_samples)
        rec_schaake = reconcile(schaake_shuffle(raw_samples, train, bottom_ids, h, n_samples, rng))
        rec_gauss = reconcile(gaussian_copula_shuffle(raw_samples, train, bottom_ids, h, n_samples, rng))
        dep = bottom_dependence(train, bottom_ids)      # hierarchy dependence for this window

        for u in nodes:
            rows.append(dict(group=group, uid=u, window=w, is_bottom=u in set(bottom_ids),
                             dependence=dep,
                             crps_raw=crps_sample(raw_samples[u], test[u]) / scale[u],
                             crps_indep=crps_sample(rec_indep[idx[u]], test[u]) / scale[u],
                             crps_schaake=crps_sample(rec_schaake[idx[u]], test[u]) / scale[u],
                             crps_gauss=crps_sample(rec_gauss[idx[u]], test[u]) / scale[u]))
    return pd.DataFrame(rows)
