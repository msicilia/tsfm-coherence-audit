"""Two high cross-series dependence hierarchies: Solar and Electricity.

These populate the high end of the cross-series dependence axis, which the public
benchmark hierarchies barely reach (Traffic, at 0.44, is the highest of them).
Dependence is the mean absolute off-diagonal correlation of the bottom series' first
differences, the metric used throughout:

    * Solar (137 PV stations, gluonts ``solar_nips``)   -> bottom dependence ~0.70
    * Electricity (370 clients grouped into feeders,
      gluonts ``electricity_nips``)                     -> bottom dependence ~0.63

Solar is coupled at the natural bottom level (all stations track the same sun), so
its bottom series are the stations themselves. Raw electricity *clients* are only
mid-coupled (~0.27); aggregating them into feeder groups (realistic in power
systems, where a feeder serves many meters) lifts the bottom dependence to ~0.63.
Electricity is also the energy-domain hierarchy used for the global ML baselines.

Both return ``(Y_long, S_df, tags, meta)`` in exactly the shape
``datasetsforecast.hierarchical.HierarchicalData.load`` and
``m5_hierarchy.build_m5_hierarchy`` return, so the coherence engine consumes them
unchanged.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .paths import DATA


class _Meta:
    """Mimics datasetsforecast's HierarchicalInfo group meta."""

    def __init__(self, horizon: int, seasonality: int):
        self.horizon = horizon
        self.seasonality = seasonality


# --------------------------------------------------------------------------- #
# Shared construction
# --------------------------------------------------------------------------- #
def _gluonts_wide(name: str, cache: str, max_len: int) -> pd.DataFrame:
    """Load a gluonts multivariate dataset as a wide (time x series) frame, cached.

    Series are right-aligned to their common length and capped at ``max_len`` most
    recent steps (the full electricity/solar histories are longer than we need and
    make the summing checks slow).
    """
    if os.path.exists(cache):
        return pd.read_parquet(cache)

    from gluonts.dataset.repository import get_dataset

    ds = get_dataset(name)
    rows = [np.asarray(e["target"], dtype=float) for e in ds.train]
    L = min(len(r) for r in rows)
    L = min(L, max_len)
    mat = np.stack([r[-L:] for r in rows])          # (n_series, L)
    wide = pd.DataFrame(mat.T, columns=[f"s{i}" for i in range(mat.shape[0])])
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    wide.to_parquet(cache, index=False)
    return wide


def _assemble(bottom_wide: pd.DataFrame, groups: dict[str, list[str]],
              mid_of: dict[str, str], horizon: int, seasonality: int):
    """Build (Y_long, S, tags, meta) for a 3-level hierarchy Total -> mid -> bottom.

    ``bottom_wide``  : (time x bottom_id) values of the bottom series.
    ``groups``       : mid_id -> list of bottom_ids under it.
    ``mid_of``       : bottom_id -> its mid_id.
    Verifies max|S y_bottom - y_all| == 0 by construction before returning.
    """
    bottom_ids = list(bottom_wide.columns)
    mids = list(groups.keys())
    all_ids = ["Total"] + mids + bottom_ids

    # summing matrix S[node, bottom] = 1 iff bottom rolls up into node
    S = pd.DataFrame(0.0, index=all_ids, columns=bottom_ids)
    for b in bottom_ids:
        S.loc["Total", b] = 1.0
        S.loc[mid_of[b], b] = 1.0
        S.loc[b, b] = 1.0

    # node value frames: aggregates are exact sums of their bottom members
    node_vals = {"Total": bottom_wide.sum(axis=1)}
    for m in mids:
        node_vals[m] = bottom_wide[groups[m]].sum(axis=1)
    for b in bottom_ids:
        node_vals[b] = bottom_wide[b]

    frames = []
    for uid in all_ids:
        v = node_vals[uid].to_numpy(dtype=float)
        frames.append(pd.DataFrame({"unique_id": uid,
                                    "ds": np.arange(len(v)),
                                    "y": v}))
    Y = pd.concat(frames, ignore_index=True)

    # self-check: the *data* must satisfy the constraint exactly (else S is wrong)
    Smat = S.to_numpy(dtype=float)
    y_bottom = bottom_wide[bottom_ids].to_numpy(dtype=float).T        # (n_bottom, T)
    y_all = np.stack([node_vals[u].to_numpy(dtype=float) for u in all_ids])  # (n_all, T)
    resid = float(np.max(np.abs(Smat @ y_bottom - y_all)))
    if resid > 1e-6:
        raise ValueError(f"summing matrix inconsistent with data: max|S y - y_all|={resid:g}")

    tags = {"Total": np.array(["Total"]),
            "Mid": np.array(mids),
            "Bottom": np.array(bottom_ids)}
    return Y, S, tags, _Meta(horizon, seasonality)


def _contiguous_groups(ids: list[str], n_groups: int, prefix: str):
    """Partition ``ids`` into ``n_groups`` contiguous buckets. Returns (groups, mid_of)."""
    groups: dict[str, list[str]] = {}
    mid_of: dict[str, str] = {}
    edges = np.linspace(0, len(ids), n_groups + 1).astype(int)
    for g in range(n_groups):
        members = ids[edges[g]:edges[g + 1]]
        if not members:
            continue
        mid = f"{prefix}{g}"
        groups[mid] = members
        for b in members:
            mid_of[b] = mid
    return groups, mid_of


# --------------------------------------------------------------------------- #
# Solar: bottom = 137 PV stations (naturally strongly coupled, ~0.70)
# --------------------------------------------------------------------------- #
def build_solar_hierarchy(data_dir: str | None = None, n_zones: int = 14,
                          horizon: int = 24, seasonality: int = 24, max_len: int = 3000):
    """Total -> zone -> station (bottom). Bottom = the PV stations themselves,
    which all track the same solar irradiance and are thus strongly coupled."""
    data_dir = str(DATA) if data_dir is None else data_dir
    cache = os.path.join(data_dir, "solar_wide.parquet")
    wide = _gluonts_wide("solar_nips", cache, max_len)
    bottom_ids = list(wide.columns)
    groups, mid_of = _contiguous_groups(bottom_ids, n_zones, "zone")
    return _assemble(wide, groups, mid_of, horizon, seasonality)


# --------------------------------------------------------------------------- #
# Electricity: bottom = feeder groups of clients (~0.63); energy-domain ML venue
# --------------------------------------------------------------------------- #
def build_electricity_hierarchy(data_dir: str | None = None, n_feeders: int = 40,
                                n_regions: int = 5, horizon: int = 24,
                                seasonality: int = 24, max_len: int = 3000):
    """Total -> region -> feeder (bottom). Raw clients are only mid-coupled (~0.27),
    so the bottom level is feeder groups (contiguous blocks of clients summed),
    lifting bottom dependence to ~0.63. This is also the energy-domain hierarchy on
    which the global ML baseline (LightGBM/TCN) competes with the foundation models."""
    data_dir = str(DATA) if data_dir is None else data_dir
    cache = os.path.join(data_dir, "electricity_wide.parquet")
    clients = _gluonts_wide("electricity_nips", cache, max_len)

    # aggregate clients -> feeders (bottom level)
    client_ids = list(clients.columns)
    feeder_groups, _ = _contiguous_groups(client_ids, n_feeders, "feeder")
    feeder_wide = pd.DataFrame(
        {fid: clients[members].sum(axis=1) for fid, members in feeder_groups.items()})

    # feeders -> regions (mid level)
    feeder_ids = list(feeder_wide.columns)
    regions, mid_of = _contiguous_groups(feeder_ids, n_regions, "region")
    return _assemble(feeder_wide, regions, mid_of, horizon, seasonality)
