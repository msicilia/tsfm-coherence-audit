"""Build a hierarchical dataset from M5 retail (a sixth hierarchy).

The five Monash hierarchies span little of the cross-series dependence axis that
the probabilistic-coherence law is estimated on (only Traffic is strongly
coupled). M5 retail gives an independent, strongly-coupled hierarchy: stores in a
state share demand shocks (holidays, SNAP days, seasons).

Structure (strict hierarchy, 70 bottom series, 14 aggregates, 84 nodes):
    Total  ->  State (3)  ->  Store (10)  ->  Store x Dept (70, bottom)

The 47M-row item-level table is aggregated once and cached, so re-runs are cheap.
"""
from __future__ import annotations

import os

import numpy as np

from .paths import DATA
import pandas as pd

HORIZON = 28          # M5 competition horizon
SEASONALITY = 7       # daily data


class _Meta:
    """Mimics datasetsforecast's HierarchicalInfo group meta."""
    def __init__(self, horizon=HORIZON, seasonality=SEASONALITY):
        self.horizon = horizon
        self.seasonality = seasonality


def build_m5_hierarchy(data_dir: str | None = None, cache: str | None = None):
    """Return (Y_long, S_df, tags, meta) in the same shape as HierarchicalData.load."""
    data_dir = str(DATA) if data_dir is None else data_dir
    cache = cache or os.path.join(data_dir, "m5_hierarchy_bottom.parquet")
    bottom = _bottom_series(data_dir, cache)          # (ds, store_id, dept_id, y)

    bottom["unique_id"] = bottom["store_id"].astype(str) + "/" + bottom["dept_id"].astype(str)
    bottom_ids = sorted(bottom["unique_id"].unique())
    store_of = {b: b.split("/")[0] for b in bottom_ids}
    state_of = {b: b.split("/")[0].split("_")[0] for b in bottom_ids}

    stores = sorted(set(store_of.values()))
    states = sorted(set(state_of.values()))
    all_ids = ["Total"] + states + stores + bottom_ids

    # summing matrix: S[node, bottom] = 1 if bottom rolls up into node
    S = pd.DataFrame(0.0, index=all_ids, columns=bottom_ids)
    for b in bottom_ids:
        S.loc["Total", b] = 1.0
        S.loc[state_of[b], b] = 1.0
        S.loc[store_of[b], b] = 1.0
        S.loc[b, b] = 1.0

    # long frame for every node = S @ bottom, built by aggregation
    wide = bottom.pivot_table(index="ds", columns="unique_id", values="y",
                              aggfunc="sum", fill_value=0.0)
    wide = wide.reindex(columns=bottom_ids, fill_value=0.0).sort_index()
    node_vals = {"Total": wide.sum(axis=1)}
    for s in states:
        cols = [b for b in bottom_ids if state_of[b] == s]
        node_vals[s] = wide[cols].sum(axis=1)
    for st in stores:
        cols = [b for b in bottom_ids if store_of[b] == st]
        node_vals[st] = wide[cols].sum(axis=1)
    for b in bottom_ids:
        node_vals[b] = wide[b]

    frames = []
    for uid in all_ids:
        v = node_vals[uid]
        frames.append(pd.DataFrame({"unique_id": uid,
                                    "ds": np.arange(len(v)),
                                    "y": v.to_numpy(dtype=float)}))
    Y = pd.concat(frames, ignore_index=True)

    tags = {"Total": np.array(["Total"]),
            "State": np.array(states),
            "Store": np.array(stores),
            "Store/Dept": np.array(bottom_ids)}
    return Y, S, tags, _Meta()


def _bottom_series(data_dir: str, cache: str) -> pd.DataFrame:
    """Aggregate item-level M5 to store x dept daily totals (cached)."""
    if os.path.exists(cache):
        return pd.read_parquet(cache)

    from datasetsforecast.m5 import M5
    Y, _X, meta = M5.load(directory=data_dir)
    cols = set(meta.columns)
    if not {"store_id", "dept_id"}.issubset(cols):
        raise ValueError(f"M5 metadata missing store_id/dept_id; has {sorted(cols)}")
    m = meta.drop_duplicates("unique_id").set_index("unique_id")
    # map with categoricals to keep the 47M-row frame small
    Y = Y.copy()
    Y["store_id"] = Y["unique_id"].map(m["store_id"]).astype("category")
    Y["dept_id"] = Y["unique_id"].map(m["dept_id"]).astype("category")
    bottom = (Y.groupby(["store_id", "dept_id", "ds"], observed=True)["y"]
              .sum().reset_index())
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    bottom.to_parquet(cache, index=False)
    return bottom
