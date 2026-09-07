"""Real datasets with *declared* structural constraints.

We pair each dataset with the hard constraints that hold for it by construction
(not inferred from the pattern): M-competition series are non-negative; a
hierarchy carries summing/coherence constraints. These declarations are what a
constraint-aware system gets to use and a vanilla TSFM ignores.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .paths import DATA

DATA_DIR = str(DATA)


@dataclass
class SeriesItem:
    uid: str
    y: np.ndarray              # full series (float)
    period: int                # seasonal period
    non_negative: bool = True


@dataclass
class Hierarchy:
    """A hierarchical dataset: long-format values + summing matrix S.

    S maps bottom-level series to every node: y_all = S @ y_bottom.
    Coherence constraint: forecasts must satisfy the same linear relation.
    """
    Y: pd.DataFrame                       # columns: unique_id, ds, y
    S: pd.DataFrame                       # index: all node ids, columns: bottom ids
    period: int
    bottom_ids: list = field(default_factory=list)
    agg_ids: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Flat datasets (non-negativity constraint)
# --------------------------------------------------------------------------- #
def load_m3(group: str = "Monthly", max_series: Optional[int] = None) -> list[SeriesItem]:
    """M3 competition. All series are non-negative (sales, demographics, etc.)."""
    from datasetsforecast.m3 import M3

    period = {"Monthly": 12, "Quarterly": 4, "Yearly": 1, "Other": 1}[group]
    df, *_ = M3.load(directory=DATA_DIR, group=group)
    items = []
    for uid, g in df.groupby("unique_id", sort=False):
        y = g["y"].to_numpy(dtype=float)
        items.append(SeriesItem(uid=str(uid), y=y, period=period, non_negative=True))
        if max_series and len(items) >= max_series:
            break
    return items


def load_m4(group: str = "Monthly", max_series: Optional[int] = None,
            min_len: int = 0) -> list[SeriesItem]:
    """M4 competition. Long series (esp. Monthly/Weekly/Daily) -> supports a wide
    history-length sweep. All non-negative."""
    from datasetsforecast.m4 import M4

    period = {"Monthly": 12, "Quarterly": 4, "Yearly": 1,
              "Weekly": 52, "Daily": 7, "Hourly": 24}[group]
    df, *_ = M4.load(directory=DATA_DIR, group=group)
    items = []
    for uid, g in df.groupby("unique_id", sort=False):
        y = g["y"].to_numpy(dtype=float)
        if len(y) >= min_len:
            items.append(SeriesItem(uid=str(uid), y=y, period=period, non_negative=True))
        if max_series and len(items) >= max_series:
            break
    return items


def load_m5_sample(n_series: int = 800, min_len: int = 200,
                   seed: int = 0) -> list[SeriesItem]:
    """M5 retail: intermittent, strictly non-negative integer sales counts.

    This is the regime where TSFMs genuinely violate non-negativity: series
    sit near zero with many gaps. We sample `n_series` bottom-level series.
    """
    from datasetsforecast.m5 import M5

    Y, *_ = M5.load(directory=DATA_DIR)
    uids = Y["unique_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    pick = rng.choice(uids, size=min(n_series, len(uids)), replace=False)
    sub = Y[Y["unique_id"].isin(pick)]
    items = []
    for uid, g in sub.groupby("unique_id", sort=False):
        y = g["y"].to_numpy(dtype=float)
        if len(y) >= min_len:
            items.append(SeriesItem(uid=str(uid), y=y, period=7, non_negative=True))
    return items


# --------------------------------------------------------------------------- #
# Hierarchical dataset (coherence constraint)
# --------------------------------------------------------------------------- #
def load_tourism_hierarchy() -> Hierarchy:
    """TourismSmall hierarchical dataset with its summing matrix S."""
    from datasetsforecast.hierarchical import HierarchicalData

    Y, S, _tags = _load_hier("TourismSmall")
    bottom_ids = list(S.columns)
    agg_ids = [i for i in S.index if i not in set(bottom_ids)]
    return Hierarchy(Y=Y, S=S, period=4, bottom_ids=bottom_ids, agg_ids=agg_ids)


def _load_hier(group: str):
    from datasetsforecast.hierarchical import HierarchicalData

    out = HierarchicalData.load(directory=DATA_DIR, group=group)
    # API returns (Y_df, S_df) or (Y_df, S_df, tags) across versions
    if len(out) == 3:
        Y, S, tags = out
    else:
        Y, S = out
        tags = None
    return Y, S, tags
