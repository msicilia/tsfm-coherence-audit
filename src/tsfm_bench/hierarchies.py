"""Single registry for every hierarchy the benchmark uses.

One place maps a hierarchy name to a loader returning
``(Y_long, S_df, tags, meta)``, the shape produced by
``datasetsforecast.hierarchical.HierarchicalData.load``. A study runs over
``ALL_HIERARCHIES`` uniformly and a new hierarchy is added in exactly one spot.

Dependence column = mean absolute off-diagonal correlation of bottom-series first
differences (the paper's cross-series dependence metric), recorded here from
scoping so the run order can be dependence-sorted without recomputing.
"""
from __future__ import annotations

from .paths import DATA


# Monash hierarchies shipped by datasetsforecast, in ascending scoped dependence.
_MONASH = ["Wiki2", "TourismLarge", "Labour", "TourismSmall", "Traffic"]

# name -> approximate bottom-series dependence (from scoping); for ordering/reporting only.
SCOPED_DEPENDENCE = {
    "Wiki2": 0.04, "TourismLarge": 0.11, "Labour": 0.19, "TourismSmall": 0.31,
    "M5": 0.34, "Traffic": 0.44, "Electricity": 0.63, "Solar": 0.70,
}

# Every hierarchy, dependence-ascending. The two energy hierarchies occupy the high
# end of the dependence axis; Electricity is also the venue for the global baselines.
ALL_HIERARCHIES = _MONASH + ["M5", "Electricity", "Solar"]


def load(name: str, data_dir: str | None = None):
    """Return ``(Y_long, S_df, tags, meta)`` for a hierarchy by name."""
    data_dir = str(DATA) if data_dir is None else data_dir
    if name == "M5":
        from .m5_hierarchy import build_m5_hierarchy
        return build_m5_hierarchy(data_dir)
    if name == "Solar":
        from .hierarchies_energy import build_solar_hierarchy
        return build_solar_hierarchy(data_dir)
    if name == "Electricity":
        from .hierarchies_energy import build_electricity_hierarchy
        return build_electricity_hierarchy(data_dir)

    from datasetsforecast.hierarchical import HierarchicalData, HierarchicalInfo
    Y, S, tags = HierarchicalData.load(data_dir, name)
    meta = HierarchicalInfo.get_group(name)
    return Y, S, tags, meta
