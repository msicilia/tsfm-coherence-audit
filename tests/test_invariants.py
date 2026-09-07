"""Invariants the experiment engines rely on.

Each test pins a property that is easy to break silently: rolling-origin window
caps, the summing-matrix identity, and forecast-wrapper output shapes.
"""
from __future__ import annotations

import numpy as np
import pytest

from tsfm_bench.temporal import DIVISORS_12, aggregate_from_end, build_temporal_S


# --------------------------------------------------------------------------- #
# Temporal hierarchy: end-alignment and the summing identity
# --------------------------------------------------------------------------- #
def test_aggregate_is_end_aligned():
    """Blocks must be anchored to the END of the series; the oldest remainder is
    dropped. If this drifts, train and forecast blocks are offset and every
    temporal-coherence number is meaningless."""
    y = np.arange(1, 14, dtype=float)          # 1..13, k=3 -> drop the leading 1
    assert aggregate_from_end(y, 3).tolist() == [9.0, 18.0, 27.0, 36.0]


def test_aggregate_exact_multiple_keeps_everything():
    y = np.arange(1, 13, dtype=float)          # 12 points, k=4 -> nothing dropped
    out = aggregate_from_end(y, 4)
    assert len(out) == 3 and out.sum() == y.sum()


@pytest.mark.parametrize("k", DIVISORS_12)
def test_S_matches_direct_aggregation(k):
    """S @ bottom must equal level-wise aggregation for every level."""
    S, labels = build_temporal_S(12)
    rng = np.random.default_rng(0)
    bottom = rng.random(12) * 10
    idx = [i for i, l in enumerate(labels) if l.startswith(f"k{k}_")]
    assert np.allclose((S @ bottom)[idx], aggregate_from_end(bottom, k))


def test_temporal_S_shape_and_levels():
    S, labels = build_temporal_S(12)
    assert S.shape == (28, 12)                 # 12 bottom + 16 aggregates
    assert sum(1 for l in labels if l.startswith("k1_")) == 12
    assert sum(1 for l in labels if l.startswith("k12_")) == 1
    # every row selects exactly k consecutive bottom slots
    for row, lab in zip(S, labels):
        assert row.sum() == int(lab.split("_")[0][1:])


# --------------------------------------------------------------------------- #
# Adaptive rolling-origin cap
# --------------------------------------------------------------------------- #
def _n_eff(requested: int, min_len: int, h: int) -> int:
    """Mirror of the cap in coherence.run_coherence / prob_coherence."""
    return int(max(1, min(requested, (min_len - 5) // h)))


def test_window_cap_short_series_does_not_empty_the_study():
    """Requesting 8 origins on TourismSmall (len 36, h 4) must not filter out
    every series and surfaced as an opaque KeyError: 'model'."""
    n = _n_eff(8, min_len=36, h=4)
    assert 1 <= n <= 8
    assert 36 > 4 * n + 4                       # every retained series supports n windows


@pytest.mark.parametrize("min_len,h,req,expected", [
    (36, 4, 8, 7),      # TourismSmall: capped below the request
    (1969, 28, 8, 8),   # M5: request satisfied
    (20, 14, 8, 1),     # pathological: never drops below one window
])
def test_window_cap_values(min_len, h, req, expected):
    assert _n_eff(req, min_len, h) == expected


# --------------------------------------------------------------------------- #
# Model wrapper output contract
# --------------------------------------------------------------------------- #
def test_quantile_shape_contract_rejects_transposed_output():
    """Chronos-2 returns (1, h, L) while others return (1, L, h); a silently
    transposed array corrupts every CRPS number downstream."""
    from tsfm_bench.models import _check_quantiles
    h, levels = 8, [0.1, 0.5, 0.9]
    good = np.sort(np.random.default_rng(0).random((2, h, len(levels))), axis=2)
    assert _check_quantiles(good, 2, h, levels).shape == (2, h, len(levels))
    with pytest.raises(ValueError):
        _check_quantiles(good.transpose(0, 2, 1), 2, h, levels)


def test_quantile_contract_repairs_crossing_quantiles():
    """Crossing is a real model artefact, not a bug in our code: repair by sorting
    and count it, rather than aborting a study."""
    from tsfm_bench.models import QUANTILE_CROSSINGS, _check_quantiles
    h, levels = 4, [0.1, 0.5, 0.9]
    bad = np.zeros((1, h, 3))
    bad[..., 0], bad[..., 1], bad[..., 2] = 3.0, 2.0, 1.0   # decreasing
    before = QUANTILE_CROSSINGS["repaired"]
    out = _check_quantiles(bad, 1, h, levels)
    assert np.all(np.diff(out, axis=2) >= 0)                # monotone after repair
    assert QUANTILE_CROSSINGS["repaired"] == before + 1     # and it was counted


def test_quantile_tolerance_is_relative_not_absolute():
    """Large-magnitude series (M5 sales in the thousands) must not trip the check on
    floating-point noise; an absolute epsilon did exactly that."""
    from tsfm_bench.models import _check_quantiles
    h, levels = 4, [0.1, 0.5, 0.9]
    big = np.zeros((1, h, 3))
    big[..., 0], big[..., 1], big[..., 2] = 1000.0, 1000.0 - 1e-9, 2000.0
    out = _check_quantiles(big, 1, h, levels)               # noise-level dip: tolerated
    assert out.shape == (1, h, 3)


# --------------------------------------------------------------------------- #
# Paths are absolute / cwd-independent
# --------------------------------------------------------------------------- #
def test_paths_are_absolute(tmp_path, monkeypatch):
    from tsfm_bench.paths import DATA, RESULTS, ROOT
    monkeypatch.chdir(tmp_path)                 # run from an unrelated directory
    assert ROOT.is_absolute() and DATA.is_absolute() and RESULTS.is_absolute()
    assert (ROOT / "pyproject.toml").exists()
