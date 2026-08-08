"""Regime clustering, checked on embeddings we built ourselves.

Offline, no keys, no downloads: the synthetic grids here have known regimes, so
the answer is known before the code runs. The tests that matter are the ones
guarding against a regime map that looks fine and means nothing, reshuffled ids,
a swallowed minority class, an uncalibrated threshold reading as safe.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfas.regime import (Regime, calibrate_thresholds, choose_k, cluster_regimes,
                         kmeans, valid_cells)


def _grid(h=60, w=60, d=16, seed=0):
    """Three regimes in known places: lagoon left, built middle, sand right."""
    rng = np.random.default_rng(seed)
    sig = {0: -3.0, 1: 0.0, 2: 3.0}
    grid = np.zeros((h, w, d), "float32")
    truth = np.zeros((h, w), "int32")
    for j, lo, hi in ((0, 0, w // 3), (1, w // 3, 2 * w // 3), (2, 2 * w // 3, w)):
        grid[:, lo:hi, :] = sig[j] + rng.normal(0, 0.3, (h, hi - lo, d))
        truth[:, lo:hi] = j
    return grid, truth


def test_it_finds_the_regimes_that_are_actually_there():
    grid, truth = _grid()
    labels, regimes = cluster_regimes(grid, k=3, seed=0)

    assert labels.shape == truth.shape
    assert len(regimes) == 3
    # Ids are arbitrary, so score by agreement: each true regime must land almost
    # entirely inside one cluster.
    for t in (0, 1, 2):
        found = labels[truth == t]
        dominant = np.bincount(found).max() / len(found)
        assert dominant > 0.98, f"true regime {t} scattered across clusters ({dominant:.2f})"


def test_the_same_seed_gives_the_same_map():
    """A regime map that reshuffles its ids cannot be calibrated.

    Yesterday's threshold for regime 3 would land on today's different ground,
    silently. Determinism is a correctness property here, not a convenience.
    """
    grid, _ = _grid()
    a, ra = cluster_regimes(grid, k=4, seed=7)
    b, rb = cluster_regimes(grid, k=4, seed=7)
    assert np.array_equal(a, b)
    assert [r.id for r in ra] == [r.id for r in rb]
    assert [r.cells for r in ra] == [r.cells for r in rb]


def test_a_small_regime_is_not_swallowed():
    """Makoko is small and does not behave like Ikoyi. It must survive clustering.

    k-means chases mass, and the lagoon is the largest uniform thing in a Lagos
    tile. Random seeding drops centres into it and lets a small distinct regime
    get absorbed; k-means++ seeding by distance is what prevents that. If this
    regresses, the ground we care most about is the ground that disappears.
    """
    rng = np.random.default_rng(1)
    grid = rng.normal(0, 0.3, (60, 60, 16)).astype("float32")   # one big bland mass
    grid[:5, :5, :] += 8.0                                       # a small, very distinct patch
    labels, regimes = cluster_regimes(grid, k=3, seed=0)

    patch = labels[:5, :5]
    assert len(np.unique(patch)) == 1, "the patch should be one regime"
    pid = int(patch[0, 0])
    elsewhere = labels.copy()
    elsewhere[:5, :5] = -1
    assert (elsewhere == pid).sum() == 0, "and that regime should be only the patch"


def test_area_and_fraction_are_real():
    grid, _ = _grid()
    labels, regimes = cluster_regimes(grid, k=3, seed=0, cell_m=10.0)

    # Cells are exact and must tile the labelled ground with nothing double-counted.
    assert sum(r.cells for r in regimes) == int((labels >= 0).sum())
    # Fractions are rounded to 4dp for reading, so they sum to 1 only within that
    # rounding, at most 5e-5 of drift per regime.
    assert abs(sum(r.fraction for r in regimes) - 1.0) < 5e-5 * len(regimes)
    for r in regimes:
        assert r.cells == int((labels == r.id).sum())
        assert abs(r.area_km2 - r.cells * 0.0001) < 1e-9    # 10 m cells


def test_centroids_come_back_in_tessera_units():
    """Standardising is an internal step; a centroid must stay comparable to a raw
    embedding or it cannot be matched against one later."""
    grid, _ = _grid()
    _, regimes = cluster_regimes(grid, k=3, seed=0)
    means = sorted(float(r.centroid.mean()) for r in regimes)
    # The synthetic regimes sit at -3, 0, +3 in the raw space.
    assert means[0] < -2 and abs(means[1]) < 1 and means[2] > 2, means


def test_cells_without_an_embedding_are_marked_not_guessed():
    grid, _ = _grid()
    grid[0, :, :] = np.nan
    labels, regimes = cluster_regimes(grid, k=3, seed=0)

    assert (labels[0, :] == -1).all(), "missing data must read as -1, not a regime"
    assert (labels[1:, :] >= 0).all()
    assert sum(r.cells for r in regimes) == int((labels >= 0).sum())


def test_tesseras_nodata_fill_is_water_and_is_not_a_regime():
    """TESSERA returns all-zero over water, not NaN. isfinite() passes on it.

    This is not hypothetical: on a real coastal Lagos tile the fill is
    570,024 of 1,224,342 cells, 46.6%, the Atlantic and the lagoon. Treated as
    data it eats a whole cluster, drags the standardisation toward zero for every
    real cell, halves every regime's reported share, and flattens the inertia
    curve choose_k() draws. It fails silently and looks like a plausible regime.
    """
    grid, _ = _grid()
    grid[:10, :, :] = 0.0                     # the fill, exactly as TESSERA emits it
    labels, regimes = cluster_regimes(grid, k=3, seed=0)

    assert np.isfinite(grid[:10]).all(), "the fill is finite, that is the trap"
    assert (labels[:10, :] == -1).all(), "the fill must be nodata, not a regime"
    assert (labels[10:, :] >= 0).all(), "and real ground must survive it"
    assert sum(r.cells for r in regimes) == int((labels >= 0).sum())
    # Shares are of observed ground, not of the tile, or every regime reads half
    # its true size.
    assert abs(sum(r.fraction for r in regimes) - 1.0) < 5e-5 * len(regimes)


def test_valid_cells_keeps_a_genuine_zero_reading():
    """Only an all-zero *vector* is fill. A single zero component is a real value.

    Over-eager masking would silently drop real ground whose embedding happens to
    cross zero in one dimension out of 128.
    """
    x = np.zeros((3, 8), "float32")
    x[0] = 0.0                    # fill
    x[1, 3] = 0.5                 # one component set: a real observation
    x[2] = np.nan                 # non-finite
    keep = valid_cells(x)
    assert list(keep) == [False, True, False]


def test_a_tile_that_is_all_water_refuses_rather_than_inventing_regimes():
    grid = np.zeros((20, 20, 8), "float32")
    try:
        cluster_regimes(grid, k=3)
    except ValueError as e:
        assert "nodata" in str(e).lower() or "observed" in str(e).lower()
    else:
        raise AssertionError("an all-nodata tile should refuse, not cluster the fill")


def test_empty_clusters_are_reseeded_not_dropped():
    """k regimes in, k regimes out, or the caller's k means nothing."""
    x = np.repeat(np.arange(4, dtype="float32")[:, None] * 10, 8, axis=1)
    x = np.repeat(x, 25, axis=0)          # 4 tight blobs, 100 points
    labels, centres = kmeans(x, 4, seed=0)
    assert len(centres) == 4
    assert len(np.unique(labels)) == 4


def test_kmeans_refuses_more_clusters_than_points():
    try:
        kmeans(np.zeros((3, 8), "float32"), 5)
    except ValueError as e:
        assert "at least" in str(e)
    else:
        raise AssertionError("k > n should refuse rather than invent clusters")


def test_more_regimes_fit_tighter():
    """Inertia must fall with k, or choose_k() is telling you nothing."""
    grid, _ = _grid()
    inertia = choose_k(grid, ks=(2, 3, 4, 6), seed=0, fit_sample=3000)
    vals = [inertia[k] for k in (2, 3, 4, 6)]
    assert vals == sorted(vals, reverse=True), f"inertia should fall with k: {inertia}"
    # The synthetic grid has 3 real regimes, so the fall should be steep to 3 and
    # shallow after, that is the elbow choose_k() exists to show.
    assert (vals[0] - vals[1]) > (vals[2] - vals[3])


def test_a_regime_learns_its_threshold_from_call_ins():
    grid, truth = _grid()
    labels, regimes = cluster_regimes(grid, k=3, seed=0)
    left = int(labels[30, 5])            # a cell in the leftmost true regime

    reports = [(30, 5, 55.0, True), (31, 6, 40.0, True), (32, 7, 48.0, True),
               (30, 8, 12.0, False)]     # a dry-day report must not set a threshold
    out = calibrate_thresholds(regimes, labels, reports)

    got = {r.id: r.threshold_mm for r in out}
    assert got[left] == 40.0, "the threshold is the lowest rain that actually flooded"
    assert all(r.calibrated for r in out if r.id == left)


def test_an_uncalibrated_regime_is_unknown_not_safe():
    """None must never read as 'this ground does not flood'.

    A regime with no confirmed call-ins knows nothing about itself. Anything that
    treats that silence as an all-clear inverts the system's whole purpose.
    """
    grid, _ = _grid()
    labels, regimes = cluster_regimes(grid, k=3, seed=0)
    out = calibrate_thresholds(regimes, labels, [])

    assert all(r.threshold_mm is None for r in out)
    assert all(not r.calibrated for r in out)


def test_a_regime_needs_enough_reports_before_it_claims_a_threshold():
    grid, _ = _grid()
    labels, regimes = cluster_regimes(grid, k=3, seed=0)
    rid = int(labels[30, 5])

    thin = calibrate_thresholds(regimes, labels, [(30, 5, 50.0, True)], min_reports=3)
    assert {r.id: r.threshold_mm for r in thin}[rid] is None, "one call-in is not calibration"

    enough = calibrate_thresholds(
        regimes, labels,
        [(30, 5, 50.0, True), (31, 5, 52.0, True), (32, 5, 47.0, True)], min_reports=3)
    assert {r.id: r.threshold_mm for r in enough}[rid] == 47.0


def test_reports_off_the_grid_are_ignored_not_crashed():
    grid, _ = _grid()
    labels, regimes = cluster_regimes(grid, k=3, seed=0)
    out = calibrate_thresholds(regimes, labels,
                               [(999, 999, 60.0, True), (-1, 0, 60.0, True)])
    assert all(r.threshold_mm is None for r in out)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
