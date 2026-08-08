"""The join: does one voice note move the map, and only where it should.

Offline, no keys, no models. The tests guard the two failures that would matter
most on air, a call-in that changes nothing, and a call-in that changes
everything.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfas.assimilate import (Observation, Posterior, assimilate, prior_band,
                             reach_of, summarise)
from cfas.regime import Regime, cluster_regimes


def _map():
    """Three regimes in known columns, from cfas.regime's own clustering."""
    rng = np.random.default_rng(0)
    grid = np.zeros((30, 30, 8), "float32")
    for j, lo, hi in ((0, 0, 10), (1, 10, 20), (2, 20, 30)):
        grid[:, lo:hi, :] = (j - 1) * 4.0 + rng.normal(0, 0.2, (30, hi - lo, 8))
    labels, regimes = cluster_regimes(grid, k=3, seed=0)
    return labels, regimes


def _with_thresholds(regimes, mm=50.0):
    return [Regime(id=r.id, cells=r.cells, area_km2=r.area_km2, fraction=r.fraction,
                   centroid=r.centroid, threshold_mm=mm) for r in regimes]


def test_one_call_in_moves_a_whole_regime():
    """The claim the design rests on: a report generalises across ground that
    behaves alike. One caller speaks for every cell in their regime."""
    labels, regimes = _map()
    regimes = _with_thresholds(regimes)
    rid = int(labels[15, 5])

    quiet = assimilate(regimes, labels, [], rain_mm=0.0)
    assert quiet[rid].band == "LOW"

    loud = assimilate(regimes, labels,
                      [Observation("e reach my chest, e dey rush", 15, 5)],
                      rain_mm=0.0)
    assert loud[rid].band == "HIGH"
    assert loud[rid].moved and loud[rid].direct and loud[rid].support == 1
    assert reach_of(loud, rid) > 0, "one call-in must cover real ground"
    assert "raised by 1 call-in" in loud[rid].note


def test_it_moves_only_that_regime():
    """A report in Ikoyi must not raise Makoko. Regimes are the unit of reach."""
    labels, regimes = _map()
    regimes = _with_thresholds(regimes)
    rid = int(labels[15, 5])

    post = assimilate(regimes, labels,
                      [Observation("e reach my chest, e dey rush", 15, 5)],
                      rain_mm=0.0)
    for other in (r.id for r in regimes if r.id != rid):
        assert post[other].band == "LOW", "an unrelated regime must not move"
        assert not post[other].direct and post[other].support == 0


def test_an_observation_overrides_a_calm_prior():
    """Physics said dry. A person is standing in it. The person wins."""
    labels, regimes = _map()
    regimes = _with_thresholds(regimes, mm=200.0)   # a very high threshold: prior LOW
    rid = int(labels[15, 25])

    post = assimilate(regimes, labels,
                      [Observation("water reach my waist, e dey carry rubbish", 15, 25)],
                      rain_mm=0.0)
    assert post[rid].prior_band == "LOW"
    assert post[rid].band == "HIGH"
    assert post[rid].moved


def test_observations_escalate_and_never_de_escalate():
    """A shallow report on one street is not evidence the next street is dry."""
    labels, regimes = _map()
    regimes = _with_thresholds(regimes, mm=10.0)
    rid = int(labels[15, 5])

    post = assimilate(regimes, labels, [], rain_mm=50.0)
    assert post[rid].band == "HIGH", "heavy rain over a low threshold"

    calmed = assimilate(regimes, labels,
                        [Observation("water just reach my ankle, e no dey move", 15, 5)],
                        rain_mm=50.0)
    assert calmed[rid].band == "HIGH", "a mild report must not pull a HIGH band down"


def test_the_worst_report_in_a_regime_sets_the_band():
    labels, regimes = _map()
    regimes = _with_thresholds(regimes)
    rid = int(labels[15, 5])

    post = assimilate(regimes, labels, [
        Observation("water reach my ankle, e no dey move", 15, 5),
        Observation("e reach my chest, e dey rush", 16, 6),
        Observation("water reach my knee", 17, 7),
    ], rain_mm=0.0)
    assert post[rid].band == "HIGH"
    assert post[rid].support == 3, "every measurable call-in counts as support"


def test_an_uncalibrated_regime_is_unknown_not_safe():
    """No learned threshold means the satellite knows nothing. Say so."""
    labels, regimes = _map()      # thresholds all None
    post = assimilate(regimes, labels, [], rain_mm=80.0)
    assert all(p.band == "UNKNOWN" for p in post.values())
    assert all("not safe" in p.note for p in post.values())


def test_a_call_in_gives_an_uncalibrated_regime_its_first_band():
    """The city can speak before the satellite has learned anything at all.

    This is the cold-start path: on day one no regime has a threshold, and the
    only thing that can band the map is somebody calling in.
    """
    labels, regimes = _map()
    rid = int(labels[15, 5])
    post = assimilate(regimes, labels,
                      [Observation("e reach my knee, e dey rush", 15, 5)], rain_mm=0.0)
    assert post[rid].prior_band == "UNKNOWN"
    assert post[rid].band == "MEDIUM", "the caller banded ground the satellite could not"
    assert post[rid].moved


def test_an_unmeasurable_call_contributes_nothing_rather_than_zero():
    """We heard them; we could not measure them. That is not an all-clear."""
    labels, regimes = _map()
    regimes = _with_thresholds(regimes)
    rid = int(labels[15, 5])
    post = assimilate(regimes, labels,
                      [Observation("water plenty for road o, e bad", 15, 5)], rain_mm=0.0)
    assert post[rid].support == 0, "an unmeasurable report is not support"
    assert post[rid].band == "LOW", "and it must not move the band either way"


def test_a_caller_standing_in_the_lagoon_is_dropped():
    labels, regimes = _map()
    regimes = _with_thresholds(regimes)
    labels[0, 0] = -1                     # nodata: water
    post = assimilate(regimes, labels,
                      [Observation("e reach my chest, e dey rush", 0, 0)], rain_mm=0.0)
    assert all(p.support == 0 for p in post.values())


def test_reports_off_the_grid_are_ignored_not_crashed():
    labels, regimes = _map()
    regimes = _with_thresholds(regimes)
    post = assimilate(regimes, labels, [
        Observation("e reach my chest, e dey rush", 999, 999),
        Observation("e reach my chest, e dey rush", -5, 2),
    ], rain_mm=0.0)
    assert all(p.support == 0 for p in post.values())


def test_the_prior_ladder_climbs_with_rain():
    r = Regime(id=0, cells=1, area_km2=1.0, fraction=1.0, centroid=None,
               threshold_mm=50.0)
    assert prior_band(r, 0.0) == "LOW"
    assert prior_band(r, 29.0) == "LOW"
    assert prior_band(r, 30.0) == "MEDIUM"    # 0.6 * 50
    assert prior_band(r, 49.9) == "MEDIUM"
    assert prior_band(r, 50.0) == "HIGH"
    assert prior_band(r, 120.0) == "HIGH"


def test_summarise_puts_the_worst_first():
    labels, regimes = _map()
    regimes = _with_thresholds(regimes)
    post = assimilate(regimes, labels,
                      [Observation("e reach my chest, e dey rush", 15, 5)], rain_mm=0.0)
    rows = summarise(post)
    assert rows[0][1] == "HIGH", "the band that matters must lead"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
