"""The leak, and the arrival rate below which nothing can be decided.

Wald assumes the hypothesis holds still. A flood does not, which is the whole reason
this system exists, so evidence decays and the decay has a consequence that is easy
to miss and expensive to miss: below a critical arrival rate the accumulator has a
steady state under the threshold, and no amount of patience gets there.

These tests exist so that result cannot quietly regress into a system that looks
calm while being incapable.
"""
import math

import pytest

from cfas import waggle as w

AREAS = {0: 22.83, 1: 12.93, 2: 23.28, 3: 14.88, 4: 36.17}


def test_decay_is_exponential_with_the_hydrological_constant():
    assert w.decay(1.0, 0.0) == pytest.approx(1.0)
    assert w.decay(1.0, w.TAU_MIN) == pytest.approx(math.exp(-1), abs=1e-9)
    assert w.decay(1.0, 2 * w.TAU_MIN) == pytest.approx(math.exp(-2), abs=1e-9)


def test_decay_refuses_evidence_from_the_future():
    with pytest.raises(ValueError):
        w.decay(1.0, -5.0)


def test_older_evidence_is_worth_less():
    assert w.decay(1.0, 10) > w.decay(1.0, 20) > w.decay(1.0, 40)


def test_lambda_star_matches_the_design_table():
    """Reports per minute each regime needs before it can ever warn."""
    expected_minutes_between = {0: 12.3, 1: 23.5, 2: 12.1, 3: 19.1, 4: 8.9}
    for rid, area in AREAS.items():
        got = 1.0 / w.lambda_star(area)
        assert got == pytest.approx(expected_minutes_between[rid], abs=0.1), f"regime {rid}"


def test_the_largest_regime_needs_the_fastest_reporting():
    """36 km² has the most to lose, so it clears the highest bar and fails first."""
    rates = {rid: w.lambda_star(a) for rid, a in AREAS.items()}
    assert max(rates, key=rates.get) == 4
    assert min(rates, key=rates.get) == 1


def _simulate(area_km2, minutes_between, hours=24, tau=None):
    """Feed HIGH reports at a fixed rate and return the highest LLR ever reached.

    Evidence is recomputed from scratch at each arrival, exactly as the server does
    from the ledger, so this exercises the real accumulation path.
    """
    tau = tau or w.TAU_MIN
    horizon = hours * 60
    arrivals = [i * minutes_between for i in range(int(horizon / minutes_between) + 1)]
    peak = 0.0
    for k, now in enumerate(arrivals):
        obs = [("HIGH", now - t) for t in arrivals[:k + 1]]
        peak = max(peak, w.accumulate(0, area_km2, obs, tau_min=tau).llr)
    return peak


def test_below_lambda_star_the_threshold_is_never_reached():
    """The central result: a phase transition, not a slowdown.

    Twenty-four hours of steady reporting just under the critical rate never crosses.
    This is the state in which a quiet screen means the system has already failed.
    """
    for rid, area in AREAS.items():
        critical = 1.0 / w.lambda_star(area)
        a, _ = w.thresholds(area)
        peak = _simulate(area, minutes_between=critical * 1.25)
        assert peak < a, f"regime {rid} crossed below its critical rate"


def test_above_lambda_star_the_threshold_is_reached():
    for rid, area in AREAS.items():
        critical = 1.0 / w.lambda_star(area)
        a, _ = w.thresholds(area)
        peak = _simulate(area, minutes_between=critical * 0.5)
        assert peak >= a, f"regime {rid} never crossed above its critical rate"


def test_unreachable_is_reported_rather_than_swallowed():
    """A regime that cannot warn has to say so. Silence and calm look identical."""
    area = AREAS[4]
    slow = 1.0 / w.lambda_star(area) * 2.0     # half the rate it needs
    obs = [("HIGH", i * slow) for i in range(4)]
    e = w.accumulate(4, area, obs)
    assert e.state == "UNREACHABLE"
    assert e.lambda_obs < e.lambda_star


def test_a_single_report_is_not_declared_unreachable():
    """One report says nothing about a rate, so it must not condemn the regime."""
    e = w.accumulate(4, AREAS[4], [("HIGH", 0.0)])
    assert e.state == "ACCUMULATING"


def test_arrival_rate_is_what_moves_a_regime_out_of_unreachable():
    """The agentic layer's entire specification, as a test.

    Same regime, same reports, same evidence per report. The only thing that changes
    is how fast they arrive, and that alone decides whether the city is ever warned.
    """
    area = AREAS[4]
    critical = 1.0 / w.lambda_star(area)
    a, _ = w.thresholds(area)
    assert _simulate(area, critical * 1.5) < a
    assert _simulate(area, critical * 0.5) >= a


def test_a_longer_memory_lowers_the_bar():
    """tau is hydrology, and if floods lasted longer the system would need less."""
    area = AREAS[0]
    assert w.lambda_star(area, tau_min=60.0) < w.lambda_star(area, tau_min=30.0)
