"""The evidence arithmetic, pinned.

These tests are the contract for the one part of the system that decides anything.
If they pass, the same evidence produces the same verdict on every run, and the
numbers in the design document are the numbers the code actually computes.
"""
import math

import pytest

from cfas import waggle as w

AREAS = {0: 22.83, 1: 12.93, 2: 23.28, 3: 14.88, 4: 36.17}


def test_ceiling_matches_the_measured_skill():
    """kappa is log(TPR/FPR) and nothing else."""
    assert w.ceiling() == pytest.approx(math.log(0.82 / 0.48), abs=1e-9)
    assert w.ceiling() == pytest.approx(0.5355, abs=1e-3)


def test_ceiling_is_zero_at_chance():
    """A regime map no better than a coin carries no information about its regime.

    This is the same point at which Youden's J is zero, so this and
    generalise._informedness agree on what "no skill" means.
    """
    assert w.ceiling(tpr=0.5, fpr=0.5) == pytest.approx(0.0, abs=1e-12)


def test_band_weights_match_the_design_table():
    assert w.llr("HIGH") == pytest.approx(0.505, abs=1e-3)
    assert w.llr("MEDIUM") == pytest.approx(0.339, abs=1e-3)
    assert w.llr("LOW") == pytest.approx(-0.338, abs=1e-3)


def test_a_low_report_is_evidence_against():
    """Cross-inhibition is not a bolted-on mechanism, it is what the ratio does.

    A report more probable under "not flooding" than under "flooding" carries negative
    weight, which is the honeybee stop signal arriving as arithmetic.
    """
    assert w.llr("LOW") < 0
    assert w.llr("HIGH") > 0
    assert w.llr("MEDIUM") > 0


def test_unknown_is_censored_to_zero():
    """An unmeasurable report must never become evidence of safety.

    The arithmetic would make UNKNOWN strongly negative. We refuse that number: it is
    dominated by nuisance parameters nobody has measured, and treating it as evidence
    of a dry street is exactly the failure cfas/hazard.py already guards against.
    """
    assert w.llr("UNKNOWN") == 0.0


@pytest.mark.parametrize("band", ["HIGH", "MEDIUM"])
@pytest.mark.parametrize("s,f", [(0.6, 0.2), (0.9, 0.05), (0.99, 0.001),
                                 (0.999, 1e-6), (1 - 1e-9, 1e-12)])
def test_no_report_ever_beats_the_ceiling(band, s, f, monkeypatch):
    """The ceiling theorem, property-tested against ever more reliable reporters.

    However good the human instrument gets, the evidence it can carry about a 23 km²
    regime saturates at log(TPR/FPR). This is the result that says the binding
    constraint is the regime map rather than the people, and it is worth a test
    because it is the argument the whole roadmap rests on.
    """
    monkeypatch.setitem(w.P_BAND_GIVEN_FLOODED, band, s)
    monkeypatch.setitem(w.P_BAND_GIVEN_DRY, band, f)
    assert w.llr(band) <= w.ceiling() + 1e-12


def test_llr_approaches_the_ceiling_for_a_perfect_reporter(monkeypatch):
    monkeypatch.setitem(w.P_BAND_GIVEN_FLOODED, "HIGH", 1 - 1e-9)
    monkeypatch.setitem(w.P_BAND_GIVEN_DRY, "HIGH", 1e-9)
    assert w.llr("HIGH") == pytest.approx(w.ceiling(), abs=1e-4)


def test_alpha_scales_so_expected_cost_is_constant():
    """The regime with the most ground to lose clears the highest bar."""
    assert w.alpha_for(20.0) == pytest.approx(0.25)
    assert w.alpha_for(40.0) < w.alpha_for(10.0)
    assert w.alpha_for(36.17) * 36.17 == pytest.approx(w.alpha_for(22.83) * 22.83)


def test_thresholds_match_the_design_table():
    expected_a = {0: 1.50, 1: 0.93, 2: 1.52, 3: 1.07, 4: 1.96}
    for rid, area in AREAS.items():
        a, b = w.thresholds(area)
        assert a == pytest.approx(expected_a[rid], abs=0.01), f"regime {rid}"
        assert b < 0, "the stand-down boundary is below zero by construction"


def test_no_single_report_crosses_any_threshold():
    """The rule the console used to state in a tooltip, now enforced by arithmetic.

    One voice can move the map, and it can never on its own wake the city.
    """
    strongest = w.llr("HIGH")
    for rid, area in AREAS.items():
        a, _ = w.thresholds(area)
        assert strongest < a, f"one report crossed regime {rid}"


@pytest.mark.parametrize("area", [0.01, 0.5, 1, 2, 5, 8.44, 10, 12.93, 22.83,
                                  36.17, 60, 500, 10_000])
def test_no_single_report_crosses_at_any_area(area):
    """The invariant holds for any regime, not only the five we happened to draw.

    Ajegunle's smallest regime is 8.44 km², and under the constant-cost rule alone its
    alpha would be permissive enough to put the threshold under the weight of one HIGH
    report. Testing only the regimes on today's tile would have missed that, so this
    sweeps the whole range instead.
    """
    a, _ = w.thresholds(area)
    assert w.llr("HIGH") < a, f"one report crossed a {area} km² regime"


def test_alpha_cap_is_derived_from_the_current_calibration(monkeypatch):
    """Recalibration must not silently invalidate the single-report invariant."""
    monkeypatch.setitem(w.P_BAND_GIVEN_FLOODED, "HIGH", 0.95)
    monkeypatch.setitem(w.P_BAND_GIVEN_DRY, "HIGH", 0.001)
    assert w.llr("HIGH") < w.thresholds(0.5)[0]


def test_evidence_state_transitions():
    area = AREAS[1]
    a, b = w.thresholds(area)

    fresh = w.accumulate(1, area, [("HIGH", 0.0), ("HIGH", 1.0)])
    assert fresh.llr >= a and fresh.state == "ESCALATE"

    contradicted = w.accumulate(1, area, [("LOW", 0.0)] * 12)
    assert contradicted.llr <= b and contradicted.state == "STAND_DOWN"

    quiet = w.accumulate(1, area, [("HIGH", 0.0)])
    assert quiet.state == "ACCUMULATING"
    assert quiet.gap > 0


def test_gap_is_what_the_agent_is_buying():
    e = w.accumulate(0, AREAS[0], [("HIGH", 0.0)])
    assert e.gap == pytest.approx(e.threshold_a - e.llr, abs=1e-9)
    crossed = w.accumulate(0, AREAS[0], [("HIGH", 0.0)] * 6)
    assert crossed.gap == 0.0
