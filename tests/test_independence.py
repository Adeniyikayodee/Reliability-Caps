"""One person cannot manufacture a warning. That is the guarantee, and it is arithmetic.

Reach and vulnerability are the same property here: a report bands up to 36 km², so
whatever makes the system valuable also makes it worth attacking. The defence is not a
heuristic about suspicious behaviour, it is the fact that log-likelihood ratios only
add for independent observations, and a second call from the same phone is not one.

The single most important assertion in this repository is at the bottom: no reporter,
at any band, with any number of calls, in any regime, can cross a threshold alone.
"""
import pytest

from cfas import waggle as w

AREAS = {0: 22.83, 1: 12.93, 2: 23.28, 3: 14.88, 4: 36.17, 5: 8.44}
SMALLEST_A = min(w.thresholds(a)[0] for a in AREAS.values())

FAR = dict(lat=6.4550, lon=3.3509)          # Boundary
NEAR_FAR = dict(lat=6.4553, lon=3.3512)     # ~45 m away
ELSEWHERE = dict(lat=6.4215, lon=3.3005)    # ~6 km away


def R(band, age=0.0, who=None, **pos):
    return w.Report(band=band, age_min=age, reporter=who, **pos)


# ---------------------------------------------------------------- same reporter
def test_a_repeat_call_adds_no_evidence():
    """A bee that dances twenty times is still one bee at the cavity."""
    one = w.accumulate(0, AREAS[0], [R("HIGH", who="aisha", **FAR)])
    ten = w.accumulate(0, AREAS[0], [R("HIGH", age=i, who="aisha", **FAR)
                                     for i in range(10)])
    assert ten.llr == pytest.approx(one.llr, abs=1e-9)
    assert ten.n_eff == pytest.approx(1.0)


def test_the_strongest_call_from_a_reporter_is_the_one_kept():
    """Observations escalate and never de-escalate, including within one person.

    Somebody who reports chest-deep water and then calls back calmer has not undone
    what they saw, and a later mild report must not quietly retract an earlier alarm.
    """
    e = w.accumulate(0, AREAS[0], [R("HIGH", age=2, who="aisha", **FAR),
                                   R("LOW", age=0, who="aisha", **FAR)])
    assert e.llr == pytest.approx(w.decay(w.llr("HIGH"), 2), abs=1e-9)
    assert e.llr > 0


@pytest.mark.parametrize("band", ["HIGH", "MEDIUM", "LOW"])
@pytest.mark.parametrize("n", [1, 2, 5, 20, 100])
@pytest.mark.parametrize("area", list(AREAS.values()))
def test_one_reporter_can_never_cross_any_threshold(band, n, area):
    """The guarantee, swept over every band, every volume, every regime.

    If this ever fails, one phone can warn a settlement, and every other safeguard in
    the system is downstream of a hole.
    """
    obs = [R(band, age=i * 0.5, who="attacker", **FAR) for i in range(n)]
    e = w.accumulate(0, area, obs)
    a, _ = w.thresholds(area)
    assert e.llr < a, f"{n}x {band} from one reporter crossed a {area} km² regime"


def test_anonymous_reports_are_not_merged():
    """Absent identity we assume independence, which is the honest default.

    Collapsing every unidentified call into one would silently discard real evidence
    from a settlement where nobody is registered. Position still discounts them, so the
    permissiveness is bounded.
    """
    e = w.accumulate(0, AREAS[0], [R("HIGH", **ELSEWHERE), R("HIGH", **FAR)])
    assert e.n_eff == pytest.approx(2.0)


# ---------------------------------------------------------------- proximity
def test_two_people_at_the_same_corner_are_not_two_observations():
    far = w.accumulate(0, AREAS[0], [R("HIGH", who="a", **FAR),
                                     R("HIGH", who="b", **ELSEWHERE)])
    near = w.accumulate(0, AREAS[0], [R("HIGH", who="a", **FAR),
                                      R("HIGH", who="b", **NEAR_FAR)])
    assert near.llr < far.llr
    # n_eff is reported rounded, so compare at the precision it actually carries
    assert near.n_eff == pytest.approx(1 + w._marginal(2, w.RHO_NEAR), abs=1e-3)


def test_marginal_value_falls_away_within_a_cluster():
    """1.00, 0.33, 0.17, 0.10: the fifth voice on one corner is worth almost nothing."""
    ms = [w._marginal(k, w.RHO_NEAR) for k in range(1, 6)]
    assert ms == sorted(ms, reverse=True)
    assert ms[0] == pytest.approx(1.0)
    assert ms[1] == pytest.approx(1 / 3, abs=1e-6)


def test_the_same_reporter_is_the_limiting_case_of_proximity():
    """rho = 1 is what "the same person" means, and it falls out of the same formula."""
    assert w._marginal(2, 1.0) == 0.0
    assert w.RHO_SAME_REPORTER == 1.0


def test_spread_out_reporters_are_worth_more_than_clustered_ones():
    """The reason the recruiter should ask someone far from whoever already called."""
    clustered = w.accumulate(0, AREAS[0], [R("HIGH", who=f"p{i}", lat=6.4550 + i * 1e-4,
                                             lon=3.3509) for i in range(4)])
    spread = w.accumulate(0, AREAS[0], [R("HIGH", who=f"p{i}", lat=6.44 + i * 0.01,
                                          lon=3.33) for i in range(4)])
    assert spread.llr > clustered.llr
    assert spread.n_eff > clustered.n_eff


# ---------------------------------------------------------------- the floor
def test_two_separated_colluders_can_cross_and_that_is_stated():
    """The system's real adversarial floor, asserted rather than hidden.

    Two people far enough apart do reach a threshold in the smaller regimes. That is
    the honest limit: independent liars beat every human warning system too, and
    raising the floor further would cost lead time on real floods. The skeptic and the
    operator's dismissal exist for exactly this case.
    """
    e = w.accumulate(1, AREAS[1], [R("HIGH", who="a", **FAR),
                                   R("HIGH", who="b", **ELSEWHERE)])
    assert e.llr >= w.thresholds(AREAS[1])[0]


def test_one_report_is_below_every_threshold_in_the_system():
    assert w.llr("HIGH") < SMALLEST_A
