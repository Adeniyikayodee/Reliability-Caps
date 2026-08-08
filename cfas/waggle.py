"""When is the evidence enough to wake somebody up?

`cfas/assimilate.py` answers what the map says. This module answers the harder
question underneath it: whether what the map says is worth acting on yet. That is a
threshold decision under asymmetric cost, and it is the one part of the system that
must never depend on a model, because the same evidence has to produce the same
verdict on every run and a test has to be able to pin it down.

The shape is taken from honeybee nest-site selection, where a swarm accumulates
evidence for a candidate site and commits when a quorum is reached rather than when
the scouts agree. Marshall et al. (2009) showed that the bees' recruitment dynamics
converge on the drift-diffusion model, which is a continuous special case of Wald's
sequential probability ratio test: the test that, among all tests, minimises decision
time for a given error rate. So the swarm supplies the architecture and Wald supplies
the units, and the units are nats of log-likelihood ratio.

Three things here are load-bearing and none of them are obvious.

**A report is about a cell; the decision is about a regime.** Composing those two
stages of uncertainty puts a hard ceiling on what any single report can be worth:

    kappa = log(TPR / FPR)

where TPR and FPR come straight out of `cfas/generalise.leave_one_out`. No report,
however honest and however well measured, carries more than kappa nats about its
regime. That makes the validation code the calibration instrument for the decision,
and it means the binding constraint on this system is the regime map rather than the
people reporting into it.

**Evidence has to leak.** Wald assumes the hypothesis holds still. Flood state is
exactly what does not: it is why the system exists. An observation from forty minutes
ago is evidence about a world that may have drained, so it decays with a hydrological
time constant, and the leak is arithmetic rather than a judgement call. A model asked
"is this report still relevant?" answers differently on identical evidence.

**The leak implies a rate below which nothing can ever be decided.** With arrival
rate lambda the accumulator settles at Lambda*lambda*tau, so if that sits under the
threshold, no amount of waiting reaches it:

    lambda_star = A / (Lambda * tau)

That is a phase transition rather than a slowdown, and it is invisible to any design
that leaves the decay out. It is also the entire specification of the agentic layer,
whose one job is to push lambda above lambda_star.

Everything in here is a pure function of the ledger. No network, no clock of its own,
no model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------- calibration
# Regime-to-cell skill: the exchange rate between a report and evidence about its
# regime. These are fallbacks, deliberately conservative, used only when no
# calibration file has been loaded. `calibrate.py` in the deployed system measures them from the real
# tile with `generalise.leave_one_out` and `load_calibration()` installs the result,
# so the numbers a deployment runs on are measured rather than typed here.
TPR = 0.82
FPR = 0.48
CALIBRATION: dict | None = None


def load_calibration(path) -> dict:
    """Install measured TPR/FPR from a calibration file. Returns what was loaded.

    Module-level state is usually a smell, but the alternative is threading a
    calibration object through every call site of a pure arithmetic module, and the
    value genuinely is a property of the deployment rather than of a request. It is
    set once at startup and read everywhere; `calibration_note()` keeps it honest by
    making the provenance visible to anyone who asks.
    """
    global TPR, FPR, CALIBRATION
    import json
    import pathlib
    cal = json.loads(pathlib.Path(path).read_text())
    if not (0 < cal["fpr"] < 1 and 0 < cal["tpr"] <= 1):
        raise ValueError(f"calibration out of range: {cal['tpr']}, {cal['fpr']}")
    if cal["tpr"] <= cal["fpr"]:
        raise ValueError("calibration has no skill: TPR must exceed FPR, or a report "
                         "carries no evidence about its regime and every threshold "
                         "below is meaningless")
    TPR, FPR, CALIBRATION = cal["tpr"], cal["fpr"], cal
    return cal


def calibration_note() -> str:
    """One line on where the current numbers came from. For the operator and the log."""
    if CALIBRATION is None:
        return "uncalibrated: using fallback constants, not measured on this tile"
    c = CALIBRATION
    return (f"kappa {c['kappa_nats']} nats (J {c['informedness_j']}, "
            f"{c['gap_over_shuffle']} over shuffle, {c['ground_truth']} ground truth)")

# Cell-to-report skill: what a resident standing in a flooded (or dry) cell says.
# Provisional in the same way, and fitted from verified reports once a pilot exists.
# Rows sum to one: these are the four things a report can turn out to be.
P_BAND_GIVEN_FLOODED = {"HIGH": 0.55, "MEDIUM": 0.30, "LOW": 0.10, "UNKNOWN": 0.05}
P_BAND_GIVEN_DRY     = {"HIGH": 0.02, "MEDIUM": 0.08, "LOW": 0.25, "UNKNOWN": 0.65}

# How long an observation stays informative about now, in minutes. Hydrology sets
# this, not decision theory. Fitted from the call-in ledger once one exists.
TAU_MIN = 30.0

# Error policy. Beta is the tolerated miss rate. Alpha is the tolerated rate of
# escalating to an operator with no flood, and it is deliberately permissive because
# this threshold governs escalation rather than airing: a human is the second stage,
# and setting alpha as though the machine were the last line of defence would throw
# that away and cost real lead time.
BETA = 0.02
ALPHA_REF = 0.25          # at the reference area below
AREA_REF_KM2 = 20.0       # alpha_g * area_g is held constant at ALPHA_REF * AREA_REF

BANDS = ("HIGH", "MEDIUM", "LOW", "UNKNOWN")


def ceiling(tpr: float | None = None, fpr: float | None = None) -> float:
    """The most any single report can be worth about its regime, in nats.

    Zero exactly when TPR equals FPR, which is where Youden's J is also zero, so this
    and the informedness statistic `cfas/generalise.py` already computes vanish
    together and `against_shuffle` is testing whether this is above chance.
    """
    # Resolved in the body, never as a default argument. A default binds once at import
    # and would then ignore load_calibration entirely, so the server would announce the
    # measured kappa while every threshold below quietly ran on the fallbacks.
    tpr = TPR if tpr is None else tpr
    fpr = FPR if fpr is None else fpr
    if tpr <= 0 or fpr <= 0:
        raise ValueError("TPR and FPR must be positive to form a likelihood ratio")
    return math.log(tpr / fpr)


def llr(band: str, tpr: float | None = None, fpr: float | None = None) -> float:
    """Evidence a report of `band` carries about its regime, in nats.

    Composes the two stages: the regime says which cells are wet, the resident says
    what they see in the cell they are standing in.

    UNKNOWN is censored to zero rather than taking the value the arithmetic gives it.
    The likelihood says an unmeasurable report is evidence against flooding, because
    people standing in floods tend to describe the water, but that number is dominated
    by nuisance parameters nobody has measured: a dead phone, panic, a language
    mismatch, a child reporting. Under that misspecification and this loss function the
    conservative reading is that we learned nothing, which is also the rule
    `cfas/hazard.py` already enforces at the instrument. An unmeasurable report is a
    reason to ask a follow-up question, never evidence of safety.
    """
    if band not in BANDS:
        raise KeyError(f"unknown band: {band}")
    if band == "UNKNOWN":
        return 0.0
    tpr = TPR if tpr is None else tpr      # see ceiling(): defaults must not bind here
    fpr = FPR if fpr is None else fpr
    s = P_BAND_GIVEN_FLOODED[band]
    f = P_BAND_GIVEN_DRY[band]
    num = tpr * s + (1 - tpr) * f
    den = fpr * s + (1 - fpr) * f
    return math.log(num / den)


ALPHA_MARGIN = 0.95       # how far below the cap alpha is held; see alpha_cap


def alpha_cap(beta: float = BETA, band: str = "HIGH",
              margin: float | None = None) -> float:
    """The largest alpha that still keeps a single report below the threshold.

    A small regime gets a permissive alpha under the constant-cost rule, and taken far
    enough that permissiveness drops the escalation boundary under the weight of one
    HIGH report, at which point one voice alone could wake the city. Solving A > Lambda
    for alpha bounds it:

        alpha < (1 - beta) * exp(-Lambda)

    Deriving the cap rather than picking a round number means the invariant survives
    recalibration: when Phase 2 measures a different kappa, this moves with it instead
    of quietly going stale. Ajegunle's smallest regime is 8.44 km², which lands inside
    the capped range, so this is load-bearing today rather than defensive.

    `margin` is the safety factor that keeps A strictly above Lambda instead of equal to
    it, since equality is the case where a capped part needs an unbounded reporting rate.
    It is not a free choice made once and forgotten: every capped part contributes
    exactly one over `-tau log(1 - Lambda/A)` to the reporting rate a city must sustain,
    and that quantity is set by this number alone. Reading it from a module global rather
    than from a default argument is what lets an experiment sweep it, which
    `experiments/exp13_alpha_margin.py` does.
    """
    m = ALPHA_MARGIN if margin is None else margin
    return (1 - beta) * math.exp(-llr(band)) * m


def alpha_for(area_km2: float) -> float:
    """False-escalation rate for a regime, scaled so expected cost is constant.

    A false alarm over 36 km² costs more than one over 13 km², so the regime with the
    most ground to lose clears a higher bar. Holding alpha*area constant does that
    without anyone hand-picking a threshold per regime, and the cap above keeps the
    small end honest.
    """
    if area_km2 <= 0:
        raise ValueError("area must be positive")
    return min(ALPHA_REF * AREA_REF_KM2 / area_km2, alpha_cap())


def thresholds(area_km2: float, beta: float = BETA) -> tuple[float, float]:
    """Wald's boundaries (escalate, stand down) in nats, for a regime of this size."""
    a = alpha_for(area_km2)
    return math.log((1 - beta) / a), math.log(beta / (1 - a))


def lambda_star(area_km2: float, tau_min: float = TAU_MIN,
                band: str = "HIGH", **kw) -> float:
    """Reports per minute below which this regime can never reach its threshold.

    The obvious form of this is the continuous steady state, Lambda*lambda*tau, and it
    is wrong here in a way that matters. That expression is the *mean* level of the
    accumulator, but nobody reads the mean: the decision is evaluated at the instant a
    report lands, when the accumulator is at its peak. Summing the geometric series of
    decayed arrivals at interval dt gives that peak as

        L_peak = Lambda / (1 - exp(-dt / tau))

    and setting it equal to the escalation boundary gives the critical interval

        dt* = -tau * ln(1 - Lambda / A)

    which is longer than the continuous form implies. Using the continuous version
    would overstate the rate this system needs and make it look less capable than it
    is; a simulation across twenty-four hours in tests/test_leak.py is what caught it.

    Returned in reports per minute, so a caller compares it against an observed
    arrival rate directly.
    """
    a, _ = thresholds(area_km2)
    per_report = llr(band, **kw)
    if per_report <= 0:
        return math.inf
    if per_report >= a:
        return 0.0                      # one report already crosses; any rate serves
    return 1.0 / (-tau_min * math.log(1 - per_report / a))


def decay(value: float, minutes: float, tau_min: float = TAU_MIN) -> float:
    """Exponential leak. `minutes` is age, so older evidence is worth less."""
    if minutes < 0:
        raise ValueError("evidence cannot be from the future")
    return value * math.exp(-minutes / tau_min)


# ---------------------------------------------------------------- independence
# Log-likelihood ratios add only for observations that are conditionally independent
# given the hypothesis. Two reports from one phone are one observation counted twice,
# and two people describing the same puddle are close to it. The swarm gets this free,
# since a bee cannot fake being at a site or be in two places; a central accumulator
# has to pay for it, and this is the payment.
RHO_SAME_REPORTER = 1.0     # one person is one observation, however often they call
RHO_NEAR = 0.5              # two people at the same corner see much the same water
NEAR_M = 100.0


@dataclass(frozen=True)
class Report:
    """One observation, with what it takes to know whether it is independent."""
    band: str
    age_min: float
    reporter: str | None = None
    lat: float | None = None
    lon: float | None = None


def _as_report(o) -> Report:
    return o if isinstance(o, Report) else Report(band=o[0], age_min=float(o[1]))


def _metres(a: Report, b: Report) -> float:
    """Rough ground distance. Equirectangular is ample over a 12 km tile."""
    if None in (a.lat, a.lon, b.lat, b.lon):
        return math.inf                       # unknown position: assume independent
    mean_lat = math.radians((a.lat + b.lat) / 2)
    dx = math.radians(a.lon - b.lon) * math.cos(mean_lat)
    dy = math.radians(a.lat - b.lat)
    return math.hypot(dx, dy) * 6_371_000


def _marginal(k: int, rho: float) -> float:
    """What the k-th correlated observation adds, on top of the k-1 before it.

    Effective sample size for k observations of correlation rho is k/(1+(k-1)rho), so
    the marginal contribution is the difference between successive terms. At rho = 0.5
    the sequence runs 1.0, 0.33, 0.17, 0.10: a second voice from the same corner is
    worth a third of the first, and the fifth is worth almost nothing. At rho = 1 every
    term after the first is zero, which is the same-reporter case.
    """
    if k <= 1:
        return 1.0
    if rho >= 1.0:
        return 0.0
    return k / (1 + (k - 1) * rho) - (k - 1) / (1 + (k - 2) * rho)


def independence_weights(reports) -> list[float]:
    """A weight per report so that correlated observations do not count twice.

    Two passes. First, one voice per reporter: a repeat call refreshes how recent the
    evidence is but adds none of it, exactly as a bee that dances twenty times is still
    one bee at the cavity. The call kept is the strongest, since observations escalate
    and never de-escalate, so a later calmer report from the same person cannot quietly
    undo their earlier alarm. Second, among the survivors, reports within NEAR_M of one
    already counted are discounted by the marginal effective sample size above.

    This is the anti-Sybil guarantee, and it is arithmetic rather than policy: one
    person contributes at most Lambda(HIGH), which is below the smallest threshold in
    the system. Nobody can manufacture a warning alone at any band with any number of
    calls.
    """
    reps = [_as_report(r) for r in reports]
    weights = [0.0] * len(reps)

    # pass one: the strongest surviving call from each reporter, anonymous calls kept
    best: dict[str, int] = {}
    keep: list[int] = []
    for i, r in enumerate(reps):
        if r.reporter is None:
            keep.append(i)
            continue
        prev = best.get(r.reporter)
        if prev is None or decay(llr(r.band), r.age_min) > decay(llr(reps[prev].band),
                                                                reps[prev].age_min):
            best[r.reporter] = i
    keep.extend(best.values())
    keep.sort()

    # pass two: proximity. Each kept report joins the cluster of the first counted
    # report it is near, and pays the marginal rate for its position in that cluster.
    cluster_size: dict[int, int] = {}
    anchors: list[int] = []
    for i in keep:
        anchor = next((a for a in anchors if _metres(reps[i], reps[a]) <= NEAR_M), None)
        if anchor is None:
            anchors.append(i)
            cluster_size[i] = 1
            weights[i] = 1.0
        else:
            cluster_size[anchor] += 1
            weights[i] = _marginal(cluster_size[anchor], RHO_NEAR)
    return weights


@dataclass(frozen=True)
class Evidence:
    """One regime's accumulator. Derived from the ledger, never mutated in place."""
    regime_id: int
    llr: float                 # nats, leak-adjusted
    threshold_a: float         # escalate at or above
    threshold_b: float         # stand down at or below
    area_km2: float
    n_reports: int             # observations still carrying weight
    n_eff: float               # effective independent observations
    lambda_obs: float          # reports per minute, observed
    lambda_star: float         # reports per minute, required

    @property
    def state(self) -> str:
        """ACCUMULATING, UNREACHABLE, ESCALATE or STAND_DOWN.

        UNREACHABLE is the one worth staring at. It means this regime cannot reach its
        threshold at the rate reports are arriving, so the quiet accumulator on the
        screen is not a system waiting, it is a system that has already failed and has
        no way to say so. Everything the agentic layer does exists to leave this state.
        """
        if self.llr >= self.threshold_a:
            return "ESCALATE"
        if self.llr <= self.threshold_b:
            return "STAND_DOWN"
        # One report says nothing about a rate, so a regime is never condemned on a
        # sample of one; it accumulates until there is an interval to measure.
        if self.n_reports > 1 and self.lambda_obs < self.lambda_star:
            return "UNREACHABLE"
        return "ACCUMULATING"

    @property
    def gap(self) -> float:
        """Nats still needed to escalate. What the agentic layer is buying."""
        return max(0.0, self.threshold_a - self.llr)


def accumulate(regime_id: int, area_km2: float, observations, *,
               tau_min: float = TAU_MIN, weight_of=None) -> Evidence:
    """Fold a regime's observations into one decision variable.

    `observations` is an iterable of `Report`, or of bare (band, age_minutes) pairs for
    callers with no identity to give. Log-likelihood ratios add, so the accumulator is
    a sum; each term is discounted for how long ago it was seen and again for how much
    of it is already carried by another report. `weight_of` overrides the independence
    model for tests that want to isolate one effect from the other.
    """
    obs = [_as_report(o) for o in observations]
    weights = weight_of(obs) if weight_of else independence_weights(obs)
    total = sum(decay(llr(r.band), r.age_min, tau_min) * wt
                for r, wt in zip(obs, weights))
    a, b_thr = thresholds(area_km2)

    # Arrival rate over the window the evidence actually spans. One report tells us
    # nothing about a rate, so it reads as zero and the regime stays ACCUMULATING
    # rather than being declared unreachable on a sample of one.
    span = max((r.age_min for r in obs), default=0.0)
    rate = (len(obs) / span) if (span > 0 and len(obs) > 1) else 0.0

    return Evidence(
        regime_id=regime_id, llr=total, threshold_a=a, threshold_b=b_thr,
        area_km2=area_km2, n_reports=len(obs), n_eff=round(sum(weights), 3),
        lambda_obs=rate, lambda_star=lambda_star(area_km2, tau_min),
    )


def reports_to_go(e: Evidence, band: str = "HIGH") -> int | None:
    """How many further reports of `band`, landing now, would cross the threshold.

    This is the number an operator actually reads. "Needs two more independent
    reports" is a sentence somebody can act on in a storm; "L = 1.44 nats" is a
    receipt, and the literature on non-expert uncertainty displays is blunt about
    which of the two gets misread as a depth. Returns 0 once the threshold is already
    crossed, and None when reports of this band cannot close the gap at all.
    """
    if e.gap <= 0:
        return 0
    per_report = llr(band)
    if per_report <= 0:
        return None
    return math.ceil(e.gap / per_report)


def summarise(evidences) -> list[dict]:
    """The decision state of every regime, worst first. For the operator's desk."""
    order = {"ESCALATE": 3, "UNREACHABLE": 2, "ACCUMULATING": 1, "STAND_DOWN": 0}
    rows = [{"regime_id": e.regime_id, "state": e.state, "llr": round(e.llr, 3),
             "threshold_a": round(e.threshold_a, 3), "gap": round(e.gap, 3),
             "n_reports": e.n_reports, "area_km2": e.area_km2,
             "lambda_obs": round(e.lambda_obs, 4),
             "lambda_star": round(e.lambda_star, 4)}
            for e in evidences]
    rows.sort(key=lambda r: (-order[r["state"]], -r["llr"]))
    return rows
