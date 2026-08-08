"""One voice note, and the map moves.

This is the join between the two halves, and the whole claim of the urban design
sits in it.

The satellite supplies a prior: ground that behaves alike, learned from TESSERA
(cfas/regime.py), with a rainfall threshold each regime has learned from past
call-ins. A person supplies an observation: water at their knee, and it is
rushing (cfas/hazard.py). The observation is not a probability to be blended , 
it is a fact, reported by someone standing in it. So it does not average against
the prior. It overrides it.

The leverage is what a regime *means*. A regime is ground that behaves alike, so
a report from one street is evidence about every street that behaves like it. One
caller in Ikoyi speaks for the ground in Ikoyi's regime across the tile. That is
what buys lead time in a city with no radar: not a better forecast, but a
sensor network that generalises across ground the satellite already grouped.

That is a strong claim and it is deliberately falsifiable. Every posterior carries
its `support` (how many reports) and `direct` (whether this regime was itself
observed, or inherited the band from a sibling cell), so a regime raised on one
call is visibly a regime raised on one call. cfas/generalise.py is what settles
whether the generalisation earns its keep, and if it does not, the honest
outcome is a smaller k, not a louder claim.

Precision over recall here, unlike the village. A false alarm on Third Mainland
costs a million commuter-hours and burns credibility in one morning. But a
confirmed report of chest-deep water is not a false alarm risk, it is water, at
someone's chest. Observations escalate; they never de-escalate. Nobody's silence
is evidence that the road is clear.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .hazard import Hazard, assess_report

BANDS = ("LOW", "MEDIUM", "HIGH")
RANK = {b: i for i, b in enumerate(BANDS)}
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Observation:
    """One call-in, placed on the grid."""
    transcript: str
    row: int
    col: int
    lang: str = "pcm"          # pidgin by default; the contact language under stress
    at: str | None = None      # timestamp, caller's clock; passed through, never parsed


@dataclass(frozen=True)
class Posterior:
    """What a regime's band is now, and how much it rests on."""
    regime_id: int
    band: str
    prior_band: str
    support: int = 0                      # call-ins that landed in this regime
    direct: bool = False                  # observed here, or inherited from a sibling
    area_km2: float = 0.0
    hazards: list = field(default_factory=list)
    note: str = ""

    @property
    def moved(self) -> bool:
        """Did the city change this band, or is it still the satellite's guess?"""
        return self.band != self.prior_band


def prior_band(regime, rain_mm):
    """The satellite's guess, before anybody speaks.

    A regime that has learned its threshold bands against it. A regime that has
    not learned one says UNKNOWN, never LOW. An uncalibrated regime knows
    nothing about itself, and letting that silence read as safety would invert the
    system's purpose.
    """
    if regime.threshold_mm is None:
        return UNKNOWN
    if rain_mm >= regime.threshold_mm:
        return "HIGH"
    if rain_mm >= 0.6 * regime.threshold_mm:
        return "MEDIUM"
    return "LOW"


def _raise_to(current, observed):
    """Observations escalate and never de-escalate.

    Someone standing in water is the strongest evidence available. But someone
    reporting shallow water on their street is not evidence that the next street
    is dry, so a LOW report cannot pull a HIGH regime down.
    """
    if current == UNKNOWN:
        return observed
    if observed == UNKNOWN:
        return current
    return observed if RANK[observed] > RANK[current] else current


def assimilate(regimes, labels, observations, *, rain_mm=0.0, stature_m=None):
    """Prior from orbit, posterior from people. Pure; no network.

    `regimes` and `labels` come from cluster_regimes(); `observations` are
    Observations placed on the same grid. Returns {regime_id: Posterior}.

    Each call-in is measured (cfas/hazard.py), located in a regime, and the whole
    regime takes the band, because a regime is ground that behaves alike, and
    that is the assumption doing the work here. An UNKNOWN hazard, one with no
    body reference to measure, contributes nothing rather than contributing zero.
    """
    kw = {"stature_m": stature_m} if stature_m else {}
    by_id = {r.id: r for r in regimes}
    heard: dict[int, list[Hazard]] = {}

    for obs in observations:
        if not (0 <= obs.row < labels.shape[0] and 0 <= obs.col < labels.shape[1]):
            continue                      # off the grid
        rid = int(labels[obs.row, obs.col])
        if rid < 0 or rid not in by_id:
            continue                      # nodata: the caller is standing in the lagoon
        h = assess_report(obs.transcript, **kw)
        if h.band == UNKNOWN:
            continue                      # we heard them; we could not measure them
        heard.setdefault(rid, []).append(h)

    out = {}
    for r in regimes:
        pb = prior_band(r, rain_mm)
        hs = heard.get(r.id, [])
        band = pb
        for h in hs:
            band = _raise_to(band, h.band)
        note = ""
        if hs and band != pb:
            note = f"raised by {len(hs)} call-in{'s' if len(hs) > 1 else ''}"
        elif pb == UNKNOWN and not hs:
            note = "no threshold learned, nobody called: unknown, not safe"
        out[r.id] = Posterior(regime_id=r.id, band=band, prior_band=pb,
                              support=len(hs), direct=bool(hs),
                              area_km2=r.area_km2, hazards=hs, note=note)
    return out


def reach_of(posteriors, regime_id):
    """The ground one regime's band covers, in km2.

    This is the number the design lives or dies on: how much of the city a single
    caller speaks for. Large is the point, and large is also the risk, which is
    why cfas/generalise.py exists.
    """
    p = posteriors.get(regime_id)
    return p.area_km2 if p else 0.0


def summarise(posteriors):
    """The map's state in one line per regime, worst first."""
    order = {UNKNOWN: -1, **RANK}
    rows = sorted(posteriors.values(), key=lambda p: (-order[p.band], -p.area_km2))
    return [(p.regime_id, p.band, p.prior_band, p.support, round(p.area_km2, 2), p.note)
            for p in rows]
