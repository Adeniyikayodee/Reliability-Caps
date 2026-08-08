"""A spoken report turned into a measured hazard.

Nobody says "the water is 0.48 metres deep." They say *e reach my knee*. That is
not vagueness to be tolerated, it is a measurement, taken against the most widely
distributed instrument in Lagos, and one every caller carries. Human stature is
regular enough that a body reference fixes depth to within a few centimetres, and
the reference classes people reach for by themselves, ankle, knee, waist, chest , 
happen to land on the thresholds that decide whether you drive, walk, or stay.

The mapping is anthropometric, not invented. Body landmark heights are stable
fractions of stature (Drillis & Contini, reproduced in Winter, *Biomechanics and
Motor Control of Human Movement*, 4th ed., 2009): ankle 0.039H, knee 0.285H, hip
0.530H, chest 0.720H, shoulder 0.818H. Fix H and every cue becomes a number.

Depth alone is only half a hazard, and this is where the folk rule fails. "Six
inches knocks you down, two feet sweeps your car" is real NWS and FEMA guidance
but neither agency cites a study for it, and the engineering standard is the
product of depth and velocity, not depth (Abt et al. 1989, flume experiments with
20 adult volunteers; UK Environment Agency FD2320; Australian Rainfall & Runoff).
Still water at the knee is a nuisance. Moving water at the knee kills.

So a report needs two tokens, and Pidgin already carries both:

    "e reach my knee"      -> depth   0.476 m
    "e dey rush"           -> velocity 1.5 m/s
                           -> HR = 0.95, dangerous for some (MEDIUM)

The second token is the one no existing system asks for. PetaBencana takes a body
reference and a photo, which fixes depth and leaves velocity unmeasured; here the
motion word is a first-class input because it is half the answer.

Two hazards come out, not one, because the urban action is about movement:

    pedestrian  UK EA FD2320:  HR = D * (V + 0.5) + debris
    vehicle     flotation:     ~0.3 m floats a small car, ~0.5 m a 4WD
                               (Shand et al., Australian Rainfall & Runoff)

They disagree often, and the disagreement is the message. Knee-deep still water is
safe to wade and will float your car. "You fit waka, no drive" is a real warning
that a single band cannot express.

Pure: no network, no model, no dependency. The transcript arrives as text, so
whatever produced it, N-ATLAS, Whisper, Cactus, a human typing, stays the
caller's business. Every lexicon entry below wants a native speaker's eye before
it goes on air; the Yoruba especially is a starting point, not an authority.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Adult mean stature. Nigeria runs close to 1.67 m for men, 1.58 m for women; we
# take a single figure because the caller is unknown and the error it buys (a few
# cm) is far smaller than the gap between one body landmark and the next.
STATURE_M = 1.67

# Body landmark heights as fractions of stature (Winter 2009, after Drillis &
# Contini). These are the instrument's calibration; do not tune them to taste.
BODY_FRACTION = {
    "ankle": 0.039,
    "calf": 0.150,
    "knee": 0.285,
    "thigh": 0.420,
    "waist": 0.530,
    "chest": 0.720,
    "shoulder": 0.818,
    "neck": 0.870,
    "head": 1.000,
}

# Velocity classes. A caller cannot report m/s, but the distinction between water
# that sits, water that moves, and water that carries things away is one anybody
# can make and is exactly the distinction the hazard formula needs. Representative
# values are deliberately conservative within each class.
VELOCITY_CLASS = {
    "still": 0.1,
    "flowing": 0.5,
    "rushing": 1.5,
    "carrying": 2.5,
}

# Debris factor, UK EA FD2320. 0 for open ground, 0.5 where debris is likely,
# 1.0 where the caller reports things in the water.
DEBRIS_NONE, DEBRIS_LIKELY, DEBRIS_REPORTED = 0.0, 0.5, 1.0

# Vehicle flotation (Shand et al., Australian Rainfall & Runoff). Still-water
# thresholds; moving water floats a car shallower than this.
CAR_FLOATS_M = 0.30
SUV_FLOATS_M = 0.50

# UK EA FD2320 hazard-rating cuts, folded onto the three CFAS bands so a call-in
# and a satellite read speak the same language.
HR_MEDIUM = 0.75   # dangerous for some (children, elderly)
HR_HIGH = 1.25     # dangerous for most people

BANDS = ("LOW", "MEDIUM", "HIGH")
RANK = {b: i for i, b in enumerate(BANDS)}


@dataclass(frozen=True)
class Hazard:
    """What a caller's words amount to, in numbers a decision can use."""
    depth_m: float | None
    velocity_ms: float | None
    dv: float | None            # depth x velocity, m2/s
    hazard_rating: float | None  # UK EA FD2320
    band: str
    walkable: bool | None
    drivable: bool | None
    depth_cue: str | None       # the word we actually heard
    velocity_cue: str | None
    debris: float
    advice: str


# --- the lexicon ------------------------------------------------------------
# Nigerian Pidgin first: it is the contact language a Lagos caller reaches for
# under stress, and its body vocabulary is English-lexified, which makes it the
# easiest half of this problem. Yoruba is the indigenous language of Lagos and
# predominates in the general population (Akande & Salami 2010), so it leads on
# reach even where Pidgin leads on convenience. Every line below is provisional
# until a native speaker signs it off; see docs/lagos.md.
DEPTH_CUES = {
    # landmark: (english/pidgin, yoruba)
    "ankle": ("ankle", "ankles", "foot", "feet", "leg", "kokose", "kokosẹ̀", "ese", "ẹsẹ̀"),
    "calf": ("calf", "shin", "half leg"),
    "knee": ("knee", "knees", "orunkun", "orúnkún"),
    "thigh": ("thigh", "lap", "itan"),
    "waist": ("waist", "belle", "belly", "hip", "ibadi", "ìbàdí", "egbe", "ẹgbẹ́"),
    "chest": ("chest", "breast", "aya", "àyà"),
    "shoulder": ("shoulder", "ejika", "èjìká"),
    "neck": ("neck", "throat", "orun", "ọrùn"),
    "head": ("head", "overhead", "over my head", "ori", "orí"),
}

VELOCITY_CUES = {
    "still": ("just dey stand", "dey stand", "no dey move", "still", "standing",
              "no dey flow", "e stop", "duro", "dúró"),
    "flowing": ("dey flow", "dey move", "flowing", "moving", "dey go", "n san"),
    "rushing": ("dey rush", "rushing", "fast", "strong", "e strong", "dey run",
                "sare", "sáré", "dey pursue"),
    "carrying": ("dey carry", "carry person", "carry car", "carry moto",
                 "carry am", "sweep", "dey sweep", "wash am", "gbe", "gbé"),
}

# Debris is a term in the EA formula and callers volunteer it unprompted.
DEBRIS_CUES = ("carry", "rubbish", "refuse", "pure water", "log", "wood",
               "tire", "tyre", "plank", "bottle", "nylon", "gbogbo nkan")


def _normalise(text):
    """Lowercase and strip tone marks, so 'orúnkún' and 'orunkun' both land.

    Callers type without diacritics and ASR emits them inconsistently, so the
    lexicon has to meet both. We match on the stripped form and keep the original
    only for reporting.
    """
    text = unicodedata.normalize("NFD", str(text).lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _present(norm, phrase):
    """Is `phrase` in the already-normalised `norm` as a whole word?

    Whole-word, never substring. A plain `in` test reads "refused" as the debris
    cue "refuse", "entire" as "tire", and "woodland" as "wood", each of which
    silently adds a full 1.0 to the hazard rating, enough to carry knee-deep
    flowing water from LOW to HIGH on a caller who mentioned none of it.
    """
    return re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", norm) is not None


def _find_cue(text, cues):
    """The first cue present in the text, longest phrase first.

    Longest-first matters: "no dey move" must beat "dey move", or a caller saying
    the water is still gets read as flowing, the exact inversion that turns a
    safe report into a fatal one.
    """
    norm = _normalise(text)
    best, best_len, best_word = None, 0, None
    for key, words in cues.items():
        for w in words:
            wn = _normalise(w)
            if len(wn) <= best_len:
                continue
            if _present(norm, wn):
                best, best_len, best_word = key, len(wn), w
    return best, best_word


def depth_from_cue(landmark, stature_m=STATURE_M):
    """A body landmark to metres of water. Pure arithmetic on stature."""
    if landmark not in BODY_FRACTION:
        raise KeyError(f"unknown landmark: {landmark}")
    return round(BODY_FRACTION[landmark] * stature_m, 3)


def hazard_rating(depth_m, velocity_ms, debris=DEBRIS_NONE):
    """UK EA FD2320: HR = D * (V + 0.5) + debris factor.

    The +0.5 is what keeps still water hazardous at depth: a metre of water that
    is not moving still drowns, and a formula that is purely D*V would call it
    safe.
    """
    return round(depth_m * (velocity_ms + 0.5) + debris, 3)


def band_of(hazard_rating_value):
    """The FD2320 rating, banded. Stability only, see escalate_for_depth()."""
    if hazard_rating_value < HR_MEDIUM:
        return "LOW"
    if hazard_rating_value < HR_HIGH:
        return "MEDIUM"
    return "HIGH"


def escalate_for_depth(band, depth_m, stature_m=STATURE_M):
    """Raise the band on depth alone, whatever the water is doing.

    FD2320 measures *stability*: the risk of being knocked down and swept. It
    genuinely rates deep standing water as low, the EA's own Low/Caution class
    reads "shallow flowing water or deep standing water", because you can stand
    up in it and walk out. As physics that is correct. As a broadcast band it is
    indefensible: nobody hears "LOW" with water at their chest and trusts us again.

    So depth escalates on its own, against the body rather than an absolute, which
    keeps the rule true for any stature:

        chest-deep  -> at least MEDIUM.  It is over a child's head, and MEDIUM in
                       FD2320 is precisely "dangerous for some (i.e. children)".
        neck-deep   -> HIGH.             An adult is losing footing; the hazard
                       stops being stability and becomes drowning.

    This mirrors the rainfall floor in risk.fuse(), where a strong forecast raises
    the band by itself rather than being averaged away.
    """
    # Thresholds go through depth_from_cue so they carry the same rounding as the
    # reading they are compared against. Computed raw, chest lands at 1.2024 while
    # the reading rounds to 1.202, and a chest-deep report fails its own test by
    # four ten-thousandths of a metre.
    if depth_m >= depth_from_cue("neck", stature_m):
        return "HIGH"
    if depth_m >= depth_from_cue("chest", stature_m) and RANK[band] < RANK["MEDIUM"]:
        return "MEDIUM"
    return band


def _advice(band, walkable, drivable):
    """The sentence that survives being repeated.

    Short, concrete, and about movement, the urban action is whether to travel,
    not whether to climb. Pidgin phrasing because it is what gets retold at the
    junction; a native speaker reviews these before broadcast.
    """
    if not walkable and not drivable:
        return "No waka, no drive. Stay where you dey."
    if walkable and not drivable:
        return "You fit waka, but no drive. Water go carry moto."
    if band == "LOW":
        return "Road still dey pass. Waka with care."
    return "Waka with care. No drive if you fit avoid am."


def assess_report(transcript, *, stature_m=STATURE_M, default_velocity="flowing"):
    """A call-in transcript to a banded Hazard. Pure; no network.

    Returns a Hazard with `band` = "UNKNOWN" when no body reference is present , 
    a report we cannot measure must not silently become a report of no danger.
    Velocity falls back to `default_velocity` when the caller gives depth but no
    motion word, because half a measurement beats discarding the call.
    """
    landmark, depth_word = _find_cue(transcript, DEPTH_CUES)
    vclass, vel_word = _find_cue(transcript, VELOCITY_CUES)

    if landmark is None:
        # No instrument, no reading. Say so rather than guessing at zero. We still
        # report the velocity if the caller gave one, it is a real measurement,
        # it costs nothing to keep, and the follow-up question only needs depth.
        return Hazard(depth_m=None,
                      velocity_ms=VELOCITY_CLASS[vclass] if vclass else None,
                      dv=None, hazard_rating=None,
                      band="UNKNOWN", walkable=None, drivable=None,
                      depth_cue=None, velocity_cue=vel_word, debris=DEBRIS_NONE,
                      advice="Ask the caller how high the water reach on their body.")

    depth = depth_from_cue(landmark, stature_m)
    vclass = vclass or default_velocity
    velocity = VELOCITY_CLASS[vclass]

    norm = _normalise(transcript)
    debris = (DEBRIS_REPORTED if any(_present(norm, _normalise(c)) for c in DEBRIS_CUES)
              else DEBRIS_NONE)

    hr = hazard_rating(depth, velocity, debris)
    band = escalate_for_depth(band_of(hr), depth, stature_m)

    # Two verdicts, because they disagree and the disagreement is the point.
    # Australian guidance caps adults at 1.2 m depth and 3.0 m/s regardless of the
    # product, so a deep slow river is impassable even at a low rating.
    walkable = hr < HR_HIGH and depth < 1.2 and velocity < 3.0
    drivable = depth < CAR_FLOATS_M

    return Hazard(depth_m=depth, velocity_ms=velocity,
                  dv=round(depth * velocity, 3), hazard_rating=hr, band=band,
                  walkable=walkable, drivable=drivable,
                  depth_cue=depth_word, velocity_cue=vel_word, debris=debris,
                  advice=_advice(band, walkable, drivable))
