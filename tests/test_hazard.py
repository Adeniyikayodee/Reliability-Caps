"""A caller's words to a hazard, checked against the standards they claim.

Offline, no keys, no models: every input here is a line someone might actually
say. The tests that matter most are the ones where depth and velocity disagree,
because that disagreement is the whole argument for reading both.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfas.hazard import (BODY_FRACTION, CAR_FLOATS_M, STATURE_M, assess_report,
                         band_of, depth_from_cue, hazard_rating)


def test_the_body_is_calibrated_not_invented():
    """Landmarks come from Winter's stature fractions, and they must stay there."""
    # Knee at 0.285 of stature: 0.285 * 1.67 = 0.476 m.
    assert abs(depth_from_cue("knee") - 0.476) < 0.002
    assert abs(depth_from_cue("waist") - 0.885) < 0.002
    assert abs(depth_from_cue("chest") - 1.202) < 0.002
    # And the ladder is monotonic, which is the only thing that makes it a scale.
    order = ["ankle", "calf", "knee", "thigh", "waist", "chest", "shoulder", "neck", "head"]
    depths = [depth_from_cue(k) for k in order]
    assert depths == sorted(depths), "body landmarks must rise monotonically"
    assert all(BODY_FRACTION[k] <= 1.0 for k in BODY_FRACTION)


def test_a_taller_caller_reads_deeper_water():
    """Stature is a parameter, not a constant, and the arithmetic must follow it."""
    assert depth_from_cue("knee", stature_m=1.90) > depth_from_cue("knee", stature_m=1.50)


def test_still_water_is_not_safe_water():
    """The EA formula's +0.5 exists so depth alone still registers.

    A pure depth-times-velocity reading would call a metre of standing water
    harmless. It drowns people. This is the term that prevents that.
    """
    hr = hazard_rating(1.0, 0.0)
    assert hr == 0.5, "standing water must carry a rating"
    assert hazard_rating(1.0, 0.0) > hazard_rating(0.2, 0.0)


def test_the_same_depth_bands_differently_when_it_moves():
    """This is the case the folk rule cannot express, and the reason for two tokens.

    Knee-deep across the three velocity classes walks the whole band ladder on
    velocity alone. If these ever collapse to one band, the motion word has
    stopped being read and half the measurement is gone.
    """
    still = assess_report("the water reach my knee, e just dey stand")
    rushing = assess_report("the water reach my knee, e dey rush")
    carrying = assess_report("the water reach my knee, e dey carry moto")

    depths = {still.depth_m, rushing.depth_m, carrying.depth_m}
    assert len(depths) == 1, "same body reference, same depth"
    assert still.velocity_ms < rushing.velocity_ms < carrying.velocity_ms
    assert still.hazard_rating < rushing.hazard_rating < carrying.hazard_rating
    assert (still.band, rushing.band, carrying.band) == ("LOW", "MEDIUM", "HIGH"), (
        f"still={still.band} rushing={rushing.band} carrying={carrying.band}, "
        "velocity must move the band on its own")


def test_walkable_and_drivable_disagree_and_that_is_the_message():
    """Knee-deep still water: safe to wade, floats your car.

    A single band cannot say this, which is why two verdicts come out. This is
    the actual Lagos warning, the action is about movement, not shelter.
    """
    h = assess_report("e reach my knee, e just dey stand")
    assert h.walkable is True
    assert h.drivable is False, f"{h.depth_m} m must float a car (>{CAR_FLOATS_M})"
    assert "no drive" in h.advice.lower()


def test_ankle_deep_still_water_is_fine_to_drive():
    h = assess_report("water just reach my ankle, e no dey move")
    assert h.depth_m < CAR_FLOATS_M
    assert h.drivable is True and h.walkable is True
    assert h.band == "LOW"


def test_deep_still_water_escalates_past_what_the_formula_says():
    """FD2320 rates chest-deep standing water LOW, and it is not wrong to.

    The rating measures stability, and you can stand up in chest-deep still water
    and walk out of it, the EA's own Low/Caution class says "deep standing
    water" in as many words. But LOW is an unbroadcastable word for it, so depth
    escalates on its own. Chest is over a child's head, which is exactly FD2320's
    "dangerous for some (i.e. children)".
    """
    h = assess_report("e reach my chest, e just dey stand")
    assert h.hazard_rating < 0.75, "the raw rating really is in the LOW range"
    assert band_of(h.hazard_rating) == "LOW", "...and the formula says so"
    assert h.band == "MEDIUM", "but the broadcast band must not"
    assert h.drivable is False


def test_the_depth_floor_is_body_relative_not_absolute():
    """A shorter caller's chest is lower ground, and the rule must follow them."""
    short = assess_report("e reach my chest, e just dey stand", stature_m=1.50)
    tall = assess_report("e reach my chest, e just dey stand", stature_m=1.90)
    assert short.depth_m < tall.depth_m
    assert short.band == tall.band == "MEDIUM", "chest is chest, whoever is standing"


def test_a_report_we_cannot_measure_is_not_a_report_of_safety():
    """No body reference means no reading. Silence must not read as LOW.

    This is the failure that kills: an unparsed call defaulting to zero depth
    would turn every ambiguous report into an all-clear.
    """
    h = assess_report("plenty water for road, e bad well well")
    assert h.band == "UNKNOWN"
    assert h.depth_m is None and h.hazard_rating is None
    assert h.walkable is None and h.drivable is None
    assert "how high" in h.advice.lower(), "it should ask for the missing measurement"


def test_negation_is_not_missed():
    """'no dey move' must beat 'dey move', or still water reads as flowing.

    Longest-match-first is what protects this. If it regresses, a caller
    reporting calm water gets a rushing-water band, the exact inversion that
    turns a safe report into a panic.
    """
    h = assess_report("the water reach my waist but e no dey move")
    assert h.velocity_cue is not None
    assert h.velocity_ms == 0.1, f"read '{h.velocity_cue}' as {h.velocity_ms} m/s"


def test_pidgin_reads_natively():
    h = assess_report("oga, water don reach my belle for Ojuelegba, e dey rush pass")
    assert h.depth_cue in ("belle", "belly")
    assert abs(h.depth_m - depth_from_cue("waist")) < 1e-6
    assert h.velocity_ms == 1.5
    assert h.band == "HIGH"


def test_yoruba_reads_with_and_without_tone_marks():
    """ASR emits diacritics inconsistently, so both forms must land the same."""
    marked = assess_report("omi ti de orúnkún mi")
    plain = assess_report("omi ti de orunkun mi")
    assert marked.depth_m == plain.depth_m == depth_from_cue("knee")
    assert marked.band == plain.band


def test_debris_raises_the_rating():
    """Debris is a term in FD2320, and callers volunteer it unprompted."""
    clean = assess_report("e reach my knee, e dey flow")
    dirty = assess_report("e reach my knee, e dey flow, e dey carry rubbish")
    assert dirty.hazard_rating > clean.hazard_rating
    assert dirty.debris > clean.debris


def test_debris_cues_match_whole_words_only():
    """A debris cue hidden inside an ordinary word must not add 1.0 to the rating.

    "refused" contains "refuse", "entire" contains "tire", "woodland" contains
    "wood". A substring test reads all three as debris, which carries knee-deep
    flowing water from LOW to HIGH on a caller who reported no debris at all.
    """
    clean = assess_report("e reach my knee, e dey flow")
    for trap in ("dem refused to move, e reach my knee, e dey flow",
                 "the entire street flood, e reach my knee, e dey flow",
                 "for woodland side, e reach my knee, e dey flow"):
        h = assess_report(trap)
        assert h.debris == clean.debris == 0.0, trap
        assert h.hazard_rating == clean.hazard_rating, trap
        assert h.band == clean.band == "LOW", trap

    # and the genuine cue still fires
    assert assess_report("e reach my knee, e dey flow, rubbish dey inside").debris == 1.0


def test_water_carrying_things_is_the_top_velocity_class():
    h = assess_report("e reach my knee for Alaba, the water dey carry moto")
    assert h.velocity_cue == "carry moto"
    assert h.velocity_ms == 2.5
    assert h.band == "HIGH"


def test_a_velocity_without_depth_keeps_the_half_it_measured():
    """No body reference, but the motion word is still a real reading.

    We cannot band this, and we do not pretend to. But throwing away the half we
    heard would make the follow-up question dumber than it needs to be.
    """
    h = assess_report("the water dey carry moto for Alaba")
    assert h.band == "UNKNOWN"
    assert h.depth_m is None and h.hazard_rating is None
    assert h.velocity_ms == 2.5, "the velocity was measured; keep it"
    assert "how high" in h.advice.lower()


def test_depth_without_motion_still_yields_a_reading():
    """Half a measurement beats discarding the call."""
    h = assess_report("water reach my knee for here")
    assert h.band != "UNKNOWN"
    assert h.velocity_cue is None
    assert h.velocity_ms == 0.5, "should fall back to the flowing default"


def test_bands_are_ordered_and_cover_the_line():
    assert band_of(0.0) == "LOW"
    assert band_of(0.74) == "LOW"
    assert band_of(0.75) == "MEDIUM"
    assert band_of(1.24) == "MEDIUM"
    assert band_of(1.25) == "HIGH"
    assert band_of(9.0) == "HIGH"


def test_deep_slow_water_is_impassable_even_at_a_modest_rating():
    """Australian guidance caps adults at 1.2 m regardless of the product."""
    h = assess_report("e reach my neck, e just dey stand")
    assert h.depth_m > 1.2
    assert h.walkable is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
