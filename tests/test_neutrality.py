"""The independence guarantee, at the level of words.

If a recruited resident is handed the answer, their reply is an echo and the evidence
arithmetic is silently wrong. These tests are what make that impossible rather than
discouraged, and the property test at the bottom is the one that matters: it holds for
every cue the measurement model knows, in all three languages, and it keeps holding
when someone adds a new cue to cfas/hazard.py without reading this file.
"""
from itertools import chain

import pytest

from cfas import neutral
from cfas.hazard import DEBRIS_CUES, DEPTH_CUES, VELOCITY_CUES, assess_report


def ok(q):
    good, _ = neutral.is_neutral(q)
    return good


# ---------------------------------------------------------------- what passes
@pytest.mark.parametrize("q", [
    "Water don reach where for your body?",
    "How the water dey for your street now?",
    "Wetin you dey see for road now?",
    "What can you see outside?",
    "Omi ti dé ibo?",
    "Can you walk to the junction?",
])
def test_open_questions_pass(q):
    assert ok(q), f"refused a neutral question: {q}"


def test_the_filter_over_blocks_on_purpose():
    """A harmless "still" is refused, and that is the intended trade.

    "Can you still walk to the junction?" carries no leading intent, but "still" is
    how the measurement model reads standing water, so it goes. The costs are wildly
    asymmetric: a false positive costs the recruiter one rewrite, and the refusal note
    tells it exactly which word to drop, while a false negative silently invalidates
    every log-likelihood ratio downstream and leaves the arithmetic looking healthy.
    Context-sensitivity here would be precisely the cleverness that opens the hole.
    """
    good, reason = neutral.is_neutral("Can you still walk to the junction?")
    assert not good and reason == "still"
    assert ok("Can you walk to the junction?")


# ---------------------------------------------------------------- what is refused
@pytest.mark.parametrize("q,term", [
    ("Is the water at your chest?", "chest"),
    ("Has the water reached your knee?", "knee"),
    ("Ṣé omi ti dé orúnkún rẹ?", "orunkun"),
    ("Is the water rushing past you?", "rushing"),
    ("Is it a high flood there?", "high"),
])
def test_leading_questions_are_refused(q, term):
    good, reason = neutral.is_neutral(q)
    assert not good, f"let a leading question through: {q}"
    assert reason


def test_a_proposition_to_agree_with_is_refused():
    """No cue word in it, and still not a question a person answers by looking."""
    good, reason = neutral.is_neutral("We hear say the water bad for your side, na true?")
    assert not good and reason


def test_statements_are_refused():
    assert not ok("Tell me about the water outside")


def test_over_long_questions_are_refused():
    """Someone standing in moving water reads the first few words."""
    assert not ok("Could you please take a moment to describe in as much detail as you "
                  "are able exactly what you are currently able to observe outside")


def test_refusal_names_the_offending_term():
    """A recruiter told which word it tripped on rewrites; one told 'no' retries blind."""
    good, reason = neutral.is_neutral("Is the water at your waist?")
    assert not good
    assert reason in ("waist", "belle", "belly", "hip", "ibadi", "egbe")
    assert reason in neutral.refusal_note(reason)


# ------------------------------------------------- the guarantee, as a property
@pytest.mark.parametrize("cue", sorted(set(chain(
    *DEPTH_CUES.values(), *VELOCITY_CUES.values(), DEBRIS_CUES))))
def test_no_question_containing_a_measurement_cue_can_be_sent(cue):
    """The load-bearing property: the blocklist IS the measurement lexicon.

    For every single cue the hazard model can read out of a reply, a question
    containing it is refused. This is what makes the question channel and the
    measurement channel provably disjoint at the lexical level, and it holds across
    English, Nigerian Pidgin and Yorùbá at once because the lexicon does.
    """
    assert not ok(f"Is the water {cue} where you are?"), f"cue leaked through: {cue}"


def test_the_filter_tracks_the_lexicon_rather_than_a_copy(monkeypatch):
    """Add a cue to the measurement model and the filter covers it, uninstructed.

    A hand-maintained copy of the lexicon would pass every test above and then rot the
    first time somebody extended cfas/hazard.py. Building the blocklist at call time is
    what stops that, so it is worth a test of its own.
    """
    assert ok("Is the water skibidi where you are?")
    monkeypatch.setitem(DEPTH_CUES, "invented", ("skibidi",))
    assert not ok("Is the water skibidi where you are?")


def test_a_neutral_question_still_elicits_a_measurable_answer():
    """Neutrality must not cost us the measurement.

    A filter that only allowed questions too vague to answer would protect
    independence by making the channel useless. The reply to a neutral question still
    has to run through the instrument and come out with a band.
    """
    assert ok("Water don reach where for your body?")
    h = assess_report("water don reach my knee, e dey rush")
    assert h.band != "UNKNOWN" and h.depth_m
