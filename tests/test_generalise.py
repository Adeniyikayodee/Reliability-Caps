"""Does the regime generalisation earn its keep, and can the tool tell?

Offline, no keys. The load-bearing test is not that the score is high; it is that
the tool reports high skill when regimes really do predict flooding and low skill
when they do not. A calibrator that always says "great" proves nothing, so both
cases are here.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfas.generalise import Skill, against_shuffle, leave_one_out


def _two_regimes(h=40, w=40):
    """A labels grid split down the middle: regime 0 left, regime 1 right."""
    labels = np.zeros((h, w), "int32")
    labels[:, w // 2:] = 1
    return labels


def _strips(h=60, w=60, n=6):
    """A labels grid in n vertical strips, n regimes of equal size."""
    labels = np.zeros((h, w), "int32")
    for j in range(n):
        labels[:, j * w // n:(j + 1) * w // n] = j
    return labels


def test_when_a_regime_floods_together_the_generalisation_holds():
    """Regime 0 floods, regime 1 stays dry. A regime-mate should predict you right."""
    labels = _two_regimes()
    reports = ([(5, 5, True), (10, 8, True), (15, 3, True)]      # all of regime 0: flood
               + [(5, 30, False), (12, 35, False), (20, 33, False)])  # all of regime 1: dry
    s = leave_one_out(labels, reports)

    assert s.recall == 1.0, "every held-out flood had a flooded regime-mate"
    assert s.false_alarm == 0.0, "and no dry cell was warned"
    assert s.reach_km2 > 0


def test_when_floods_ignore_regimes_the_tool_says_so():
    """Floods scattered without regard to regime: the generalisation must fail.

    This is the test that makes the tool credible. If it reported skill here, its
    praise elsewhere would be worthless.
    """
    labels = _strips(h=60, w=60, n=6)
    # Floods placed at random across six regimes: membership carries no signal.
    # Six groups and plenty of reports so majority-vote is stable, not coin-flip.
    rng = np.random.default_rng(3)
    reports = []
    for _ in range(180):
        r = int(rng.integers(0, 60))
        c = int(rng.integers(0, 60))
        reports.append((r, c, bool(rng.random() < 0.5)))
    real, real_j, shuf_mean, shuf_std, gap = against_shuffle(labels, reports, trials=200)

    # The claim is one-sided: the tool must not report *positive* skill where none
    # exists. A gap at or below zero is the correct verdict of no information.
    # (Recall alone would read high here and lie, which is why J is the statistic.)
    assert gap is not None
    assert gap < 0.2, f"uninformative regimes must not show positive skill, gap={gap}"


def test_real_structure_beats_the_shuffle():
    """The number worth quoting: skill over the control, not skill alone."""
    labels = _two_regimes()
    reports = ([(r, c, True) for r in range(3, 18, 3) for c in (3, 6, 9)]     # regime 0 floods
               + [(r, c, False) for r in range(3, 18, 3) for c in (30, 33, 36)])  # regime 1 dry
    real, real_j, shuf_mean, shuf_std, gap = against_shuffle(labels, reports, trials=200)

    assert real.recall == 1.0
    assert real_j is not None and real_j > 0.9, "clean separation should score J near 1"
    assert shuf_mean is not None and shuf_mean < 0.4, "shuffling should collapse informedness"
    assert gap > 0.5, f"real regimes should clear the control by a wide margin, gap={gap}"


def test_a_report_never_predicts_itself():
    """Leave-one-out means the held-out cell's own mark cannot leak into its band.

    A single flood alone in its regime has no other voice, so it must be untested,
    not scored as a self-fulfilling hit.
    """
    labels = _two_regimes()
    reports = [(5, 5, True),                       # alone in regime 0
               (5, 30, False), (10, 32, False)]    # regime 1 has company
    s = leave_one_out(labels, reports)
    # The lone flood cannot be a true positive off its own back.
    assert s.counts["tp"] == 0, "a solo report must not predict itself"


def test_a_lone_report_is_untested_not_scored():
    labels = _two_regimes()
    s = leave_one_out(labels, [(5, 5, True)])
    assert sum(s.counts.values()) == 0, "one report cannot be leave-one-out scored"
    assert not s.informative
    assert s.n_regimes_tested == 0


def test_reach_is_the_area_a_confirmed_report_speaks_for():
    labels = _two_regimes(h=40, w=40)     # each regime is 40x20 = 800 cells
    reports = [(5, 5, True), (10, 8, True), (5, 30, False), (12, 35, False)]
    s = leave_one_out(labels, reports, cell_m=100.0)
    # Only regime 0 flooded; at 100 m cells, 800 cells = 8 km2.
    assert abs(s.reach_km2 - 8.0) < 1e-6


def test_off_grid_and_nodata_reports_do_not_score():
    labels = _two_regimes()
    labels[0, 0] = -1                      # nodata
    reports = [(0, 0, True),               # nodata cell
               (999, 999, True),           # off grid
               (5, 5, True), (10, 8, True)]
    s = leave_one_out(labels, reports)
    # Only the two real regime-0 reports can be scored against each other.
    assert s.counts["tp"] == 2
    assert s.counts["fp"] == s.counts["fn"] == s.counts["tn"] == 0


def test_a_dry_regime_mate_does_not_warn():
    """Warning requires a regime-mate who actually confirmed a flood."""
    labels = _two_regimes()
    reports = [(5, 30, False), (12, 35, False), (20, 33, False)]   # all dry, one regime
    s = leave_one_out(labels, reports)
    assert s.counts["tn"] == 3, "dry mates predict dry, correctly"
    assert s.counts["fp"] == 0


def test_leakage_is_reported_when_a_regime_is_mixed():
    """A regime with one flood and one dry cell: the dry cell gets a false alarm.

    This is the price the design pays for reach, and precision-over-recall means
    it must be visible, not hidden.
    """
    labels = np.zeros((10, 10), "int32")   # one regime, everything
    reports = [(1, 1, True), (2, 2, True), (8, 8, False)]
    s = leave_one_out(labels, reports)
    assert s.counts["fp"] >= 1, "a dry cell among floods should show as leakage"
    assert s.false_alarm is not None and s.false_alarm > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
