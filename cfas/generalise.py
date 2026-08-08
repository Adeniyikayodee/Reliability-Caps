"""Does one caller really speak for their regime? Measure it, do not assert it.

The urban design makes one strong, load-bearing claim: a regime is ground that
behaves alike, so a flood report from one street is evidence about every street in
the same regime (cfas/assimilate.py). That is what lets a single call-in raise a
band over 12 km2, and it is the first thing a sceptic should attack. This module
is the attack, run on ourselves.

The test is leave-one-out. Hold back one confirmed report, predict its cell from
every *other* report in the same regime, and check the prediction against its own
mark. A report never gets to predict itself, so what is measured is genuinely the
generalisation across ground nobody stood on, exactly the leap the design
depends on.

The prediction here is the *majority* of the other reports in the regime, and
that is deliberately not the rule the live system runs. cfas/assimilate.py
escalates on a *single* confirmed report, because for a live warning one person
in chest-deep water is enough and missing a flood costs more than warning early.
That operational rule maximises recall and cannot, by construction, measure
whether a regime is homogeneous, with common floods and large regimes every
regime holds some flood report, so "any report warns" warns everywhere and tells
you nothing. The question here is different: does knowing a cell's regime predict
whether it floods? The optimal answer to that is the majority of its neighbours,
so that is what the calibrator scores. One rule is for warning; this one is for
knowing whether the warning's reach is earned.

One number is not enough, because a high score could be an artifact of the flood
being everywhere that day. So the regime grouping is scored against a control:
the same reports, the same group sizes, but membership shuffled at random. If
regimes carry real flood information, they beat the shuffle. If they do not, they
tie it, and the honest response to a tie is a different regime map or a smaller
k, not a louder claim. The gap between the two is the skill that TESSERA's
grouping actually adds.

Two costs are reported plainly beside the skill, because generalisation is a
trade, not a free lunch:

    reach     km2 a single confirmed report speaks for. Large is the point.
    leakage   the false-alarm rate the generalisation introduces: cells a regime
              warned on a neighbour's report that did not themselves flood. Large
              is the price, and precision-over-recall (docs/lagos.md) means this
              is the number that caps how far a report may travel.

The scoring is the standard contingency table (Jolliffe & Stephenson, 2012), the
same one the archived rural calibrator (cfas/legacy/calibrate.py) uses; only the
unit of prediction is different. There it is a community-day; here it is a
held-out cell, predicted from its regime.

Pure: numpy and the standard library, no network. Reports arrive as (row, col,
flooded) triples, a confirmed wet or dry mark placed on the grid, the same
ground truth the rural calibrator reads from feedback.jsonl.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Skill:
    """What the generalisation is worth, and what it costs."""
    recall: float | None          # held-out floods a regime-mate's report caught
    precision: float | None       # regime warnings that proved real
    false_alarm: float | None     # dry cells a regime warned anyway (leakage)
    reach_km2: float              # ground one confirmed report speaks for
    n_reports: int
    n_regimes_tested: int
    counts: dict

    @property
    def informative(self) -> bool:
        """Was there enough to say anything at all?"""
        return self.counts["tp"] + self.counts["fn"] > 0


def _counts_to_metrics(c):
    r = lambda a, b: a / b if b else None
    return (r(c["tp"], c["tp"] + c["fn"]),          # recall
            r(c["tp"], c["tp"] + c["fp"]),          # precision
            r(c["fp"], c["fp"] + c["tn"]))          # false alarm (leakage)


def _informedness(c):
    """Youden's J = recall + specificity - 1 = TPR - FPR.

    The right statistic to test against the shuffle, where recall is not. When
    floods are common and regimes are large, recall stays high under a random
    shuffle, a random group almost always contains another flood, so recall
    cannot tell a real grouping from noise. J can: it is the rate a regime warns
    true floods minus the rate it warns dry cells, so a grouping that carries no
    flood information scores ~0 however common floods are, and only a grouping
    that separates wet ground from dry approaches 1.
    """
    tpr = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else None
    fpr = c["fp"] / (c["fp"] + c["tn"]) if (c["fp"] + c["tn"]) else None
    if tpr is None or fpr is None:
        return None
    return tpr - fpr


def leave_one_out(labels, reports, *, cell_m=10.0):
    """Score the regime generalisation, one held-out report at a time. Pure.

    `reports` is an iterable of (row, col, flooded). For each report, its cell is
    predicted from the *other* reports in the same regime by majority vote: if more
    than half of them confirmed a flood, this cell is predicted to flood. The
    prediction is then checked against this report's own mark. Majority, not "any"
   , see the module docstring for why the diagnostic rule differs from the live
    escalation rule.

    A report alone in its regime has no other voice to be predicted from, so it is
    counted as untested rather than scored, generalisation needs at least two.
    """
    pts = [(int(r), int(c), bool(f)) for r, c, f in reports]
    rids = [int(labels[r, c]) if 0 <= r < labels.shape[0] and 0 <= c < labels.shape[1]
            else -1 for r, c, _ in pts]

    by_regime: dict[int, list[int]] = {}
    for i, rid in enumerate(rids):
        if rid >= 0:
            by_regime.setdefault(rid, []).append(i)

    c = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    tested_regimes = set()
    for i, (_, _, truth) in enumerate(pts):
        rid = rids[i]
        if rid < 0:
            continue                       # off-grid or nodata: no regime to speak for it
        mates = [j for j in by_regime[rid] if j != i]
        if not mates:
            continue                       # nobody else in this regime; nothing to predict from
        tested_regimes.add(rid)
        # Majority vote of the other reports in this regime. A tie does not warn:
        # the neighbours are split, so the regime is not evidence either way.
        flooded_mates = sum(pts[j][2] for j in mates)
        warned = flooded_mates > len(mates) / 2
        key = ("tp" if truth else "fp") if warned else ("fn" if truth else "tn")
        c[key] += 1

    recall, precision, false_alarm = _counts_to_metrics(c)

    # Reach: ground a single confirmed report speaks for, averaged over the
    # regimes that actually carried a confirmed flood. This is the amplification
    # the point-baseline (a report speaks only for its own 10 m cell) does not get.
    cell_km2 = (cell_m / 1000.0) ** 2
    flooded_regions = {rids[i] for i, (_, _, f) in enumerate(pts) if f and rids[i] >= 0}
    reach = float(np.mean([int((labels == rid).sum()) * cell_km2
                           for rid in flooded_regions])) if flooded_regions else 0.0

    return Skill(recall=recall, precision=precision, false_alarm=false_alarm,
                 reach_km2=round(reach, 3), n_reports=len(pts),
                 n_regimes_tested=len(tested_regimes), counts=c)


def against_shuffle(labels, reports, *, trials=200, seed=0, cell_m=10.0):
    """The same score, with regime membership shuffled. The control.

    Preserves how many regimes there are and how big each is, but severs the tie
    between a cell and the regime TESSERA put it in. The statistic is informedness
    (Youden's J), not recall: recall survives a shuffle whenever floods are common,
    so it cannot separate a real grouping from noise, whereas J is ~0 for any
    grouping that carries no flood information and rises only for one that
    genuinely tells wet ground from dry.

    The gap, real J minus shuffled J, is the skill TESSERA's grouping adds, and
    it is the number worth quoting. A gap near zero is the honest signal to try a
    different regime map or a smaller k, not to claim reach the data does not
    support.

    Returns (real_skill, real_J, shuffled_J_mean, shuffled_J_std, gap).
    """
    real = leave_one_out(labels, reports, cell_m=cell_m)
    real_j = _informedness(real.counts)

    pts = [(int(r), int(c), bool(f)) for r, c, f in reports]
    on_grid = [(r, c, f) for r, c, f in pts
               if 0 <= r < labels.shape[0] and 0 <= c < labels.shape[1]
               and labels[r, c] >= 0]
    real_ids = np.array([int(labels[r, c]) for r, c, _ in on_grid])
    truths = np.array([f for _, _, f in on_grid])
    uniq = np.unique(real_ids)

    rng = np.random.default_rng(seed)
    js = []
    for _ in range(trials):
        shuffled = real_ids.copy()
        rng.shuffle(shuffled)              # same group sizes, membership randomised
        c = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for g in uniq:
            members = np.nonzero(shuffled == g)[0]
            if len(members) < 2:
                continue
            for i in members:
                mates = members[members != i]
                warned = truths[mates].sum() > len(mates) / 2      # same majority rule
                truth = bool(truths[i])
                key = ("tp" if truth else "fp") if warned else ("fn" if truth else "tn")
                c[key] += 1
        j = _informedness(c)
        if j is not None:
            js.append(j)

    real_j = None if real_j is None else round(real_j, 3)
    if not js:
        return real, real_j, None, None, None
    mean, std = float(np.mean(js)), float(np.std(js))
    gap = None if real_j is None else round(real_j - mean, 3)
    return real, real_j, round(mean, 3), round(std, 3), gap
