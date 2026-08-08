"""Effect of solicitation policy on time to a decision.

Unsolicited reports arrive as a bursty point process whose median time to a decision sits
outside the window in which a warning is still useful, so a querying layer exists to
raise the arrival rate. Two questions follow.

Whether solicitation moves the decision time enough to be worth the intrusion, given that
a question wakes a real person during the worst hour of their week and the budget is
therefore small.

Whether ranking recipients by expected evidence earns its place against simpler rules.
Four policies are compared on identical incidents: no querying, ask the nearest
respondent, ask at random, ask the most distant, and an oracle. Correlated replies are
discounted by marginal effective sample size, so a near neighbour whose account is
already carried by the report that prompted the question contributes little.

The cost of a leading question is measured separately, by letting a fraction of solicited
replies echo the band of the report that prompted them and recording the realised false
escalation rate against the rate the boundary was designed for.

Everything runs through the deterministic evidence core.

Writes results/exp5_recruitment.json.
"""
from __future__ import annotations

import json
import math

import numpy as np

import common as P
import terrain as T
from cfas import waggle

TPR, FPR = 0.918, 0.3889
TAU = 30.0
PART = 4                       # 36.17 km2, the part with the least slack
AREA = 36.17
LAT0, LON0 = 6.4482, 3.3335
LEAD = 60.0                    # minutes a warning still has value
HORIZON = 180.0
N_COMPOUNDS = 40               # people live in compounds on streets, not on a lattice
PER_COMPOUND = 8
COMPOUND_SPREAD_M = 45.0       # inside the 100 m radius the independence rule discounts
LAMBDA_UNSOLICITED = 1 / 25.0  # one unsolicited report every 25 minutes, zone-wide
P_REPLY = 0.60
REPLY_DELAY_MIN = 3.0
QUESTION_BUDGET = 4
MIN_GAP_BETWEEN_QUESTIONS = 5.0
N_TRIALS = 3000
POLICIES = ("none", "random", "nearest", "spread", "oracle")


def latlon(row: int, col: int, r0: int, c0: int) -> tuple[float, float]:
    lat = LAT0 + (row - r0) * P.CELL_M / 111_320.0
    lon = LON0 + (col - c0) * P.CELL_M / (111_320.0 * math.cos(math.radians(LAT0)))
    return lat, lon


def draw_band(flooded: bool, rng) -> str:
    table = waggle.P_BAND_GIVEN_FLOODED if flooded else waggle.P_BAND_GIVEN_DRY
    bands = list(table)
    return bands[int(rng.choice(len(bands), p=[table[b] for b in bands]))]


def metres(a, b) -> float:
    mean_lat = math.radians((a[0] + b[0]) / 2)
    dx = math.radians(a[1] - b[1]) * math.cos(mean_lat)
    dy = math.radians(a[0] - b[0])
    return math.hypot(dx, dy) * 6_371_000


def choose(policy: str, residents, asked: set, ledger, rng, wet) -> int | None:
    """Which resident to ask next, under each policy. None when nobody is left."""
    pool = [i for i in range(len(residents)) if i not in asked]
    if not pool:
        return None
    if policy == "random":
        return int(rng.choice(pool))
    spoken = [(x["lat"], x["lon"]) for x in ledger if x["lat"] is not None]
    if not spoken:
        return int(rng.choice(pool))
    dist = {i: min(metres((residents[i]["lat"], residents[i]["lon"]), s) for s in spoken)
            for i in pool}
    if policy == "nearest":
        return min(pool, key=lambda i: dist[i])
    if policy == "spread":
        return max(pool, key=lambda i: dist[i])
    if policy == "oracle":
        seen = [i for i in pool if wet[i]]
        return max(seen or pool, key=lambda i: dist[i])
    raise ValueError(policy)


def trial(policy: str, residents, threshold: float, rng, *, leading: float = 0.0,
          flooding: bool = True) -> dict:
    """One incident. Returns when the decision was reached and what it cost.

    Events are the arrivals themselves, and the evidence is recomputed from the whole
    ledger at each one, which is what the server does. `leading` is the probability that
    a solicited reply echoes the band of the report that prompted the question instead of
    describing the water, which is the failure the neutrality filter exists to prevent.

    Cell states are drawn per incident from the calibrated model rather than read off a
    single flood realisation: under the hypothesis that this part is flooding a cell is
    wet with probability TPR, and otherwise with probability FPR. Those two numbers are
    the definition of what the part-level hypothesis means, so simulating anything else
    would be testing the decision rule against a question it was not asked.
    """
    wet = rng.random(len(residents)) < (TPR if flooding else FPR)
    ledger: list[dict] = []                    # band, arrival time, reporter, position
    pending: list[tuple[float, int]] = []      # (arrival time, resident index)
    asked: set[int] = set()
    spent, last_question = 0, -1e9
    next_unsolicited = rng.exponential(1.0 / LAMBDA_UNSOLICITED)

    while True:
        t = min([next_unsolicited] + [p[0] for p in pending])
        if t > HORIZON:
            break
        if t == next_unsolicited:
            i = int(rng.integers(len(residents)))
            next_unsolicited = t + rng.exponential(1.0 / LAMBDA_UNSOLICITED)
            echo = False
        else:
            j = int(np.argmin([p[0] for p in pending]))
            _, i = pending.pop(j)
            echo = rng.random() < leading

        r = residents[i]
        if echo and ledger:
            band = max(ledger, key=lambda x: waggle.llr(x["band"]))["band"]
        else:
            band = draw_band(bool(wet[i]), rng)
        ledger.append({"band": band, "t": t, "reporter": f"r{i}",
                       "lat": r["lat"], "lon": r["lon"]})

        obs = [waggle.Report(band=x["band"], age_min=t - x["t"], reporter=x["reporter"],
                             lat=x["lat"], lon=x["lon"]) for x in ledger]
        if waggle.accumulate(PART, AREA, obs, tau_min=TAU).llr >= threshold:
            return {"decided_min": t, "questions": spent, "n_reports": len(ledger)}

        if (policy != "none" and spent < QUESTION_BUDGET
                and t - last_question >= MIN_GAP_BETWEEN_QUESTIONS):
            pick = choose(policy, residents, asked, ledger, rng, wet)
            if pick is not None:
                asked.add(pick)
                spent += 1
                last_question = t
                if rng.random() < P_REPLY:
                    pending.append((t + rng.exponential(REPLY_DELAY_MIN), pick))
    return {"decided_min": None, "questions": spent, "n_reports": len(ledger)}


def main():
    labels = np.load(P.REPO / "data/labels.npy")
    ys, xs = np.nonzero(labels == PART)
    rng = np.random.default_rng(0)
    r0, c0 = int(np.median(ys)), int(np.median(xs))

    # People live in compounds along streets. Placing them on a lattice would put almost
    # every pair beyond the 100 m radius at which the independence rule discounts, which
    # would hide the effect the roster ranking exists to exploit.
    seeds = rng.choice(len(ys), N_COMPOUNDS, replace=False)
    residents = []
    for s_ in seeds:
        br, bc = int(ys[s_]), int(xs[s_])
        for _ in range(PER_COMPOUND):
            dr, dc = rng.normal(0, COMPOUND_SPREAD_M / P.CELL_M, 2)
            lat, lon = latlon(br + int(dr), bc + int(dc), r0, c0)
            residents.append({"lat": lat, "lon": lon})

    with P.calibrated(TPR, FPR):
        threshold, _ = waggle.thresholds(AREA)
        alpha = waggle.alpha_for(AREA)
        results = {}
        for policy in POLICIES:
            rng = np.random.default_rng(11)
            runs = [trial(policy, residents, threshold, rng) for _ in range(N_TRIALS)]
            dec = [r["decided_min"] for r in runs if r["decided_min"] is not None]
            in_lead = sum(1 for d in dec if d <= LEAD) / len(runs)
            results[policy] = {
                "p_decided_within_lead": round(in_lead, 4),
                "p_decided_within_horizon": round(len(dec) / len(runs), 4),
                "median_decision_min": round(float(np.median(dec)), 1) if dec else None,
                "mean_questions": round(float(np.mean([r["questions"] for r in runs])), 2),
                "mean_reports": round(float(np.mean([r["n_reports"] for r in runs])), 2)}
            print(f"  {policy:<9} P(decide<{LEAD:.0f}min) {in_lead:.3f}   "
                  f"median {results[policy]['median_decision_min']}   "
                  f"questions {results[policy]['mean_questions']:.2f}")

        # what a leading question costs, over a zone that is not flooding
        leading = []
        for q in (0.0, 0.15, 0.3, 0.5, 0.7, 1.0):
            rng = np.random.default_rng(23)
            runs = [trial("spread", residents, threshold, rng, leading=q, flooding=False)
                    for _ in range(N_TRIALS)]
            rate = sum(1 for r in runs if r["decided_min"] is not None) / len(runs)
            leading.append({"echo_probability": q, "false_escalation_rate": round(rate, 4),
                            "design_alpha": round(alpha, 4),
                            "inflation": round(rate / alpha, 2) if alpha else None})
            print(f"  echo {q:.2f}: false escalation {rate:.4f} against "
                  f"design alpha {alpha:.4f}")

    out = {"config": {"part": PART, "area_km2": AREA, "tau_min": TAU,
                      "lead_min": LEAD, "horizon_min": HORIZON,
                      "n_residents": len(residents), "n_compounds": N_COMPOUNDS,
                      "compound_spread_m": COMPOUND_SPREAD_M,
                      "lambda_unsolicited_per_min": LAMBDA_UNSOLICITED,
                      "p_reply": P_REPLY, "reply_delay_min": REPLY_DELAY_MIN,
                      "question_budget": QUESTION_BUDGET, "n_trials": N_TRIALS,
                      "threshold_nats": round(threshold, 4)},
           "policies": results, "leading_questions": leading}
    P.RESULTS.mkdir(parents=True, exist_ok=True)
    (P.RESULTS / "exp5_recruitment.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n-> {P.RESULTS / 'exp5_recruitment.json'}")


if __name__ == "__main__":
    main()
