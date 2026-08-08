"""Critical reporting interval under periodic and Poisson arrivals.

For arrivals at a fixed interval the accumulator read at the instant of the n-th arrival
is a geometric sum with supremum Lambda / (1 - exp(-dt / tau)), so the escalation
boundary is reached if and only if

    dt < dt* = -tau * ln(1 - Lambda / A).

That is verified here on every part of the map, by simulation either side of the
boundary, together with the factor by which the mean-level form of the same criterion,
Lambda * lambda * tau = A, understates dt*.

Unsolicited reports arrive as a point process rather than on a schedule, and a point
process is bursty. First-passage times are therefore also estimated under Poisson
arrivals across a range of rates, and the arrival interval that reaches a decision inside
a stated lead time with probability 0.9 is reported per part. A Chernoff bound on the
stationary shot noise, from Campbell's theorem, is evaluated alongside the simulation.

Writes results/exp4_metastability.json.
"""
from __future__ import annotations

import json
import math

import numpy as np

import common as P
from cfas import waggle

AREAS = {0: 22.83, 1: 12.93, 2: 23.28, 3: 14.88, 4: 36.17, 5: 8.44}
TAU = 30.0
TPR, FPR = 0.918, 0.3889          # the deployed calibration
LEAD_TIMES = (30.0, 60.0, 120.0)  # minutes a warning still has value
CONFIDENCE = 0.90
N_PATHS = 6000
RATIOS = (0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)


def ein(c: float, terms: int = 400) -> float:
    """Ein(c) = sum c^n / (n n!). Entire, so the series converges for every argument."""
    total, term = 0.0, 1.0
    for n in range(1, terms):
        term *= c / n                       # term now holds c^n / n!
        total += term / n
        if abs(term / n) < 1e-16 * max(1.0, abs(total)):
            break
    return total


def chernoff_stationary(a: float, rate: float, jump: float, tau: float = TAU) -> float:
    """Bound on P(L >= a) for the stationary accumulator, at one instant.

    This is the standing-level quantity. It is not a bound on the running maximum over a
    window, and the two are only comparable once that is said plainly, because a window
    long enough gives many nearly independent chances at the same tail.
    """
    best = 0.0
    for theta in np.linspace(1e-4, 80.0 / max(jump, 1e-9), 6000):
        best = min(best, -theta * a + rate * tau * ein(theta * jump))
    return min(1.0, math.exp(best))


def periodic_peak(jump: float, dt: float, tau: float = TAU) -> float:
    """Supremum of the accumulator under arrivals every dt minutes."""
    return jump / (1 - math.exp(-dt / tau))


def simulate(a: float, jump: float, rate: float, horizon: float, n_paths: int,
             tau: float = TAU, seed: int = 0):
    """First passage under Poisson arrivals, plus the stationary occupancy above a.

    The accumulator is evaluated only at arrival instants, which is where it peaks and
    where the server evaluates it, so nothing is discretised and the estimate is exact up
    to Monte Carlo error. Occupancy is measured as the time-weighted fraction of the path
    spent at or above the threshold, which is the quantity the Chernoff bound covers.
    """
    rng = np.random.default_rng(seed)
    times, above, total = [], 0.0, 0.0
    for _ in range(n_paths):
        t, level, first = 0.0, 0.0, None
        while t < horizon:
            gap = rng.exponential(1.0 / rate)
            decayed = level * math.exp(-gap / tau)
            if level >= a:                       # time spent above before decaying under
                above += min(gap, tau * math.log(level / a)) if level > a else 0.0
            t += gap
            if t > horizon:
                total += horizon - (t - gap)
                break
            total += gap
            level = decayed + jump
            if level >= a and first is None:
                first = t
        times.append(first)
    reached = [t for t in times if t is not None]
    return {"p_cross": len(reached) / n_paths,
            "median_first_passage_min": round(float(np.median(reached)), 1) if reached else None,
            "mean_first_passage_min": round(float(np.mean(reached)), 1) if reached else None,
            "occupancy_above": above / total if total else 0.0}


def rate_for_lead(a: float, jump: float, lead: float, conf: float, tau: float = TAU,
                  seed: int = 0) -> float | None:
    """Smallest arrival rate at which the decision is reached inside `lead` w.p. `conf`.

    Bisection on the interval, since the crossing probability is monotone in the rate.
    """
    lo, hi = 0.05, 200.0                       # minutes between reports
    if simulate(a, jump, 1.0 / lo, lead, 1500, tau, seed=seed)["p_cross"] < conf:
        return None
    for _ in range(26):
        mid = math.sqrt(lo * hi)
        p = simulate(a, jump, 1.0 / mid, lead, 1500, tau, seed=seed)["p_cross"]
        lo, hi = (mid, hi) if p >= conf else (lo, mid)
    return lo


def main():
    with P.calibrated(TPR, FPR):
        jump = waggle.llr("HIGH")
        parts = []
        for pid, area in sorted(AREAS.items()):
            a, _ = waggle.thresholds(area)
            dt_star = -TAU * math.log(1 - jump / a)
            parts.append({"part": pid, "area_km2": area, "threshold_nats": round(a, 4),
                          "single_report_nats": round(jump, 4),
                          "critical_interval_min": round(dt_star, 2),
                          "mean_field_interval_min": round(TAU * jump / a, 2),
                          "conservatism_of_mean_field": round(dt_star / (TAU * jump / a), 3)})

        # 1. the deterministic boundary is exact
        for r in parts:
            d = r["critical_interval_min"]
            r["periodic_peak_inside"] = round(periodic_peak(jump, d * 0.98), 4)
            r["periodic_peak_outside"] = round(periodic_peak(jump, d * 1.02), 4)
            r["deterministic_boundary_exact"] = bool(
                r["periodic_peak_inside"] > r["threshold_nats"] > r["periodic_peak_outside"])

        # 2. Poisson arrivals through the boundary, at each lead time
        curves = {}
        for r in parts:
            a, d = r["threshold_nats"], r["critical_interval_min"]
            pts = []
            for ratio in RATIOS:
                interval = d * ratio
                s = simulate(a, jump, 1.0 / interval, max(LEAD_TIMES), N_PATHS,
                             seed=1000 + r["part"])
                row = {"interval_over_critical": ratio, "interval_min": round(interval, 2),
                       "median_first_passage_min": s["median_first_passage_min"],
                       "occupancy_above_threshold": round(s["occupancy_above"], 5),
                       "chernoff_stationary": round(
                           chernoff_stationary(a, 1.0 / interval, jump), 6)}
                for lead in LEAD_TIMES:
                    row[f"p_cross_within_{int(lead)}min"] = round(
                        simulate(a, jump, 1.0 / interval, lead, N_PATHS,
                                 seed=2000 + r["part"])["p_cross"], 4)
                pts.append(row)
            curves[r["part"]] = pts

        # 3. the rate lead time actually demands
        for r in parts:
            a = r["threshold_nats"]
            for lead in LEAD_TIMES:
                got = rate_for_lead(a, jump, lead, CONFIDENCE, seed=3000 + r["part"])
                r[f"interval_for_{int(lead)}min_lead"] = None if got is None else round(got, 2)

    out = {"config": {"tau_min": TAU, "tpr": TPR, "fpr": FPR, "n_paths": N_PATHS,
                      "lead_times_min": list(LEAD_TIMES), "confidence": CONFIDENCE,
                      "ratios": list(RATIOS), "areas_km2": AREAS},
           "parts": parts, "poisson": curves}
    P.RESULTS.mkdir(parents=True, exist_ok=True)
    (P.RESULTS / "exp4_metastability.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"{'part':>5}{'km2':>8}{'A':>8}{'dt*':>8}{'exact':>7}"
          + "".join(f"{'lead ' + str(int(l)):>11}" for l in LEAD_TIMES))
    for r in parts:
        print(f"{r['part']:>5}{r['area_km2']:>8.2f}{r['threshold_nats']:>8.3f}"
              f"{r['critical_interval_min']:>8.1f}"
              f"{str(r['deterministic_boundary_exact']):>7}"
              + "".join(f"{str(r[f'interval_for_{int(l)}min_lead']):>11}" for l in LEAD_TIMES))

    print(f"\nPoisson arrivals, part 4 at {AREAS[4]} km2, dt* = "
          f"{parts[4]['critical_interval_min']:.1f} min")
    print(f"{'dt/dt*':>8}{'interval':>10}{'med FPT':>10}"
          + "".join(f"{'P(<' + str(int(l)) + ')':>10}" for l in LEAD_TIMES)
          + f"{'occupancy':>11}{'Chernoff':>11}")
    for p in curves[4]:
        print(f"{p['interval_over_critical']:>8.2f}{p['interval_min']:>10.1f}"
              f"{str(p['median_first_passage_min']):>10}"
              + "".join(f"{p['p_cross_within_' + str(int(l)) + 'min']:>10.3f}" for l in LEAD_TIMES)
              + f"{p['occupancy_above_threshold']:>11.5f}{p['chernoff_stationary']:>11.6f}")
    print(f"\n-> {P.RESULTS / 'exp4_metastability.json'}")


if __name__ == "__main__":
    main()
