"""Sensitivity of the effort curve to the alpha cap margin.

Under the constant-cost rule a small part draws a permissive alpha, and past a certain
smallness the cap in waggle.alpha_cap takes over and holds its escalation boundary at
Lambda - log(m), where m is the margin by which the cap keeps the boundary clear of a
single report. Every capped part then demands the same fixed rate whatever its area and
whatever the representation, so beyond the granularity at which most parts are capped the
effort curve approaches a straight line whose slope is set by m alone.

This sweeps m and reports, for each prior family, the granularity that minimises effort,
the number of parts capped there, and the share of the required rate those parts
contribute. A criterion whose selected granularity moves with m is reporting the safety
factor rather than the representation, and this is the check that distinguishes the two.

Arithmetic on a finished run. Reads results/exp2_granularity.json, which stores the
calibration and part areas for this purpose, and writes results/exp13_alpha_margin.json.
"""
from __future__ import annotations

import json
import math

import numpy as np

import common as P
from cfas import waggle

MARGINS = (0.80, 0.90, 0.95, 0.99, 0.999)
TAU = 30.0
# The deployed evidence per HIGH report, the same figure Table 6 and Figure 1 use. The
# floor a capped part demands depends on it, so it has to be the deployed calibration
# and not whatever pair happens to be installed in the module at import time.
LAMBDA_DEPLOYED = 0.807


def curve_at(margin: float, rows: list[dict]) -> dict:
    """R(k) recomputed under one safety factor, paired by realisation throughout."""
    out = {}
    with P.alpha_margin(margin):
        for row in rows:
            areas = {int(i): a for i, a in row["areas_km2"].items()}
            per = []
            for cal in row["tpr_fpr_by_realisation"]:
                if cal is None:
                    continue
                e = P.city_effort(areas, cal[0], cal[1], tau_min=TAU)
                per.append((e["city_reports_per_hour"], e["n_capped"],
                            e["capped_share_of_rate"]))
            if not per:
                continue
            out[row["k"]] = {
                "R_mean": round(float(np.mean([p[0] for p in per])), 3),
                "R_sd": round(float(np.std([p[0] for p in per])), 3),
                "n": len(per),
                "n_capped": int(np.median([p[1] for p in per])),
                "n_parts": row["n_parts"],
                "capped_share_of_rate": round(float(np.mean([p[2] for p in per])), 4)}
    return out


def main() -> None:
    src_path = P.RESULTS / "exp2_granularity.json"
    if not src_path.exists():
        raise SystemExit("run exp2_granularity.py first; this reads its results")
    src = json.load(open(src_path))
    if "areas_km2" not in src["curves"]["TESSERA-128"][0]:
        raise SystemExit("rerun exp2_granularity.py first; it now stores the calibration")

    # The rate a single capped part demands, which is the whole rising arm in one number.
    floors = {}
    for m in MARGINS:
        lam = LAMBDA_DEPLOYED
        a = lam - math.log(m)
        dt = -TAU * math.log(1 - lam / a)
        floors[m] = {"boundary_nats": round(a, 4),
                     "critical_interval_min": round(dt, 2),
                     "rate_per_capped_part_per_hour": round(60.0 / dt, 4)}

    out = {"margins": list(MARGINS), "tau_min": TAU,
           "lambda_deployed_nats": LAMBDA_DEPLOYED,
           "floor_per_capped_part": floors, "families": {}}
    for family, rows in src["curves"].items():
        out["families"][family] = {}
        for m in MARGINS:
            c = curve_at(m, rows)
            best = min(c, key=lambda k: c[k]["R_mean"])
            out["families"][family][str(m)] = {
                "curve": c, "argmin_k": best, "R_at_argmin": c[best]["R_mean"]}
            print(f"{family:<12} margin {m:<5} argmin k={best:<3} "
                  f"R={c[best]['R_mean']:.2f}  "
                  f"capped at argmin {c[best]['n_capped']}/{c[best]['n_parts']} parts, "
                  f"{c[best]['capped_share_of_rate']:.0%} of the rate")

    (P.RESULTS / "exp13_alpha_margin.json").write_text(json.dumps(out, indent=1))
    print(f"\n-> {P.RESULTS / 'exp13_alpha_margin.json'}")


if __name__ == "__main__":
    main()
