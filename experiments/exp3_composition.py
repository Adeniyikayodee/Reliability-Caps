"""Sensitivity of each prior to how hazard variance divides.

A representation can only price a hazard it can see. Relief is visible to a terrain model
and invisible to a surface one; surface character is the reverse; an unmapped culvert is
visible to neither. Which prior is worth building therefore depends on how the hazard
divides between the three, which is a property of the settlement.

Unobserved variance is held at 0.30 throughout, since no prior touches it, and the
remaining 0.70 is moved from relief to surface character in steps. Each prior is cut once
and scored at every point on the sweep, so the partitions are identical across the axis
and only the hazard moves.

Writes results/exp3_composition.json.
"""
from __future__ import annotations

import json
import time

import numpy as np

import common as P
import terrain as T

SHARES_SURFACE = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
SHARE_OBSERVED = 0.7          # the rest is unobserved by every prior
HAZARD_SEEDS = tuple(range(10))
K = 6
N_REPORTS = 4000
NOISE = 0.20
TAU = 30.0
TRIALS = 100


def main():
    grid = P.load_tile()
    land = P.land_mask(grid)
    meta = json.loads((P.REPO / "data/meta.json").read_text())
    elev = T.elevation_on_grid(meta, land.shape)
    stack = T.terrain_stack(elev, land)
    wet = T.wetness_index(elev, land)

    tess = P.tessera_partition(grid, K, seed=0)
    fixed = [tess, P.terrain_partition(stack, land, K, seed=0),
             P.tessellation_partition(land, K), P.shuffled_partition(tess, seed=0)]

    out = {"config": {"k": K, "shares_surface": list(SHARES_SURFACE),
                      "share_observed": SHARE_OBSERVED,
                      "hazard_seeds": list(HAZARD_SEEDS), "n_reports": N_REPORTS,
                      "report_noise": NOISE, "tau_min": TAU, "shuffle_trials": TRIALS},
           "sweep": []}

    t0 = time.time()
    for ss in SHARES_SURFACE:
        st = SHARE_OBSERVED - ss
        point = {"share_surface": ss, "share_terrain": round(st, 3),
                 "share_unobserved": round(1 - SHARE_OBSERVED, 3), "priors": {}}
        per_prior: dict[str, list] = {}
        oracle: list = []
        for seed in HAZARD_SEEDS:
            truth = P.flood_field(grid, land, wet, seed=seed,
                                  share_terrain=st, share_surface=ss)
            driver = P.flood_driver(grid, land, wet, seed=seed,
                                    share_terrain=st, share_surface=ss)
            reports = P.sample_reports(truth, land, n=N_REPORTS, noise=NOISE, seed=seed)
            for part in fixed + [P.driver_partition(driver, land, K)]:
                m = P.measure_kappa(part.labels, reports, shuffle_trials=TRIALS, seed=7)
                eff = None
                if m.get("usable") and (m.get("p_permutation") or 1) < 0.05:
                    eff = P.city_effort(part.areas(), m["tpr"], m["fpr"],
                                        tau_min=TAU)["city_reports_per_hour"]
                rec = {"kappa_excess_nats": m["kappa_excess_nats"],
                       "cleared": bool(eff is not None), "city_reports_per_hour": eff}
                (oracle if part.name == "driver-oracle" else
                 per_prior.setdefault(part.name, [])).append(rec)
        per_prior["driver-oracle"] = oracle

        for name, rs in per_prior.items():
            effs = [r["city_reports_per_hour"] for r in rs if r["cleared"]]
            point["priors"][name] = {
                "kappa_excess_nats": [
                    round(float(np.mean([r["kappa_excess_nats"] for r in rs])), 4),
                    round(float(np.std([r["kappa_excess_nats"] for r in rs])), 4)],
                "cleared_null": f"{sum(r['cleared'] for r in rs)}/{len(rs)}",
                "city_reports_per_hour": round(float(np.mean(effs)), 3) if effs else None}
        out["sweep"].append(point)
        row = " ".join(f"{n.split('-')[0][:4]} {v['kappa_excess_nats'][0]:.2f}"
                       for n, v in point["priors"].items())
        print(f"  surface {ss:.1f} / terrain {st:.1f}:  {row}   ({time.time()-t0:.0f}s)")

    P.RESULTS.mkdir(parents=True, exist_ok=True)
    (P.RESULTS / "exp3_composition.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"\n{'surface share':>14}{'TESSERA':>12}{'terrain':>12}{'tessellation':>14}{'oracle':>10}")
    for p in out["sweep"]:
        g = lambda n: p["priors"][n]["kappa_excess_nats"][0]
        print(f"{p['share_surface']:>14.1f}{g('TESSERA-128'):>12.3f}{g('terrain'):>12.3f}"
              f"{g('tessellation'):>14.3f}{g('driver-oracle'):>10.3f}")
    print(f"\n{time.time() - t0:.0f}s -> {P.RESULTS / 'exp3_composition.json'}")


if __name__ == "__main__":
    main()
