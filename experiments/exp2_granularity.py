"""Required city-wide reporting rate against partition granularity.

Clustering is conventionally cut where the inertia curve bends. That criterion is
independent of the decision the partition serves, and the decision has two opposing
appetites. Finer parts are more homogeneous, so a report says more about its neighbours
and kappa rises. Finer parts are also smaller, so each carries a lower escalation
threshold and is individually easier to decide, while there are more of them for the same
settlement to keep supplied.

The quantity that resolves the two is the reporting rate the whole settlement must
sustain for every part to remain capable of warning,

    R(k) = sum over parts of 1 / dt*(part)

evaluated across hazard realisations at each k, alongside within-cluster dispersion on
the same axis.

Per-realisation values are stored rather than only their means, so that two granularities
can be compared paired on the hazard each was measured against. The count of realisations
in which every part was decidable is stored with them, since a realisation in which one
is not contributes no finite rate and is excluded from the mean. The calibration and the
part areas are stored so that exp13 can sweep the alpha cap without repeating the
clustering.

Writes results/exp2_granularity.json.
"""
from __future__ import annotations

import json
import time

import numpy as np

import common as P
import terrain as T

KS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24)
HAZARD_SEEDS = tuple(range(10))
CLUSTER_SEEDS = (0, 1, 2)
N_REPORTS = 4000
NOISE = 0.20
TAU = 30.0
TRIALS = 100


def inertia(grid, land, ks, seed=0, sample=20_000):
    """Within-cluster dispersion per k, on the same standardised space k-means fits."""
    from cfas.regime import kmeans
    flat = grid.reshape(-1, grid.shape[-1])[land.reshape(-1)]
    z = (flat - flat.mean(0)) / (flat.std(0) + 1e-6)
    rng = np.random.default_rng(seed)
    z = z[rng.choice(len(z), sample, replace=False)]
    out = {}
    for k in ks:
        lab, cen = kmeans(z, k, seed=seed)
        out[k] = round(float(((z - cen[lab]) ** 2).sum(1).mean()), 4)
    return out


def ari(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index between two partitions, on their shared land cells."""
    m = (a >= 0) & (b >= 0)
    x, y = a[m].ravel()[::13], b[m].ravel()[::13]
    nx, ny = x.max() + 1, y.max() + 1
    tab = np.zeros((nx, ny), "int64")
    np.add.at(tab, (x, y), 1)
    comb = lambda v: (v * (v - 1) // 2).sum()
    n = tab.sum()
    idx, ea, eb = comb(tab), comb(tab.sum(1)), comb(tab.sum(0))
    exp = ea * eb / (n * (n - 1) / 2)
    return float((idx - exp) / ((ea + eb) / 2 - exp))


def main():
    grid = P.load_tile()
    land = P.land_mask(grid)
    meta = json.loads((P.REPO / "data/meta.json").read_text())
    elev = T.elevation_on_grid(meta, land.shape)
    stack = T.terrain_stack(elev, land)
    wet = T.wetness_index(elev, land)

    hazards = []
    for s in HAZARD_SEEDS:
        truth = P.flood_field(grid, land, wet, seed=s)
        hazards.append(P.sample_reports(truth, land, n=N_REPORTS, noise=NOISE, seed=s))

    inert = inertia(grid, land, KS)
    print("inertia:", inert)

    out = {"config": {"ks": list(KS), "hazard_seeds": list(HAZARD_SEEDS),
                      "cluster_seeds": list(CLUSTER_SEEDS), "n_reports": N_REPORTS,
                      "report_noise": NOISE, "tau_min": TAU,
                      "shuffle_trials": TRIALS},
           "inertia": inert, "curves": {}, "stability": {}}

    t0 = time.time()
    for family in ("TESSERA-128", "terrain"):
        curve = []
        for k in KS:
            parts = [P.tessera_partition(grid, k, seed=cs) if family == "TESSERA-128"
                     else P.terrain_partition(stack, land, k, seed=cs)
                     for cs in CLUSTER_SEEDS]
            if family == "TESSERA-128":
                out["stability"].setdefault(family, {})[k] = round(
                    float(np.mean([ari(parts[0].labels, p.labels) for p in parts[1:]])), 4)

            base, areas = parts[0], parts[0].areas()
            # Per realisation, and aligned by index across every k, so a comparison
            # between two granularities is paired on the hazard it was measured against.
            # Averaging first and comparing afterwards throws that pairing away, and the
            # spread between realisations here is wider than the differences of interest.
            ks_, ok = [], 0
            effs = [None] * len(hazards)
            cal = [None] * len(hazards)
            capped = None
            for i, reports in enumerate(hazards):
                m = P.measure_kappa(base.labels, reports, shuffle_trials=TRIALS, seed=7)
                ks_.append(m["kappa_excess_nats"])
                if m.get("usable") and (m.get("p_permutation") or 1) < 0.05:
                    ok += 1
                    cal[i] = [m["tpr"], m["fpr"]]
                    e = P.city_effort(areas, m["tpr"], m["fpr"], tau_min=TAU)
                    effs[i] = e["city_reports_per_hour"]
                    capped = {"n_capped": e["n_capped"],
                              "capped_share_of_rate": e["capped_share_of_rate"]}
            got = [e for e in effs if e is not None]
            row = {"k": k, "n_parts": len(areas),
                   "min_area_km2": round(min(areas.values()), 3),
                   "median_area_km2": round(float(np.median(list(areas.values()))), 3),
                   "kappa_excess_nats": [round(float(np.mean(ks_)), 4),
                                         round(float(np.std(ks_)), 4)],
                   "cleared_null": f"{ok}/{len(hazards)}",
                   "city_reports_per_hour": [round(float(np.mean(got)), 3),
                                             round(float(np.std(got)), 3)] if got else None,
                   "reports_per_hour_by_realisation": effs,
                   "n_effort_realisations": len(got),
                   "alpha_cap": capped,
                   # Kept so the alpha-margin sweep is arithmetic on a finished run
                   # instead of a second hour of clustering; see exp13.
                   "tpr_fpr_by_realisation": cal,
                   "areas_km2": {str(i): round(a, 4) for i, a in sorted(areas.items())},
                   "inertia": inert[k]}
            curve.append(row)
            print(f"  {family:<12} k={k:<3} kappa* {row['kappa_excess_nats'][0]:.3f}  "
                  f"min area {row['min_area_km2']:.2f} km2  "
                  f"R {row['city_reports_per_hour'][0] if effs else None} rep/h  "
                  f"({time.time() - t0:.0f}s)")
        out["curves"][family] = curve

    P.RESULTS.mkdir(parents=True, exist_ok=True)
    (P.RESULTS / "exp2_granularity.json").write_text(json.dumps(out, indent=2) + "\n")

    for family, curve in out["curves"].items():
        usable = [r for r in curve if r["city_reports_per_hour"]]
        if usable:
            best = min(usable, key=lambda r: r["city_reports_per_hour"][0])
            print(f"\n{family}: R is least at k={best['k']} "
                  f"({best['city_reports_per_hour'][0]:.2f} reports/h); "
                  f"deployed k=6 costs "
                  f"{[r for r in curve if r['k'] == 6][0]['city_reports_per_hour'][0]:.2f}")
    print(f"\n{time.time() - t0:.0f}s -> {P.RESULTS / 'exp2_granularity.json'}")


if __name__ == "__main__":
    main()
