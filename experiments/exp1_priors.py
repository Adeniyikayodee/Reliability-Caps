"""Evidence per report under each candidate spatial prior.

Everything except the partition is held fixed: the same tile, the same hazard field, the
same report locations, the same reading noise, the same permutation control, the same
seeds. The partition is the only variable, so any movement in kappa is attributable to
it.

Five priors and a ceiling are scored. The learned representation at full and at reduced
dimension; a terrain prior cut from Copernicus GLO-30; a regular tessellation of matched
granularity; a permutation control; and a partition cut from the hazard driver itself,
which bounds what any grouping at this granularity could achieve.

The same measurement is repeated under the self-consistent label model, in which the
largest parts of the partition under test are declared to be in hazard, so that the two
can be read side by side.

Writes results/exp1_priors.json and results/exp1_circular_repeated.json.
"""
from __future__ import annotations

import json
import time

import numpy as np

import common as P
import terrain as T

HAZARD_SEEDS = tuple(range(20))
CLUSTER_SEED = 0
K = 6
N_REPORTS = 2000
NOISE = 0.20
TAU = 30.0
SHARE_TERRAIN = 0.35
SHARE_SURFACE = 0.35
ORDER = ["TESSERA-128", "TESSERA-32", "TESSERA-16", "terrain", "tessellation",
         "random", "driver-oracle"]


def fixed_priors(grid, land, stack, k, seed=CLUSTER_SEED):
    """Every candidate whose partition does not depend on the hazard realisation."""
    t = P.tessera_partition(grid, k, seed=seed)
    return [t,
            P.tessera_partition(grid, k, seed=seed, dims=32),
            P.tessera_partition(grid, k, seed=seed, dims=16),
            P.terrain_partition(stack, land, k, seed=seed),
            P.tessellation_partition(land, k),
            P.shuffled_partition(t, seed=seed)]


def score(part, reports, seed, tau=TAU):
    m = P.measure_kappa(part.labels, reports, seed=seed + 100)
    areas = part.areas()
    if m.get("usable"):
        eff = P.city_effort(areas, m["tpr"], m["fpr"], tau_min=tau)
        m["city_reports_per_hour"] = eff["city_reports_per_hour"]
        m["parts"] = eff["parts"]
    m["partition"] = part.name
    m["seed"] = seed
    m["n_parts"] = len(areas)
    m["areas_km2"] = [round(a, 2) for _, a in sorted(areas.items())]
    return m


def agg(rs, key):
    v = [r.get(key) or 0.0 for r in rs]
    return round(float(np.mean(v)), 4), round(float(np.std(v)), 4)


def main():
    grid = P.load_tile()
    land = P.land_mask(grid)
    meta = json.loads((P.REPO / "data/meta.json").read_text())
    elev = T.elevation_on_grid(meta, land.shape)
    stack = T.terrain_stack(elev, land)
    wet = T.wetness_index(elev, land)
    print(f"tile {grid.shape}, land {land.sum() * P.CELL_KM2:.1f} km2, "
          f"water {(~land).sum() * P.CELL_KM2:.1f} km2")
    print(f"relief: mean {elev[land].mean():.2f} m, sd {elev[land].std():.2f} m, "
          f"range {elev[land].min():.1f} to {elev[land].max():.1f} m")

    fixed = fixed_priors(grid, land, stack, K)
    rows: dict[str, list] = {p.name: [] for p in fixed}
    rows["driver-oracle"] = []

    t0 = time.time()
    for seed in HAZARD_SEEDS:
        truth = P.flood_field(grid, land, wet, seed=seed, share_terrain=SHARE_TERRAIN,
                              share_surface=SHARE_SURFACE)
        driver = P.flood_driver(grid, land, wet, seed=seed,
                                share_terrain=SHARE_TERRAIN, share_surface=SHARE_SURFACE)
        reports = P.sample_reports(truth, land, n=N_REPORTS, noise=NOISE, seed=seed)
        for part in fixed + [P.driver_partition(driver, land, K)]:
            rows[part.name].append(score(part, reports, seed))
        if seed % 5 == 0:
            print(f"  hazard seed {seed}: {truth.sum() / land.sum():.1%} of land, "
                  f"{time.time() - t0:.0f}s")

    circ = {}
    for part in fixed:
        if part.name == "random":
            continue
        truth = P.flood_field_circular(part.labels, part.areas(), n_flooding=3)
        reports = P.sample_reports(truth, land, n=N_REPORTS, noise=NOISE, seed=0)
        circ[part.name] = score(part, reports, 0)

    summary = {}
    for name in ORDER:
        rs = rows[name]
        ok = [r for r in rs if r.get("usable") and (r.get("p_permutation") or 1) < 0.05]
        summary[name] = {
            "seeds_clearing_null": f"{len(ok)}/{len(rs)}",
            "kappa_nats": agg(rs, "kappa_nats"),
            "kappa_null_mean": agg(rs, "kappa_null_mean"),
            "kappa_excess_nats": agg(rs, "kappa_excess_nats"),
            "informedness_j": agg(rs, "informedness_j"),
            "gap_over_shuffle": agg(rs, "gap_over_shuffle"),
            "tpr": agg(rs, "tpr"), "fpr": agg(rs, "fpr"),
            "reach_km2": agg(rs, "reach_km2"),
            "city_reports_per_hour": agg(ok, "city_reports_per_hour") if ok else None,
            "n_parts": rs[0]["n_parts"]}

    P.RESULTS.mkdir(parents=True, exist_ok=True)
    (P.RESULTS / "exp1_priors.json").write_text(json.dumps(
        {"config": {"k": K, "cluster_seed": CLUSTER_SEED,
                    "hazard_seeds": list(HAZARD_SEEDS), "n_reports": N_REPORTS,
                    "report_noise": NOISE, "tau_min": TAU,
                    "share_terrain": SHARE_TERRAIN, "share_surface": SHARE_SURFACE,
                    "relief_sd_m": round(float(elev[land].std()), 3),
                    "relief_mean_m": round(float(elev[land].mean()), 3),
                    "land_km2": round(float(land.sum()) * P.CELL_KM2, 2)},
         "summary": summary, "runs": rows, "circular_label_model": circ},
        indent=2, default=str) + "\n")

    print(f"\n{'prior':<15}{'kappa':>16}{'null':>8}{'excess':>16}{'cleared':>9}{'rep/h':>16}")
    for name in ORDER:
        s = summary[name]
        ch = "" if s["city_reports_per_hour"] is None else \
            f"{s['city_reports_per_hour'][0]:>9.1f} ±{s['city_reports_per_hour'][1]:<5.1f}"
        print(f"{name:<15}{s['kappa_nats'][0]:>9.3f} ±{s['kappa_nats'][1]:<5.3f}"
              f"{s['kappa_null_mean'][0]:>8.3f}"
              f"{s['kappa_excess_nats'][0]:>9.3f} ±{s['kappa_excess_nats'][1]:<5.3f}"
              f"{s['seeds_clearing_null']:>9}{ch:>16}")
    print("\ncircular label model, for comparison with the deployed figure:")
    for name, m in circ.items():
        print(f"  {name:<15} kappa {m['kappa_nats']:.3f}  excess {m['kappa_excess_nats']:.3f}")
    print(f"\n{time.time() - t0:.0f}s -> {P.RESULTS / 'exp1_priors.json'}")


if __name__ == "__main__":
    main()
