"""The same measurement across eight flood-exposed settlements.

Every other experiment here runs on a single tile over one settlement. This one repeats
the measurement on eight, holding k, the seeds, the standardisation, the hazard
composition, the number of marks, the reading error and the permutation control fixed, so
that the ground is the only variable.

Each city is fetched, scored and released in turn, so only one tile is held in memory at
a time. Results are written after each city, which makes the run safe to interrupt and
resume. Granularity is swept per city as well, so that each prior can be compared at its
own best granularity rather than only at a common one.

Requires network throughout. A full run caches roughly 1.3 GB of tiles and rasters.

Writes results/exp6_cities.json.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

import common as P
import terrain as T
from cfas.regime import fetch_embedding_tile

# Flood-exposed, densely settled ground, chosen to sit inside a single 0.1 degree tile.
CITIES = [
    ("Lagos, Ajegunle",      6.4482,   3.3335),
    ("Accra, Odaw",          5.5800,  -0.2100),
    ("Nairobi, Mathare",    -1.2600,  36.8600),
    ("Kampala, Bwaise",      0.3500,  32.5600),
    ("Dhaka, Korail",       23.7800,  90.4200),
    ("Jakarta, Kp. Melayu", -6.2200, 106.8600),
    ("Manila, Marikina",    14.6300, 121.0700),
    ("Karachi, Lyari",      24.8800,  67.0300),
]
YEAR = 2024
KS = (4, 6, 8, 12, 16)
K_MAIN = 6
HAZARD_SEEDS = tuple(range(8))
N_REPORTS = 2000
NOISE = 0.20
TAU = 30.0
TRIALS = 100
CLUSTER_SEED = 0


def city_grid(lat: float, lon: float):
    """The tile and the metadata the elevation reprojection needs, or None."""
    got = fetch_embedding_tile(lat, lon, YEAR, dataset_version="v1")
    if got is None:
        return None
    grid, transform, crs = got
    meta = {"transform": [transform.a, transform.b, transform.c,
                          transform.d, transform.e, transform.f],
            "crs": str(crs)}
    return grid, meta


def evaluate(name: str, grid, meta) -> dict:
    land = P.land_mask(grid)
    slug = name.split(",")[0].lower().replace(" ", "_")
    elev = T.elevation_on_grid(meta, land.shape, cache_name=f"elev_{slug}")
    stack = T.terrain_stack(elev, land)
    wet = T.wetness_index(elev, land)

    P._MICRO = None                 # the latent surface classes belong to this tile
    hazards = []
    for s in HAZARD_SEEDS:
        truth = P.flood_field(grid, land, wet, seed=s)
        hazards.append(P.sample_reports(truth, land, n=N_REPORTS, noise=NOISE, seed=s))

    out = {"city": name, "shape": list(grid.shape),
           "land_km2": round(float(land.sum()) * P.CELL_KM2, 2),
           "water_km2": round(float((~land).sum()) * P.CELL_KM2, 2),
           "relief_mean_m": round(float(elev[land].mean()), 2),
           "relief_sd_m": round(float(elev[land].std()), 2),
           "relief_p95_m": round(float(np.percentile(elev[land], 95)), 2),
           "priors": {}, "granularity": {}}

    def score(part):
        ks, effs, cleared = [], [], 0
        for reports in hazards:
            m = P.measure_kappa(part.labels, reports, shuffle_trials=TRIALS, seed=7)
            ks.append(m["kappa_excess_nats"])
            if m.get("usable") and (m.get("p_permutation") or 1) < 0.05:
                cleared += 1
                effs.append(P.city_effort(part.areas(), m["tpr"], m["fpr"],
                                          tau_min=TAU)["city_reports_per_hour"])
        return {"kappa_excess_nats": [round(float(np.mean(ks)), 4),
                                      round(float(np.std(ks)), 4)],
                "cleared_null": f"{cleared}/{len(hazards)}",
                "city_reports_per_hour": round(float(np.mean(effs)), 3) if effs else None,
                "n_parts": len(part.areas())}

    tess = P.tessera_partition(grid, K_MAIN, seed=CLUSTER_SEED)
    for part in [tess,
                 P.terrain_partition(stack, land, K_MAIN, seed=CLUSTER_SEED),
                 P.tessellation_partition(land, K_MAIN),
                 P.shuffled_partition(tess, seed=CLUSTER_SEED)]:
        out["priors"][part.name] = score(part)

    for k in KS:
        out["granularity"][k] = {
            "TESSERA-128": score(P.tessera_partition(grid, k, seed=CLUSTER_SEED)),
            "terrain": score(P.terrain_partition(stack, land, k, seed=CLUSTER_SEED))}
    return out


def main():
    P.RESULTS.mkdir(parents=True, exist_ok=True)
    path = P.RESULTS / "exp6_cities.json"
    done = json.loads(path.read_text()) if path.exists() else {"config": {}, "cities": {}}
    done["config"] = {"year": YEAR, "ks": list(KS), "k_main": K_MAIN,
                      "hazard_seeds": list(HAZARD_SEEDS), "n_reports": N_REPORTS,
                      "report_noise": NOISE, "tau_min": TAU,
                      "shuffle_trials": TRIALS, "cluster_seed": CLUSTER_SEED,
                      "share_terrain": 0.35, "share_surface": 0.35,
                      "coordinates": {n: [la, lo] for n, la, lo in CITIES}}

    t0 = time.time()
    for name, lat, lon in CITIES:
        if name in done["cities"]:
            print(f"[{name}] already done, skipping")
            continue
        print(f"\n[{name}] fetching {lat}, {lon}", flush=True)
        got = city_grid(lat, lon)
        if got is None:
            done["cities"][name] = {"city": name, "error": "no v1 coverage"}
            print(f"[{name}] no coverage")
            path.write_text(json.dumps(done, indent=2) + "\n")
            continue
        grid, meta = got
        try:
            row = evaluate(name, grid, meta)
        except Exception as exc:
            row = {"city": name, "error": f"{type(exc).__name__}: {exc}"}
            print(f"[{name}] failed: {row['error']}")
        finally:
            del grid                       # the tile is 630 MB; do not hold two at once
        done["cities"][name] = row
        path.write_text(json.dumps(done, indent=2) + "\n")   # written after each city
        if "error" not in row:
            pr = row["priors"]
            print(f"[{name}] relief sd {row['relief_sd_m']} m   "
                  f"TESSERA {pr['TESSERA-128']['kappa_excess_nats'][0]:.2f}   "
                  f"terrain {pr['terrain']['kappa_excess_nats'][0]:.2f}   "
                  f"tessellation {pr['tessellation']['kappa_excess_nats'][0]:.2f}   "
                  f"({time.time() - t0:.0f}s)", flush=True)

    ok = {n: r for n, r in done["cities"].items() if "error" not in r}
    print(f"\n{'city':<22}{'relief sd':>10}{'TESSERA':>10}{'terrain':>10}"
          f"{'tessel':>9}{'random':>9}{'best k':>8}")
    for n, r in ok.items():
        g = lambda p: r["priors"][p]["kappa_excess_nats"][0]
        usable = {k: v["TESSERA-128"]["city_reports_per_hour"]
                  for k, v in r["granularity"].items()
                  if v["TESSERA-128"]["city_reports_per_hour"]}
        best = min(usable, key=usable.get) if usable else None
        print(f"{n:<22}{r['relief_sd_m']:>10.2f}{g('TESSERA-128'):>10.2f}"
              f"{g('terrain'):>10.2f}{g('tessellation'):>9.2f}{g('random'):>9.2f}"
              f"{str(best):>8}")
    print(f"\n{time.time() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
