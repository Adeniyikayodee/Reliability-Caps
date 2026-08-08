"""Per-part generalisation against the character of the ground.

A single figure per settlement says whether to build the system and nothing about where
inside it the method holds. Each part of each city's partition is therefore scored on its
own, by the share of held-out marks in it whose state matches the majority of the other
marks in it, and regressed against attributes of the ground it covers:

    compactness     4*pi*A / P^2, whether the part is one place or several
    fragments       connected components per square kilometre
    relief range    internal relief, from Copernicus GLO-30
    water distance  distance to standing water
    area            in square kilometres
    dispersion      within-cluster spread in the representation, the quantity k-means
                    minimises

Reported as rank correlations with a Bonferroni correction across the six tests, and as
standardised coefficients from a linear fit on all six.

Writes results/exp9_stratify.json.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
from scipy import ndimage, stats

import common as P
import terrain as T
from cfas.regime import fetch_embedding_tile
from exp6_cities import CITIES, YEAR

K = 6
HAZARD_SEEDS = tuple(range(12))
N_REPORTS = 3000
NOISE = 0.20
ATTRS = ("area_km2", "compactness", "fragments_per_km2", "water_km", "dispersion",
         "relief_range_m")


def attributes(labels, pid, grid, land, elev):
    """The describable character of one part."""
    m = labels == pid
    n = int(m.sum())
    area = n * P.CELL_KM2

    # Perimeter by counting cell faces that touch something else. Exact on a grid, and
    # not sensitive to a contour-tracing convention.
    pad = np.pad(m, 1)
    edges = ((pad[1:-1, 1:-1] & ~pad[:-2, 1:-1]).sum() + (pad[1:-1, 1:-1] & ~pad[2:, 1:-1]).sum()
             + (pad[1:-1, 1:-1] & ~pad[1:-1, :-2]).sum() + (pad[1:-1, 1:-1] & ~pad[1:-1, 2:]).sum())
    perim_km = edges * P.CELL_M / 1000.0
    compactness = 4 * math.pi * area / (perim_km ** 2) if perim_km else 0.0

    lab_cc, n_cc = ndimage.label(m)
    dist_water = ndimage.distance_transform_edt(land) * P.CELL_M / 1000.0

    vec = grid[m]
    centroid = vec.mean(0)
    dispersion = float(np.linalg.norm(vec - centroid, axis=1).mean())

    e = elev[m]
    return {"area_km2": round(area, 3),
            "compactness": round(compactness, 4),
            "fragments_per_km2": round(n_cc / area, 3),
            "water_km": round(float(dist_water[m].mean()), 3),
            "dispersion": round(dispersion, 4),
            "relief_range_m": round(float(np.percentile(e, 90) - np.percentile(e, 10)), 3),
            "cells": n}


def agreement(labels, reports):
    """Per-part share of held-out marks the rest of the part predicts correctly."""
    hit, tot = {}, {}
    by_part = {}
    for i, (r, c, f) in enumerate(reports):
        pid = int(labels[r, c])
        if pid >= 0:
            by_part.setdefault(pid, []).append((i, bool(f)))
    for pid, members in by_part.items():
        if len(members) < 2:
            continue
        truths = [t for _, t in members]
        total_wet = sum(truths)
        for t in truths:
            others_wet = total_wet - t
            warned = others_wet > (len(truths) - 1) / 2
            hit[pid] = hit.get(pid, 0) + int(warned == t)
            tot[pid] = tot.get(pid, 0) + 1
    return {pid: hit[pid] / tot[pid] for pid in tot}


def main():
    rows = []
    t0 = time.time()
    for name, lat, lon in CITIES:
        got = fetch_embedding_tile(lat, lon, YEAR, dataset_version="v1")
        if got is None:
            print(f"[{name}] no coverage")
            continue
        grid, transform, crs = got
        land = P.land_mask(grid)
        meta = {"transform": [transform.a, transform.b, transform.c,
                              transform.d, transform.e, transform.f], "crs": str(crs)}
        slug = name.split(",")[0].lower().replace(" ", "_")
        elev = T.elevation_on_grid(meta, land.shape, cache_name=f"elev_{slug}")
        wet = T.wetness_index(elev, land)

        P._MICRO = None
        part = P.tessera_partition(grid, K, seed=0)
        per_part_scores: dict[int, list] = {}
        for s in HAZARD_SEEDS:
            truth = P.flood_field(grid, land, wet, seed=s)
            reports = P.sample_reports(truth, land, n=N_REPORTS, noise=NOISE, seed=s)
            for pid, a in agreement(part.labels, reports).items():
                per_part_scores.setdefault(pid, []).append(a)

        for pid in sorted(per_part_scores):
            row = {"city": name, "part": pid,
                   "agreement": round(float(np.mean(per_part_scores[pid])), 4),
                   "agreement_sd": round(float(np.std(per_part_scores[pid])), 4)}
            row.update(attributes(part.labels, pid, grid, land, elev))
            rows.append(row)
        print(f"[{name}] {len(per_part_scores)} parts  ({time.time()-t0:.0f}s)", flush=True)
        del grid

    y = np.array([r["agreement"] for r in rows])
    corr = {}
    for a in ATTRS:
        x = np.array([r[a] for r in rows])
        rho, p = stats.spearmanr(x, y)
        corr[a] = {"spearman_rho": round(float(rho), 3), "p": round(float(p), 4)}

    # A plain linear fit on standardised attributes, to see how much of the variation in
    # per-part generalisation the describable character of the ground accounts for at all.
    X = np.column_stack([[r[a] for r in rows] for a in ATTRS]).astype(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - resid.var() / y.var()
    n, k = len(y), X.shape[1] - 1
    adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)

    out = {"config": {"k": K, "hazard_seeds": list(HAZARD_SEEDS),
                      "n_reports": N_REPORTS, "report_noise": NOISE,
                      "n_parts": len(rows), "n_cities": len({r['city'] for r in rows})},
           "spearman": corr,
           "linear_fit": {"r2": round(float(r2), 3), "adjusted_r2": round(float(adj), 3),
                          "standardised_coefficients": {
                              a: round(float(b), 4) for a, b in zip(ATTRS, beta[1:])}},
           "parts": rows}
    P.RESULTS.mkdir(parents=True, exist_ok=True)
    (P.RESULTS / "exp9_stratify.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"\n{len(rows)} parts across {out['config']['n_cities']} cities")
    print(f"{'attribute':<20}{'rho':>8}{'p':>9}{'std coef':>11}")
    for a in ATTRS:
        print(f"{a:<20}{corr[a]['spearman_rho']:>8.2f}{corr[a]['p']:>9.4f}"
              f"{out['linear_fit']['standardised_coefficients'][a]:>11.4f}")
    print(f"\nlinear fit R2 {r2:.3f}, adjusted {adj:.3f}")
    print(f"-> {P.RESULTS / 'exp9_stratify.json'}")


if __name__ == "__main__":
    main()
