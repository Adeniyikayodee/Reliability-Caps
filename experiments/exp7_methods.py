"""Sensitivity of the comparison to the partitioning algorithm.

Every partition elsewhere in this suite comes from Lloyd's algorithm with k-means++
seeding at default settings, which leaves open whether a result is a property of the
representations or of one clustering method run at its defaults.

Five algorithms of genuinely different character are therefore run over both feature
spaces at two granularities, and each family is represented by its best member rather
than by its first:

    kmeans      Lloyd, the incumbent
    pca         k-means after PCA whitening, which removes the covariance structure
    gmm         Gaussian mixture with diagonal covariance, a soft assignment
    ward        Ward agglomerative, which is hierarchical rather than centroid-based
    smoothed    k-means followed by a 90 m modal filter, since a warning zone has to be
                contiguous ground and none of the others knows that

Writes results/exp7_methods.json.
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy import ndimage, stats

import common as P
import terrain as T

METHODS = ("kmeans", "whitened", "gmm", "ward", "smoothed")
KS = (6, 12)
HAZARD_SEEDS = tuple(range(20))
N_REPORTS = 2000
NOISE = 0.20
TAU = 30.0
TRIALS = 100
FIT_SAMPLE = 50_000
WARD_SAMPLE = 12_000
SEED = 0
SMOOTH_CELLS = 9              # a 90 m modal window, below the finest regime dimension


def _standardise(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def _assign(z, centres):
    lab = np.empty(len(z), "int32")
    for i in range(0, len(z), 200_000):
        b = z[i:i + 200_000]
        d2 = (b ** 2).sum(1)[:, None] - 2 * b @ centres.T + (centres ** 2).sum(1)[None, :]
        lab[i:i + 200_000] = np.argmin(d2, 1)
    return lab


def partition(features: np.ndarray, land: np.ndarray, k: int, method: str,
              *, seed: int = SEED) -> P.Partition:
    """One partition of the land cells, by one algorithm, over one feature space."""
    flat = features.reshape(-1, features.shape[-1]).astype("float32")
    ok = land.reshape(-1)
    z = _standardise(flat[ok])
    rng = np.random.default_rng(seed)

    if method == "whitened":
        c = np.cov(z[rng.choice(len(z), min(FIT_SAMPLE, len(z)), replace=False)].T)
        w, v = np.linalg.eigh(c + 1e-6 * np.eye(len(c)))
        z = (z @ v) / np.sqrt(np.maximum(w, 1e-6))

    fit = z if len(z) <= FIT_SAMPLE else z[rng.choice(len(z), FIT_SAMPLE, replace=False)]

    if method in ("kmeans", "whitened", "smoothed"):
        from cfas.regime import kmeans
        _, centres = kmeans(fit, k, seed=seed)
    elif method == "gmm":
        # A mixture over three terrain layers collapses components long before one over
        # 128 embedding dimensions does, so the regulariser is raised until the fit is
        # defined rather than fixed at a value that happens to suit the wider space.
        from sklearn.mixture import GaussianMixture
        centres = None
        for reg in (1e-4, 1e-3, 1e-2, 1e-1):
            try:
                g = GaussianMixture(n_components=k, covariance_type="diag",
                                    random_state=seed, max_iter=200,
                                    reg_covar=reg).fit(fit.astype("float64"))
                centres = g.means_.astype("float32")
                break
            except ValueError:
                continue
        if centres is None:
            raise RuntimeError(f"gmm did not converge at k={k} for any regulariser")
    elif method == "ward":
        from sklearn.cluster import AgglomerativeClustering
        sub = fit if len(fit) <= WARD_SAMPLE else fit[rng.choice(len(fit), WARD_SAMPLE,
                                                                replace=False)]
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(sub)
        centres = np.stack([sub[lab == j].mean(0) for j in range(k)]).astype("float32")
    else:
        raise ValueError(method)

    lab = np.full(len(flat), -1, "int32")
    lab[ok] = _assign(z, centres)
    labels = lab.reshape(land.shape)

    if method == "smoothed":
        # A warning zone has to be ground somebody can be told to leave, so speckle is a
        # defect rather than detail. Modal filtering keeps the cluster identities and
        # removes isolated cells, which is what a practitioner would do before shipping.
        stacked = np.stack([ndimage.uniform_filter((labels == j).astype("float32"),
                                                   SMOOTH_CELLS) for j in range(k)])
        out = np.argmax(stacked, 0).astype("int32")
        out[~land] = -1
        labels = out
    return P.Partition(method, labels, k)


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

    spaces = {"embedding": grid, "terrain": stack}
    per_seed: dict[str, list] = {}
    rows = []
    t0 = time.time()
    for space, feats in spaces.items():
        for method in METHODS:
            for k in KS:
                tag = f"{space}/{method}/k{k}"
                try:
                    part = partition(feats, land, k, method)
                except Exception as exc:
                    # One algorithm failing on one feature space is information, not a
                    # reason to lose the other nineteen combinations.
                    rows.append({"space": space, "method": method, "k": k,
                                 "error": f"{type(exc).__name__}: {exc}"})
                    print(f"  {tag:<26} failed: {exc}", flush=True)
                    continue
                ks_, effs, cleared = [], [], 0
                for reports in hazards:
                    m = P.measure_kappa(part.labels, reports, shuffle_trials=TRIALS, seed=7)
                    ks_.append(m["kappa_excess_nats"])
                    if m.get("usable") and (m.get("p_permutation") or 1) < 0.05:
                        cleared += 1
                        effs.append(P.city_effort(part.areas(), m["tpr"], m["fpr"],
                                                  tau_min=TAU)["city_reports_per_hour"])
                per_seed[tag] = ks_
                rows.append({"space": space, "method": method, "k": k,
                             "n_parts": len(part.areas()),
                             "min_area_km2": round(min(part.areas().values()), 3),
                             "kappa_excess_nats": [round(float(np.mean(ks_)), 4),
                                                   round(float(np.std(ks_)), 4)],
                             "cleared_null": f"{cleared}/{len(hazards)}",
                             "city_reports_per_hour": round(float(np.mean(effs)), 3)
                             if effs else None})
                print(f"  {tag:<26} kappa* {np.mean(ks_):.3f} +/- {np.std(ks_):.3f}  "
                      f"R {rows[-1]['city_reports_per_hour']}  ({time.time()-t0:.0f}s)",
                      flush=True)

    def best(space):
        cand = [r for r in rows if r["space"] == space and "error" not in r]
        return max(cand, key=lambda r: r["kappa_excess_nats"][0])

    be, bt = best("embedding"), best("terrain")
    be_tag = f"{be['space']}/{be['method']}/k{be['k']}"
    bt_tag = f"{bt['space']}/{bt['method']}/k{bt['k']}"
    x, y = np.array(per_seed[be_tag]), np.array(per_seed[bt_tag])
    d = x - y
    paired = {"embedding_best": be_tag, "terrain_best": bt_tag,
              "mean_difference_nats": round(float(d.mean()), 4),
              "sd_of_differences": round(float(d.std(ddof=1)), 4),
              "effect_over_noise": round(float(abs(d.mean()) / d.std(ddof=1)), 3),
              "wilcoxon_p": round(float(stats.wilcoxon(x, y).pvalue), 5)}

    base = np.array(per_seed["embedding/kmeans/k6"])
    gain = {"embedding_best_minus_kmeans_k6": round(float((x - base).mean()), 4),
            "wilcoxon_p": round(float(stats.wilcoxon(x, base).pvalue), 5)}

    P.RESULTS.mkdir(parents=True, exist_ok=True)
    (P.RESULTS / "exp7_methods.json").write_text(json.dumps(
        {"config": {"methods": list(METHODS), "ks": list(KS), "seed": SEED,
                    "hazard_seeds": list(HAZARD_SEEDS), "n_reports": N_REPORTS,
                    "report_noise": NOISE, "tau_min": TAU, "shuffle_trials": TRIALS,
                    "smooth_cells": SMOOTH_CELLS},
         "rows": rows, "best_versus_best": paired,
         "tuning_gain_over_default": gain, "per_seed_kappa": per_seed},
        indent=2) + "\n")

    print(f"\nbest of each family, paired over {len(HAZARD_SEEDS)} realisations")
    print(f"  embedding: {be_tag:<26} kappa* {be['kappa_excess_nats'][0]:.3f}")
    print(f"  terrain:   {bt_tag:<26} kappa* {bt['kappa_excess_nats'][0]:.3f}")
    print(f"  difference {paired['mean_difference_nats']:+.3f} nats, "
          f"{paired['effect_over_noise']:.2f}x noise, p {paired['wilcoxon_p']}")
    print(f"  tuning the embedding side gained "
          f"{gain['embedding_best_minus_kmeans_k6']:+.3f} nats over default k-means at "
          f"k=6 (p {gain['wilcoxon_p']})")
    print(f"\n{time.time()-t0:.0f}s -> {P.RESULTS / 'exp7_methods.json'}")


if __name__ == "__main__":
    main()
