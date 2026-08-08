"""A hand-built Sentinel-2 composite as an additional prior.

The comparison elsewhere places a learned representation beside free global relief and a
regular tessellation. This adds the baseline that asks the sharper question: an annual
composite of the same Sentinel-2 observations the representation was trained on, reduced
the way a remote sensing group would have done it before foundation models existed.

The composite is deliberately a competent one. Six surface reflectance bands, each
summarised by its median and its tenth and ninetieth percentiles so that seasonality
survives, plus five standard indices, giving 23 channels on the same grid, from twelve
scenes cloud and shadow masked from the scene classification layer.

The hazard's surface component is defined twice, once from a fine clustering of the
embedding and once from a fine clustering of the composite, so that the advantage each
representation gains from having drawn the latent classes can be read off directly.

Requires network for the optical scenes. Writes results/exp8_spectral.json.
"""
from __future__ import annotations

import json
import time
import urllib.request

import numpy as np
from scipy import stats

import common as P
import terrain as T
from exp7_methods import partition

STAC = "https://earth-search.aws.element84.com/v1/search"
BBOX = [3.30, 6.40, 3.40, 6.50]
YEAR = 2024
MAX_CLOUD = 40
MAX_SCENES = 12
BANDS = ("blue", "green", "red", "nir", "swir16", "swir22")
KEEP_SCL = (4, 5, 6, 7)          # vegetated, bare, water, unclassified
KS = (6, 12)
HAZARD_SEEDS = tuple(range(20))
N_REPORTS = 2000
NOISE = 0.20
TAU = 30.0
TRIALS = 100
CACHE = P.RESULTS.parent / "cache" / "spectral_ajegunle.npy"


def scenes() -> list[dict]:
    req = urllib.request.Request(STAC, headers={"Content-Type": "application/json"},
        data=json.dumps({"collections": ["sentinel-2-l2a"], "bbox": BBOX,
                         "datetime": f"{YEAR}-01-01T00:00:00Z/{YEAR}-12-31T23:59:59Z",
                         "query": {"eo:cloud_cover": {"lt": MAX_CLOUD}},
                         "limit": 40}).encode())
    feats = json.loads(urllib.request.urlopen(req, timeout=90).read())["features"]
    feats.sort(key=lambda f: f["properties"].get("eo:cloud_cover", 100))
    return feats[:MAX_SCENES]


def composite(meta: dict, shape: tuple[int, int]) -> np.ndarray:
    """An annual Sentinel-2 summary on the tile grid: (H, W, 23). Cached."""
    if CACHE.exists():
        return np.load(CACHE)

    import rasterio
    from rasterio.warp import Resampling, reproject
    from affine import Affine

    transform = Affine(*meta["transform"][:6])

    def read(href, resampling=Resampling.bilinear):
        dst = np.zeros(shape, "float32")
        with rasterio.open(href) as src:
            reproject(rasterio.band(src, 1), dst, src_transform=src.transform,
                      src_crs=src.crs, dst_transform=transform, dst_crs=meta["crs"],
                      resampling=resampling)
        return dst

    feats = scenes()
    print(f"  {len(feats)} scenes, cloud "
          f"{[round(f['properties']['eo:cloud_cover']) for f in feats]}")
    stacks = {b: [] for b in BANDS}
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_USE_HEAD="NO"):
        for i, f in enumerate(feats):
            t0 = time.time()
            scl = read(f["assets"]["scl"]["href"], Resampling.nearest)
            good = np.isin(np.rint(scl).astype("int16"), KEEP_SCL)
            for b in BANDS:
                a = read(f["assets"][b]["href"])
                stacks[b].append(np.where(good, a, np.nan))
            print(f"    scene {i+1}/{len(feats)} {f['properties']['datetime'][:10]} "
                  f"clear {good.mean():.0%}  ({time.time()-t0:.0f}s)", flush=True)

    chans, names = [], []
    med = {}
    for b in BANDS:
        arr = np.stack(stacks[b])
        with np.errstate(all="ignore"):
            q = np.nanpercentile(arr, [10, 50, 90], axis=0)
        for tag, layer in zip(("p10", "median", "p90"), q):
            chans.append(np.nan_to_num(layer))
            names.append(f"{b}_{tag}")
        med[b] = np.nan_to_num(q[1])

    def idx(a, b):
        return (a - b) / (a + b + 1e-6)

    for name, layer in (
            ("ndvi", idx(med["nir"], med["red"])),
            ("ndwi", idx(med["green"], med["nir"])),
            ("mndwi", idx(med["green"], med["swir16"])),
            ("ndbi", idx(med["swir16"], med["nir"])),
            ("bsi", idx(med["swir16"] + med["red"], med["nir"] + med["blue"]))):
        chans.append(layer)
        names.append(name)

    out = np.dstack(chans).astype("float32")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE, out)
    print(f"  composite {out.shape}: {', '.join(names)}")
    return out


def main():
    grid = P.load_tile()
    land = P.land_mask(grid)
    meta = json.loads((P.REPO / "data/meta.json").read_text())
    print("building the Sentinel-2 annual composite")
    spec = composite(meta, land.shape)
    stack = T.terrain_stack(T.elevation_on_grid(meta, land.shape), land)
    wet = T.wetness_index(T.elevation_on_grid(meta, land.shape), land)

    # The two surface-class sources: one favours the embedding, one favours the composite.
    micro_tessera, _ = __import__("cfas.regime", fromlist=["cluster_regimes"]).cluster_regimes(
        grid, k=P.MICRO_K, seed=P.MICRO_SEED)
    micro_spectral = partition(spec, land, P.MICRO_K, "kmeans", seed=P.MICRO_SEED).labels

    priors = {}
    for space, feats in (("embedding", grid), ("spectral", spec), ("terrain", stack)):
        for k in KS:
            priors[f"{space}/k{k}"] = partition(feats, land, k, "kmeans")

    out = {"config": {"year": YEAR, "max_cloud": MAX_CLOUD, "n_scenes": MAX_SCENES,
                      "bands": list(BANDS), "channels": int(spec.shape[-1]),
                      "ks": list(KS), "hazard_seeds": list(HAZARD_SEEDS),
                      "n_reports": N_REPORTS, "report_noise": NOISE, "tau_min": TAU},
           "results": {}, "paired": {}}

    for tag, micro in (("surface classes from the embedding", micro_tessera),
                       ("surface classes from the composite", micro_spectral)):
        P._MICRO = micro
        hazards = [P.sample_reports(P.flood_field(grid, land, wet, seed=s), land,
                                    n=N_REPORTS, noise=NOISE, seed=s)
                   for s in HAZARD_SEEDS]
        per_seed = {}
        print(f"\n{tag}")
        for name, part in priors.items():
            ks_, effs = [], []
            for reports in hazards:
                m = P.measure_kappa(part.labels, reports, shuffle_trials=TRIALS, seed=7)
                ks_.append(m["kappa_excess_nats"])
                if m.get("usable") and (m.get("p_permutation") or 1) < 0.05:
                    effs.append(P.city_effort(part.areas(), m["tpr"], m["fpr"],
                                              tau_min=TAU)["city_reports_per_hour"])
            per_seed[name] = ks_
            out["results"].setdefault(tag, {})[name] = {
                "kappa_excess_nats": [round(float(np.mean(ks_)), 4),
                                      round(float(np.std(ks_)), 4)],
                "city_reports_per_hour": round(float(np.mean(effs)), 3) if effs else None}
            print(f"  {name:<16} kappa* {np.mean(ks_):.3f} +/- {np.std(ks_):.3f}  "
                  f"R {out['results'][tag][name]['city_reports_per_hour']}", flush=True)

        pairs = {}
        for a, b in (("embedding/k6", "spectral/k6"), ("embedding/k12", "spectral/k12"),
                     ("spectral/k6", "terrain/k6")):
            x, y = np.array(per_seed[a]), np.array(per_seed[b])
            d = x - y
            pairs[f"{a} - {b}"] = {
                "mean_nats": round(float(d.mean()), 4),
                "effect_over_noise": round(float(abs(d.mean()) / d.std(ddof=1)), 3),
                "wilcoxon_p": round(float(stats.wilcoxon(x, y).pvalue), 5)}
            print(f"    {a} - {b}: {d.mean():+.3f} nats, "
                  f"{abs(d.mean())/d.std(ddof=1):.2f}x noise, p {pairs[f'{a} - {b}']['wilcoxon_p']}")
        out["paired"][tag] = pairs

    (P.RESULTS / "exp8_spectral.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n-> {P.RESULTS / 'exp8_spectral.json'}")


if __name__ == "__main__":
    main()
