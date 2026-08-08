"""The city sorted into ground that behaves alike.

A TESSERA tile over Ajegunle is 1106 x 1107 x 128: one and a quarter
million ten-metre observations, each a 128-dimensional read of what the surface
actually is. cfas/legacy/risk.py:136 averages all of it into a single vector and bands the
result, which is a fair description of a village and a useless one for a city.
Ojota floods while Ikeja, eight kilometres off, stays dry, and one number cannot
hold both.

The obvious repair is a catchment map, and Lagos will not give you one. There is
no public drain network, and the elevation data disagrees with itself by more than
the city rises, Elmoustafa et al. (2015) found ~6 m of disagreement across a
Lekki catchment holding 20 m of relief, enough to reverse a creek's computed flow
direction. Route water over that and you are modelling noise. See docs/lagos.md.

So the zone here is not ground that *drains* together. It is ground that
*behaves* alike: reclaimed sand, stilt settlement, hardstanding, lagoon margin.
TESSERA sees that directly from Sentinel-1 and Sentinel-2 at 10 m, with no
elevation model in the path and nothing to lie about. Cluster the embeddings and
the regimes fall out, learned, not modelled.

What the satellite cannot tell you is where the water actually goes, and that is
the half the city knows. Each regime carries a rainfall threshold the call-in
ledger fills in from evidence: this cluster floods at 40 mm, that one does not.
The prior comes from orbit; the posterior comes from people (cfas/hazard.py).

TESSERA is annual, so a regime map is a slow, stable thing, cut it once a season,
not once a storm.

Pure below the fetch: numpy and the standard library. The clustering takes an
array, so it tests offline against ground we build ourselves.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TESSERA_CELL_M = 10.0     # TESSERA's native resolution
# Regimes. k=6 is the default, chosen against three criteria and re-verified on
# the Ajegunle pilot tile. The inertia elbow (kneedle) sits at 6; seed-to-seed
# stability is sound (ARI ~0.76); and, the constraint that actually binds, every
# regime stays large enough to collect call-ins (on Ajegunle the smallest is
# ~8 km2). At k>=9 the smallest regime falls below ~2 km2 and may never reach the
# reports it needs to learn a threshold, so k=8 is a hard ceiling and 6 is the
# pick, fewer, larger regimes each calibrate faster in a cold-start season, for
# almost no loss of fit. Rerun choose_k() on a new city; it is tile-specific.
DEFAULT_K = 6
FIT_SAMPLE = 50_000       # cells to fit on; assignment still covers every cell


@dataclass(frozen=True)
class Regime:
    """One flood regime: ground the satellite says behaves alike."""
    id: int
    cells: int
    area_km2: float
    fraction: float                 # share of the tile
    centroid: np.ndarray            # the 128-d signature of this regime
    threshold_mm: float | None = None   # rainfall that floods it; learned, not known

    @property
    def label(self) -> str:
        return f"regime-{self.id:02d}"

    @property
    def calibrated(self) -> bool:
        """False until call-ins have taught this regime its threshold."""
        return self.threshold_mm is not None


def fetch_embedding_tile(lat, lon, year, dataset_version: str = "v1"):
    """The full TESSERA tile with its geotransform: (grid, transform, crs).

    Returns the (H, W, 128) grid, the affine transform that maps a pixel to a
    projected coordinate, and the tile's CRS (UTM for Lagos, EPSG:32631). None on
    failure. GeoTessera hands back all three; the archived rural
    legacy/risk.fetch_embedding() kept only the mean of the grid and dropped the
    rest. The transform and crs are exactly what place a caller on the grid, so
    keeping them is what makes geolocation possible (see cfas/locate.py).
    """
    try:
        from geotessera import GeoTessera
        # The dataset version is pinned by the caller rather than defaulted, because a
        # tile fetched under a different version is a different prior, and the whole
        # evidence calibration downstream is a property of the prior. An unpinned fetch
        # is how the map silently changes underneath a published kappa.
        emb, crs, transform = GeoTessera(
            dataset_version=dataset_version).fetch_embedding(lon=lon, lat=lat, year=year)
        return np.asarray(emb, "float32"), transform, crs
    except Exception:
        return None


def fetch_embedding_grid(lat, lon, year, dataset_version: str = "v1"):
    """The full (H, W, 128) TESSERA tile, or None on failure.

    The grid alone, for callers that only cluster. Use fetch_embedding_tile when
    you also need the geotransform to map locations to cells.
    """
    tile = fetch_embedding_tile(lat, lon, year, dataset_version)
    return None if tile is None else tile[0]


def valid_cells(flat):
    """Which rows are real observations, as against TESSERA's nodata fill.

    TESSERA is a *land* model, so it declines to embed water and returns an
    all-zero vector there, not NaN. On the Ajegunle pilot tile that is a few
    percent (the harbour creeks); on a coastal tile like Victoria Island it can be
    nearly half (the Atlantic and lagoon).
    A plain isfinite() check passes every one of them.

    Left in, the fill does four things, all quiet: it eats a whole cluster, it
    drags the standardisation toward zero for every real cell, it halves every
    regime's reported share of the tile, and it corrupts the inertia curve that
    choose_k() exists to show you.

    This is also the water mask, free. There is no need to detect water from the
    embeddings or fetch a surface-water layer to subtract, the foundation model's
    own coverage is the answer, and ground it declines to describe is ground that
    is already flood.
    """
    flat = np.asarray(flat)
    return np.isfinite(flat).all(1) & ~(flat == 0).all(1)


def _standardise(x):
    """Zero mean, unit variance per dimension.

    TESSERA dimensions carry different scales, and k-means measures plain
    Euclidean distance, so without this the widest dimension quietly becomes the
    only one that votes.
    """
    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True)
    return (x - mu) / (sd + 1e-6), mu, sd


def _kmeans_plusplus(x, k, rng):
    """k-means++ seeding: spread the first centres out, proportional to distance.

    Random seeding on a tile this size reliably drops two centres inside the
    lagoon, it is the largest uniform thing in the frame, and leaves the built
    fabric sharing one. Seeding by distance is what stops the interesting ground
    being collapsed into a single regime.
    """
    n = len(x)
    centres = [x[rng.integers(n)]]
    d2 = ((x - centres[0]) ** 2).sum(1)
    for _ in range(1, k):
        total = d2.sum()
        if total <= 0:                       # every point already sits on a centre
            centres.append(x[rng.integers(n)])
            continue
        i = int(rng.choice(n, p=d2 / total))
        centres.append(x[i])
        d2 = np.minimum(d2, ((x - centres[-1]) ** 2).sum(1))
    return np.stack(centres)


def kmeans(x, k, *, iters=50, seed=0, tol=1e-4):
    """Lloyd's algorithm with k-means++ seeding. Pure numpy, no new dependency.

    Returns (labels, centres). Deterministic for a given seed, which matters: a
    regime map that reshuffles its own ids between runs cannot be calibrated,
    because yesterday's threshold would land on today's different ground.
    """
    x = np.asarray(x, "float32")
    if len(x) < k:
        raise ValueError(f"need at least k={k} points, got {len(x)}")
    rng = np.random.default_rng(seed)
    centres = _kmeans_plusplus(x, k, rng)

    labels = np.zeros(len(x), "int32")
    for _ in range(iters):
        # Assign: nearest centre, by squared distance expanded to avoid an
        # (n, k, d) intermediate that would not fit for a full tile.
        d2 = (x ** 2).sum(1)[:, None] - 2 * x @ centres.T + (centres ** 2).sum(1)[None, :]
        new = np.argmin(d2, 1).astype("int32")
        if np.array_equal(new, labels):
            break
        labels = new
        shift = 0.0
        for j in range(k):
            hit = labels == j
            if not hit.any():
                # An empty cluster: re-seed it on the worst-fit point rather than
                # dropping it, so k regimes in means k regimes out.
                centres[j] = x[np.argmax(d2.min(1))]
                continue
            nc = x[hit].mean(0)
            shift = max(shift, float(np.abs(nc - centres[j]).max()))
            centres[j] = nc
        if shift < tol:
            break
    return labels, centres


def cluster_regimes(grid, k=DEFAULT_K, *, seed=0, cell_m=TESSERA_CELL_M,
                    fit_sample=FIT_SAMPLE):
    """A TESSERA grid to a regime map. Pure; no network.

    `grid` is (H, W, 128). Returns (labels (H, W), regimes). Fitting runs on a
    random subsample for tractability, a full tile is 1.2M x 128, but every
    cell is assigned, so the map is complete and only the centres are estimated.
    """
    grid = np.asarray(grid, "float32")
    if grid.ndim != 3:
        raise ValueError(f"grid must be (H, W, D), got {grid.shape}")
    h, w, d = grid.shape
    flat = grid.reshape(-1, d)

    valid = valid_cells(flat)
    if not valid.any():
        raise ValueError("no observed cells in the grid: all nodata or non-finite")

    z, mu, sd = _standardise(flat[valid])
    rng = np.random.default_rng(seed)
    fit = z if len(z) <= fit_sample else z[rng.choice(len(z), fit_sample, replace=False)]
    _, centres = kmeans(fit, k, seed=seed)

    # Assign every cell against the centres the sample fitted, in blocks so a
    # full tile does not need an (n, k) distance matrix all at once.
    labels_valid = np.empty(len(z), "int32")
    for i in range(0, len(z), 200_000):
        blk = z[i:i + 200_000]
        d2 = (blk ** 2).sum(1)[:, None] - 2 * blk @ centres.T + (centres ** 2).sum(1)[None, :]
        labels_valid[i:i + 200_000] = np.argmin(d2, 1)

    labels = np.full(h * w, -1, "int32")     # -1 is nodata: water, and TESSERA knows
    labels[valid] = labels_valid
    labels = labels.reshape(h, w)

    cell_km2 = (cell_m / 1000.0) ** 2
    total = int(valid.sum())
    regimes = []
    for j in range(k):
        n = int((labels_valid == j).sum())
        if n == 0:
            continue
        # Report the centroid in the embedding's own units, not standardised
        # space, so it stays comparable to a raw TESSERA vector.
        regimes.append(Regime(id=j, cells=n, area_km2=round(n * cell_km2, 4),
                              fraction=round(n / total, 4),
                              centroid=(centres[j] * sd[0] + mu[0]).astype("float32")))
    return labels, regimes


def choose_k(grid, ks=(4, 6, 8, 10, 12), *, seed=0, fit_sample=20_000):
    """Inertia per k, for picking the number of regimes with eyes open.

    There is no true k. Too few and Makoko shares a regime with Ikoyi; too many
    and each regime holds too few call-ins to ever learn a threshold. The elbow
    is a judgement, and the binding constraint is usually calibration, not fit:
    pick the largest k whose smallest regime still collects reports.
    """
    grid = np.asarray(grid, "float32")
    flat = grid.reshape(-1, grid.shape[-1])
    flat = flat[valid_cells(flat)]      # the fill would flatten the curve it draws
    z, _, _ = _standardise(flat)
    rng = np.random.default_rng(seed)
    if len(z) > fit_sample:
        z = z[rng.choice(len(z), fit_sample, replace=False)]

    out = {}
    for k in ks:
        labels, centres = kmeans(z, k, seed=seed)
        out[k] = float(((z - centres[labels]) ** 2).sum(1).mean())
    return out


def calibrate_thresholds(regimes, labels, reports, *, min_reports=3):
    """Teach each regime the rainfall that floods it, from call-ins.

    This is the half the satellite cannot supply. `reports` is an iterable of
    (row, col, rain_mm, flooded), a call-in located on the grid, the rainfall
    that preceded it, and whether the caller's hazard cleared the alert band.

    A regime's threshold is the lowest rainfall that actually flooded it. That is
    deliberately the optimistic edge: for a warning, missing a flood costs more
    than warning early, and the calibrator (cfas/generalise.py) is what tells you
    when that trade has gone too far in one direction.

    Regimes with fewer than `min_reports` confirmations keep threshold None, and
    None must read as "unknown", never as "safe".
    """
    floods = {}
    for row, col, rain_mm, flooded in reports:
        if not flooded:
            continue
        if not (0 <= row < labels.shape[0] and 0 <= col < labels.shape[1]):
            continue
        rid = int(labels[row, col])
        if rid < 0:
            continue                      # a cell with no embedding
        floods.setdefault(rid, []).append(float(rain_mm))

    out = []
    for r in regimes:
        seen = floods.get(r.id, [])
        thr = round(min(seen), 1) if len(seen) >= min_reports else None
        out.append(Regime(id=r.id, cells=r.cells, area_km2=r.area_km2,
                          fraction=r.fraction, centroid=r.centroid,
                          threshold_mm=thr))
    return out
