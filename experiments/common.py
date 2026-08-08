"""Shared machinery for the experiments.

Three things live here that the deployed code deliberately does not carry.

**Exogenous flood fields.** `calibrate.py` in the deployed system generates its ground truth by
declaring that the three largest regimes flood. That is circular: the partition under
evaluation is also the partition that generated the labels, so any partition scores
well on its own labels and the measured kappa is a property of the label model rather
than of the representation. For a comparison between candidate priors the flood field
has to be drawn from a process none of them can see. `flood_field` does that, from
proximity to standing water and a spatially correlated residual, both defined on the
grid rather than on any partition. The circular model is kept as `flood_field_circular`
because it is the one the deployed system reports, and quoting the two side by side is
the honest way to report an upper bound next to an estimate.

**Candidate partitions.** A kappa in isolation says nothing. It is interpretable
against the alternatives available to somebody building the same system: a regular
tessellation of matched granularity, a random partition of matched part sizes, and a
partition cut directly from the variable that drives the labels, which is the ceiling.

**Safe calibration.** `cfas.waggle` holds TPR and FPR as module globals, and
`alpha_cap` reads them, so thresholds computed before a calibration is installed silently
mix two calibrations. `calibrated` installs a pair and restores the previous one.
"""
from __future__ import annotations

import contextlib
import math
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from cfas import waggle
from cfas.generalise import _counts_to_metrics, _informedness, leave_one_out
from cfas.regime import cluster_regimes

TILE = REPO / "global_0.1_degree_representation/2024/grid_3.35_6.45"
CELL_M = 10.0
CELL_KM2 = (CELL_M / 1000.0) ** 2
RESULTS = pathlib.Path(__file__).resolve().parent / "results"


# ------------------------------------------------------------------ the tile
def load_tile() -> np.ndarray:
    """The Ajegunle 2024 TESSERA tile, dequantised to float32 (H, W, 128)."""
    q = np.load(TILE / "grid_3.35_6.45.npy")
    s = np.load(TILE / "grid_3.35_6.45_scales.npy")
    return (q.astype("float32") * s[:, :, None]).astype("float32")


def land_mask(grid: np.ndarray) -> np.ndarray:
    """True where TESSERA embedded the cell. Its nodata is the water mask."""
    flat = grid.reshape(-1, grid.shape[-1])
    ok = np.isfinite(flat).all(1) & ~(flat == 0).all(1)
    return ok.reshape(grid.shape[:2])


# ------------------------------------------------------------- flood fields
def _standardise(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / (a.std() + 1e-9)


MICRO_K = 24          # surface classes the hazard follows, finer than any prior tested
MICRO_SEED = 99


_MICRO: np.ndarray | None = None


def micro_classes(grid: np.ndarray) -> np.ndarray:
    """The latent surface classes the hazard follows. Cut once and reused.

    Fixed across hazard realisations on purpose: the ground does not change between
    storms, only which classes happen to flood does.
    """
    global _MICRO
    if _MICRO is None:
        _MICRO, _ = cluster_regimes(grid, k=MICRO_K, seed=MICRO_SEED)
    return _MICRO


def surface_driver(grid: np.ndarray, land: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Flood propensity as a function of latent surface class.

    The design hypothesis behind a learned prior is that ground of the same character
    behaves alike: reclaimed sand, stilt settlement, hardstanding and lagoon margin do
    not respond to the same rain in the same way. Encoding that hypothesis means
    drawing the hazard from a surface classification, and the honest way to do it is at
    a granularity finer than anything under test, with propensities assigned at random,
    so a candidate partition has to recover structure it was not given.
    """
    micro = micro_classes(grid)
    rng = np.random.default_rng(seed)
    propensity = rng.standard_normal(MICRO_K)
    field = np.where(micro >= 0, propensity[np.clip(micro, 0, None)], 0.0)
    return _standardise(ndimage.gaussian_filter(field.astype("float32"), 3.0))


def spatial_driver(land: np.ndarray, *, corr_km: float = 0.8,
                   seed: int = 0) -> np.ndarray:
    """Everything the hazard depends on that nothing in this study observes.

    Drainage that was never mapped, culverts that are blocked this week, a wall built
    last year. Smooth at the scale things vary within a settlement, and invisible to
    every candidate prior including the terrain one.
    """
    rng = np.random.default_rng(seed + 7919)
    sigma_px = corr_km * 1000.0 / CELL_M
    return _standardise(ndimage.gaussian_filter(rng.standard_normal(land.shape), sigma_px))


def flood_field(grid: np.ndarray, land: np.ndarray, wetness: np.ndarray, *,
                p_flood: float = 0.40, share_terrain: float = 0.35,
                share_surface: float = 0.35, seed: int = 0) -> np.ndarray:
    """A flood extent built from three named parts, in stated proportions.

    Real relief carries the first part, latent surface class the second, and unobserved
    local factors the rest. Each candidate prior can see at most one of the three, and
    the oracle partition sees the sum, so no prior is given the answer and none is
    denied a route to it. The shares are the experiment: they say how much of the
    hazard is expressed in the ground's appearance as against its shape, and which
    prior is worth building depends entirely on where a city sits on that axis.

    Shares are variances and must not exceed one; the remainder is unobserved.
    """
    if share_terrain + share_surface > 1.0:
        raise ValueError("terrain and surface shares exceed the total variance")
    share_unobs = 1.0 - share_terrain - share_surface
    z = (math.sqrt(share_terrain) * _standardise(np.where(land, wetness, 0.0))
         + math.sqrt(share_surface) * surface_driver(grid, land, seed=seed)
         + math.sqrt(share_unobs) * spatial_driver(land, seed=seed))
    cut = np.quantile(z[land], 1 - p_flood)
    return (z > cut) & land


def flood_driver(grid: np.ndarray, land: np.ndarray, wetness: np.ndarray, *,
                 share_terrain: float = 0.35, share_surface: float = 0.35,
                 seed: int = 0) -> np.ndarray:
    """The continuous field behind `flood_field`, for cutting the oracle partition."""
    share_unobs = 1.0 - share_terrain - share_surface
    return (math.sqrt(share_terrain) * _standardise(np.where(land, wetness, 0.0))
            + math.sqrt(share_surface) * surface_driver(grid, land, seed=seed)
            + math.sqrt(share_unobs) * spatial_driver(land, seed=seed))


def flood_field_circular(labels: np.ndarray, areas: dict[int, float],
                         n_flooding: int = 3) -> np.ndarray:
    """The label model the deployed calibrator uses: the largest regimes flood.

    Retained for comparability with the published number, and reported as an upper
    bound because the partition under test is also the partition that drew the labels.
    """
    order = sorted(areas, key=lambda i: -areas[i])[:n_flooding]
    return np.isin(labels, order)


def sample_reports(truth: np.ndarray, land: np.ndarray, *, n: int = 2000,
                   noise: float = 0.20, seed: int = 0):
    """Residents' marks: a sample of cells, each read with probability `noise` of error.

    Sampling is uniform over land. A real ledger is not uniform, and §Limitations says
    so; uniformity keeps the comparison between partitions clean, because a sampling
    bias correlated with the field would advantage whichever partition happens to align
    with the bias.
    """
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(land)
    idx = rng.choice(len(ys), min(n, len(ys)), replace=False)
    out = []
    for i in idx:
        r, c = int(ys[i]), int(xs[i])
        t = bool(truth[r, c])
        out.append((r, c, (not t) if rng.random() < noise else t))
    return out


# --------------------------------------------------------------- partitions
@dataclass(frozen=True)
class Partition:
    """A candidate prior: an integer map with -1 for cells it declines to describe."""
    name: str
    labels: np.ndarray
    k: int

    def areas(self) -> dict[int, float]:
        ids, counts = np.unique(self.labels[self.labels >= 0], return_counts=True)
        return {int(i): float(n) * CELL_KM2 for i, n in zip(ids, counts)}


def tessera_partition(grid: np.ndarray, k: int, *, seed: int = 0,
                      dims: int | None = None) -> Partition:
    """k-means over TESSERA embeddings, optionally over a leading prefix of dimensions.

    The prefix is the Matryoshka question: TESSERA orders its dimensions by information,
    so a 16-dimensional prefix is a cheaper prior, and whether it costs anything that
    matters is answerable in the same units as everything else here.
    """
    g = grid if dims is None else grid[:, :, :dims]
    labels, _ = cluster_regimes(g, k=k, seed=seed)
    name = "TESSERA-128" if dims is None else f"TESSERA-{dims}"
    return Partition(name, labels.astype("int32"), k)


def tessellation_partition(land: np.ndarray, k: int) -> Partition:
    """A regular grid of roughly k equal blocks over the land mask.

    This is the operational status quo in most cities: a warning is issued for a ward,
    and a ward is a polygon drawn for administration rather than for hydrology. It has
    the same granularity as the learned map and none of its content, so the difference
    between the two is what the representation contributes.
    """
    h, w = land.shape
    rows = int(round(math.sqrt(k)))
    cols = int(math.ceil(k / rows))
    ri = np.minimum((np.arange(h) * rows) // h, rows - 1)
    ci = np.minimum((np.arange(w) * cols) // w, cols - 1)
    lab = (ri[:, None] * cols + ci[None, :]).astype("int32")
    lab[~land] = -1
    return Partition("tessellation", lab, int(len(np.unique(lab[lab >= 0]))))


def shuffled_partition(p: Partition, seed: int = 0) -> Partition:
    """The permutation control: identical part sizes, membership severed from the ground."""
    rng = np.random.default_rng(seed)
    lab = p.labels.copy()
    idx = np.nonzero(lab >= 0)
    vals = lab[idx].copy()
    rng.shuffle(vals)
    lab[idx] = vals
    return Partition("random", lab, p.k)


def terrain_partition(stack: np.ndarray, land: np.ndarray, k: int,
                      *, seed: int = 0) -> Partition:
    """k-means over elevation, height above nearest water, and slope.

    The classical hydrological prior, and the one a practitioner reaches for first. It
    is scored here on the same ground and in the same units as the learned prior, which
    is the only way to settle whether free global relief is worth anything in a
    settlement this flat.
    """
    z = stack.reshape(-1, stack.shape[-1]).astype("float32")
    ok = land.reshape(-1)
    zz = (z[ok] - z[ok].mean(0)) / (z[ok].std(0) + 1e-6)
    from cfas.regime import kmeans
    rng = np.random.default_rng(seed)
    fit = zz if len(zz) <= 50_000 else zz[rng.choice(len(zz), 50_000, replace=False)]
    _, centres = kmeans(fit, k, seed=seed)
    d2 = (zz ** 2).sum(1)[:, None] - 2 * zz @ centres.T + (centres ** 2).sum(1)[None, :]
    lab = np.full(len(z), -1, "int32")
    lab[ok] = np.argmin(d2, 1)
    return Partition("terrain", lab.reshape(land.shape), k)


def driver_partition(driver: np.ndarray, land: np.ndarray, k: int) -> Partition:
    """Quantiles of the field that generated the labels. The ceiling, not a baseline.

    A partition cut from the driver itself is the best any grouping of this granularity
    could do, so it turns kappa from a number into a fraction of what was available.
    """
    edges = np.quantile(driver[land], np.linspace(0, 1, k + 1)[1:-1])
    lab = np.digitize(driver, edges).astype("int32")
    lab[~land] = -1
    return Partition("driver-oracle", lab, k)


# ------------------------------------------------------------ kappa, safely
def measure_kappa(labels: np.ndarray, reports, *, shuffle_trials: int = 200,
                  seed: int = 1) -> dict:
    """Leave-one-out TPR and FPR, kappa, and informedness against a permutation control.

    The control shuffles report-to-part membership while holding the part sizes, which
    is the null that matters: informedness survives nothing but a grouping that really
    separates wet ground from dry, whereas recall survives any grouping at all when
    floods are common.
    """
    skill = leave_one_out(labels, reports)
    tpr, fpr = skill.recall, skill.false_alarm
    if not tpr or not fpr or tpr <= fpr:
        # A partition whose held-out prediction is no better than its own base rate
        # supports no positive evidence at all, so its ceiling is zero and its critical
        # rate is unbounded. Recording that as kappa = 0 rather than as a missing value
        # keeps it in the averages, where it belongs: a prior that fails on two seeds
        # in five has not earned the mean of the three it survived.
        return {"usable": False, "reason": "no skill above base rate",
                "tpr": tpr, "fpr": fpr, "kappa_nats": 0.0, "kappa_excess_nats": 0.0,
                "kappa_null_mean": 0.0, "p_permutation": 1.0, "informedness_j": 0.0,
                "gap_over_shuffle": 0.0, "counts": skill.counts,
                "reach_km2": skill.reach_km2, "n_reports": skill.n_reports,
                "city_reports_per_hour": None}

    pts = [(int(r), int(c), bool(f)) for r, c, f in reports]
    on = [(r, c, f) for r, c, f in pts
          if 0 <= r < labels.shape[0] and 0 <= c < labels.shape[1] and labels[r, c] >= 0]
    ids = np.array([int(labels[r, c]) for r, c, _ in on])
    truths = np.array([f for _, _, f in on])
    rng = np.random.default_rng(seed)
    js, kappas = [], []
    for _ in range(shuffle_trials):
        sh = ids.copy()
        rng.shuffle(sh)
        c = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for g in np.unique(sh):
            m = np.nonzero(sh == g)[0]
            if len(m) < 2:
                continue
            for i in m:
                mates = m[m != i]
                warned = truths[mates].sum() > len(mates) / 2
                t = bool(truths[i])
                c["tp" if t else "fp"] += 1 if warned else 0
                c["fn" if t else "tn"] += 0 if warned else 1
        j = _informedness(c)
        if j is not None:
            js.append(j)
        st, sf = _counts_to_metrics(c)[0], _counts_to_metrics(c)[2]
        kappas.append(math.log(st / sf) if st and sf and st > sf else 0.0)

    real_j = _informedness(skill.counts)
    kappa = math.log(tpr / fpr)
    # A permutation that preserves part sizes but severs membership still scores above
    # zero, because leave-one-out majority voting on a finite sample is biased upward
    # whenever the base rate is away from a half. The excess over the permuted mean is
    # therefore the quantity to quote, and the one-sided p-value says whether a prior
    # cleared its own null at all. Quoting the raw kappa credits a random grouping with
    # evidence it does not carry, which is exactly the error the control exists to catch.
    null_mean = float(np.mean(kappas)) if kappas else 0.0
    p = (1 + sum(1 for x in kappas if x >= kappa)) / (1 + len(kappas)) if kappas else None
    return {"usable": True, "tpr": round(tpr, 4), "fpr": round(fpr, 4),
            "kappa_nats": round(kappa, 4),
            "kappa_null_mean": round(null_mean, 4),
            "kappa_null_sd": round(float(np.std(kappas)), 4) if kappas else None,
            "kappa_excess_nats": round(max(0.0, kappa - null_mean), 4),
            "p_permutation": round(p, 4) if p is not None else None,
            "kappa_neg_nats": round(math.log((1 - tpr) / (1 - fpr)), 4)
            if fpr < 1 and tpr < 1 else None,
            "informedness_j": round(real_j, 4),
            "shuffle_j_mean": round(float(np.mean(js)), 4) if js else None,
            "shuffle_j_sd": round(float(np.std(js)), 4) if js else None,
            "gap_over_shuffle": round(real_j - float(np.mean(js)), 4) if js else None,
            "n_reports": skill.n_reports, "n_parts_tested": skill.n_regimes_tested,
            "reach_km2": skill.reach_km2, "counts": skill.counts}


@contextlib.contextmanager
def alpha_margin(margin: float):
    """Install a different safety factor on the alpha cap for the duration of a block.

    The cap holds the escalation boundary of a small part strictly above the weight of
    one report, and the margin says by how much. Every part small enough for the cap to
    bind then contributes the same fixed amount to a city's required reporting rate, so
    this one constant sets the whole rising arm of the effort curve against granularity.
    Sweeping it is the check that the granularity result is about the representation and
    not about a number chosen once in the deployed system.
    """
    prev = waggle.ALPHA_MARGIN
    waggle.ALPHA_MARGIN = margin
    try:
        yield
    finally:
        waggle.ALPHA_MARGIN = prev


@contextlib.contextmanager
def calibrated(tpr: float, fpr: float):
    """Install a calibration for the duration of a block, then put back what was there.

    `alpha_cap` reads the module globals, so thresholds read before installation would
    be derived from a different pair than the evidence per report. That mixture is
    silent, and its visible symptom is a small part whose bar drops under the weight of
    one report, which the single-report invariant forbids.
    """
    prev = (waggle.TPR, waggle.FPR)
    waggle.TPR, waggle.FPR = tpr, fpr
    try:
        yield
    finally:
        waggle.TPR, waggle.FPR = prev


# ---------------------------------------------------- decidability of a city
def critical_interval(area_km2: float, *, tau_min: float = 30.0,
                      band: str = "HIGH") -> float:
    """Longest mean gap between reports that still lets this part reach its threshold.

    Requires a calibration to be installed; see `calibrated`.
    """
    a, _ = waggle.thresholds(area_km2)
    lam = waggle.llr(band)
    if lam >= a:
        return math.inf
    return -tau_min * math.log(1 - lam / a)


def city_effort(areas: dict[int, float], tpr: float, fpr: float, *,
                tau_min: float = 30.0, band: str = "HIGH") -> dict:
    """Total reporting rate the whole city needs, and the per-part detail.

    The quantity a city cares about is not whether one zone can be warned. It is how
    many calls per hour the settlement as a whole has to produce for every zone to be
    capable of warning, which is the sum of the per-part critical rates. Finer
    partitions make each part easier and there are more of them, so this sum is where
    the granularity trade actually shows up.
    """
    with calibrated(tpr, fpr):
        lam = waggle.llr(band)
        cap = waggle.alpha_cap(band=band)
        parts = []
        for pid, area in sorted(areas.items()):
            a, _ = waggle.thresholds(area)
            dt = critical_interval(area, tau_min=tau_min, band=band)
            parts.append({"id": pid, "area_km2": round(area, 3),
                          "threshold_nats": round(a, 4),
                          "single_report_nats": round(lam, 4),
                          "capped": bool(waggle.alpha_for(area) >= cap - 1e-12),
                          "reports_to_cross": math.ceil(a / lam) if lam > 0 else None,
                          "critical_interval_min": None if math.isinf(dt) else round(dt, 2)})
    total_per_hour = sum(60.0 / p["critical_interval_min"] for p in parts
                         if p["critical_interval_min"])
    # A part whose boundary sits at the cap contributes a fixed amount to this sum, the
    # same amount for every such part, so the count is what says how much of the total is
    # the representation and how much is the cap. It is reported rather than buried.
    return {"parts": parts, "n_parts": len(parts),
            "n_capped": sum(p["capped"] for p in parts),
            "capped_share_of_rate": round(
                sum(60.0 / p["critical_interval_min"] for p in parts
                    if p["capped"] and p["critical_interval_min"]) / total_per_hour, 4)
            if total_per_hour else None,
            "city_reports_per_hour": round(total_per_hour, 3),
            "single_report_nats": round(lam, 4)}


def decidable_at(areas: dict[int, float], tpr: float, fpr: float,
                 interval_min: float, *, tau_min: float = 30.0) -> int:
    """How many parts can ever warn when unsolicited reports arrive this far apart."""
    with calibrated(tpr, fpr):
        return sum(1 for area in areas.values()
                   if critical_interval(area, tau_min=tau_min) >= interval_min)
