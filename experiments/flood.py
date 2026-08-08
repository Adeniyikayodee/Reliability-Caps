"""Observed flood extents, from Sentinel-1, on the same grid as the embeddings.

Every hazard extent in the first version of this study was synthetic. That is the
weakest thing about it, and no amount of care in the sweep answers the objection, so
this module replaces the generated field with one that was measured while water was on
the ground.

**What is measured.** Sentinel-1 radiometrically terrain corrected gamma0, VV
polarisation, at 10 m, from the Planetary Computer's `sentinel-1-rtc` collection. Open
water is smooth at C band and reflects away from the sensor, so it returns far less
energy than any land surface. A flood is therefore a fall in backscatter against the
same ground on the same orbit at a dry date, and detecting it is a comparison rather
than a classification.

**Why the comparison is against a per-pixel baseline and not against a single dry
scene.** Backscatter over a settlement varies with wind, soil moisture and the exact
incidence angle, so one dry scene carries the weather of the day it was taken. The
baseline here is the per-pixel median over every dry scene in the same relative orbit,
which holds the viewing geometry fixed and averages the weather out.

**Two thresholds, both stated.** A cell is flooded when the event scene falls below
`WATER_DB` in absolute terms and sits more than `DROP_DB` below its own baseline. The
first condition alone would mark permanently smooth ground such as airport aprons; the
second alone would mark any surface that merely dried out. `sensitivity()` recomputes
the extent across a grid of both, because a hazard field
that moves under its own thresholds would not support a comparison between priors.

**The coupling that had to be designed out.** TESSERA is trained on Sentinel-1 and
Sentinel-2 over a full year, so a flood inside the embedding year is a flood the learned
prior has partly seen. Every event used here is from a year other than the embedding
year, which is 2024. That is the reason `EVENTS` carries no 2024 date even where a
well-documented 2024 flood exists, and it is the difference between an exogenous hazard
and a leak.

**What this does not see.** Water standing between buildings often raises backscatter
instead of lowering it, because a wall and a flat water surface make a double bounce
straight back to the sensor. So this detects open floodwater on streets, yards and open
ground, and under-reports water inside the densest built fabric. The consequence is a
hazard field that misses part of the true extent. It is not a consequence that favours
any one candidate prior, which is what the comparison needs, and `Limitations` says so.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import pathlib
import urllib.request

import numpy as np

CACHE = pathlib.Path(__file__).resolve().parent / "cache"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = ("https://planetarycomputer.microsoft.com/api/sas/v1/token/"
       "sentinel1euwestrtc/sentinel1-grd-rtc")

WATER_DB = -15.0        # absolute gamma0 VV below which C band sees open water
DROP_DB = 3.0           # fall against the dry baseline required as well
MIN_BASELINE = 6        # dry scenes needed before a per-pixel median is worth having
EMBEDDING_YEAR = 2024   # no event may come from this year; see the module docstring

# Documented urban flood events, one per settlement, each outside the embedding year.
# `window` is the search span handed to the archive, wide enough to catch the first
# usable pass after the water arrived and narrow enough not to catch the next storm.
EVENTS = {
    "Lagos, Ajegunle":     {"date": "2022-07-10", "window": ("2022-07-05", "2022-07-25"),
                            "event": "July 2022 Lagos rainy-season flooding"},
    "Accra, Odaw":         {"date": "2023-06-18", "window": ("2023-06-15", "2023-07-05"),
                            "event": "June 2023 Odaw basin flooding"},
    "Nairobi, Mathare":    {"date": "2020-04-29", "window": ("2020-04-25", "2020-05-15"),
                            "event": "April to May 2020 Nairobi river flooding"},
    "Kampala, Bwaise":     {"date": "2022-05-05", "window": ("2022-05-01", "2022-05-20"),
                            "event": "May 2022 Lubigi channel flooding"},
    "Dhaka, Korail":       {"date": "2020-07-20", "window": ("2020-07-15", "2020-08-05"),
                            "event": "July 2020 Bangladesh monsoon flooding"},
    "Jakarta, Kp. Melayu": {"date": "2020-01-01", "window": ("2020-01-01", "2020-01-20"),
                            "event": "1 January 2020 Ciliwung flood"},
    "Manila, Marikina":    {"date": "2020-11-12", "window": ("2020-11-12", "2020-11-30"),
                            "event": "Typhoon Vamco, Marikina river record stage"},
    "Karachi, Lyari":      {"date": "2020-08-27", "window": ("2020-08-25", "2020-09-15"),
                            "event": "August 2020 Karachi record monthly rainfall"},
}

_TOKEN: str | None = None


# --------------------------------------------------------------- the archive
def _token() -> str:
    """A read token for the RTC container. Anonymous, and good for about an hour."""
    global _TOKEN
    if _TOKEN is None:
        with urllib.request.urlopen(SAS, timeout=60) as r:
            _TOKEN = json.load(r)["token"]
    return _TOKEN


def _post(body: dict) -> dict:
    req = urllib.request.Request(STAC, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def search(bbox, start: str, end: str, limit: int = 500) -> list[dict]:
    """Every RTC scene intersecting the box between two dates, newest last."""
    out, body = [], {"collections": ["sentinel-1-rtc"], "bbox": list(bbox),
                     "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 100}
    while True:
        page = _post(body)
        out.extend(page.get("features", []))
        nxt = [l for l in page.get("links", []) if l.get("rel") == "next"]
        if not nxt or len(out) >= limit:
            break
        body = nxt[0]["body"]
    out.sort(key=lambda f: f["properties"]["datetime"])
    return out


def _orbit(f: dict) -> tuple:
    p = f["properties"]
    return (p.get("sat:relative_orbit"), p.get("sat:orbit_state"))


def covers(f: dict, bbox) -> bool:
    """Whether a scene's own footprint contains the whole tile.

    A track that clips the tile is worse than useless here, because the missing corner
    is not missing at random: it is the same corner every pass, so it would drop the same
    ground from the baseline and from the event alike.
    """
    from shapely.geometry import box, shape
    try:
        return shape(f["geometry"]).contains(box(*bbox))
    except Exception:
        return False


def dominant_orbit(features: list[dict], bbox=None) -> tuple:
    """The track that visits this tile most often and sees all of it when it does.

    Holding the relative orbit fixed holds the incidence angle and the look direction
    fixed with it, which is what makes a difference between two dates a difference in
    the ground rather than in the geometry. Kampala is the reason for the footprint
    test: its most frequent track cuts the tile in half and never covers it once.
    """
    usable = [f for f in features if bbox is None or covers(f, bbox)]
    if not usable:
        usable = features
    counts: dict[tuple, int] = {}
    for f in usable:
        counts[_orbit(f)] = counts.get(_orbit(f), 0) + 1
    return max(counts, key=counts.get)


# ------------------------------------------------------------ onto the grid
def read_on_grid(href: str, meta: dict, shape: tuple[int, int]) -> np.ndarray:
    """One VV asset reprojected onto a tile's own 10 m grid, in dB. NaN off-scene."""
    import rasterio
    from affine import Affine
    from rasterio.warp import Resampling, reproject

    transform = Affine(*meta["transform"][:6])
    dst = np.full(shape, np.nan, "float32")
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_USE_HEAD="NO", GDAL_HTTP_MAX_RETRY="3",
                      GDAL_HTTP_RETRY_DELAY="2"):
        with rasterio.open("/vsicurl/" + href + "?" + _token()) as src:
            reproject(source=rasterio.band(src, 1), destination=dst,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=meta["crs"],
                      src_nodata=src.nodata if src.nodata is not None else 0.0,
                      dst_nodata=np.nan, resampling=Resampling.bilinear)
    out = np.where(dst > 0, dst, np.nan)
    return (10.0 * np.log10(out, where=np.isfinite(out),
                            out=np.full(shape, np.nan, "float32"))).astype("float32")


def _stack(features: list[dict], meta: dict, shape: tuple[int, int],
           note: str = "") -> np.ndarray:
    frames = []
    for i, f in enumerate(features):
        try:
            frames.append(read_on_grid(f["assets"]["vv"]["href"], meta, shape))
            print(f"      {note}{i + 1}/{len(features)} "
                  f"{f['properties']['datetime'][:10]}", flush=True)
        except Exception as exc:                       # one bad scene is not a failure
            print(f"      {note}{i + 1}/{len(features)} skipped: {exc!r}"[:140], flush=True)
    if not frames:
        raise RuntimeError("no usable scenes")
    return np.stack(frames)


# ------------------------------------------------------------- the detector
def water_series(meta: dict, shape: tuple[int, int], land: np.ndarray,
                 start: str, end: str, *, orbit: tuple | None = None,
                 months: tuple[int, ...] | None = None,
                 decimate: int = 8) -> list[dict]:
    """How much of this tile looked like open water on every pass in a span.

    Reading at a coarse scale is enough to rank dates, and ranking dates is the point.
    A published account of a flood tells you when people were in the water; it does not
    tell you whether a satellite was overhead while the water was still there. This
    puts the choice of event scene on the radar record instead of on the news archive.
    """
    from affine import Affine

    t = Affine(*meta["transform"][:6])
    coarse = {"transform": [t.a * decimate, t.b, t.c, t.d, t.e * decimate, t.f],
              "crs": meta["crs"], "bbox_wgs84": meta["bbox_wgs84"]}
    cshape = (shape[0] // decimate, shape[1] // decimate)
    cland = land[:cshape[0] * decimate, :cshape[1] * decimate]
    cland = cland.reshape(cshape[0], decimate, cshape[1], decimate).mean((1, 3)) > 0.5

    rows = []
    for f in search(meta["bbox_wgs84"], start, end):
        if orbit is not None and _orbit(f) != orbit:
            continue
        date = f["properties"]["datetime"][:10]
        if months and int(date[5:7]) not in months:
            continue
        try:
            db = read_on_grid(f["assets"]["vv"]["href"], coarse, cshape)
        except Exception:
            continue
        ok = np.isfinite(db) & cland
        if ok.sum() < 0.5 * cland.sum():
            continue
        rows.append({"date": date, "orbit": list(_orbit(f)),
                     "water_share": round(float(((db < WATER_DB) & ok).sum() / ok.sum()), 4)})
        print(f"      {date} {str(_orbit(f)):18} water {rows[-1]['water_share']:.4f}",
              flush=True)
    return rows


def observed_flood(city: str, meta: dict, shape: tuple[int, int], land: np.ndarray, *,
                   water_db: float = WATER_DB, drop_db: float = DROP_DB,
                   refresh: bool = False) -> dict:
    """The flood extent of this city's event, as a boolean array on its own grid.

    Returns the mask, the dry baseline, the event scene, and the provenance needed to
    reproduce or to argue with any of it.
    """
    slug = city.split(",")[0].lower().replace(" ", "_")
    out = CACHE / f"flood_{slug}.npz"
    if out.exists() and not refresh:
        z = np.load(out, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        d["provenance"] = json.loads(str(d["provenance"]))
        return d

    ev = EVENTS[city]
    if ev["date"][:4] == str(EMBEDDING_YEAR):
        raise ValueError(f"{city}: event is inside the embedding year")

    bbox = meta["bbox_wgs84"]
    year = int(ev["date"][:4])
    # A baseline drawn from the same calendar year keeps the built fabric fixed, and a
    # window either side of the event keeps the season broadly comparable.
    dry = search(bbox, f"{year}-01-01", f"{year}-12-31")
    orbit = dominant_orbit(dry)
    dry = [f for f in dry if _orbit(f) == orbit]

    hit = search(bbox, ev["window"][0], ev["window"][1])
    hit = [f for f in hit if _orbit(f) == orbit]
    if not hit:
        raise RuntimeError(f"{city}: no scene on orbit {orbit} inside the event window")
    post = hit[0]
    post_date = post["properties"]["datetime"][:10]

    # Everything on that track except the event itself and the fortnight around it.
    d0 = _dt.date.fromisoformat(post_date)
    dry = [f for f in dry
           if abs((_dt.date.fromisoformat(f["properties"]["datetime"][:10]) - d0).days) > 14]
    if len(dry) < MIN_BASELINE:
        raise RuntimeError(f"{city}: only {len(dry)} dry scenes on orbit {orbit}")

    print(f"    orbit {orbit}, event {post_date}, {len(dry)} dry scenes", flush=True)
    base = np.nanmedian(_stack(dry, meta, shape, "dry "), axis=0)
    ev_db = read_on_grid(post["assets"]["vv"]["href"], meta, shape)

    permanent = base < water_db
    flooded = (ev_db < water_db) & ((base - ev_db) > drop_db) & land & ~permanent
    flooded &= np.isfinite(ev_db) & np.isfinite(base)

    prov = {"city": city, "event": ev["event"], "event_date": post_date,
            "requested_date": ev["date"], "orbit": list(orbit),
            "n_dry_scenes": len(dry), "water_db": water_db, "drop_db": drop_db,
            "flooded_km2": round(float(flooded.sum()) * 0.0001, 3),
            "flooded_share_of_land": round(float(flooded.sum() / max(land.sum(), 1)), 4),
            "permanent_water_km2": round(float((permanent & land).sum()) * 0.0001, 3)}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, flooded=flooded, baseline_db=base.astype("float32"),
                        event_db=ev_db.astype("float32"),
                        provenance=json.dumps(prov))
    return {"flooded": flooded, "baseline_db": base, "event_db": ev_db,
            "provenance": prov}


def detectability(meta: dict, shape: tuple[int, int], land: np.ndarray,
                  event_date: str, orbit: tuple, *, thr_db: float = 4.0) -> dict:
    """Whether the radar saw the flood at all, against a null of its own dry scenes.

    A fall in backscatter is not by itself evidence of water. Any scene differs from the
    baseline, and a rainy scene differs more than a dry one in both directions at once,
    so a bare count of falling cells will find a flood on any wet day. The statistic that
    separates water from weather is the *excess* of falling cells over rising ones, since
    open water darkens and nothing systematically brightens.

    The null is built by scoring every dry scene on the same track the same way, against
    the median of the others. If the event does not stand outside that spread, the
    instrument did not see the flood, and no threshold applied to the same image will
    change that.
    """
    import datetime as _d

    feats = [f for f in search(meta["bbox_wgs84"], f"{event_date[:4]}-01-01",
                               f"{event_date[:4]}-12-31") if _orbit(f) == orbit]
    ev = [f for f in feats if f["properties"]["datetime"][:10] == event_date]
    if not ev:
        raise RuntimeError(f"no scene on {event_date} for orbit {orbit}")
    d0 = _d.date.fromisoformat(event_date)
    dry = [f for f in feats
           if abs((_d.date.fromisoformat(f["properties"]["datetime"][:10]) - d0).days) > 14]
    if len(dry) < MIN_BASELINE:
        raise RuntimeError(f"only {len(dry)} dry scenes on orbit {orbit}")

    stack = _stack(dry, meta, shape, "dry ")
    ev_db = read_on_grid(ev[0]["assets"]["vv"]["href"], meta, shape)

    # A pass whose footprint clips the tile is dropped rather than allowed to erase the
    # mask. Requiring every scene to be finite at every cell is what left Manila with no
    # cells at all on the first run.
    keep = [i for i in range(len(stack))
            if (np.isfinite(stack[i]) & land).sum() > 0.9 * land.sum()]
    if len(keep) < MIN_BASELINE:
        raise RuntimeError(f"only {len(keep)} dry scenes cover the tile on orbit {orbit}")
    stack = stack[keep]
    base_all = np.nanmedian(stack, axis=0)
    ok = np.isfinite(ev_db) & np.isfinite(base_all) & land
    if ok.sum() < 0.5 * land.sum():
        raise RuntimeError("the event scene does not cover the tile")

    def excess(residual):
        m = ok & np.isfinite(residual)
        r = residual - np.median(residual[m])
        return (((r > thr_db) & m).sum() - ((r < -thr_db) & m).sum()) / m.sum() * 100

    null = [excess(np.nanmedian(np.delete(stack, i, axis=0), axis=0) - stack[i])
            for i in range(len(stack))]
    score = excess(base_all - ev_db)
    dry = [dry[i] for i in keep]
    mu, sd = float(np.mean(null)), float(np.std(null))
    return {"event_date": event_date, "orbit": list(orbit), "n_dry_scenes": len(dry),
            "threshold_db": thr_db, "excess_fall_pp": round(score, 3),
            "null_mean_pp": round(mu, 3), "null_sd_pp": round(sd, 3),
            "null_max_pp": round(float(max(null)), 3),
            "z": round((score - mu) / sd, 2) if sd > 0 else None,
            "detected": bool(sd > 0 and (score - mu) / sd > 2.0)}


def sensitivity(cached: dict, land: np.ndarray,
                waters=(-14.0, -15.0, -16.0, -17.0),
                drops=(2.0, 3.0, 4.0)) -> list[dict]:
    """The same extent recomputed across the two thresholds that define it.

    A hazard field is only a fair basis for comparing priors if it is a property of the
    water and not of the two numbers chosen to find it. This is the check.
    """
    base, ev = cached["baseline_db"], cached["event_db"]
    rows = []
    for w in waters:
        for d in drops:
            m = (ev < w) & ((base - ev) > d) & land & ~(base < w)
            m &= np.isfinite(ev) & np.isfinite(base)
            rows.append({"water_db": w, "drop_db": d,
                         "flooded_km2": round(float(m.sum()) * 0.0001, 3),
                         "share": round(float(m.sum() / max(land.sum(), 1)), 4)})
    return rows
