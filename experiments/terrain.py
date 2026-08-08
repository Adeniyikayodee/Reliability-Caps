"""Elevation, and the classical hydrological prior built from it.

A conventional catchment pipeline has nothing to stand on in a settlement whose public
elevation data disagrees with itself by more than the ground rises. That is a testable
claim rather than an assumption, so the terrain prior is built here and scored beside the
learned one on the same ground.

Copernicus GLO-30 is a digital *surface* model at 30 m, so over a dense settlement it
carries roof heights as well as ground. That is a limitation of the only free elevation
product with global coverage, and it is the limitation a practitioner in an ungauged
city actually faces, which is the reason to score it rather than to exclude it.

Three fields are derived and cached:

    elevation     metres, resampled to the 10 m TESSERA grid
    hand          metres above the nearest standing water, using TESSERA's own nodata
                  mask as the drainage network. A crude height-above-nearest-drainage,
                  and crude is the honest form of it at this resolution.
    slope         local gradient, which is what separates a fill platform from a
                  drainage line when absolute height is unreliable.
"""
from __future__ import annotations

import pathlib

import numpy as np
from scipy import ndimage

CACHE = pathlib.Path(__file__).resolve().parent / "cache"
COG = ("https://copernicus-dem-30m.s3.amazonaws.com/"
       "Copernicus_DSM_COG_10_N06_00_E003_00_DEM/"
       "Copernicus_DSM_COG_10_N06_00_E003_00_DEM.tif")


def _tile_url(lat_floor: int, lon_floor: int) -> str:
    """Copernicus tiles are named for their south-west corner, in whole degrees."""
    ns = f"N{lat_floor:02d}" if lat_floor >= 0 else f"S{-lat_floor:02d}"
    ew = f"E{lon_floor:03d}" if lon_floor >= 0 else f"W{-lon_floor:03d}"
    name = f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"
    return f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


def elevation_on_grid(meta: dict, shape: tuple[int, int], *,
                      cache_name: str = "elevation_10m") -> np.ndarray:
    """Copernicus GLO-30 elevation resampled onto a tile's 10 m grid. Cached.

    A 0.1 degree embedding tile usually sits inside one 1 degree elevation tile and
    occasionally straddles two or four, so every tile covering the bounding box is
    reprojected in turn and the result filled where it is still empty. Tiles that do not
    exist, which is the case over open water, are skipped rather than treated as an
    error, since a coastal city will legitimately have some.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{cache_name}.npy"
    if out.exists():
        return np.load(out)

    import math
    import rasterio
    from rasterio.warp import Resampling, reproject, transform_bounds
    from affine import Affine

    transform = Affine(*meta["transform"][:6])
    h, w = shape
    bounds = (transform.c, transform.f + transform.e * h,
              transform.c + transform.a * w, transform.f)
    west, south, east, north = transform_bounds(meta["crs"], "EPSG:4326", *bounds)

    dst = np.zeros(shape, "float32")
    filled = np.zeros(shape, bool)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_USE_HEAD="NO"):
        for la in range(math.floor(south), math.floor(north) + 1):
            for lo in range(math.floor(west), math.floor(east) + 1):
                tmp = np.zeros(shape, "float32")
                try:
                    with rasterio.open(f"/vsicurl/{_tile_url(la, lo)}") as src:
                        reproject(source=rasterio.band(src, 1), destination=tmp,
                                  src_transform=src.transform, src_crs=src.crs,
                                  dst_transform=transform, dst_crs=meta["crs"],
                                  resampling=Resampling.bilinear)
                except Exception:
                    continue                      # no tile here, which over sea is normal
                take = (~filled) & (tmp != 0)
                dst[take] = tmp[take]
                filled |= take
    if not filled.any():
        raise RuntimeError("no Copernicus coverage for this tile")
    np.save(out, dst)
    return dst


def terrain_stack(elev: np.ndarray, land: np.ndarray) -> np.ndarray:
    """(H, W, 3) of elevation, height above nearest water, and slope.

    The height above nearest water uses the elevation at the closest cell TESSERA
    declined to embed, which is the nearest standing water. `distance_transform_edt`
    returns those indices directly, so no flow routing is involved, which is deliberate:
    routing over a surface model in a settlement this flat is the step the design
    argument says produces noise.
    """
    _, (iy, ix) = ndimage.distance_transform_edt(land, return_indices=True)
    hand = elev - elev[iy, ix]
    gy, gx = np.gradient(elev.astype("float32"), 10.0)
    slope = np.hypot(gy, gx)
    return np.dstack([elev, hand, slope]).astype("float32")


def wetness_index(elev: np.ndarray, land: np.ndarray) -> np.ndarray:
    """A standardised propensity to hold water: low ground, low above drainage, flat.

    Used as one component of the flood driver in the label model, so that part of the
    hazard is set by real measured relief rather than by a synthetic field. Its sign is
    fixed by hydrology: water collects where the ground is low, close to drainage, and
    slack enough not to shed it.
    """
    st = terrain_stack(elev, land)
    z = lambda a: (a - a[land].mean()) / (a[land].std() + 1e-9)
    return (-z(st[:, :, 0]) - z(st[:, :, 1]) - 0.5 * z(st[:, :, 2])) / 2.5
