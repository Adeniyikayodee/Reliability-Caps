"""Placing a caller on the grid, checked against a tile we build ourselves.

Offline, no keys, no network: a synthetic north-up UTM tile with a known
geotransform, so the right cell for a given lon/lat is known before the code runs.
The geocoder is the only networked part and is left to a live check. The tests
that matter guard the failure that would misplace a warning: a location read into
the wrong cell, or an off-tile location silently snapped onto the grid.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from cfas.locate import Locator, cell_from_lonlat

affine = pytest.importorskip("affine")
Affine = affine.Affine

# A synthetic tile matching the real Lagos one: 10 m pixels, UTM zone 31N, north
# up, 200 x 200. Origin chosen so the tile sits over Lagos (the Ajegunle area).
CRS = "EPSG:32631"
ORIGIN_X, ORIGIN_Y, PIX = 544000.0, 718000.0, 10.0
TRANSFORM = Affine(PIX, 0.0, ORIGIN_X, 0.0, -PIX, ORIGIN_Y)
SHAPE = (200, 200)


def _lonlat_of_pixel(row, col):
    """The lon/lat at the centre of a pixel, computed independently of the code."""
    import pyproj
    from rasterio.transform import xy
    x, y = xy(TRANSFORM, row, col)
    to_ll = pyproj.Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    return to_ll.transform(x, y)


def test_a_lonlat_lands_on_its_own_pixel():
    """Round trip: take a pixel's lon/lat, and the code should return that pixel."""
    for row, col in ((0, 0), (50, 120), (199, 199), (137, 42)):
        lon, lat = _lonlat_of_pixel(row, col)
        got = cell_from_lonlat(lon, lat, TRANSFORM, CRS)
        assert got == (row, col), f"({lon:.5f},{lat:.5f}) -> {got}, expected ({row},{col})"


def test_north_up_orientation_holds():
    """Moving north raises latitude and lowers the row; moving east raises col."""
    lon0, lat0 = _lonlat_of_pixel(100, 100)
    north = cell_from_lonlat(lon0, lat0 + 0.0003, TRANSFORM, CRS)
    east = cell_from_lonlat(lon0 + 0.0003, lat0, TRANSFORM, CRS)
    assert north[0] < 100, "further north is a smaller row"
    assert east[1] > 100, "further east is a larger col"


def test_locator_returns_in_bounds_cells():
    loc = Locator(TRANSFORM, CRS, SHAPE)
    lon, lat = _lonlat_of_pixel(75, 90)
    assert loc.cell(lon, lat) == (75, 90)


def test_a_location_off_the_tile_is_rejected_not_snapped():
    """A caller outside the tile must read as off-tile, never as the nearest edge.

    Snapping an off-tile fix onto a border cell would place a warning on ground
    nobody reported, which is the quiet failure this guards.
    """
    loc = Locator(TRANSFORM, CRS, SHAPE)
    # A point well west and south of the tile origin.
    far_lon, far_lat = _lonlat_of_pixel(0, 0)
    off = loc.cell(far_lon - 0.05, far_lat - 0.05)
    assert off is None


def test_bounds_are_reported_in_lonlat():
    loc = Locator(TRANSFORM, CRS, SHAPE)
    west, south, east, north = loc.bounds_lonlat()
    assert west < east and south < north
    # The synthetic tile sits over Lagos, near 3.4 E, 6.4 N (the Ajegunle area).
    assert 3.3 < west < 3.6 and 6.3 < north < 6.6


def test_geocode_degrades_without_geopy(monkeypatch):
    """A missing geocoder returns None rather than raising, so the loop survives."""
    import builtins
    real_import = builtins.__import__

    def no_geopy(name, *a, **k):
        if name.startswith("geopy"):
            raise ImportError("geopy absent")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_geopy)
    from cfas.locate import geocode
    assert geocode("Ozumba Mbadiwe") is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        try:
            fn()
        except TypeError:
            continue  # skip fixtures when run directly
        print("ok", fn.__name__)
    print("done")
