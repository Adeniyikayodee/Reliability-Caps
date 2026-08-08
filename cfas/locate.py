"""Place a caller on the grid.

A call-in has two halves, a report and a place. `cfas/hazard.py` reads the report;
this module reads the place. It turns a location into a `(row, col)` cell in the
TESSERA regime grid, using the geotransform the tile carries, so that
`cfas/assimilate.py` can drop the report into the right regime.

The geotransform is the piece the earlier code threw away. GeoTessera returns a
tile as `(grid, transform, crs)`: an affine transform that maps a pixel to a
projected coordinate, and the tile's CRS, which for Lagos is UTM zone 31N
(EPSG:32631). `cfas/regime.fetch_embedding_tile()` now keeps both, and the two
steps below use them:

    lon, lat (WGS84)  ->  x, y (the tile's CRS)  ->  row, col (pixel)

The reprojection runs on pyproj, and the pixel step on rasterio's affine inverse.
Both are offline; only the geocoder, which turns a place name into a lon/lat, uses
the network. The coordinate maths is pure and testable against a synthetic tile.

A location the demo accepts:

    a place name          "Boundary, Ajegunle"               (geocoded)
    a lon/lat fix         from a phone that shares its position
    a raw cell            for testing and for a dispatcher clicking the map
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@lru_cache(maxsize=8)
def _transformer(crs_str):
    """A cached WGS84-to-tile-CRS transformer. always_xy gives (lon, lat) order in."""
    import pyproj
    return pyproj.Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)


def cell_from_lonlat(lon, lat, transform, crs):
    """A (lon, lat) in WGS84 to a (row, col) pixel in the tile. Pure; no network.

    Reprojects the point into the tile's CRS, then applies the affine inverse to
    land on a pixel. The result may fall outside the tile; the caller checks
    bounds (see `Locator.cell`).
    """
    from rasterio.transform import rowcol
    x, y = _transformer(str(crs)).transform(lon, lat)
    row, col = rowcol(transform, x, y)
    return int(row), int(col)


def geocode(place, *, city="Lagos", country="Nigeria"):
    """A place name to (lon, lat) through OpenStreetMap Nominatim. Network.

    Tries the finest query first and widens it. Returns None when nothing matches
    or geopy is absent. The city and country scope the search so that a short
    landmark name resolves inside Lagos rather than somewhere else on earth.
    """
    try:
        from geopy.geocoders import Nominatim
    except ImportError:
        return None
    geo = Nominatim(user_agent="cfas/1.0")
    for query in (f"{place}, {city}, {country}", f"{place}, {country}", place):
        try:
            loc = geo.geocode(query, timeout=10)
        except Exception:
            return None
        if loc:
            return loc.longitude, loc.latitude
    return None


@dataclass(frozen=True)
class Locator:
    """Maps a location to a cell on one tile. Built from a fetched tile.

    Hold the transform, the crs, and the grid shape from
    `cfas.regime.fetch_embedding_tile()`, and this resolves a lon/lat or a place
    name to an in-bounds `(row, col)`, or None when the place lies off the tile.
    """
    transform: object            # affine.Affine, from the fetched tile
    crs: object                  # rasterio CRS or an EPSG string
    shape: tuple                 # (H, W) of the grid

    def cell(self, lon, lat):
        """A lon/lat fix to an in-bounds cell, or None if it lies off the tile."""
        row, col = cell_from_lonlat(lon, lat, self.transform, self.crs)
        if 0 <= row < self.shape[0] and 0 <= col < self.shape[1]:
            return row, col
        return None

    def place(self, name, **kw):
        """A place name to an in-bounds cell, geocoded, or None. Network."""
        ll = geocode(name, **kw)
        if ll is None:
            return None
        return self.cell(*ll)

    def bounds_lonlat(self):
        """The tile's (west, south, east, north) in lon/lat, for a sanity check."""
        import pyproj
        from rasterio.transform import xy
        h, w = self.shape
        to_ll = pyproj.Transformer.from_crs(str(self.crs), "EPSG:4326", always_xy=True)
        corners = [xy(self.transform, r, c) for r, c in ((0, 0), (h - 1, w - 1))]
        lons, lats = zip(*(to_ll.transform(x, y) for x, y in corners))
        return min(lons), min(lats), max(lons), max(lats)
