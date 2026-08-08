"""Whether Sentinel-1 registers a documented flood over each settlement.

For each settlement this takes a documented flood, finds the first pass on the relative
orbit whose footprint contains the whole tile, and compares it against a per-pixel
baseline built from every dry scene on that same orbit in the same year, which holds the
incidence angle and the look direction fixed.

The statistic is not the count of cells whose backscatter fell. Any scene differs from a
baseline, and a wet scene differs more in both directions at once, so a bare count
registers a flood on any wet day. The statistic is the excess of falling cells over
rising ones, since open water is smooth at C band and darkens while nothing
systematically brightens. The null is every dry scene on the same orbit scored the same
way against the median of the others, which gives a per-settlement reference for what
ordinary date-to-date variation produces.

Every event is drawn from a year other than the embedding year, so that nothing measured
here can have entered the representation's inputs.

Requires network. Writes results/exp11_sar_detectability.json.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import flood as F
import common as P
from cfas.regime import fetch_embedding_tile

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


def main() -> None:
    meta_all = json.load(open(F.CACHE / "city_meta.json"))
    rows = []
    for name, lat, lon in CITIES:
        print(f"\n{name}", flush=True)
        meta = meta_all[name]
        got = fetch_embedding_tile(lat, lon, 2024, dataset_version="v1")
        if got is None:
            print("  tile unavailable")
            continue
        land = P.land_mask(got[0])
        shape = land.shape
        del got

        ev = F.EVENTS[name]
        cands = F.search(meta["bbox_wgs84"], *ev["window"])
        if not cands:
            print("  no pass in the event window")
            continue
        orbit = F.dominant_orbit(
            F.search(meta["bbox_wgs84"], f"{ev['date'][:4]}-01-01", f"{ev['date'][:4]}-12-31"),
            bbox=meta["bbox_wgs84"])
        hit = [f for f in cands if F._orbit(f) == orbit]
        if not hit:
            print(f"  no pass on the dominant orbit {orbit}")
            continue
        try:
            r = F.detectability(meta, shape, land, hit[0]["properties"]["datetime"][:10],
                                orbit)
        except Exception as exc:
            print(f"  failed: {exc!r}")
            continue
        r.update({"city": name, "event": ev["event"],
                  "lag_days": int((np.datetime64(r["event_date"])
                                   - np.datetime64(ev["date"])).astype(int))})
        rows.append(r)
        print(f"  {r['event_date']} lag {r['lag_days']}d  excess-fall "
              f"{r['excess_fall_pp']:+.2f}pp  null {r['null_mean_pp']:+.2f}"
              f"±{r['null_sd_pp']:.2f}  z={r['z']}  detected={r['detected']}", flush=True)

    out = {"method": "excess of falling over rising cells at 4 dB, against a null of "
                     "dry scenes on the same relative orbit",
           "detected_rule": "z > 2 against the dry-scene null",
           "embedding_year_excluded": F.EMBEDDING_YEAR,
           "n_detected": sum(r["detected"] for r in rows), "n_cities": len(rows),
           "cities": rows}
    P.RESULTS.mkdir(parents=True, exist_ok=True)
    (P.RESULTS / "exp11_sar_detectability.json").write_text(json.dumps(out, indent=1))
    print(f"\ndetected in {out['n_detected']} of {out['n_cities']} settlements")


if __name__ == "__main__":
    main()
