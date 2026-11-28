"""Real-world probability for the live map (Architecture v2, step A5e).

The models are trained on a case-control sample (fires plus chosen controls), so
their probability is relative to that sample's fire share rho. The standard
correction (arXiv 1312.6481; King & Zeng) shifts only the intercept:

    logit(p_real) = logit(p_model) + logit(pi) - logit(rho)

pi = historical chance that a wildfire starts in one 0.1-degree map tile on one day,
per region (ml.gates.region_of) and calendar month, from USFS FPA FOD ignitions in
development years only (the locked test years are excluded).

  python -m ml.absolute_risk build --fpa FPA_FOD.sqlite --out ml/models/base_rates_v0.json
"""
import argparse
import json
import sqlite3

import numpy as np
import pandas as pd

from ml import gates

LOCKED_YEARS = (2024, 2025)
DEV_YEARS = (2001, 2023)


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def to_real(p_model, region, month, rates, rho):
    """Intercept-corrected probability. rates = the built JSON's 'pi'."""
    pi = np.array([rates.get(r, {}).get(str(int(m)), np.nan) for r, m in zip(region, month)])
    z = _logit(p_model) + _logit(pi) - _logit(rho)
    return 1 / (1 + np.exp(-z))


def _elevations(points):
    """USGS 3DEP 10 m elevation for many points via Earth Engine, 500 per call."""
    from ml.fire_danger import _ee, _reduce
    ee = _ee()
    img = ee.ImageCollection("USGS/3DEP/10m_collection").mosaic().select("elevation")
    out = np.full(len(points), np.nan)
    for c0 in range(0, len(points), 500):
        chunk = [(i, la, lo) for i, (la, lo) in enumerate(points[c0:c0 + 500], start=c0)]
        res = _reduce(img, chunk, 30)
        for i, _, _ in chunk:
            v = res.get(str(i), {}).get("elevation")
            out[i] = np.nan if v is None else v
    return out


def build(fpa, out):
    from ml.geo_checks import on_ca_land
    tiles = [(round(32.5 + i * 0.1, 2), round(-124.0 + j * 0.1, 2)) for i in range(96) for j in range(101)]
    tiles = [t for t in tiles if on_ca_land(*t)]
    t_region = gates.region_of(np.array([t[0] for t in tiles]), np.nan_to_num(_elevations(tiles), nan=0.0))
    q = pd.read_sql(f"select FIRE_YEAR, DISCOVERY_DATE, LATITUDE, LONGITUDE from Fires where STATE='CA' "
                    f"and FIRE_YEAR between {DEV_YEARS[0]} and {DEV_YEARS[1]}", sqlite3.connect(fpa))
    q["month"] = pd.to_datetime(q.DISCOVERY_DATE - 2440587.5, unit="D").dt.month
    q["region"] = gates.region_of(q.LATITUDE.values,
                                  np.nan_to_num(_elevations(list(zip(q.LATITUDE, q.LONGITUDE))), nan=0.0))
    years = DEV_YEARS[1] - DEV_YEARS[0] + 1
    days = {m: pd.Period(f"2001-{m:02d}").days_in_month for m in range(1, 13)}
    n_tiles = pd.Series(t_region).value_counts().to_dict()
    counts = q.groupby(["region", "month"]).size()
    pi = {}
    for r, nt in n_tiles.items():
        pi[r] = {}
        for m in range(1, 13):
            ign = int(counts.get((r, m), 0))
            # tile-days with >= 1 start ~ ignitions (two starts in one tile-day are rare at 100 km2 x 1 day)
            pi[r][str(m)] = round(max(ign, 0.5) / (nt * days[m] * years), 8)
    meta = {"pi": pi, "tiles_per_region": n_tiles, "ignitions": int(len(q)), "years": list(DEV_YEARS),
            "unit": "chance a wildfire starts in one 0.1-degree tile on one day",
            "source": "USFS FPA FOD v7 (RDS-2013-0009.7), development years only"}
    with open(out, "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({r: {k: v for k, v in list(m.items())[5:9]} for r, m in pi.items()}, indent=1))
    print(f"{len(q)} ignitions, {len(tiles)} land tiles -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--fpa", required=True)
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.fpa, a.out)
