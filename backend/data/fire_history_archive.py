"""Supplement to CAL FIRE's historic fire perimeters (FRAP).

FRAP (23,334 perimeters, 1878 on) stays the live, authoritative source. This
builds the small set it lacks, from NIFC's InterAgency Fire Perimeter History,
validated the same way as the structure-damage archive:

  * fires_missing: California wildfires in NIFC with no FRAP fire at that place
    that year (prescribed burns excluded; centre on California land; not a
    duplicate of a FRAP fire or of another NIFC record);
  * frap_year_recovered: FRAP fires with no year, where NIFC has the same fire
    (same place, one year), given that year;
  * undated: FRAP fires whose year cannot be recovered;
  * empty_years: years in which no source has any mapped fire.

  python -m data.fire_history_archive build
"""
import datetime as dt
import gzip
import json
import os
import re
import time

import requests

FRAP = ("https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/ArcGIS/rest/services/"
        "California_Historic_Fire_Perimeters/FeatureServer/0/query")
NIFC = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
        "InterAgencyFirePerimeterHistory_All_Years_View/FeatureServer/0/query")
OUT = os.path.join(os.path.dirname(__file__), "fire_history", "supplement.json.gz")
FIRST_YEAR = 1878
_RX = re.compile(r"\b(RX|BURN|PILES?|BROADCAST|UNDERBURN|EXERCISE|PRESCRIBED|HAZARD REDUCTION)\b", re.I)


def _norm(s):
    s = re.sub(r"\b(FIRE|CMPLX|COMPLEX|INCIDENT)\b", " ", str(s or "").upper())
    return re.sub(r"[^A-Z0-9]+", "", re.sub(r"\d+\s*ACRES?", " ", s))


def _get(url, params):
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=120, headers={"User-Agent": "FireScope/1.0"})
            r.raise_for_status()
            d = r.json()
            if "error" in d:
                raise RuntimeError(d["error"])
            return d
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))


def _features(url, where, fields):
    out, off = [], 0
    while True:
        page = _get(url, {"where": where, "outFields": fields, "outSR": 4326, "f": "geojson",
                          "geometryPrecision": 5, "orderByFields": "OBJECTID",
                          "resultOffset": off, "resultRecordCount": 1000})
        feats = [f for f in page.get("features") or [] if f.get("geometry")]
        out += feats
        if len(page.get("features") or []) < 1000:
            return out
        off += 1000


def _bbox(f):
    xs, ys = [], []

    def walk(c):
        if c and isinstance(c[0], (int, float)):
            xs.append(c[0]); ys.append(c[1])
        elif c:
            for x in c:
                walk(x)
    walk(f["geometry"]["coordinates"])
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


def _centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _same_fire(a, ba, b, bb, name_a, name_b):
    """Overlapping boxes, and the same name, mostly the same box (IoU >= 0.5:
    NIFC lists some fires twice, as "Unknown" and unnamed), or a centre inside
    the other's perimeter."""
    from data.zone_resolver import _feature_contains
    if ba[2] < bb[0] or bb[2] < ba[0] or ba[3] < bb[1] or bb[3] < ba[1]:
        return False
    if name_a and name_a == name_b:
        return True
    inter = (min(ba[2], bb[2]) - max(ba[0], bb[0])) * (min(ba[3], bb[3]) - max(ba[1], bb[1]))
    union = (ba[2] - ba[0]) * (ba[3] - ba[1]) + (bb[2] - bb[0]) * (bb[3] - bb[1]) - inter
    if union > 0 and inter / union >= 0.5:
        return True
    (ax, ay), (bx, by) = _centre(ba), _centre(bb)
    return _feature_contains(b, ax, ay) or _feature_contains(a, bx, by)


def build():
    from ml.geo_checks import _county_features
    from data.zone_resolver import _feature_contains
    counties = _county_features()
    if not counties:  # on_ca_land fails open; an archive must not
        raise RuntimeError("California county boundaries unavailable")
    in_ca = lambda x, y: any(_feature_contains(c, x, y) for c in counties)
    this_year = dt.date.today().year
    missing, empty_years, rejected = [], [], {"prescribed": 0, "outside California": 0,
                                               "same as a FRAP fire": 0, "NIFC duplicate": 0, "bad geometry": 0}
    nifc_cache, frap_last = {}, FIRST_YEAR
    for year in range(FIRST_YEAR, this_year + 1):
        frap = _features(FRAP, f"YEAR_={year}", "OBJECTID,FIRE_NAME")
        nifc = _features(NIFC, f"FIRE_YEAR_INT={year} AND UNIT_ID LIKE 'CA%' AND "
                               f"(FEATURE_CA IS NULL OR FEATURE_CA <> 'Prescribed Fire')",
                         "OBJECTID,GlobalID,INCIDENT,FIRE_YEAR_INT,GIS_ACRES,UNIT_ID,AGENCY,IRWINID,DATE_CUR")
        nifc_cache[year] = [(g, gb, _norm(g["properties"].get("INCIDENT"))) for g in nifc if (gb := _bbox(g))]
        fb = [(f, _bbox(f), _norm(f["properties"].get("FIRE_NAME"))) for f in frap]
        kept = []
        for f in nifc:
            p, b = f["properties"], _bbox(f)
            name = _norm(p.get("INCIDENT"))
            if _RX.search(str(p.get("INCIDENT") or "")):
                rejected["prescribed"] += 1; continue
            if not b or not (p.get("GIS_ACRES") or 0) > 0:
                rejected["bad geometry"] += 1; continue
            if not in_ca(*_centre(b)):
                rejected["outside California"] += 1; continue
            if any(_same_fire(f, b, g, gb, name, gn) for g, gb, gn in fb if gb):
                rejected["same as a FRAP fire"] += 1; continue
            if any(_same_fire(f, b, g, gb, name, gn) for g, gb, gn in kept):
                rejected["NIFC duplicate"] += 1; continue
            kept.append((f, b, name))
        for f, b, _ in kept:
            p = f["properties"]
            f["properties"] = {
                "FIRE_NAME": p.get("INCIDENT"), "YEAR_": year, "GIS_ACRES": p.get("GIS_ACRES"),
                "UNIT_ID": p.get("UNIT_ID"), "AGENCY": p.get("AGENCY"), "IRWINID": p.get("IRWINID"),
                "ALARM_DATE": None, "CONT_DATE": None, "CAUSE": None,
                "SOURCE": "NIFC InterAgency Fire Perimeter History", "SOURCE_ID": p.get("GlobalID"),
            }
            missing.append(f)
        if frap:
            frap_last = year
        if not frap and not kept:
            empty_years.append(year)
        print(f"{year}: FRAP {len(frap)}, NIFC {len(nifc)}, added {len(kept)}", flush=True)

    # FRAP fires with no year: recover the year when NIFC has the same fire in exactly one year.
    # Only NIFC records not already matched to a dated FRAP fire: otherwise the
    # undated FRAP record is a copy of that dated one and must not appear twice.
    recovered, undated, used, missing_ids = [], [], set(), {id(f) for f in missing}
    for f in _features(FRAP, "YEAR_ IS NULL OR YEAR_ = 0",
                       "OBJECTID,FIRE_NAME,GIS_ACRES,CAUSE,AGENCY,UNIT_ID,INC_NUM,ALARM_DATE,CONT_DATE"):
        b, name = _bbox(f), _norm(f["properties"].get("FIRE_NAME"))
        hits = [(y, g) for y, feats in nifc_cache.items() for g, gb, gn in feats
                if id(g) in missing_ids and _same_fire(f, b, g, gb, name, gn)] if b else []
        years = {y for y, _ in hits}
        f["properties"]["SOURCE"] = "CAL FIRE FRAP"
        f["properties"]["SOURCE_ID"] = f["properties"].get("OBJECTID")
        if len(years) == 1:
            f["properties"]["YEAR_"] = years.pop()
            f["properties"]["YEAR_SOURCE"] = "recovered from NIFC InterAgency Fire Perimeter History"
            recovered.append(f)
            used |= {id(g) for _, g in hits}  # that NIFC record is this fire, not a separate missing one
        else:
            undated.append(f)

    missing = [f for f in missing if id(f) not in used]
    data = {"built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "sources": {"frap": FRAP, "nifc": NIFC},
            "fires_missing": missing, "frap_year_recovered": recovered, "undated": undated,
            # years after FRAP's latest are unpublished, not empty
            "empty_years": [y for y in empty_years if y <= frap_last
                            and not any(r["properties"]["YEAR_"] == y for r in recovered)],
            "rejected": rejected}
    body = json.dumps(data, separators=(",", ":"), sort_keys=True).encode()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    old = load()
    if old and {k: old.get(k) for k in ("fires_missing", "frap_year_recovered", "undated", "empty_years")} == \
            {k: data[k] for k in ("fires_missing", "frap_year_recovered", "undated", "empty_years")}:
        print("unchanged"); return False
    with gzip.GzipFile(OUT, "wb", mtime=0) as f:
        f.write(body)
    print(f"added {len(missing)} missing fires, recovered {len(recovered)} years, {len(undated)} undated, "
          f"empty years {data['empty_years']}, rejected {rejected} -> {OUT}")
    return True


_MEMO = {}


def load():
    if not os.path.exists(OUT):
        return None
    m = os.path.getmtime(OUT)
    if _MEMO.get("m") != m:
        with gzip.open(OUT, "rb") as f:
            _MEMO.update(m=m, data=json.loads(f.read()))
    return _MEMO["data"]


def features_for(year_from, year_to, min_acres=0):
    """Supplement features (missing fires + year-recovered FRAP fires) for a year range."""
    d = load() or {}
    return [f for f in d.get("fires_missing", []) + d.get("frap_year_recovered", [])
            if year_from <= (f["properties"].get("YEAR_") or 0) <= year_to
            and (f["properties"].get("GIS_ACRES") or 0) >= min_acres]


if __name__ == "__main__":
    import sys
    if sys.argv[1:] != ["build"]:
        sys.exit("usage: python -m data.fire_history_archive build")
    build()
