"""CAL FIRE structure-damage archive (DINS), kept the way the training data is:
validated, one file per incident year in the repo, refreshed by a daily sync.

Source: CAL FIRE POSTFIRE_MASTER_DATA_SHARE (Damage INSpection), point records
from 2013-08-07 on (no earlier point-level structure damage exists). Inspectors
record every structure they check, so ~41% of source records are "No Damage";
the map shows only fire-damaged structures, and each incident keeps its full
inspection tally (including undamaged and inaccessible) in `inspection_summary`.

  python -m data.dins_archive backfill --start 2013 --end 2025
  python -m data.dins_archive sync          # current and previous year
"""
import argparse
import datetime as dt
import gzip
import json
import os
import time

import requests

SOURCE = ("https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
          "POSTFIRE_MASTER_DATA_SHARE/FeatureServer/0/query")
ARCHIVE_DIR = os.path.join(os.path.dirname(__file__), "dins")
FIRST_YEAR = 2013
DAMAGED = ("Destroyed (>50%)", "Major (25-50%)", "Minor (10-25%)", "Affected (>0-10%)")
FIELDS = ("OBJECTID,GLOBALID,DAMAGE,HAZARDTYPE,INCIDENTNAME,INCIDENTNUM,INCIDENTSTARTDATE,"
          "SITEADDRESS,STREETNUMBER,STREETNAME,STREETTYPE,CITY,COUNTY,"
          "STRUCTURETYPE,STRUCTURECATEGORY,YEARBUILT,"
          "WHATDIDFIRESTARTFROM,WHEREFIRESTARTEDONSTRUCTURE,DEFENSIVEACTIONS,LATITUDE,LONGITUDE")
CA_BBOX = (32.4, 42.1, -124.6, -114.0)  # lat_min, lat_max, lon_min, lon_max


def _year_where(year):
    return (f"INCIDENTSTARTDATE >= DATE '{year}-01-01' AND "
            f"INCIDENTSTARTDATE < DATE '{year + 1}-01-01'")


def _get(params):
    for attempt in range(4):
        try:
            r = requests.get(SOURCE, params={**params, "f": params.get("f", "json")}, timeout=60,
                             headers={"User-Agent": "FireScope/1.0"})
            r.raise_for_status()
            d = r.json()
            if "error" in d:
                raise RuntimeError(d["error"])
            return d
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def _damaged_where(year):
    damage = ",".join(f"'{d}'" for d in DAMAGED)
    return f"HAZARDTYPE = 'Fire' AND DAMAGE IN ({damage}) AND {_year_where(year)}"


def _valid(props):
    """Reason a record is rejected, or None."""
    if props.get("HAZARDTYPE") != "Fire":
        return "not a fire"
    if props.get("DAMAGE") not in DAMAGED:
        return "no damage recorded"
    lat, lon = props.get("LATITUDE"), props.get("LONGITUDE")
    if lat is None or lon is None or not (CA_BBOX[0] <= lat <= CA_BBOX[1] and CA_BBOX[2] <= lon <= CA_BBOX[3]):
        return "coordinates outside California"
    if not props.get("INCIDENTNAME") or not props.get("INCIDENTNUM") or not props.get("INCIDENTSTARTDATE"):
        return "missing incident name, number or start date"
    return None


def fetch_year(year):
    """Fire-damaged structures for one incident year, validated and de-duplicated,
    plus every incident's full inspection tally. Raises if the result doesn't add
    up to the source's own count."""
    features, offset = [], 0
    while True:
        page = _get({"where": _damaged_where(year), "outFields": FIELDS, "outSR": "4326", "f": "geojson",
                     "orderByFields": "OBJECTID", "resultOffset": offset, "resultRecordCount": 2000})
        feats = page.get("features") or []
        features.extend(feats)
        if len(feats) < 2000:
            break
        offset += 2000
    expected = _get({"where": _damaged_where(year), "returnCountOnly": "true"})["count"]
    if len(features) != expected:
        raise RuntimeError(f"{year}: fetched {len(features)} records, source reports {expected}")

    kept, rejected, seen = [], {}, set()
    for f in features:
        p = f.get("properties") or {}
        why = _valid(p)
        if not why and p.get("GLOBALID") in seen:
            why = "duplicate record"
        if why:
            rejected[why] = rejected.get(why, 0) + 1
            continue
        seen.add(p.get("GLOBALID"))
        p["INCIDENT_YEAR"] = year
        kept.append(f)

    stats = _get({"where": f"HAZARDTYPE = 'Fire' AND {_year_where(year)}",
                  "groupByFieldsForStatistics": "INCIDENTNUM,INCIDENTNAME,DAMAGE",
                  "outStatistics": json.dumps([{"statisticType": "count", "onStatisticField": "OBJECTID",
                                                "outStatisticFieldName": "n"}])})
    summary = {}
    for row in stats.get("features", []):
        a = row["attributes"]
        inc = summary.setdefault(a["INCIDENTNUM"], {"name": a["INCIDENTNAME"], "counts": {}})
        inc["counts"][a["DAMAGE"] or "Unrecorded"] = a["n"]
    return {"type": "FeatureCollection", "year": year, "source": SOURCE,
            "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "source_count": expected, "rejected": rejected,
            "inspection_summary": summary, "features": kept}


def path_for(year):
    return os.path.join(ARCHIVE_DIR, f"dins_{year}.json.gz")


def write_year(year):
    data = fetch_year(year)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    body = json.dumps(data, separators=(",", ":"), sort_keys=True).encode()
    p = path_for(year)
    old = None
    if os.path.exists(p):
        with gzip.open(p, "rb") as f:
            old = json.loads(f.read())
    # Only rewrite when the records changed, so a no-op sync makes no commit.
    if old and old.get("features") == data["features"] and old.get("inspection_summary") == data["inspection_summary"]:
        print(f"{year}: unchanged ({len(data['features'])} damaged structures)")
        return False
    with gzip.GzipFile(p, "wb", mtime=0) as f:
        f.write(body)
    print(f"{year}: {len(data['features'])} damaged structures, rejected {data['rejected'] or 'none'}, "
          f"{len(data['inspection_summary'])} incidents -> {p}")
    return True


def load_year(year):
    """The archived year, or None when there is no file for it."""
    p = path_for(year)
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rb") as f:
        return json.loads(f.read())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill")
    b.add_argument("--start", type=int, default=FIRST_YEAR)
    b.add_argument("--end", type=int, default=dt.date.today().year)
    sub.add_parser("sync")
    args = ap.parse_args()
    this_year = dt.date.today().year
    years = range(args.start, args.end + 1) if args.cmd == "backfill" else (this_year - 1, this_year)
    for y in years:
        write_year(y)
