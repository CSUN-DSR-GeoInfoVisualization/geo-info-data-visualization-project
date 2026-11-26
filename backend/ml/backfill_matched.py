"""Offline matched-control backfill and re-grade, using the daily ingest's own
feature code (3-source skill-weighted weather, per-feature backups, gross-error
checks), so backfilled rows are built exactly like daily rows.

Run from backend/ with FIRMS_MAP_KEY set and Earth Engine credentials available:

  # fill a date range, ~N fire/control pairs per week (resumable)
  python -m ml.backfill_matched fill --start 2018-01-01 --end 2020-12-31 --per-week 10 --out bf_2018_2020.csv
  # recompute weather for existing rows with the current blend
  python -m ml.backfill_matched regrade --inp training_data/california_matched.csv --out regraded.csv

Labels come from the FIRMS VIIRS standard-processing archive, keeping only
type 0 (vegetation fire) with nominal/high confidence: the archive marks ~26% of
California detections as static industrial heat sources (type 2).
"""
import argparse
import csv
import datetime as dt
import io
import os
import sys
import time

import requests

SP_ARCHIVE_END = dt.date(2026, 8, 31)
FIRMS = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{src}/-124.48,32.53,-114.13,42.01/{days}/{date}"


def _app():
    # get_elevation caches tiles in the app DB; a local SQLite file is enough offline.
    os.environ.setdefault("DATABASE_URL", "sqlite:///backfill_cache.db")
    from app import create_app
    from models import db
    app = create_app()
    with app.app_context():
        db.create_all()
    return app


def _firms_week(monday):
    """Vegetation-fire detections (type 0, conf n/h) for the 7 days from `monday`."""
    key = os.environ["FIRMS_MAP_KEY"]
    # The standard-processing archive lags ~3 months (ends 2026-08-31 as of
    # 2026-11-25); newer weeks come from the NRT feed the daily ingest uses, which
    # has no type field, so static sources are screened by the mask instead.
    nrt = monday > SP_ARCHIVE_END
    if nrt:
        src = "VIIRS_NOAA20_NRT"
    elif monday < dt.date(2012, 1, 20):   # before VIIRS: MODIS (1 km), from 2000-11
        src = "MODIS_SP"
    elif monday < dt.date(2018, 4, 1):
        src = "VIIRS_SNPP_SP"
    else:
        src = "VIIRS_NOAA20_SP"
    out = []
    for start, days in ((monday, 5), (monday + dt.timedelta(days=5), 2)):
        for attempt in range(4):
            try:
                r = requests.get(FIRMS.format(key=key, src=src, days=days, date=start), timeout=90)
                r.raise_for_status()
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(10 * (attempt + 1))
        for row in csv.DictReader(io.StringIO(r.text)):
            conf = row.get("confidence", "")
            # VIIRS: l/n/h; MODIS: 0-100, where NASA treats < 30 as low confidence.
            confident = int(conf) >= 30 if conf.isdigit() else conf in ("n", "h")
            if (nrt or row.get("type") == "0") and confident:
                out.append((float(row["latitude"]), float(row["longitude"]), row["acq_date"]))
    return out


from data.open_meteo import RateLimited


def fill(args):
    from routes import ml_ingest as mi
    from ml.geo_checks import on_ca_land
    app = _app()
    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            for r in csv.DictReader(f):
                done.add(r["week"])
    new_file = not os.path.exists(args.out)
    cols = ["week"] + mi.ROW_COLS + ["sensor", "pixel_m", "n_weather_sources"] + mi.RAW_WEATHER_COLS
    if not new_file:  # files started before n_weather_sources existed keep their header
        with open(args.out, newline="") as f:
            cols = next(csv.reader(f))
    f = open(args.out, "a", newline="")
    # sensor/pixel_m are provenance, never model inputs (the trainer selects
    # FEATURE_COLS by name): a MODIS detection locates a fire within a ~1 km pixel,
    # VIIRS within ~375 m, so the two are different-quality labels.
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    if new_file:
        w.writeheader()
    stats = {"pairs": 0, "fail": 0, "mismatch": 0, "quality": 0}
    monday = dt.date.fromisoformat(args.start)
    monday -= dt.timedelta(days=monday.weekday())
    end = dt.date.fromisoformat(args.end)
    with app.app_context():
        while monday <= end:
            wk = monday.isoformat()
            if wk in done:
                monday += dt.timedelta(days=7)
                continue
            try:
                fires = [p for p in _firms_week(monday) if on_ca_land(p[0], p[1]) and p[2] <= args.end]
            except Exception as e:
                print(f"{wk} FIRMS failed: {e}", file=sys.stderr)
                monday += dt.timedelta(days=7)
                continue
            pairs = 0
            fires = [p for p in fires if not mi._is_static_source(p[0], p[1])]
            for lat, lon, d in mi._thin_detections(fires, args.per_week, seed=wk):
                if pairs >= args.per_week:
                    break
                rows = []
                for label, date in ((1, d), (0, mi._prior_year(d))):
                    try:
                        raw = {}
                        feats, n_src = mi._features_for_historical(lat, lon, date, raw=raw)
                    except RateLimited:
                        raise SystemExit(f"STOPPED at {wk}: a weather source is rate-limiting us; "
                                         f"no rows written for this week. Rerun to resume.")
                    except mi.CrossSourceMismatch:
                        stats["mismatch"] += 1
                        break
                    except Exception:
                        stats["fail"] += 1
                        break
                    row = {"lat": round(lat, 5), "lon": round(lon, 5), "acq_date": date, **feats,
                           "n_weather_sources": n_src, **raw,
                           "fire": label, "source": (("firms_modis" if monday < dt.date(2012, 1, 20) else "firms_viirs")
                                                     if label else "matched_control")}
                    if mi._row_quality_issue(row):
                        stats["quality"] += 1
                        break
                    rows.append(row)
                if len(rows) == 2:  # complete pairs only; the control inherits its fire's sensor
                    modis = monday < dt.date(2012, 1, 20)
                    for row in rows:
                        w.writerow({"week": wk, **{c: row.get(c) for c in mi.ROW_COLS},
                                    "sensor": "MODIS" if modis else "VIIRS", "pixel_m": 1000 if modis else 375,
                                    "n_weather_sources": row["n_weather_sources"],
                                    **{c: row.get(c) for c in mi.RAW_WEATHER_COLS}})
                    pairs += 1
            f.flush()
            stats["pairs"] += pairs
            print(f"{wk}: {len(fires)} veg-fire detections -> {pairs} pairs  {stats}", flush=True)
            monday += dt.timedelta(days=7)
    f.close()


def static_mask(args):
    """Known static heat sources (refineries, plants, flares): every 0.01 deg cell
    with a FIRMS archive type-2 detection over the period. The NRT feed the daily
    ingest uses has no type field, so the ingest filters NRT fires against this."""
    key = os.environ["FIRMS_MAP_KEY"]
    cells = {}
    day = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    while day <= end:
        for src in ("VIIRS_NOAA20_SP", "VIIRS_SNPP_SP"):
            r = requests.get(FIRMS.format(key=key, src=src, days=5, date=day), timeout=90)
            if r.ok:
                for row in csv.DictReader(io.StringIO(r.text)):
                    if row.get("type") == "2":
                        c = (round(float(row["latitude"]), 2), round(float(row["longitude"]), 2))
                        cells[c] = cells.get(c, 0) + 1
        day += dt.timedelta(days=5)
        print(day, len(cells), flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon", "detections"])
        for (la, lo), n in sorted(cells.items()):
            w.writerow([f"{la:.2f}", f"{lo:.2f}", n])
    print(f"{len(cells)} static-source cells -> {args.out}")


def regrade(args):
    """Recompute temperature/humidity/wind for existing rows with the current
    3-source blend. Rows whose weather now fails (no source, or 2 sources that
    disagree) are dropped together with their pair partner."""
    from routes import ml_ingest as mi
    app = _app()
    with open(args.inp, newline="") as f:
        rows = list(csv.DictReader(f))
        cols = list(rows[0].keys()) + [c for c in ["n_weather_sources"] + mi.RAW_WEATHER_COLS if c not in rows[0]]
    kept, dropped = [], []
    with app.app_context():
        for i, r in enumerate(rows):
            try:
                raw = {}
                feats, n = mi._features_for_historical(float(r["lat"]), float(r["lon"]), r["acq_date"], raw=raw)
                r = {**r, "air_temp_encoded": feats["air_temp_encoded"], "wind": feats["wind"], "humidity": feats["humidity"],
                     "n_weather_sources": n, **{c: raw.get(c) for c in mi.RAW_WEATHER_COLS}}
                kept.append(r)
            except RateLimited:
                raise SystemExit(f"STOPPED at row {i}: a weather source is rate-limiting us. Rerun regrade.")
            except Exception as e:
                dropped.append((r, f"{type(e).__name__}: {e}"[:160]))
            if i % 50 == 0:
                print(f"{i}/{len(rows)} kept={len(kept)} dropped={len(dropped)}", flush=True)
    # drop pair partners of dropped rows (control = same point, prior year)
    gone = {(r["lat"], r["lon"]) for r, _ in dropped}
    kept = [r for r in kept if (r["lat"], r["lon"]) not in gone]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(kept)
    print(f"done: kept {len(kept)} of {len(rows)}; dropped {len(rows) - len(kept)}")
    for r, why in dropped[:20]:
        print("  dropped", r["lat"], r["lon"], r["acq_date"], why)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fill")
    a.add_argument("--start", required=True)
    a.add_argument("--end", required=True)
    a.add_argument("--per-week", type=int, default=10)
    a.add_argument("--out", required=True)
    b = sub.add_parser("regrade")
    b.add_argument("--inp", required=True)
    b.add_argument("--out", required=True)
    c = sub.add_parser("static-mask")
    c.add_argument("--start", required=True)
    c.add_argument("--end", required=True)
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    {"fill": fill, "regrade": regrade, "static-mask": static_mask}[args.cmd](args)
