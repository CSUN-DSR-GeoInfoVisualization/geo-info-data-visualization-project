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
    cols = ["week"] + mi.WEATHERAPP_COLS
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
                                    "n_weather_sources": row["n_weather_sources"], "weather_method": mi.WEATHER_METHOD,
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
        cols = list(rows[0].keys()) + [c for c in ["n_weather_sources", "weather_method"] + mi.RAW_WEATHER_COLS
                                       if c not in rows[0]]
    kept, dropped = [], []
    with app.app_context():
        for i, r in enumerate(rows):
            try:
                raw = {}
                # elevation is static and already on the row: reuse it rather than re-fetch
                feats, n = mi._features_for_historical(float(r["lat"]), float(r["lon"]), r["acq_date"], raw=raw,
                                                       elevation=float(r["elevation"]))
                r = {**r, "air_temp_encoded": feats["air_temp_encoded"], "wind": feats["wind"], "humidity": feats["humidity"],
                     "n_weather_sources": n, "weather_method": mi.WEATHER_METHOD,
                     **{c: raw.get(c) for c in mi.RAW_WEATHER_COLS}}
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


def _key(r):
    return (round(float(r["lat"]), 4), round(float(r["lon"]), 4), r["acq_date"], str(r["fire"]))


def _open_append(path, cols):
    """Open a CSV to append rows with `cols`. If the file was started with other
    columns (the code gained columns since), first rewrite it with `cols`, old rows
    carried over by name. Appending new rows under an old header misaligned every
    column (2026-11-26)."""
    if os.path.exists(path):
        with open(path, newline="") as f:
            rd = csv.reader(f)
            old = next(rd, None)
            if old is not None and old != cols:
                rows = [dict(zip(old, r)) for r in rd]
                if any(len(r) != len(old) for r in rows if r):
                    raise SystemExit(f"{path}: rows do not match its header; repair before appending")
                with open(path + ".tmp", "w", newline="") as g:
                    w = csv.DictWriter(g, fieldnames=cols, extrasaction="ignore")
                    w.writeheader()
                    w.writerows(rows)
                os.replace(path + ".tmp", path)
                print(f"{path}: upgraded header {len(old)} -> {len(cols)} columns", file=sys.stderr, flush=True)
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    if new_file:
        w.writeheader()
    return f, w


def raw_readings(args):
    """Add each source's raw reading to rows that predate the raw columns, without
    touching anything stored (additive). Resumable: appends to --out, keyed by
    (lat, lon, acq_date, fire). Also re-blends from the raw readings and reports
    rows whose stored blend can't be reproduced, which would mean the stored row
    was made some other way."""
    from routes import ml_ingest as mi
    with open(args.inp, newline="") as f:
        rows = list(csv.DictReader(f))
    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            done = {_key(r) for r in csv.DictReader(f)}
    cols = ["lat", "lon", "acq_date", "fire", "raw_n_sources", "reblend_ok"] + mi.RAW_WEATHER_COLS + mi.RAW_EVENT_COLS
    new_file = not os.path.exists(args.out)
    out = open(args.out, "a", newline="")
    w = csv.DictWriter(out, fieldnames=cols)
    if new_file:
        w.writeheader()
    off = 0
    for i, r in enumerate(rows):
        if _key(r) in done:
            continue
        try:
            src = mi._weather_sources(float(r["lat"]), float(r["lon"]), r["acq_date"])
            ev = mi._event_sources(float(r["lat"]), float(r["lon"]), r["acq_date"])
        except RateLimited:
            raise SystemExit(f"STOPPED at row {i}: a weather source is rate-limiting us. Rerun to resume.")
        ok = ""
        if len(src) >= 2:
            try:
                b, _ = mi._blend_weather(src)
                ok = int(abs(b["humidity"] - float(r["humidity"])) < 0.5 and abs(b["wind_speed"] - float(r["wind"])) < 0.2
                         and abs(b["temperature_celsius"] + 273.15 - float(r["air_temp_encoded"]) * 0.02) < 0.2)
            except mi.CrossSourceMismatch:
                ok = 0
        off += ok == 0
        w.writerow({"lat": r["lat"], "lon": r["lon"], "acq_date": r["acq_date"], "fire": r["fire"],
                    "raw_n_sources": len(src), "reblend_ok": ok,
                    **{f"{n}_{k}": x for n, v in src.items() for k, x in v.items()},
                    **{f"{n}_{k}": x for n, v in ev.items() for k, x in v.items()}})
        if i % 50 == 0:
            out.flush()
            print(f"{i}/{len(rows)} not-reproduced={off}", flush=True)
    out.close()
    print(f"done: {args.inp} -> {args.out}; blend not reproduced on {off} new rows")


def merge(args):
    """Build the Weather App Plan training file from row files (highest priority
    first: a later file never replaces a row an earlier one has), filling raw
    readings from --raw files where a row lacks them. Adds provenance (sensor,
    pixel_m, weather_method) where missing and refuses to write a row without
    >=2 weather sources. Never drops a row except an exact duplicate."""
    from routes import ml_ingest as mi
    raw, events = {}, {}
    from ml.fire_danger import DANGER_COLS, BENCH_COLS, NORMAL_COLS, SOIL_COLS
    extra_cols = DANGER_COLS + BENCH_COLS + NORMAL_COLS + SOIL_COLS
    for p in getattr(args, "danger", None) or []:  # extra-column files (fire-danger, normals, soil): additive
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                v = {c: r[c] for c in extra_cols if r.get(c)}
                if v:
                    events.setdefault(_key(r), {}).update(v)
    for p in args.raw or []:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                k = _key(r)
                if r["reblend_ok"] == "1":  # only readings that reproduce the stored blend
                    raw[k] = {c: r[c] for c in mi.RAW_WEATHER_COLS}
                ev = {c: r[c] for c in mi.RAW_EVENT_COLS if r.get(c)}
                if ev:
                    events.setdefault(k, {}).update(ev)
    rows, seen, seen_pair = [], set(), set()
    for p in args.inp:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                k = _key(r)
                if k in seen or (r.get("pair_of") and r["pair_of"] in seen_pair):
                    continue  # exact duplicate, or a second random spot for the same fire
                seen.add(k)
                if r.get("pair_of"):
                    seen_pair.add(r["pair_of"])
                if not any(r.get(c) for c in mi.RAW_WEATHER_COLS) and k in raw:
                    r.update(raw[k])
                if k in events:
                    r.update({c: v for c, v in events[k].items() if not r.get(c)})
                rows.append(r)
    # sensor of a control = its fire's (same point); a row without its partner (the
    # original dataset has some) is kept, additive, with the sensor of its date
    sensor = {(r["lat"], r["lon"]): ("MODIS" if r["source"] == "firms_modis" else "VIIRS")
              for r in rows if r["fire"] == "1"}
    for r in rows:
        r["sensor"] = (r.get("sensor") or sensor.get((r["lat"], r["lon"]))
                       or ("MODIS" if r["acq_date"] < "2012-01-20" else "VIIRS"))
        r["pixel_m"] = r.get("pixel_m") or (1000 if r["sensor"] == "MODIS" else 375)
        r["weather_method"] = r.get("weather_method") or mi.WEATHER_METHOD  # inputs all built under v0 weights
        if int(r["n_weather_sources"]) < 2:
            raise SystemExit(f"refusing {_key(r)}: {r['n_weather_sources']} weather source(s)")
    rows.sort(key=lambda r: (r["acq_date"], r["lat"], r["lon"], r["fire"]))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mi.WEATHERAPP_COLS + BENCH_COLS + ["pair_of"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    no_raw = sum(not any(r.get(c) for c in mi.RAW_WEATHER_COLS) for r in rows)
    print(f"merged {len(rows)} rows, "
          f"{no_raw} still without raw readings -> {args.out}")


def _perimeters(year, cache_dir):
    """Every mapped perimeter for a year (FRAP + NIFC supplement), via the live API, cached on disk."""
    import json
    path = os.path.join(cache_dir, f"perimeters_{year}.json")
    if not os.path.exists(path):
        r = requests.get("https://firescope-api.onrender.com/api/history/perimeters",
                         params={"year": year, "min_acres": 0}, timeout=300)
        r.raise_for_status()
        with open(path, "w") as f:
            json.dump(r.json().get("features", []), f)
    with open(path) as f:
        feats = json.load(f)
    return [(ft, _bbox(ft)) for ft in feats if ft.get("geometry")]


def _bbox(ft):
    xs, ys = [], []
    def walk(c):
        if c and isinstance(c[0], (int, float)):
            xs.append(c[0]); ys.append(c[1])
        elif c:
            for x in c:
                walk(x)
    walk(ft["geometry"]["coordinates"])
    return min(xs), min(ys), max(xs), max(ys)


def label_size(args):
    """Fire size for every fire row (additive columns): the mapped perimeter that
    contains the detection and was burning that day (alarm date <= day <= containment
    + 3 days; same year when undated). fire_acres = 0 means no mapped perimeter
    (FRAP maps fires from ~10 acres). fire_start = the perimeter's alarm date, so
    Model B can use only detections near a fire's start."""
    from data.zone_resolver import _feature_contains
    os.makedirs(args.cache, exist_ok=True)
    with open(args.inp, newline="") as f:
        rd = csv.DictReader(f)
        cols, rows = rd.fieldnames, list(rd)
    for c in ("control_type", "fire_acres", "fire_start", "fire_id"):
        if c not in cols:
            cols.append(c)
    iso = lambda ms: dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).date().isoformat() if ms else None
    by_year, n_big = {}, 0
    for r in rows:
        r["control_type"] = r.get("control_type") or {"matched_control": "same_place", "same_season_control": "same_season",
                                                      "spatial_control": "spatial"}.get(r["source"], "")
        if r["fire"] != "1":
            continue
        y = int(r["acq_date"][:4])
        if y not in by_year:
            by_year[y] = _perimeters(y, args.cache)
        la, lo, day = float(r["lat"]), float(r["lon"]), r["acq_date"]
        best = None
        for ft, (x0, y0, x1, y1) in by_year[y]:
            if not (x0 <= lo <= x1 and y0 <= la <= y1) or not _feature_contains(ft, lo, la):
                continue
            pr = ft["properties"]
            a, c = iso(pr.get("ALARM_DATE")), iso(pr.get("CONT_DATE"))
            if a and not (a <= day <= str(dt.date.fromisoformat(c or a) + dt.timedelta(days=3 if c else 120))):
                continue
            if best is None or (a or "") > (iso(best.get("ALARM_DATE")) or ""):
                best = pr
        r["fire_acres"] = round(float(best.get("GIS_ACRES") or 0), 1) if best else 0
        r["fire_start"] = iso(best.get("ALARM_DATE")) or "" if best else ""
        r["fire_id"] = f"{best.get('FIRE_NAME')}|{best.get('YEAR_')}|{best.get('SOURCE')}" if best else ""
        n_big += best is not None and float(best.get("GIS_ACRES") or 0) >= 1000
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    fires = sum(r["fire"] == "1" for r in rows)
    mapped = sum(1 for r in rows if r["fire"] == "1" and r["fire_id"])
    print(f"{fires} fire rows: {mapped} inside a mapped perimeter, {n_big} in fires >= 1,000 acres -> {args.out}")


def spatial(args):
    """Random-spot controls (Model A learns WHERE): for each fire row, a random
    California-land point on the SAME date, more than SPATIAL_MIN_DEG from every
    fire that day, built by the same feature code as every other row. Resumable;
    --part i/n shards the fires for parallel runs."""
    import random
    from routes import ml_ingest as mi
    from ml.geo_checks import on_ca_land
    app = _app()
    with open(args.inp, newline="") as f:
        rows = list(csv.DictReader(f))
    fires = [r for r in rows if r["fire"] == "1"]
    by_date = {}
    for r in fires:
        by_date.setdefault(r["acq_date"], []).append((float(r["lat"]), float(r["lon"])))
    i, n = map(int, args.part.split("/"))
    fires = [r for k, r in enumerate(fires) if k % n == i]
    cols = mi.WEATHERAPP_COLS + ["pair_of"]
    done = set()
    for path in [args.out] + list(getattr(args, "skip_files", None) or []):  # resharding: see every shard's work
        if os.path.exists(path):
            with open(path, newline="") as f:
                done |= {r["pair_of"] for r in csv.DictReader(f)}
    out, w = _open_append(args.out, cols)
    stats = {"written": 0, "failed": 0}
    with app.app_context():
        for k, r in enumerate(fires):
            key = f"{r['lat']},{r['lon']},{r['acq_date']}"
            if key in done:
                continue
            rng = random.Random(key)  # reproducible
            for attempt in range(3):
                for _ in range(500):
                    la, lo = rng.uniform(32.5, 42.0), rng.uniform(-124.45, -114.13)
                    if on_ca_land(la, lo) and all(abs(la - a) > SPATIAL_MIN_DEG or abs(lo - b) > SPATIAL_MIN_DEG
                                                  for a, b in by_date[r["acq_date"]]):
                        break
                else:
                    continue
                try:
                    raw = {}
                    feats, n_src = mi._features_for_historical(la, lo, r["acq_date"], raw=raw)
                except RateLimited:
                    raise SystemExit(f"STOPPED at fire {k}: a weather source is rate-limiting us. Rerun to resume.")
                except Exception:
                    continue  # another random point
                row = {"lat": round(la, 5), "lon": round(lo, 5), "acq_date": r["acq_date"], **feats, "fire": 0,
                       "source": "spatial_control", "control_type": "spatial", "sensor": r.get("sensor"),
                       "pixel_m": r.get("pixel_m"), "n_weather_sources": n_src, "weather_method": mi.WEATHER_METHOD,
                       "pair_of": key, **raw}
                if mi._row_quality_issue(row):
                    continue
                w.writerow(row)
                stats["written"] += 1
                break
            else:
                stats["failed"] += 1
            if k % 25 == 0:
                out.flush()
                print(f"{k}/{len(fires)} {stats}", flush=True)
    out.close()
    print(f"done {stats} -> {args.out}")


SPATIAL_MIN_DEG = 0.25  # ~25 km: never a control inside or next to that day's fires


def _firms_src(day):
    if day > SP_ARCHIVE_END:
        return "VIIRS_NOAA20_NRT"
    if day < dt.date(2012, 1, 20):
        return "MODIS_SP"
    return "VIIRS_SNPP_SP" if day < dt.date(2018, 4, 1) else "VIIRS_NOAA20_SP"


def growth_set(args):
    """Model B ("will it grow large"): one row per mapped California fire, at its
    earliest confident FIRMS vegetation-fire detection inside the perimeter, from
    the day before its alarm date to 3 days after. Weather is the start-day
    weather, so the model learns what makes a fire grow, not what a big fire looks
    like on day 10. Label = fire_acres (large = >= 1,000 at training time)."""
    from routes import ml_ingest as mi
    from data.zone_resolver import _feature_contains
    app = _app()
    key = os.environ["FIRMS_MAP_KEY"]
    i, n = map(int, args.part.split("/"))
    cols = mi.WEATHERAPP_COLS
    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            done = {r["fire_id"] for r in csv.DictReader(f)}
    # fires FIRMS has no detection for: remembered, so a restart doesn't re-ask FIRMS about them
    skip_path = args.out + ".no_detection"
    if os.path.exists(skip_path):
        with open(skip_path) as f:
            done |= {line.rstrip("\n") for line in f}
    skip_f = open(skip_path, "a")
    out, w = _open_append(args.out, cols)
    stats = {"written": 0, "no_detection": 0, "failed": 0}
    iso = lambda ms: dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).date()
    k = -1
    with app.app_context():
        for year in range(args.start, args.end + 1):
            for ft, (x0, y0, x1, y1) in _perimeters(year, args.cache):
                pr = ft["properties"]
                if not pr.get("ALARM_DATE"):
                    continue
                k += 1
                fid = f"{pr.get('FIRE_NAME')}|{pr.get('YEAR_')}|{pr.get('SOURCE')}|{pr.get('OBJECTID') or pr.get('SOURCE_ID')}"
                if k % n != i or fid in done:
                    continue
                alarm = iso(pr["ALARM_DATE"])
                start = alarm - dt.timedelta(days=1)
                url = FIRMS.format(key=key, src=_firms_src(start), days=5, date=start).replace(
                    "-124.48,32.53,-114.13,42.01", f"{x0:.4f},{y0:.4f},{x1:.4f},{y1:.4f}")
                try:
                    r = requests.get(url, timeout=90)
                    r.raise_for_status()
                    time.sleep(float(os.environ.get("FIRMS_INTERVAL", 3)))  # FIRMS: 5,000 transactions / 10 min per key, shared with the live site
                except Exception as e:
                    print(f"{fid} FIRMS failed: {str(e).replace(key, '<key>')}"[:300], file=sys.stderr, flush=True)
                    stats["failed"] += 1
                    continue
                hits = []
                for d in csv.DictReader(io.StringIO(r.text)):
                    conf = d.get("confidence", "")
                    ok = int(conf) >= 30 if conf.isdigit() else conf in ("n", "h")
                    if ok and d.get("type", "0") == "0" and _feature_contains(ft, float(d["longitude"]), float(d["latitude"])):
                        hits.append((d["acq_date"], d.get("acq_time", ""), float(d["latitude"]), float(d["longitude"])))
                if not hits:
                    stats["no_detection"] += 1
                    skip_f.write(fid + "\n")
                    skip_f.flush()
                    continue
                day, _, la, lo = min(hits)
                try:
                    raw = {}
                    feats, n_src = mi._features_for_historical(la, lo, day, raw=raw)
                except RateLimited:
                    raise SystemExit(f"STOPPED at {fid}: a weather source is rate-limiting us. Rerun to resume.")
                except Exception as e:
                    print(f"{fid} features failed: {type(e).__name__}: {e}"[:300], file=sys.stderr, flush=True)
                    stats["failed"] += 1
                    continue
                row = {"lat": round(la, 5), "lon": round(lo, 5), "acq_date": day, **feats, "fire": 1,
                       "source": "perimeter_start", "control_type": "", "n_weather_sources": n_src,
                       "weather_method": mi.WEATHER_METHOD, "sensor": "MODIS" if _firms_src(start) == "MODIS_SP" else "VIIRS",
                       "pixel_m": 1000 if _firms_src(start) == "MODIS_SP" else 375,
                       "fire_acres": round(float(pr.get("GIS_ACRES") or 0), 1), "fire_start": str(alarm), "fire_id": fid, **raw}
                if mi._row_quality_issue(row):
                    stats["failed"] += 1
                    continue
                w.writerow(row)
                stats["written"] += 1
                if stats["written"] % 20 == 0:
                    out.flush()
                    print(f"{year} {stats}", flush=True)
    out.close()
    skip_f.close()
    print(f"done {stats} -> {args.out}")


def fire_danger(args):
    """Fire-danger columns (ml.fire_danger) for existing rows, additive: one gridMET +
    terrain call per date for all that date's rows, our NFDRS fuel moisture per row.
    Resumable by (lat, lon, acq_date, fire); --part i/n shards by date."""
    from concurrent.futures import ThreadPoolExecutor
    from ml import fire_danger as fd
    with open(args.inp, newline="") as f:
        rows = list(csv.DictReader(f))
    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            done = {_key(r) for r in csv.DictReader(f)}
    i, n = map(int, args.part.split("/"))
    by_date = {}
    for r in rows:
        if _key(r) not in done:
            by_date.setdefault(r["acq_date"], []).append(r)
    dates = [x for k, x in enumerate(sorted(by_date)) if k % n == i]
    cols = ["lat", "lon", "acq_date", "fire"] + fd.DANGER_COLS + fd.BENCH_COLS
    new_file = not os.path.exists(args.out)
    out = open(args.out, "a", newline="")
    w = csv.DictWriter(out, fieldnames=cols, extrasaction="ignore")
    if new_file:
        w.writeheader()
    stats = {"rows": 0, "gridmet_failed": 0}
    with ThreadPoolExecutor(4) as ex:
        for k, date in enumerate(dates):
            grp = by_date[date]
            try:
                gm = fd.gridmet_at(date, [(j, float(r["lat"]), float(r["lon"])) for j, r in enumerate(grp)])
            except Exception as e:
                print(f"{date} gridMET failed: {e}"[:200], file=sys.stderr, flush=True)
                gm, stats["gridmet_failed"] = {}, stats["gridmet_failed"] + 1
            try:
                own = list(ex.map(lambda r: fd.era5_fuel(float(r["lat"]), float(r["lon"]), date), grp))
            except RateLimited:
                raise SystemExit(f"STOPPED at {date}: Open-Meteo is rate-limiting us. Rerun to resume.")
            for j, (r, o) in enumerate(zip(grp, own)):
                w.writerow({"lat": r["lat"], "lon": r["lon"], "acq_date": date, "fire": r["fire"], **gm.get(str(j), {}), **o})
                stats["rows"] += 1
            if k % 25 == 0:
                out.flush()
                print(f"{k}/{len(dates)} dates {stats}", flush=True)
    out.close()
    print(f"done {stats} -> {args.out}")


def batch_columns(args):
    """Additive Earth Engine columns for existing rows, batched (one call per group for all
    its points): --kind normals (by calendar month) or soil (by date). Resumable."""
    from ml import fire_danger as fd
    fn, cols, group = {"normals": (lambda g, pts: fd.normals_at(int(g), pts), fd.NORMAL_COLS, lambda r: r["acq_date"][5:7]),
                       "soil": (fd.soil_at, fd.SOIL_COLS[:3], lambda r: r["acq_date"]),
                       "gldas": (fd.gldas_at, fd.GLDAS_COLS, lambda r: r["acq_date"])}[args.kind]
    with open(args.inp, newline="") as f:
        rows = list(csv.DictReader(f))
    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            done = {_key(r) for r in csv.DictReader(f)}
    groups = {}
    for r in rows:
        if _key(r) not in done:
            groups.setdefault(group(r), []).append(r)
    new_file = not os.path.exists(args.out)
    out = open(args.out, "a", newline="")
    w = csv.DictWriter(out, fieldnames=["lat", "lon", "acq_date", "fire"] + cols, extrasaction="ignore")
    if new_file:
        w.writeheader()
    n = 0
    for k, g in enumerate(sorted(groups)):
        grp = groups[g]
        for c0 in range(0, len(grp), 500):  # Earth Engine payload limit
            chunk = grp[c0:c0 + 500]
            try:
                res = fn(g, [(j, float(r["lat"]), float(r["lon"])) for j, r in enumerate(chunk)])
            except Exception as e:
                print(f"{g} failed: {e}"[:200], file=sys.stderr, flush=True)
                continue
            for j, r in enumerate(chunk):
                w.writerow({"lat": r["lat"], "lon": r["lon"], "acq_date": r["acq_date"], "fire": r["fire"], **res.get(str(j), {})})
                n += 1
        if k % 25 == 0:
            out.flush()
            print(f"{k}/{len(groups)} groups, {n} rows", flush=True)
    out.close()
    print(f"done {n} rows -> {args.out}")


NLCD_YEARS = (2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021)
NLCD_AG = {"81", "82"}  # pasture/hay, cultivated crops
# Burnable wildland (A4b): forest 41-43, shrub 51-52, grassland/sedge 71-72, wetlands 90/95.
# Water, ice, developed, barren and cropland are not (LANDFIRE fuel models come with B2).
NLCD_WILDLAND = {"41", "42", "43", "51", "52", "71", "72", "90", "95"}


def label_class(args):
    """Additive label columns for fire rows (Ido 2026-11-26: models train on confirmed
    wildfires). wildfire_confirmed = inside a mapped perimeter (fire_id, from
    label-size) or matched to the USFS FPA FOD ignition record (<= 0.03 deg, detected
    between discovery - 1 day and containment + 1 day); likely_burn_or_ag = on NLCD
    pasture/cropland with no wildfire match; else unknown. Also fpa_cause,
    fpa_size_acres, nlcd_class. Controls get no label."""
    import sqlite3
    import numpy as np
    import pandas as pd
    from ml import fire_danger as fd
    d = pd.read_csv(args.inp, dtype=str, keep_default_na=False)
    q = pd.read_sql("select DISCOVERY_DATE, CONT_DATE, LATITUDE, LONGITUDE, FIRE_SIZE, NWCG_GENERAL_CAUSE "
                    "from Fires where STATE='CA' and FIRE_YEAR >= 2000", sqlite3.connect(args.fpa))
    jd = lambda x: pd.to_datetime(x - 2440587.5, unit="D")
    disc = jd(q.DISCOVERY_DATE)
    cont = jd(q.CONT_DATE).fillna(disc + pd.Timedelta(days=3))
    cont = cont.where(cont >= disc, disc).clip(upper=disc + pd.Timedelta(days=120))
    lat, lon = q.LATITUDE.values, q.LONGITUDE.values
    lo_t, hi_t = (disc - pd.Timedelta(days=1)).values, (cont + pd.Timedelta(days=1)).values
    cell = {}
    for i, (a, b) in enumerate(zip((lat / 0.05).astype(int), (lon / 0.05).astype(int))):
        cell.setdefault((a, b), []).append(i)
    fires = d.index[d["fire"] == "1"]
    for c in ("label_class", "fpa_cause", "fpa_size_acres", "nlcd_class", "burnable"):
        if c not in d:
            d[c] = ""
    matched = {}
    for i in fires:
        la, lo, t = float(d.at[i, "lat"]), float(d.at[i, "lon"]), np.datetime64(d.at[i, "acq_date"])
        best = None
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                for j in cell.get((int(la / 0.05) + da, int(lo / 0.05) + db), []):
                    if abs(lat[j] - la) < 0.03 and abs(lon[j] - lo) < 0.03 and lo_t[j] <= t <= hi_t[j]:
                        if best is None or q.FIRE_SIZE[j] > q.FIRE_SIZE[best]:
                            best = j
        if best is not None:
            matched[i] = best
            d.at[i, "fpa_cause"] = q.NWCG_GENERAL_CAUSE[best]
            d.at[i, "fpa_size_acres"] = str(q.FIRE_SIZE[best])
    spatial_rows = d.index[d["control_type"] == "spatial"] if "control_type" in d else []
    by_ver = {}
    for i in list(fires) + list(spatial_rows):
        y = int(d.at[i, "acq_date"][:4])
        by_ver.setdefault(max([v for v in NLCD_YEARS if v <= y] or [NLCD_YEARS[0]]), []).append(i)
    ee = fd._ee()
    for ver, idx in by_ver.items():
        img = (ee.Image("USGS/NLCD_RELEASES/2021_REL/NLCD/2021") if ver == 2021 else
               ee.ImageCollection("USGS/NLCD_RELEASES/2019_REL/NLCD").filter(ee.Filter.eq("system:index", str(ver))).first()
               ).select("landcover")
        for c0 in range(0, len(idx), 500):
            chunk = idx[c0:c0 + 500]
            res = fd._reduce(img, [(i, float(d.at[i, "lat"]), float(d.at[i, "lon"])) for i in chunk], 30)
            for i in chunk:
                v = res.get(str(i), {}).get("landcover")
                d.at[i, "nlcd_class"] = "" if v is None else str(int(v))
        print(f"NLCD {ver}: {len(idx)} rows", flush=True)
    for i in spatial_rows:
        d.at[i, "burnable"] = "1" if d.at[i, "nlcd_class"] in NLCD_WILDLAND else "0"
    for i in fires:
        confirmed = bool(d.at[i, "fire_id"]) or i in matched
        d.at[i, "label_class"] = ("wildfire_confirmed" if confirmed else
                                  "likely_burn_or_ag" if d.at[i, "nlcd_class"] in NLCD_AG else "unknown")
    d.to_csv(args.out, index=False)
    f = d.loc[fires]
    print(f"{len(f)} fire rows: " + ", ".join(f"{k} {v:.0%}" for k, v in f.label_class.value_counts(normalize=True).items()))
    if len(spatial_rows):
        print(f"random-spot controls on burnable wildland: {(d.loc[spatial_rows, 'burnable'] == '1').mean():.0%}")
    print("confirmed by month:", (f[f.label_class == "wildfire_confirmed"].acq_date.str[5:7].value_counts(normalize=True)
                                  .sort_index() * 100).round().astype(int).to_dict())


def fpa_set(args):
    """Model A/B rows from the USFS FPA FOD confirmed-wildfire record (A5b): each
    sampled ignition at its discovery point and date (fire=1, label wildfire_confirmed,
    final size and cause kept), its same-place control a year earlier, and a random
    California-land control on the same date (burnable flag added by label-class).
    Sample: every fire >= 100 acres, plus --n-mid of 10-100 acres and --n-small under
    10 acres (seeded). Resumable by fpa id; --part i/n shards."""
    import random
    import sqlite3
    import pandas as pd
    from routes import ml_ingest as mi
    from ml.geo_checks import on_ca_land
    q = pd.read_sql("select FOD_ID, DISCOVERY_DATE, LATITUDE, LONGITUDE, FIRE_SIZE, NWCG_GENERAL_CAUSE from Fires "
                    "where STATE='CA' and FIRE_YEAR between 2001 and 2024", sqlite3.connect(args.fpa))
    q["date"] = pd.to_datetime(q.DISCOVERY_DATE - 2440587.5, unit="D").dt.strftime("%Y-%m-%d")
    big = q[q.FIRE_SIZE >= 100]
    mid = q[(q.FIRE_SIZE >= 10) & (q.FIRE_SIZE < 100)].sample(min(args.n_mid, int((q.FIRE_SIZE >= 10).sum())), random_state=7)
    small = q[q.FIRE_SIZE < 10].sample(args.n_small, random_state=7)
    fires = pd.concat([big, mid, small]).sort_values("FOD_ID").reset_index(drop=True)
    i, n = map(int, args.part.split("/"))
    fires = fires[fires.index % n == i]
    by_date = q.groupby("date")[["LATITUDE", "LONGITUDE"]].apply(lambda g: list(zip(g.LATITUDE, g.LONGITUDE))).to_dict()
    cols = mi.WEATHERAPP_COLS + ["pair_of", "label_class", "fpa_cause", "fpa_size_acres"]
    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            done = {r["pair_of"] for r in csv.DictReader(f)}
    out, w = _open_append(args.out, cols)
    app = _app()
    stats = {"fires": 0, "failed": 0}
    with app.app_context():
        for k, r in enumerate(fires.itertuples()):
            key = f"fpa:{r.FOD_ID}"
            if key in done:
                continue
            la, lo, date = float(r.LATITUDE), float(r.LONGITUDE), r.date
            rng = random.Random(key)
            rows = []
            for _ in range(500):  # random spot on the same date, away from that day's fires
                sla, slo = rng.uniform(32.5, 42.0), rng.uniform(-124.45, -114.13)
                if on_ca_land(sla, slo) and all(abs(sla - a) > SPATIAL_MIN_DEG or abs(slo - b) > SPATIAL_MIN_DEG
                                                for a, b in by_date.get(date, [])):
                    break
            # fire first: no fire row, no group
            for fire, pla, plo, pdate, ctype in ((1, la, lo, date, ""), (0, la, lo, mi._prior_year(date), "same_place"),
                                                 (0, sla, slo, date, "spatial")):
                try:
                    raw = {}
                    feats, n_src = mi._features_for_historical(pla, plo, pdate, raw=raw)
                except RateLimited:
                    raise SystemExit(f"STOPPED at {key}: a weather source is rate-limiting us. Rerun to resume.")
                except Exception:
                    if fire:
                        break  # no fire row, no group
                    continue
                row = {"lat": round(pla, 5), "lon": round(plo, 5), "acq_date": pdate, **feats, "fire": fire,
                       "source": "fpa_fod" if fire else ("matched_control" if ctype == "same_place" else "spatial_control"),
                       "control_type": ctype, "n_weather_sources": n_src, "weather_method": mi.WEATHER_METHOD,
                       "sensor": "", "pixel_m": "", "pair_of": key, **raw}
                if fire:
                    row.update(label_class="wildfire_confirmed", fpa_cause=r.NWCG_GENERAL_CAUSE,
                               fpa_size_acres=r.FIRE_SIZE, fire_acres=r.FIRE_SIZE, fire_start=date, fire_id=key)
                if not mi._row_quality_issue(row):
                    rows.append(row)
            if any(x["fire"] == 1 for x in rows) and len(rows) >= 2:
                for x in rows:
                    w.writerow(x)
                stats["fires"] += 1
            else:
                stats["failed"] += 1
            if k % 25 == 0:
                out.flush()
                print(f"{k}/{len(fires)} {stats}", flush=True)
    out.close()
    print(f"done {stats} -> {args.out}")


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
    d = sub.add_parser("raw")
    d.add_argument("--inp", required=True)
    d.add_argument("--out", required=True)
    e = sub.add_parser("merge")
    e.add_argument("--inp", nargs="+", required=True)
    e.add_argument("--raw", nargs="*")
    e.add_argument("--danger", nargs="*")
    e.add_argument("--out", required=True)
    g = sub.add_parser("label-size")
    g.add_argument("--inp", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--cache", default="perimeter_cache")
    h = sub.add_parser("spatial")
    h.add_argument("--inp", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--part", default="0/1")
    h.add_argument("--skip-files", nargs="*")
    j = sub.add_parser("growth-set")
    j.add_argument("--start", type=int, default=2001)
    j.add_argument("--end", type=int, default=dt.date.today().year)
    j.add_argument("--out", required=True)
    j.add_argument("--cache", default="perimeter_cache")
    j.add_argument("--part", default="0/1")
    q = sub.add_parser("fire-danger")
    q.add_argument("--inp", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--part", default="0/1")
    r_ = sub.add_parser("batch")
    r_.add_argument("--kind", choices=["normals", "soil", "gldas"], required=True)
    r_.add_argument("--inp", required=True)
    r_.add_argument("--out", required=True)
    lc = sub.add_parser("label-class")
    lc.add_argument("--inp", required=True)
    lc.add_argument("--out", required=True)
    lc.add_argument("--fpa", required=True)
    fp = sub.add_parser("fpa-set")
    fp.add_argument("--fpa", required=True)
    fp.add_argument("--out", required=True)
    fp.add_argument("--part", default="0/1")
    fp.add_argument("--n-mid", type=int, default=3000)
    fp.add_argument("--n-small", type=int, default=3000)
    args = ap.parse_args()
    {"fpa-set": fpa_set, "label-class": label_class, "batch": batch_columns, "fire-danger": fire_danger, "growth-set": growth_set, "spatial": spatial, "label-size": label_size, "fill": fill, "regrade": regrade, "static-mask": static_mask, "raw": raw_readings, "merge": merge}[args.cmd](args)
