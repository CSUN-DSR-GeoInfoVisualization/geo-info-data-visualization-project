"""Daily live feature build (v3.5, Architecture v2, gate 11 parity).

Computes the model's event-day and fire-danger inputs for every California land tile,
using the SAME definitions as the training rows (routes.ml_ingest event functions,
ml.fuel_moisture), from sources available on the day:
  RTMA (Earth Engine): one batched computation for all tiles (today + yesterday);
  Open-Meteo forecast API on the 0.25-degree grid, many cells per request, with
  past_days for the 7-day rain and the 90-day fuel-moisture spin-up; fills the
  ERA5 slot (its parity with ERA5 is checked once ERA5 publishes, ~5 days later).
Each value records its source so gate 11 can verify training/live definitions.
"""
import datetime as dt
import math

import numpy as np
import requests

from ml.fuel_moisture import dead_fuel_moisture
from routes.ml_ingest import OFFSHORE_FROM_DEG

FORECAST = "https://api.open-meteo.com/v1/forecast"
DAILY = ("wind_gusts_10m_max,wind_speed_10m_max,wind_direction_10m_dominant,relative_humidity_2m_min,"
         "relative_humidity_2m_max,vapour_pressure_deficit_max,precipitation_sum,precipitation_hours,"
         "temperature_2m_max,temperature_2m_min")


def land_tiles():
    from ml.geo_checks import on_ca_land
    pts = [(round(32.5 + i * 0.1, 2), round(-124.0 + j * 0.1, 2)) for i in range(96) for j in range(101)]
    return [p for p in pts if on_ca_land(*p)]


def cell_of(lat, lon):
    return round(lat * 4) / 4, round(lon * 4) / 4


def rtma_event_grid(date_str, points):
    """{point index: {rtma_gust_event, rtma_offshore_event, rtma_rh_min_event, rtma_vpd_max_event}}
    over the date and the day before (Pacific), one Earth Engine computation for all points."""
    from ml.fire_danger import _ee, _reduce
    ee = _ee()
    from zoneinfo import ZoneInfo
    t = dt.date.fromisoformat(date_str)
    la = ZoneInfo("America/Los_Angeles")
    # exactly the two Pacific days (DST-aware), as the training reader filters them
    start = dt.datetime.combine(t - dt.timedelta(days=1), dt.time(0), la).astimezone(dt.timezone.utc)
    end = dt.datetime.combine(t + dt.timedelta(days=1), dt.time(0), la).astimezone(dt.timezone.utc)
    col = ee.ImageCollection("NOAA/NWS/RTMA").filterDate(start.strftime("%Y-%m-%dT%H:%M:%S"), end.strftime("%Y-%m-%dT%H:%M:%S"))

    def per_hour(im):
        es = lambda x: ee.Image(0.6108).multiply(x.multiply(17.27).divide(x.add(237.3)).exp())
        e_t, e_d = es(im.select("TMP")), es(im.select("DPT"))
        return ee.Image.cat(
            im.select("GUST").rename("gust"),
            im.select("WIND").multiply(im.select("WDIR").subtract(OFFSHORE_FROM_DEG).multiply(math.pi / 180).cos()).rename("off"),
            e_d.divide(e_t).multiply(100).min(100).rename("rh"),
            e_t.subtract(e_d).max(0).rename("vpd"))
    h = col.map(per_hour)
    img = ee.Image.cat(h.select("gust").max(), h.select("off").max(), h.select("rh").min(), h.select("vpd").max())
    out = {}
    for c0 in range(0, len(points), 500):
        chunk = [(i, la, lo) for i, (la, lo) in enumerate(points[c0:c0 + 500], start=c0)]
        res = _reduce(img, chunk, 2500)
        for i, _, _ in chunk:
            r = res.get(str(i), {})
            out[i] = {k: r.get(b) for k, b in (("rtma_gust_event", "gust"), ("rtma_offshore_event", "off"),
                                                ("rtma_rh_min_event", "rh"), ("rtma_vpd_max_event", "vpd"))
                      if r.get(b) is not None}
    return out


def forecast_cells(cells, date_str, per_request=20):
    """{cell: {era5_* event values, era5_rain_7d, era5_fm100, era5_fm1000}} from the Open-Meteo
    forecast API (past 90 days + today), same definitions as training."""
    out = {}
    for c0 in range(0, len(cells), per_request):
        chunk = cells[c0:c0 + per_request]
        for attempt in range(4):
            try:
                t = dt.date.fromisoformat(date_str)
                # explicit window ending on date_str: past_days counts back from TODAY, which made every
                # historical request miss its 90-day spin-up (2026-11-28 bug: empty correction fit)
                r = requests.get(FORECAST, params={
                    "latitude": ",".join(str(c[0]) for c in chunk), "longitude": ",".join(str(c[1]) for c in chunk),
                    "daily": DAILY, "start_date": str(t - dt.timedelta(days=89)), "end_date": date_str,
                    "wind_speed_unit": "ms", "timezone": "America/Los_Angeles"}, timeout=90)
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                import time
                time.sleep(10 * (attempt + 1))
        js = r.json()
        for cell, j in zip(chunk, js if isinstance(js, list) else [js]):
            x = j["daily"]
            idx = [i for i, d in enumerate(x["time"]) if d <= date_str]
            if len(idx) < 60:
                continue
            x = {k: [x[k][i] for i in idx] for k in x}
            ev = lambda k: [v for v in x[k][-2:] if v is not None]
            off = [s * math.cos(math.radians(d - OFFSHORE_FROM_DEG))
                   for s, d in zip(x["wind_speed_10m_max"][-2:], x["wind_direction_10m_dominant"][-2:])
                   if s is not None and d is not None]
            rain = [v for v in x["precipitation_sum"][-7:] if v is not None]
            days = [(dt.date.fromisoformat(d).timetuple().tm_yday, a, b, c, e, p) for d, a, b, c, e, p in
                    zip(x["time"], x["temperature_2m_max"], x["temperature_2m_min"], x["relative_humidity_2m_max"],
                        x["relative_humidity_2m_min"], x["precipitation_hours"]) if None not in (a, b, c, e, p)]
            v = {"era5_gust_event": max(ev("wind_gusts_10m_max"), default=None),
                 "era5_offshore_event": max(off, default=None),
                 "era5_rh_min_event": min(ev("relative_humidity_2m_min"), default=None),
                 "era5_vpd_max_event": max(ev("vapour_pressure_deficit_max"), default=None),
                 "era5_rain_7d": sum(rain) if len(rain) == 7 else None}
            if len(days) >= 60:
                v["era5_fm100"], v["era5_fm1000"] = dead_fuel_moisture(days, cell[0])
            out[cell] = {k: val for k, val in v.items() if val is not None}
    return out


HIST_FORECAST = "https://historical-forecast-api.open-meteo.com/v1/forecast"
CORRECTED = ("era5_gust_event", "era5_offshore_event", "era5_rh_min_event", "era5_vpd_max_event", "era5_rain_7d",
             "era5_fm100", "era5_fm1000")


def fit_forecast_correction(n, out, seed=0):
    """A6a: linear forecast -> ERA5 correction per variable (and per region when it has >= 60
    pairs), from n random (cell, date) pairs in 2022-2026 where both the archived forecast and
    ERA5 exist. Same event definitions on both sides."""
    import json
    import random
    from routes import ml_ingest as mi
    from ml import gates, fire_danger as fd
    rng = random.Random(seed)
    cells = sorted({cell_of(*t) for t in land_tiles()})
    pairs = []
    for _ in range(n):
        c = rng.choice(cells)
        d = dt.date(2022, 3, 1) + dt.timedelta(days=rng.randrange((dt.date.today() - dt.timedelta(days=10) - dt.date(2022, 3, 1)).days))
        pairs.append((c, str(d)))
    rows = []
    global FORECAST
    live_url, FORECAST = FORECAST, HIST_FORECAST
    try:
        for c, d in pairs:
            try:
                f = forecast_cells([c], d, per_request=1).get(c, {})
                e = {f"era5_{k}": v for k, v in mi._era5_event(c[0], c[1], d).items()}
                e.update(fd.era5_fuel(c[0], c[1], d))
                rows.append({"cell": c, "date": d, "f": f, "e": e})
            except Exception:
                continue
    finally:
        FORECAST = live_url
    elev = {}
    try:
        from ml.absolute_risk import _elevations
        uniq = sorted({r["cell"] for r in rows})
        elev = dict(zip(uniq, _elevations(uniq)))
    except Exception:
        pass
    corr = {}
    for k in CORRECTED:
        xy = [(r["f"][k], r["e"][k], gates.region_of(np.array([r["cell"][0]]), np.array([elev.get(r["cell"], 0.0) or 0.0]))[0])
              for r in rows if k in r["f"] and k in r["e"]]
        if len(xy) < 30:
            continue
        x, y, reg = np.array([a for a, _, _ in xy]), np.array([b for _, b, _ in xy]), np.array([g for _, _, g in xy])
        slope, icpt = np.polyfit(x, y, 1)
        entry = {"all": [round(float(slope), 4), round(float(icpt), 4)], "n": int(len(x)),
                 "mae_before": round(float(np.mean(np.abs(x - y))), 3),
                 "mae_after": round(float(np.mean(np.abs(slope * x + icpt - y))), 3)}
        for g in np.unique(reg):
            m = reg == g
            if m.sum() >= 60:
                s_, i_ = np.polyfit(x[m], y[m], 1)
                entry[str(g)] = [round(float(s_), 4), round(float(i_), 4)]
        corr[k] = entry
    json.dump({"method": "forecast_to_era5_v0", "pairs": len(rows), "corrections": corr}, open(out, "w"), indent=1)
    print(json.dumps({k: {x: v[x] for x in ("n", "mae_before", "mae_after")} for k, v in corr.items()}, indent=1))


def apply_correction(values, region, corr):
    """values: {era5_* key: forecast value} -> ERA5-equivalent using the fitted file."""
    out = dict(values)
    for k, v in values.items():
        c = corr.get("corrections", {}).get(k)
        if c is not None and c["mae_after"] < c["mae_before"]:  # apply only where it measurably helps
            slope, icpt = c.get(region, c["all"])
            out[k] = slope * v + icpt
    return out
