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
                r = requests.get(FORECAST, params={
                    "latitude": ",".join(str(c[0]) for c in chunk), "longitude": ",".join(str(c[1]) for c in chunk),
                    "daily": DAILY, "past_days": 89, "forecast_days": 1, "wind_speed_unit": "ms",
                    "timezone": "America/Los_Angeles"}, timeout=90)
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
