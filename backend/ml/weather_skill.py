"""Weekly weather-source skill check, the way forecast blends stay honest: score
each source and the production blend against real stations, and flag when the
blend degrades, a source stops answering, or measured-best weights drift away
from the ones in production (WEATHER_WEIGHTS in routes/ml_ingest.py).

Truth: 15 California ASOS airports (Iowa Environmental Mesonet) plus 7 USCRN
reference sites (temperature and humidity only; USCRN has no 10 m wind).
"""
import csv
import datetime as dt
import io
import statistics as st
from collections import defaultdict

import requests

ASOS = {
    "LGB": (33.812, -118.147), "LAX": (33.938, -118.389), "SAN": (32.734, -117.183), "SFO": (37.619, -122.375),
    "OAK": (37.721, -122.221), "SBA": (34.426, -119.843), "MRY": (36.588, -121.845), "SNA": (33.676, -117.868),
    "SAC": (38.507, -121.495), "FAT": (36.780, -119.719), "BFL": (35.434, -119.054), "RDD": (40.518, -122.299),
    "PSP": (33.829, -116.507), "TRK": (39.320, -120.140), "BIH": (37.371, -118.358),
}
USCRN = ["Bodega_6_WSW", "Fallbrook_5_NE", "Merced_23_WSW", "Redding_12_WNW", "Santa_Barbara_11_W",
         "Stovepipe_Wells_1_SW", "Yosemite_Village_12_W"]
# Validated 2026-11-25 (212 samples, leave-one-site-out): blend MAE per feature.
BASELINE = {"humidity": 4.47, "temperature_celsius": 0.59, "wind_speed": 1.98}
DEGRADED_FACTOR = 1.5
MIN_SOURCE_COVERAGE = 0.7
WEIGHT_DRIFT = 0.25
_KEYS = ("temperature_celsius", "humidity", "wind_speed")


def _window(d):
    t = dt.date.fromisoformat(d)
    return t - dt.timedelta(days=3), t + dt.timedelta(days=3)


def _daily(days):
    """days: {date: {k: [hourly]}} -> ±3-day means in the model's definitions."""
    good = [v for v in days.values() if len(v["temperature_celsius"]) >= 18 and len(v["humidity"]) >= 18]
    if len(good) < 5:
        return None
    out = {"temperature_celsius": st.mean(st.mean(v["temperature_celsius"]) for v in good),
           "humidity": st.mean(st.mean(v["humidity"]) for v in good)}
    if all(v["wind_speed"] for v in good):
        out["wind_speed"] = st.mean(max(v["wind_speed"]) for v in good)
    return out


def asos_truth(sid, d):
    a, b = _window(d)
    c = b + dt.timedelta(days=1)  # IEM's end date is exclusive
    r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py", params={
        "station": sid, "data": ["tmpf", "relh", "sknt"], "tz": "America/Los_Angeles", "format": "onlycomma",
        "missing": "empty", "year1": a.year, "month1": a.month, "day1": a.day,
        "year2": c.year, "month2": c.month, "day2": c.day,
    }, timeout=60)
    r.raise_for_status()
    days = defaultdict(lambda: {k: [] for k in _KEYS})
    for row in csv.DictReader(io.StringIO(r.text)):
        day = row["valid"][:10]
        if not (str(a) <= day <= str(b)):
            continue
        if row["tmpf"]:
            days[day]["temperature_celsius"].append((float(row["tmpf"]) - 32) / 1.8)
        if row["relh"]:
            days[day]["humidity"].append(float(row["relh"]))
        if row["sknt"]:
            days[day]["wind_speed"].append(float(row["sknt"]) * 0.5144)
    return _daily(days)


def uscrn_truth(site, d):
    a, b = _window(d)
    r = requests.get(f"https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/{a.year}/CRNH0203-{a.year}-CA_{site}.txt", timeout=60)
    r.raise_for_status()
    days, lat, lon = defaultdict(lambda: {k: [] for k in _KEYS}), None, None
    for line in r.text.splitlines():
        c = line.split()
        day = f"{c[3][:4]}-{c[3][4:6]}-{c[3][6:]}"
        if not (str(a) <= day <= str(b)):
            continue
        lon, lat = float(c[6]), float(c[7])
        if float(c[9]) > -9000:
            days[day]["temperature_celsius"].append(float(c[9]))
        if float(c[26]) > -9000:
            days[day]["humidity"].append(float(c[26]))
    return _daily(days), lat, lon


def skill_report(date_str):
    """Score each source and the production blend at every station for the ±3-day
    window around date_str. Returns a dict with per-feature MAEs and problems."""
    from routes import ml_ingest as mi
    fetch = {"rtma": mi._rtma_weather_for_date, "merra2": mi._power_weather_for_date,
             "era5": lambda la, lo, d: (lambda e: e and {"temperature_celsius": e["air_temp_encoded"] * 0.02 - 273.15,
                                                         "humidity": e["humidity"], "wind_speed": e["wind"]})(mi._weather_for_date(la, lo, d))}
    sites = [(f"asos:{s}", lambda s=s: (asos_truth(s, date_str), *ASOS[s])) for s in ASOS]
    sites += [(f"uscrn:{s}", lambda s=s: uscrn_truth(s, date_str)) for s in USCRN]
    errs = {name: {k: [] for k in _KEYS} for name in (*fetch, "blend")}
    answered = defaultdict(int)
    n_sites = 0
    for _, get_truth in sites:
        try:
            truth, lat, lon = get_truth()
        except Exception:
            continue
        if not truth:
            continue
        n_sites += 1
        srcs = {}
        for name, f in fetch.items():
            v = f(lat, lon, date_str)
            if v:
                srcs[name] = v
                answered[name] += 1
        try:
            blend, _ = mi._blend_weather(srcs) if srcs else (None, None)
        except mi.CrossSourceMismatch:
            blend = None
        for name, v in (*srcs.items(), ("blend", blend)):
            if v:
                for k in truth:
                    errs[name][k].append(abs(v[k] - truth[k]))
    mae = {n: {k: round(st.mean(e), 3) for k, e in ks.items() if e} for n, ks in errs.items()}
    problems = []
    for k, base in BASELINE.items():
        b = mae.get("blend", {}).get(k)
        if b is not None and b > base * DEGRADED_FACTOR:
            problems.append(f"blend {k} error {b:.2f} > {DEGRADED_FACTOR}x validated {base}")
    for name in fetch:
        if n_sites and answered[name] / n_sites < MIN_SOURCE_COVERAGE:
            problems.append(f"{name} answered at {answered[name]}/{n_sites} sites")
    suggested = {}
    for k in _KEYS:
        m = {n: mae[n].get(k) for n in fetch if mae.get(n, {}).get(k)}
        if len(m) >= 2:
            inv = {n: 1 / e ** 2 for n, e in m.items()}
            suggested[k] = {n: round(v / sum(inv.values()), 2) for n, v in inv.items()}
            current = mi.WEATHER_WEIGHTS[k]
            cur_tot = sum(current.values())
            drift = max(abs(suggested[k].get(n, 0) - current.get(n, 0) / cur_tot) for n in set(suggested[k]) | set(current))
            if k == "humidity" and drift > WEIGHT_DRIFT:  # only humidity is blended in production
                problems.append(f"humidity weights drifted {drift:.2f}: measured-best {suggested[k]} vs production {current}")
    if n_sites < 10:
        problems.append(f"only {n_sites} stations had usable truth")
    return {"date": date_str, "sites": n_sites, "mae": mae, "answered": dict(answered),
            "suggested_weights": suggested, "baseline": BASELINE, "problems": problems, "healthy": not problems}


def dataset_report(path, days=14):
    """Recent training rows: every fire should have its control and vice versa."""
    since = (dt.date.today() - dt.timedelta(days=days + 7)).isoformat()
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    fires = {(r["lat"], r["lon"], r["acq_date"]) for r in rows if r["fire"] == "1" and r["acq_date"] >= since}
    def prior(d):
        t = dt.date.fromisoformat(d)
        return t.replace(year=t.year - 1) if not (t.month == 2 and t.day == 29) else t.replace(year=t.year - 1, day=28)
    controls = {(r["lat"], r["lon"], r["acq_date"]) for r in rows if r["fire"] == "0"}
    unpaired = [f for f in fires if (f[0], f[1], prior(f[2]).isoformat()) not in controls]
    problems = []
    if not fires:
        problems.append(f"no fire rows ingested since {since}")
    elif len(unpaired) > 0.2 * len(fires):
        problems.append(f"{len(unpaired)} of {len(fires)} recent fires have no matched control")
    return {"since": since, "recent_fires": len(fires), "unpaired_fires": len(unpaired), "problems": problems}
