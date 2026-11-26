"""Daily station-truth table (Weather App Plan, phase 1): every California ASOS/AWOS
station in the Iowa Environmental Mesonet archive, quality-controlled, in the
model's daily definitions (mean temperature, mean humidity, max wind), so each
gridded source can be scored against it per region.

  python -m ml.station_truth --start 2001 --out stations/asos_daily.csv

Resumable: stations-years already in --out are skipped. One request at a time
(IEM asks for polite sequential use).
"""
import argparse
import csv
import datetime as dt
import io
import math
import os
import statistics as st
import sys
import time
from collections import defaultdict

import requests

NETWORK = "https://mesonet.agron.iastate.edu/geojson/network/CA_ASOS.geojson"
ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
RANGE = {"temperature_celsius": (-40, 55), "humidity": (1, 100), "wind_speed": (0, 60)}
SPIKE = {"temperature_celsius": 10.0, "humidity": 40.0, "wind_speed": 20.0}  # vs both neighbours
MIN_OBS = 18  # same completeness rule as ml.weather_skill._daily
COLS = ["station", "date", "lat", "lon", "elevation", "county", "wfo",
        "temperature_celsius", "humidity", "wind_speed", "n_temp", "n_rh", "n_wind", "qc_dropped",
        # event-day truth (Gust Backfill Plan G0): day's peak gust (max of reported gusts and
        # sustained winds: ASOS reports a gust only when gusty), max offshore push, min RH,
        # max VPD, precipitation (only with >= MIN_OBS hourly reports)
        "gust_max", "offshore_max", "rh_min", "vpd_max", "precip_mm"]


def stations():
    feats = requests.get(NETWORK, timeout=60).json()["features"]
    return [{"station": f["properties"]["sid"], "lon": f["geometry"]["coordinates"][0],
             "lat": f["geometry"]["coordinates"][1], "elevation": f["properties"].get("elevation"),
             "county": f["properties"].get("county"), "wfo": f["properties"].get("wfo"),
             "begin": (f["properties"].get("archive_begin") or "1900")[:4],
             "end": (f["properties"].get("archive_end") or "9999")[:4]} for f in feats]


def _qc(name, obs):
    """obs: [(time, value)] in time order -> (kept, dropped times). Range check,
    then single-reading spikes that differ from both neighbours."""
    lo, hi = RANGE[name]
    obs = [(t, v) for t, v in obs if lo <= v <= hi]
    keep, dropped = [], []
    for i, (t, v) in enumerate(obs):
        if 0 < i < len(obs) - 1 and abs(v - obs[i - 1][1]) > SPIKE[name] and abs(v - obs[i + 1][1]) > SPIKE[name]:
            dropped.append(t)
        else:
            keep.append((t, v))
    return keep, dropped


def daily(rows):
    """rows: IEM onlycomma dicts for one station -> {date: record}. A variable that
    is flat all day (a stuck sensor) or has < MIN_OBS readings is left empty."""
    series = {k: [] for k in RANGE}
    for r in rows:
        if r["tmpf"]:
            series["temperature_celsius"].append((r["valid"], (float(r["tmpf"]) - 32) / 1.8))
        if r["relh"]:
            series["humidity"].append((r["valid"], float(r["relh"])))
        if r["sknt"]:
            series["wind_speed"].append((r["valid"], float(r["sknt"]) * 0.5144))
    ev = defaultdict(lambda: {"gust": [], "off": [], "rh": [], "vpd": [], "p": [], "n": 0})
    for r in rows:
        e = ev[r["valid"][:10]]
        sk = float(r["sknt"]) * 0.5144 if r["sknt"] else None
        if sk is not None and RANGE["wind_speed"][0] <= sk <= RANGE["wind_speed"][1]:
            e["gust"].append(sk)
            if r.get("drct"):
                e["off"].append(sk * math.cos(math.radians(float(r["drct"]) - 45)))
        if r.get("gust"):
            g = float(r["gust"]) * 0.5144
            if g <= RANGE["wind_speed"][1]:
                e["gust"].append(g)
        if r["tmpf"] and r["relh"]:
            tc, rh = (float(r["tmpf"]) - 32) / 1.8, float(r["relh"])
            if RANGE["temperature_celsius"][0] <= tc <= RANGE["temperature_celsius"][1] and 1 <= rh <= 100:
                es = 0.6108 * math.exp(17.27 * tc / (tc + 237.3))
                e["rh"].append(rh); e["vpd"].append(es * (1 - rh / 100))
                e["n"] += 1
        if r.get("p01i") not in (None, "", "M"):
            e["p"].append(float(r["p01i"]) * 25.4)
    days = defaultdict(lambda: {k: [] for k in RANGE})
    drops = defaultdict(int)
    for k, obs in series.items():
        kept, dropped = _qc(k, obs)
        for t, v in kept:
            days[t[:10]][k].append(v)
        for t in dropped:
            drops[t[:10]] += 1
    out = {}
    for d, v in days.items():
        rec = {"n_temp": len(v["temperature_celsius"]), "n_rh": len(v["humidity"]),
               "n_wind": len(v["wind_speed"]), "qc_dropped": drops[d]}
        for k, agg in (("temperature_celsius", st.mean), ("humidity", st.mean), ("wind_speed", max)):
            vals = v[k]
            flat = k != "wind_speed" and len(vals) >= MIN_OBS and max(vals) == min(vals)  # calm all day is real
            rec[k] = round(agg(vals), 3) if len(vals) >= MIN_OBS and not flat else ""
        e = ev[d]
        full = e["n"] >= MIN_OBS
        rec.update({"gust_max": round(max(e["gust"]), 2) if len(e["gust"]) >= MIN_OBS else "",
                    "offshore_max": round(max(e["off"]), 2) if len(e["off"]) >= MIN_OBS else "",
                    "rh_min": round(min(e["rh"]), 1) if full else "", "vpd_max": round(max(e["vpd"]), 3) if full else "",
                    "precip_mm": round(sum(e["p"]), 1) if full else ""})
        out[d] = rec
    return out


def fetch_year(sid, year):
    for attempt in range(4):
        try:
            r = requests.get(ASOS, params={
                "station": sid, "data": ["tmpf", "relh", "sknt", "drct", "gust", "p01i"], "tz": "America/Los_Angeles",
                "format": "onlycomma", "missing": "empty", "report_type": [3, 4],
                "year1": year, "month1": 1, "day1": 1, "year2": year + 1, "month2": 1, "day2": 1,
            }, timeout=180, headers={"User-Agent": "FireScope/1.0"})
            r.raise_for_status()
            return list(csv.DictReader(io.StringIO(r.text)))
        except Exception:
            if attempt == 3:
                raise
            time.sleep(30 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2001)
    ap.add_argument("--end", type=int, default=dt.date.today().year)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        with open(a.out, newline="") as f:
            done = {(r["station"], r["date"][:4]) for r in csv.DictReader(f)}
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=COLS)
    if new:
        w.writeheader()
    for s in stations():
        for year in range(max(a.start, int(s["begin"])), min(a.end, int(s["end"])) + 1):
            if (s["station"], str(year)) in done:
                continue
            try:
                days = daily(fetch_year(s["station"], year))
            except Exception as e:
                print(f"{s['station']} {year} failed: {e}", file=sys.stderr, flush=True)
                continue
            for d, rec in sorted(days.items()):
                w.writerow({**{k: s[k] for k in ("station", "lat", "lon", "elevation", "county", "wfo")},
                            "date": d, **rec})
            f.flush()
            print(f"{s['station']} {year}: {len(days)} days, "
                  f"{sum(r['qc_dropped'] for r in days.values())} spikes dropped", flush=True)
    f.close()


if __name__ == "__main__":
    main()
