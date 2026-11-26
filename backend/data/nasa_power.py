"""NASA POWER daily point data for the training path.

POWER throttles an IP for about a minute when requests come too fast and has no
paid tier, so: optional pacing (POWER_MIN_INTERVAL seconds between this
process's requests), patient backoff on 429, then RateLimited -- never a silent
switch to another source. One request per (point, date) serves both the weather
window (date +/- 3 days) and the KBDI spin-up (the 30 days to date); the values
are the same daily records two separate requests would return."""
import datetime as dt
import os
import sys
import threading
import time
from collections import defaultdict
from functools import lru_cache

import requests

from data.open_meteo import RateLimited

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
# WD10M/QV2M/PS feed the event-day fire-weather features (offshore push, minimum
# humidity, VPD); same request, so no extra calls.
PARAMS = "T2M,RH2M,WS10M_MAX,T2M_MAX,PRECTOTCORR,WD10M,QV2M,PS"
_lock = threading.Lock()
_last = [0.0]


def _get(params, timeout=30):
    gap = float(os.environ.get("POWER_MIN_INTERVAL") or 0)
    # Long waits only for paced batch jobs; a web request must not hang for minutes.
    for wait in ((60, 120, 240, None) if gap else (5, None)):
        if gap:
            with _lock:
                time.sleep(max(0.0, _last[0] + gap - time.monotonic()))
                _last[0] = time.monotonic()
        r = requests.get(URL, params=params, timeout=timeout)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        if wait is None:
            raise RateLimited("NASA POWER 429 after backing off")
        print(f"POWER 429, waiting {wait}s", file=sys.stderr, flush=True)  # visible in batch logs
        time.sleep(wait)


_key_locks = defaultdict(threading.Lock)


def window(lat, lon, date_str):
    """{param: {YYYYMMDD: value}} for date-29 .. date+3 (-999 = missing). Callers in
    parallel threads wait for one request instead of each sending their own."""
    with _key_locks[(lat, lon, date_str)]:
        return _window(lat, lon, date_str)


@lru_cache(maxsize=4096)
def _window(lat, lon, date_str):
    t = dt.datetime.strptime(date_str, "%Y-%m-%d")
    r = _get({"parameters": PARAMS, "community": "AG", "latitude": lat, "longitude": lon, "format": "JSON",
              "start": (t - dt.timedelta(days=29)).strftime("%Y%m%d"),
              "end": (t + dt.timedelta(days=3)).strftime("%Y%m%d")})
    return r.json()["properties"]["parameter"]


window.cache_clear = _window.cache_clear  # tests reset the cache
