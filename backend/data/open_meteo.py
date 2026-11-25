"""Open-Meteo historical archive. With OPEN_METEO_API_KEY set (paid plan: no daily
cap) requests go to the customer endpoint; otherwise the free one (10,000/day)."""
import os

import requests


class RateLimited(RuntimeError):
    """The source refused us for volume. Callers that write datasets must stop,
    not carry on with fewer sources."""


def archive(params, timeout=15):
    key = os.environ.get("OPEN_METEO_API_KEY")
    url = ("https://customer-archive-api.open-meteo.com/v1/archive" if key
           else "https://archive-api.open-meteo.com/v1/archive")
    r = requests.get(url, params={**params, **({"apikey": key} if key else {})}, timeout=timeout)
    if r.status_code == 429:
        raise RateLimited(f"Open-Meteo archive 429: {r.text[:120]}")
    r.raise_for_status()
    return r
