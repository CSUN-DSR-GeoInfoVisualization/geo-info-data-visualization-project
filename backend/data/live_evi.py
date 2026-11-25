"""
Live EVI (Enhanced Vegetation Index) from NASA ORNL DAAC MODIS Web Service.
Product: MOD13Q1 (16-day composite, 250m resolution)
Free API — no key required.
"""

import json
import os
import time
import requests
from datetime import datetime, timedelta

_METADATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ml", "models", "model_metadata.json",
)


def _get_evi_data_year() -> int:
    try:
        with open(_METADATA_PATH) as f:
            return int(json.load(f)["data_year"])
    except (FileNotFoundError, KeyError, ValueError):
        return 2020

# Cache at module load — avoids re-reading the JSON file on every prediction
_EVI_DATA_YEAR: int = _get_evi_data_year()

_ORNL_DAAC_URL = "https://modis.ornl.gov/rst/api/v1/MOD13Q1/subset"

# MODIS EVI raw integer valid range — values outside this are fill/error values
_EVI_MIN = -2000
_EVI_MAX = 10000


def _modis_date(dt: datetime) -> str:
    """Convert a datetime to MODIS date format: AYYYY + 3-digit day-of-year."""
    return f"A{dt.year}{dt.timetuple().tm_yday:03d}"


def get_evi(lat: float, lon: float) -> float:
    """
    Fetch the spring MODIS EVI value (closest composite to May 1) for the given
    coordinates. Uses the prior May 1st to match the spring fuel-load encoding
    used during model training.

    Returns a scaled float (raw integer * 0.0001), matching the AppEEARS
    training data encoding. E.g., raw 800 → 0.08.

    Retries up to 3 times on network errors before raising.
    """
    spring_target = datetime(_EVI_DATA_YEAR, 5, 1)

    start_date = spring_target - timedelta(days=24)
    end_date   = spring_target + timedelta(days=24)

    params = {
        "latitude":     lat,
        "longitude":    lon,
        "band":         "250m_16_days_EVI",
        "startDate":    _modis_date(start_date),
        "endDate":      _modis_date(end_date),
        "kmAboveBelow": 0,
        "kmLeftRight":  0,
    }

    for attempt in range(3):
        try:
            response = requests.get(_ORNL_DAAC_URL, params=params, timeout=15)
            response.raise_for_status()
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))

    subsets = response.json().get("subset", [])
    if not subsets:
        raise ValueError("No EVI data returned from MODIS")

    # Pick the composite closest to May 1st
    best_value = None
    best_delta = float("inf")
    for entry in subsets:
        value = entry["data"][0]
        if not (_EVI_MIN <= value <= _EVI_MAX):
            continue
        try:
            entry_date = datetime.strptime(entry["calendar_date"], "%Y-%m-%d")
            delta = abs((entry_date - spring_target).days)
            if delta < best_delta:
                best_delta = delta
                best_value = value
        except (KeyError, ValueError):
            continue

    if best_value is None:
        raise ValueError("No valid EVI value found in MODIS response")

    return float(best_value) * 0.0001


def get_evi_at_ornl(lat: float, lon: float, date_str: str) -> float:
    """Newest MOD13Q1 EVI composite at or before `date_str`, from ORNL DAAC.

    Backup for data.live_evi_gee.get_evi_at (Earth Engine): same product, same
    "newest composite at or before the date" rule, so the value is the same
    quantity. Raises if ORNL has no valid composite in the preceding 32 days."""
    target = datetime.strptime(date_str, "%Y-%m-%d")
    params = {
        "latitude": lat, "longitude": lon, "band": "250m_16_days_EVI",
        "startDate": _modis_date(target - timedelta(days=32)), "endDate": _modis_date(target),
        "kmAboveBelow": 0, "kmLeftRight": 0,
    }
    for attempt in range(3):
        try:
            response = requests.get(_ORNL_DAAC_URL, params=params, timeout=20)
            response.raise_for_status()
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    valid = [(e["calendar_date"], e["data"][0]) for e in response.json().get("subset", [])
             if e.get("data") and _EVI_MIN <= e["data"][0] <= _EVI_MAX and e["calendar_date"] <= date_str]
    if not valid:
        raise ValueError(f"no valid ORNL EVI composite at or before {date_str}")
    return float(max(valid)[1]) * 0.0001
