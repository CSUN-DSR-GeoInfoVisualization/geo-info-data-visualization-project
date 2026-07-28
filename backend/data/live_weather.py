"""
Live weather data fetched from Open-Meteo (https://open-meteo.com/).
Free API — no key required.
"""

import time
from statistics import mean

import requests

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def get_weather(lat: float, lon: float) -> dict:
    """
    Fetch current wind speed (m/s), air temperature (°C), and relative humidity (%)
    for the given coordinates. Retries up to 3 times on network errors.

    Returns dict with keys: wind_speed, temperature_celsius, humidity
    """
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "current":         "wind_speed_10m,temperature_2m,relative_humidity_2m",
        "wind_speed_unit": "ms",
    }

    for attempt in range(3):
        try:
            response = requests.get(_OPEN_METEO_URL, params=params, timeout=8)
            response.raise_for_status()
            data = response.json()
            return {
                "wind_speed":          float(data["current"]["wind_speed_10m"]),
                "temperature_celsius": float(data["current"]["temperature_2m"]),
                "humidity":            float(data["current"]["relative_humidity_2m"]),
            }
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def get_weather_daily(lat: float, lon: float) -> dict:
    """Fetch the model's daily-aggregate fire weather for today: max 10m wind
    (m/s), mean air temperature (°C), and mean relative humidity (%), each
    averaged over a ±3-day window centred on today. Retries up to 3 times.

    Returns the same keys as get_weather (wind_speed, temperature_celsius,
    humidity) so it is a drop-in at the call sites that build model features.

    Why this exists (audit 2026-07-28): the model's features are DEFINED as daily
    aggregates over a ±3-day window — that is how the frozen 2020 base was built
    (wind_speed_10m_max / temperature_2m_mean / relative_humidity_2m_mean) and how
    the backtest scores past fires (routes/ml_ingest.py _weather_for_date).
    get_weather returns an INSTANTANEOUS reading at call time, so every path that
    fed it into the model was serving a different quantity than the one the model
    was trained on (measured: base wind median 4.06 m/s vs 2.10 m/s on ingest rows
    built from instantaneous readings). Model paths use this function; get_weather
    stays as-is for the display/report paths that want current conditions.

    past_days=3 + forecast_days=4 returns exactly the 7-day window centred on
    today, so live scoring and historical scoring compute the same quantity.

    Raises on failure rather than falling back to instantaneous values — a silent
    fallback would reintroduce exactly the definition mismatch this fixes. Callers
    that need resilience have their own per-location fallbacks.
    """
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "daily":           "wind_speed_10m_max,temperature_2m_mean,relative_humidity_2m_mean",
        "wind_speed_unit": "ms",
        "timezone":        "America/Los_Angeles",
        "past_days":       3,
        "forecast_days":   4,
    }

    for attempt in range(3):
        try:
            response = requests.get(_OPEN_METEO_URL, params=params, timeout=8)
            response.raise_for_status()
            daily = response.json()["daily"]
            winds = [v for v in daily["wind_speed_10m_max"] if v is not None]
            temps = [v for v in daily["temperature_2m_mean"] if v is not None]
            hums  = [v for v in daily["relative_humidity_2m_mean"] if v is not None]
            if not winds or not temps or not hums:
                raise ValueError("Open-Meteo returned no usable daily aggregates")
            return {
                "wind_speed":          float(mean(winds)),
                "temperature_celsius": float(mean(temps)),
                "humidity":            float(mean(hums)),
            }
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
