"""Elevation lookups: USGS 3DEP primary, the same 3DEP data via Earth Engine second,
Open-Elevation last, DB tile cache.

Tile granularity: 0.01° (~1.1 km at California latitudes). Elevation is static
geographically, so cache entries never expire.
"""
from __future__ import annotations

import logging
import time
from decimal import Decimal

import requests

logger = logging.getLogger(__name__)

_USGS_URL = "https://epqs.nationalmap.gov/v1/json"
_OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
_TIMEOUT = 3  # tight so batch calls fail fast and fall through to IDW


def _tile(lat: float, lon: float) -> tuple[Decimal, Decimal]:
    return Decimal(f"{round(lat, 2):.4f}"), Decimal(f"{round(lon, 2):.4f}")


def _fetch_usgs(lat: float, lon: float) -> float | None:
    try:
        r = requests.get(
            _USGS_URL,
            params={"x": lon, "y": lat, "units": "Meters", "wkid": 4326, "includeDate": False},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        v = r.json().get("value")
        return float(v) if v is not None else None
    except Exception as e:
        logger.debug("USGS 3DEP miss at %s,%s: %s", lat, lon, e)
        return None


def _fetch_ee_3dep(lat: float, lon: float) -> float | None:
    """The same USGS 3DEP 10 m data through Earth Engine. Added 2026-11-26: the USGS
    point service slowed past its timeout while Open-Elevation rate-limited us, and
    both failing dropped backfill rows."""
    try:
        from data.live_evi_gee import _ensure_ee_initialized
        _ensure_ee_initialized()
        import ee
        v = (ee.ImageCollection("USGS/3DEP/10m_collection").mosaic()
             .reduceRegion(ee.Reducer.first(), ee.Geometry.Point(lon, lat), 10).getInfo() or {}).get("elevation")
        return float(v) if v is not None else None
    except Exception as e:
        logger.debug("EE 3DEP miss at %s,%s: %s", lat, lon, e)
        return None


def _fetch_open_elevation(lat: float, lon: float) -> float | None:
    for attempt in range(3):
        try:
            r = requests.get(_OPEN_ELEVATION_URL, params={"locations": f"{lat},{lon}"}, timeout=_TIMEOUT)
            r.raise_for_status()
            results = r.json().get("results") or []
            return float(results[0]["elevation"]) if results else None
        except Exception:
            if attempt == 2:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def get_elevation(lat: float, lon: float) -> float:
    """Return elevation (meters) at (lat, lon).

    Order: DB tile cache → USGS 3DEP → 3DEP via Earth Engine → Open-Elevation.
    Result is cached for future calls. Raises only if all fail and cache is empty.
    """
    from models import db, FeatureCacheElevation

    tlat, tlon = _tile(lat, lon)
    cached = FeatureCacheElevation.query.filter_by(tile_lat=tlat, tile_lon=tlon).first()
    if cached is not None:
        return float(cached.elevation_m)

    elev = _fetch_usgs(lat, lon)
    source = "usgs_3dep"
    if elev is None:
        elev, source = _fetch_ee_3dep(lat, lon), "usgs_3dep_ee"
    if elev is None:
        elev = _fetch_open_elevation(lat, lon)
        source = "open_elevation"
    if elev is None:
        raise RuntimeError(f"USGS, Earth Engine 3DEP and Open-Elevation all failed at {lat},{lon}")

    try:
        db.session.merge(FeatureCacheElevation(
            tile_lat=tlat, tile_lon=tlon, elevation_m=elev, source=source,
        ))
        db.session.commit()
    except Exception as e:
        logger.warning("Elevation cache write failed at %s,%s: %s", lat, lon, e)
        db.session.rollback()
    return elev
