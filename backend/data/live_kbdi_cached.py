"""KBDI lookups with DB tile cache.

Wraps Sania's data/live_kbdi.get_kbdi (NASA POWER 30-day window → Keetch-Byram
integration). KBDI changes daily, so DB entries are considered fresh for 24h.
Tile granularity: 0.25° (~27 km; was 0.01°). KBDI is a 30-day soil-drought integral that varies
smoothly, and 0.01° meant ~4,100 NASA POWER requests a day for the live map: POWER returned 429
and the Open-Meteo fallback exhausted its daily quota (2026-11-27).

This is the LARGEST single performance win on the cold-zone-compute path.
Before: 200 centroids × ~10s NASA POWER round-trip = 130-200s wall time.
After:  200 centroids × <5ms DB read = <1s once warm.
"""
from __future__ import annotations

import datetime as _dt
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)

_TTL_HOURS = 24


_STALE_OK_HOURS = 72  # if the live fetch fails, a recent value beats IDW


def _tile(lat: float, lon: float) -> tuple[Decimal, Decimal]:
    return Decimal(f"{round(lat * 4) / 4:.4f}"), Decimal(f"{round(lon * 4) / 4:.4f}")


def get_kbdi_cached(lat: float, lon: float) -> float:
    """Return KBDI at (lat, lon).

    Order: DB tile cache (if <24h old) → live NASA POWER fetch → cache
    write-through. Raises only if upstream fails and no cache entry exists.
    """
    from models import db, FeatureCacheKbdi

    tlat, tlon = _tile(lat, lon)
    cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=_TTL_HOURS)

    cached = FeatureCacheKbdi.query.filter_by(tile_lat=tlat, tile_lon=tlon).first()
    at = cached.fetched_at if cached is not None else None
    if at is not None and at.tzinfo is None:  # SQLite returns naive UTC; Postgres returns aware
        at = at.replace(tzinfo=_dt.timezone.utc)
    if at and at > cutoff:
        return float(cached.kbdi)
    stale = (float(cached.kbdi), at) if at else None
    db.session.commit()  # release the pool connection before the slow network call

    # Live fetch (slow — that's why we cache), at the cell centre so the cell has one value
    from data.live_kbdi import get_kbdi as _live_get_kbdi
    try:
        value = float(_live_get_kbdi(float(tlat), float(tlon)))
    except Exception as e:
        if stale and stale[1] > cutoff - _dt.timedelta(hours=_STALE_OK_HOURS - _TTL_HOURS):
            logger.warning("KBDI live fetch failed at %s,%s (%s); using cached value from %s", lat, lon, e, stale[1])
            return stale[0]
        raise

    try:
        from models import merge_commit
        merge_commit(FeatureCacheKbdi(tile_lat=tlat, tile_lon=tlon, kbdi=value, source="nasa_power"))
    except Exception as e:
        logger.warning("KBDI cache write failed at %s,%s: %s", lat, lon, e)
        db.session.rollback()
    return value
