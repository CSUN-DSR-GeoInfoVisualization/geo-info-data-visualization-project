"""Live MODIS EVI via Google Earth Engine with DB tile cache.

This serves CURRENT EVI conditions (last 32 days). Separate from data/live_evi.py
which pulls the training-year spring composite for model-consistency lookups.

Authentication uses OAuth user credentials supplied via the
`GEE_OAUTH_CREDENTIALS` env var (JSON body from `earthengine authenticate`).
The credentials are written to the standard EE path on first import so
`ee.Initialize()` finds them.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
from decimal import Decimal
from pathlib import Path

logger = logging.getLogger(__name__)

_EE_INIT_DONE = False
_EE_PROJECT = os.environ.get("GEE_PROJECT_ID", "ee-idothecohen")
_COMPOSITE_TTL_DAYS = 16  # MOD13Q1 cadence


def _materialize_credentials_file() -> None:
    """Write GEE_OAUTH_CREDENTIALS env var to the path ee.Initialize expects."""
    raw = os.environ.get("GEE_OAUTH_CREDENTIALS")
    if not raw:
        return
    cred_path = Path.home() / ".config" / "earthengine" / "credentials"
    if cred_path.exists() and cred_path.read_text().strip() == raw.strip():
        return
    cred_path.parent.mkdir(parents=True, exist_ok=True)
    cred_path.write_text(raw)
    cred_path.chmod(0o600)


def _ensure_ee_initialized() -> None:
    global _EE_INIT_DONE
    if _EE_INIT_DONE:
        return
    _materialize_credentials_file()
    import ee  # imported lazily so unit tests without GEE deps still load this module
    ee.Initialize(project=_EE_PROJECT)
    _EE_INIT_DONE = True


def _tile(lat: float, lon: float) -> tuple[Decimal, Decimal]:
    return Decimal(f"{round(lat, 2):.4f}"), Decimal(f"{round(lon, 2):.4f}")


def _evi_composite(lat: float, lon: float, as_of: "_dt.date",
                   newest_valid: bool = False) -> tuple[float, "_dt.date"]:
    """Scaled EVI from the newest MOD13Q1 composite at or before `as_of`, plus
    that composite's date. Shared by the live and historical entry points so
    both compute the same quantity from the same product.

    newest_valid (live map only): the newest composite is masked (cloud, no
    data) at some pixels; use the newest one in the last ~4 composites that has
    a value there instead of failing. Training keeps the default until the
    dataset is rebuilt, so all rows use one definition."""
    _ensure_ee_initialized()
    import ee

    point = ee.Geometry.Point([lon, lat])
    col = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterDate((as_of - _dt.timedelta(days=64 if newest_valid else 32)).isoformat(), as_of.isoformat())
        .select("EVI")
        .sort("system:time_start", False)
    )
    if newest_valid:
        stamped = col.map(lambda i: i.addBands(
            ee.Image(ee.Number(i.get("system:time_start"))).toDouble().rename("t").updateMask(i.mask())))
        # reduce() drops the MODIS grid; restore it so we sample the same 250 m pixel as the default path
        info = stamped.reduce(ee.Reducer.firstNonNull()).reproject(col.first().projection()).reduceRegion(
            reducer=ee.Reducer.first(), geometry=point, scale=250).getInfo() or {}
        if info.get("EVI_first") is None or info.get("t_first") is None:
            raise RuntimeError(f"GEE returned no valid EVI for {lat},{lon} in the 64 days to {as_of}")
        return float(info["EVI_first"]) * 0.0001, _dt.datetime.utcfromtimestamp(info["t_first"] / 1000.0).date()
    img = col.first()
    info = img.reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=250).getInfo()
    raw = info.get("EVI") if isinstance(info, dict) else None
    if raw is None:
        raise RuntimeError(f"GEE returned no EVI for {lat},{lon} as of {as_of}")

    millis = img.get("system:time_start").getInfo()
    composite_date = (
        _dt.datetime.utcfromtimestamp(millis / 1000.0).date()
        if isinstance(millis, (int, float)) else as_of
    )
    return float(raw) * 0.0001, composite_date


def get_evi_at(lat: float, lon: float, date_str: str) -> float:
    """EVI as it was on a PAST date. Not cached — the tile cache is keyed by
    location only and holds the CURRENT composite, so writing a historical value
    into it would poison live serving.

    Needed for matched control rows: a control is the same point on an earlier
    date, and it must carry that date's vegetation state. Note the committed
    2020 base CSV instead stores the SPRING (May 1) composite as a pre-season
    fuel-load proxy (ml/build_dataset.py), which is a different quantity from
    what this and get_evi_live return.
    """
    as_of = _dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    scaled, _ = _evi_composite(lat, lon, as_of)
    return scaled


def get_evi_live(lat: float, lon: float) -> float:
    """Return current scaled EVI at (lat, lon) from MOD13Q1 (250 m).

    Order: DB tile cache (if composite_date within 16 days) → GEE live →
    cache write-through. EVI is returned scaled (raw * 0.0001) matching the
    training data encoding.
    """
    from models import db, FeatureCacheEvi

    tlat, tlon = _tile(lat, lon)
    today = _dt.date.today()
    cutoff = today - _dt.timedelta(days=_COMPOSITE_TTL_DAYS)

    cached = FeatureCacheEvi.query.filter_by(tile_lat=tlat, tile_lon=tlon).first()
    if cached is not None and cached.composite_date >= cutoff:
        return float(cached.evi)

    scaled, composite_date = _evi_composite(lat, lon, today, newest_valid=True)
    try:
        db.session.merge(FeatureCacheEvi(
            tile_lat=tlat, tile_lon=tlon, evi=scaled,
            source="gee_mod13q1", composite_date=composite_date,
        ))
        db.session.commit()
    except Exception as e:
        logger.warning("EVI cache write failed at %s,%s: %s", lat, lon, e)
        db.session.rollback()
    return scaled
