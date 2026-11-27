"""Fire-danger inputs (Weather App Plan, "Fire-danger inputs + reference benchmark").

Stored raw per source, blended at training time like the weather:
  gridmet_*  USFS-style dead fuel moisture (100 h, 1000 h), precipitation, solar
             radiation, min/max RH from gridMET (4 km, 1979+, Earth Engine)
  era5_fm*   the same fuel moisture computed by us (ml.fuel_moisture, NFDRS 1978)
             from ERA5 daily weather: the second source for the two-source rule
  slope, aspect  USGS 3DEP 10 m (static)
The reference systems (ERC, Burning Index, Red Flag Warnings) are benchmarks only
and are never stored here as inputs.
"""
import datetime as dt
import logging

from ml.fuel_moisture import dead_fuel_moisture

GRIDMET_BANDS = ("fm100", "fm1000", "pr", "srad", "rmin", "rmax")
logger = logging.getLogger(__name__)
DANGER_COLS = [f"gridmet_{b}" for b in GRIDMET_BANDS] + ["era5_fm100", "era5_fm1000", "slope", "aspect"]
# Reference systems, stored for the benchmark gate ONLY (never model inputs; ml.train_ab enforces it):
# USFS NFDRS Energy Release Component and Burning Index from gridMET.
BENCH_COLS = ["bench_erc", "bench_bi"]


def _ee():
    from data.live_evi_gee import _ensure_ee_initialized
    _ensure_ee_initialized()
    import ee
    return ee


def gridmet_at(date_str, points):
    """points: [(key, lat, lon)] sharing one date -> {key: {col: value}} (gridMET + terrain),
    one Earth Engine call for all of them."""
    ee = _ee()
    t = dt.date.fromisoformat(date_str)
    gm = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterDate(date_str, str(t + dt.timedelta(days=1))).first()
    terrain = ee.Terrain.products(ee.ImageCollection("USGS/3DEP/10m_collection").mosaic().setDefaultProjection("EPSG:4269", None, 10))
    img = gm.select(list(GRIDMET_BANDS) + ["erc", "bi"]).addBands(terrain.select(["slope", "aspect"]))
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(lon, lat), {"k": str(k)}) for k, lat, lon in points])
    out = {}
    for f in img.reduceRegions(fc, ee.Reducer.first(), 30).getInfo()["features"]:
        p = f["properties"]
        out[p["k"]] = {**{f"gridmet_{b}": p.get(b) for b in GRIDMET_BANDS}, "slope": p.get("slope"), "aspect": p.get("aspect"),
                       "bench_erc": p.get("erc"), "bench_bi": p.get("bi")}
    return out


def era5_fuel(lat, lon, date_str):
    """Our NFDRS 100 h / 1000 h fuel moisture from 90 days of ERA5 daily weather."""
    from data.open_meteo import archive
    t = dt.date.fromisoformat(date_str)
    x = archive({"latitude": lat, "longitude": lon, "start_date": str(t - dt.timedelta(days=89)), "end_date": date_str,
                 "daily": "temperature_2m_max,temperature_2m_min,relative_humidity_2m_max,"
                          "relative_humidity_2m_min,precipitation_hours",
                 "timezone": "America/Los_Angeles"}, timeout=60).json()["daily"]
    days = [(dt.date.fromisoformat(d).timetuple().tm_yday, a, b, c, e, p) for d, a, b, c, e, p in
            zip(x["time"], x["temperature_2m_max"], x["temperature_2m_min"], x["relative_humidity_2m_max"],
                x["relative_humidity_2m_min"], x["precipitation_hours"]) if None not in (a, b, c, e, p)]
    if len(days) < 60:  # 1000-hour moisture needs the spin-up
        return {}
    mc100, mc1000 = dead_fuel_moisture(days, lat)
    return {"era5_fm100": mc100, "era5_fm1000": mc1000}


def danger_sources(lat, lon, date_str):
    """Every fire-danger column for one point (daily ingest). Missing sources are absent."""
    out = {}
    try:
        out.update(gridmet_at(date_str, [("p", lat, lon)]).get("p", {}))
    except Exception as e:  # logged (reaches the failure digest); the ERA5 source still answers
        logger.warning("gridMET fire-danger failed @ %.3f,%.3f %s: %s", lat, lon, date_str, e)
    out.update(era5_fuel(lat, lon, date_str))  # RateLimited propagates
    return {k: v for k, v in out.items() if v is not None}


# Temperature vs normal (B4): monthly 1991-2020 normals from two sources.
NORMAL_COLS = ["era5l_tnorm", "era5l_rhnorm", "prism_tnorm", "prism_rhnorm"]
# Soil moisture (B1): ERA5-Land layers 1-2 (2001+) and SMAP L4 surface (2015-04+).
SOIL_COLS = ["era5l_sm1", "era5l_sm2", "smap_sm"]


def _rh(t, td):
    import math
    return 100 * math.exp(17.625 * td / (243.04 + td)) / math.exp(17.625 * t / (243.04 + t))


def _reduce(img, points, scale):
    ee = _ee()
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(lon, lat), {"k": str(k)}) for k, lat, lon in points])
    # A one-band image comes back as property "first", not its band name; name it.
    red = ee.Reducer.first().setOutputs(img.bandNames()) if img.bandNames().size().getInfo() == 1 else ee.Reducer.first()
    return {f["properties"]["k"]: f["properties"] for f in img.reduceRegions(fc, red, scale).getInfo()["features"]}


def normals_at(month, points):
    """points: [(key, lat, lon)] -> {key: NORMAL_COLS}; one call per source for all points."""
    ee = _ee()
    era = (ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filter(ee.Filter.calendarRange(1991, 2020, "year"))
           .filter(ee.Filter.calendarRange(month, month, "month")).select(["temperature_2m", "dewpoint_temperature_2m"]).mean())
    prism = ee.ImageCollection("OREGONSTATE/PRISM/Norm91m").filter(ee.Filter.eq("system:index", f"{month:02d}")).first()
    a, b = _reduce(era, points, 11132), _reduce(prism.select(["tmean", "tdmean"]), points, 800)
    out = {}
    for k, _, _ in points:
        k = str(k)
        e, p = a.get(k, {}), b.get(k, {})
        r = {}
        if e.get("temperature_2m") is not None and e.get("dewpoint_temperature_2m") is not None:
            t, td = e["temperature_2m"] - 273.15, e["dewpoint_temperature_2m"] - 273.15
            r.update(era5l_tnorm=round(t, 3), era5l_rhnorm=round(_rh(t, td), 2))
        if p.get("tmean") is not None and p.get("tdmean") is not None:
            r.update(prism_tnorm=round(p["tmean"], 3), prism_rhnorm=round(_rh(p["tmean"], p["tdmean"]), 2))
        out[k] = r
    return out


def soil_at(date_str, points):
    ee = _ee()
    t = dt.date.fromisoformat(date_str)
    nxt = str(t + dt.timedelta(days=1))
    era = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(date_str, nxt).first().select(
        ["volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2"])
    a = _reduce(era, points, 11132)
    b = {}
    if date_str >= "2015-04-01":
        smap = ee.ImageCollection("NASA/SMAP/SPL4SMGP/008").filterDate(date_str, nxt).select("sm_surface").mean()
        b = _reduce(smap, points, 11000)
    return {str(k): {c: v for c, v in (("era5l_sm1", a.get(str(k), {}).get("volumetric_soil_water_layer_1")),
                                       ("era5l_sm2", a.get(str(k), {}).get("volumetric_soil_water_layer_2")),
                                       ("smap_sm", b.get(str(k), {}).get("sm_surface"))) if v is not None}
            for k, _, _ in points}
