"""FireScope v3.5: the shipped A x B model (Architecture v2), its daily live grid, tile lookup.

  train:  python -m ml.v35 train --data merged_v35.csv --growth growth_merged.csv
              --weights event_weights_v0.json --out ml/models/v35_ab.pkl
  live:   build_grid(date) inside the app (POST /api/research/admin/build-v35-grid, daily cron)

Risk = P(a wildfire that grows >= 1,000 acres starts in this 0.1-degree tile today), from
P_A(start) x P_B(large | start), calibrated, then shifted to the tile's region x month FPA FOD
start rate (A5e). Tiers are NFDRS-style percentile classes (Low < 60th, Moderate 60-80,
High 80-90, Very High 90-97, Extreme >= 97th) of that probability against ordinary days
(the no-fire training rows) in the same region. The display score maps those cut points onto
the site's 0.2 colour bands, so existing map colours and labels stay correct.
"""
import argparse
import datetime as dt
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd

from ml import gates

logger = logging.getLogger(__name__)
ARTIFACT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "v35_ab.pkl")
GRID_KEY = "v35_grid"
GRID_MAX_AGE_H = 36
PCT_CUTS = [0, 60, 80, 90, 97, 100]          # percentile class edges (Ido, 2026-11-26)
DISPLAY = [0, 0.2, 0.4, 0.6, 0.8, 1.0]       # the site's Low/Moderate/High/Very High/Extreme bands


def display_score(pct):
    """Percentile (0-100) -> 0-1 score whose 0.2 bands are exactly the percentile classes."""
    return float(np.interp(pct, PCT_CUTS, DISPLAY))


def train(data, growth, weights_path, out):
    from ml import train_ab as T
    weights = json.load(open(weights_path))["weights"]
    d = pd.read_csv(data, low_memory=False, dtype={"lat": str, "lon": str})
    d["pair_key"] = d["lat"] + "," + d["lon"] + "," + d["acq_date"]
    d["lat"], d["lon"] = pd.to_numeric(d["lat"]), pd.to_numeric(d["lon"])
    d = T.build_inputs(d, weights)
    g = pd.read_csv(growth, low_memory=False)
    g["control_type"] = ""
    g = pd.concat([T.build_inputs(g, weights), d[d["source"].eq("fpa_fod")]], ignore_index=True)
    a, b, _, _ = T.combined_rows(d, g)
    comb = T.fit_combined(a, b)  # shipped model: every row, locked test included (evaluated already)
    # climatology = every no-fire row (29k matched, random-spot, same-season days), not only the
    # 6.6k paired with confirmed fires (fire-season-biased: ordinary days read as Low, 2026-11-28).
    # shortcut: region-level, sampled days; upgrade to per-tile year-round score history (A6b).
    ctrl = d[d["fire"] == 0]
    p = comb.predict_proba(ctrl[comb.cols].values, pi=T._row_pi(ctrl))[:, 1]
    region = gates.region_of(ctrl["lat"].values, pd.to_numeric(ctrl["elevation"]).values)
    clim = {r: np.percentile(p[region == r], np.arange(101)).tolist() for r in np.unique(region)}
    meta = {"version": "v3.5", "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
            "rows_a": int(len(a)), "rows_b": int(len(b)), "cols": comb.cols,
            "climatology_rows": {r: int((region == r).sum()) for r in clim}}
    joblib.dump({"model": comb, "weights": weights, "climatology": clim, "meta": meta}, out, compress=3)
    print(json.dumps(meta, indent=1))


_loaded = {}


def load():
    if "a" not in _loaded:
        _loaded["a"] = joblib.load(ARTIFACT)
    return _loaded["a"]


def score(rows):
    """rows: DataFrame with lat, lon, elevation, acq_date and the raw source columns ->
    DataFrame of probability, percentile, risk_score (display), label."""
    from ml import train_ab as T
    from ml.inference import risk_label
    art = load()
    d = T.build_inputs(rows.copy(), art["weights"])
    comb = art["model"]
    p = comb.predict_proba(d[comb.cols].values.astype(float), pi=T._row_pi(d))[:, 1]
    region = gates.region_of(d["lat"].values, pd.to_numeric(d["elevation"]).values)
    pct = np.array([np.interp(pi, art["climatology"][r], np.arange(101)) if r in art["climatology"] else np.nan
                    for pi, r in zip(p, region)])
    disp = np.array([display_score(x) if np.isfinite(x) else 0.0 for x in pct])
    return pd.DataFrame({"probability": p, "percentile": pct, "risk_score": disp,
                         "label": [risk_label(x) for x in disp]}, index=d.index)


def _gridmet_rain_7d(date_str, points):
    """7-day precipitation sum (mm) from gridMET: the live second rain source (training's
    MERRA-2 is ~weeks behind real time). shortcut: different source than training's merra2 slot,
    same definition; parity measured once MERRA-2 catches up (gate 11 known issue)."""
    from ml.fire_danger import _ee, _reduce
    ee = _ee()
    t = dt.date.fromisoformat(date_str)
    img = (ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterDate(str(t - dt.timedelta(days=6)), str(t + dt.timedelta(days=1)))
           .select("pr").sum())
    out = {}
    for c0 in range(0, len(points), 500):
        out.update(_reduce(img, points[c0:c0 + 500], 4000))
    return {k: v.get("pr") for k, v in out.items()}


def _latest(fn, date, back=3):
    """Earth Engine products publish with a lag: try date, then earlier days."""
    for k in range(back + 1):
        try:
            r = fn(str(date - dt.timedelta(days=k)))
            if r and any(r.values()):
                return r
        except Exception as e:
            logger.info("v35 grid: %s not ready for %s (%s)", getattr(fn, "__name__", fn), date - dt.timedelta(days=k), e)
    return {}


def _chunked(fn, points):
    out = {}
    for c0 in range(0, len(points), 500):
        out.update(fn(points[c0:c0 + 500]))
    return out


def raw_grid(date, get_feature, get_weather):
    """Every raw source column for each California land tile on `date`, same definitions as
    the training rows. get_feature(lat, lon, name) and get_weather(lat, lon) are the server's
    cached readers (passed in so this module has no Flask dependency)."""
    from concurrent.futures import ThreadPoolExecutor
    from ml import fire_danger as fd
    from ml import live_features as L
    tiles = L.land_tiles()
    pts = [(str(i), la, lo) for i, (la, lo) in enumerate(tiles)]
    ds = str(date)
    df = pd.DataFrame({"lat": [t[0] for t in tiles], "lon": [t[1] for t in tiles], "acq_date": ds, "control_type": ""})

    def static(t):
        try:
            return [get_feature(t[0], t[1], k) for k in ("evi", "elevation", "kbdi")]
        except Exception:
            return [np.nan] * 3
    with ThreadPoolExecutor(max_workers=16) as ex:
        df[["evi", "elevation", "kbdi"]] = list(ex.map(static, tiles))
    cells = sorted({L.cell_of(*t) for t in tiles})

    def wx(c):
        try:
            return c, get_weather(*c)
        except Exception:
            return c, None
    with ThreadPoolExecutor(max_workers=8) as ex:
        cw = dict(ex.map(wx, cells))
    for k in ("air_temp_encoded", "wind", "humidity"):
        df[k] = [(cw.get(L.cell_of(*t)) or {}).get(k, np.nan) for t in tiles]

    rt = L.rtma_event_grid(ds, tiles)
    for k in ("rtma_gust_event", "rtma_offshore_event", "rtma_rh_min_event", "rtma_vpd_max_event"):
        df[k] = [rt.get(i, {}).get(k, np.nan) for i in range(len(tiles))]
    corr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "forecast_correction_v0.json")
    corr = json.load(open(corr_path))
    fc = L.forecast_cells(cells, ds)
    region = gates.region_of(df["lat"].values, pd.to_numeric(df["elevation"]).fillna(0).values)
    era = [L.apply_correction(fc.get(L.cell_of(*t), {}), r, corr) for t, r in zip(tiles, region)]
    for k in L.CORRECTED:
        df[k] = [e.get(k, np.nan) for e in era]

    gm = _latest(lambda s: _chunked(lambda p: fd.gridmet_at(s, p), pts), date)
    soil = _latest(lambda s: _chunked(lambda p: fd.soil_at(s, p), pts), date - dt.timedelta(days=4), back=5)
    norm = _chunked(lambda p: fd.normals_at(date.month, p), pts)
    rain = _latest(lambda s: _gridmet_rain_7d(s, pts), date)
    for i in range(len(tiles)):
        k = str(i)
        for src in (gm.get(k, {}), soil.get(k, {}), norm.get(k, {})):
            for c, v in src.items():
                if c != "k" and v is not None:
                    df.loc[i, c] = v
        df.loc[i, "merra2_rain_7d"] = rain.get(k, np.nan)
    return df


def build_grid(date, get_feature, get_weather):
    """Score every tile -> JSON-able payload for the endpoint cache."""
    raw = raw_grid(date, get_feature, get_weather)
    s = score(raw)
    art = load()
    cols = art["model"].cols
    from ml import train_ab as T
    built = T.build_inputs(raw.copy(), art["weights"])
    filled = built[cols].notna().mean(axis=1)
    tiles = {}
    for i, r in raw.iterrows():
        tiles[f"{r.lat:.1f},{r.lon:.1f}"] = {
            "risk_score": round(float(s.risk_score[i]), 4), "label": s.label[i],
            "probability": float(f"{s.probability[i]:.4g}"), "percentile": round(float(s.percentile[i]), 1),
            "inputs_filled": round(float(filled[i]), 2),
            "features": {k: (None if pd.isna(r.get(k)) else float(r.get(k)))
                         for k in ("evi", "air_temp_encoded", "wind", "humidity", "elevation", "kbdi")}}
    return {"date": str(date), "model": "v3.5", "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "tiles": tiles, "inputs_filled_mean": round(float(filled.mean()), 3)}


def lookup(grid, lat, lon):
    """Nearest scored tile to a point (zone centroid), within 0.3 degrees."""
    tiles = grid.get("tiles", {})
    la, lo = round(lat, 1), round(lon, 1)
    for r in range(4):
        for dla in np.arange(-r, r + 1) / 10:
            for dlo in np.arange(-r, r + 1) / 10:
                hit = tiles.get(f"{la + dla:.1f},{lo + dlo:.1f}")
                if hit:
                    return hit
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--growth", required=True)
    t.add_argument("--weights", required=True)
    t.add_argument("--out", default=ARTIFACT)
    a = ap.parse_args()
    train(a.data, a.growth, a.weights, a.out)
