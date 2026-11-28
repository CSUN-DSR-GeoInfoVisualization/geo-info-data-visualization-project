"""Models A and B (Weather App Plan). Evaluation only: writes nothing but its report.

  A, "will a wildfire start here today": confirmed wildfires (label_class, Ido
     2026-11-26) vs same-place, same-season and burnable random-spot controls.
  B, "will this fire grow large": each mapped fire at its first detection
     (growth set) plus small unmapped confirmed detections; large = >= LARGE_ACRES.

Inputs are built from raw per-source columns at load time, each needing >= 2
sources (else blank): event-day weather, dead fuel moisture (gridMET + our
NFDRS), temperature/humidity vs normal (ERA5-Land + PRISM normals), soil
moisture (ERA5-Land + SMAP, 2015+), terrain. Reference systems (bench_*) are
benchmarks only. A locked test set (LOCKED_YEARS + LOCKED_REGION) is never used
for selection and is scored once.

  python -m ml.train_ab --data merged.csv [--growth growth.csv] [--weights event_weights.json]
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

from ml import gates
from ml.train_monotonic import train_monotonic, FEATURE_COLS, MIN_INPUT_EFFECT, tier_report, tier_failures
from routes.ml_ingest import EVENT_KEYS, WEATHER_SOURCES

WEATHERAPP = "ml/training_data/weather_app_plan/california_weatherapp.csv"
GROWTH = "ml/training_data/weather_app_plan/california_growth.csv"
LARGE_ACRES = 1000
GUST_PAIRS_FROM = "2015-02-01"   # a fire's year-earlier control must also be in the RTMA gust era
SOIL_FROM = "2015-04-01"         # SMAP starts; before, one soil source -> blank (gap plan G-SM)
LOCKED_YEARS = (2024, 2025)
LOCKED_REGION = "south-high"
NEW = ["fm100", "fm1000", "temp_anom", "rh_anom", "soil", "slope", "southness"]
COLS = FEATURE_COLS + list(EVENT_KEYS) + NEW
# +1 risk never falls as the input rises, -1 never rises, 0 free
CST = ([0, 1, 1, -1, 0, 1] + [1, 1, -1, 1, -1]      # base six; gust, offshore, rh_min, vpd_max, rain_7d
       + [-1, -1, 1, -1, -1, 0, 0])                # fm100, fm1000, temp_anom, rh_anom, soil, slope, southness
ERA = {"gust_event": GUST_PAIRS_FROM, "soil": SOIL_FROM}


def _num(d, c):
    return pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)


def _blend(d, cols, weights=None):
    """Weighted mean of the sources present, blank unless >= 2 answered."""
    v = pd.DataFrame({c: _num(d, c) for c in cols})
    w = pd.Series({c: (weights or {}).get(c, 1.0) for c in cols})
    ok = v.notna()
    out = (v.fillna(0) * w).sum(axis=1) / (ok * w).sum(axis=1).replace(0, np.nan)
    out[ok.sum(axis=1) < 2] = np.nan
    return out


def build_inputs(d, weights=None):
    for k in EVENT_KEYS:
        d[k] = _blend(d, [f"{s}_{k}" for s in WEATHER_SOURCES],
                      {f"{s}_{k}": w for s, w in (weights or {}).get(k, {}).items()})
    pair_day = pd.to_datetime(d["acq_date"])
    same_place = d["control_type"].eq("same_place")
    pair_day[same_place] = pair_day[same_place] + pd.DateOffset(years=1)
    d.loc[pair_day < GUST_PAIRS_FROM, "gust_event"] = np.nan
    d["fm100"] = _blend(d, ["gridmet_fm100", "era5_fm100"])
    d["fm1000"] = _blend(d, ["gridmet_fm1000", "era5_fm1000"])
    t = _num(d, "air_temp_encoded") * 0.02 - 273.15
    d["temp_anom"] = t - _blend(d, ["era5l_tnorm", "prism_tnorm"])
    d["rh_anom"] = _num(d, "humidity") - _blend(d, ["era5l_rhnorm", "prism_rhnorm"])
    d["soil"] = _blend(d, ["era5l_sm1", "smap_sm"])
    d.loc[pair_day < SOIL_FROM, "soil"] = np.nan
    d["slope"] = _num(d, "slope")
    d["southness"] = -np.cos(np.radians(_num(d, "aspect")))  # +1 south-facing (sunnier, drier)
    return d


def model_a_rows(d, burnable_only=True):
    """Confirmed wildfires and the controls paired with them."""
    f = d[(d["fire"] == 1) & (d["label_class"] == "wildfire_confirmed")]
    keys = set(f["lat"].round(5).astype(str) + "," + f["lon"].round(5).astype(str))
    fire_keys = set(f["pair_key"])
    c = d[d["fire"] == 0]
    loc = c["lat"].round(5).astype(str) + "," + c["lon"].round(5).astype(str)
    same = c[c["control_type"].isin(["same_place", "same_season"]) & loc.isin(keys)]
    sp = c[c["control_type"].eq("spatial") & c["pair_of"].isin(fire_keys)]
    if burnable_only:
        sp = sp[sp["burnable"].astype(str).isin(["1", "1.0"])]
    return pd.concat([f, same, sp])


def _metrics(y, p):
    return {"auroc": round(roc_auc_score(y, p), 3), "brier": round(brier_score_loss(y, p), 3), "n": int(len(y))}


def _fit(d, label, cols, cst):
    keep = [i for i, c in enumerate(cols) if d[c].notna().any()]
    cols, cst = [cols[i] for i in keep], [cst[i] for i in keep]
    assert not gates.leakage_failures(cols), gates.leakage_failures(cols)
    m, s = train_monotonic(d[cols].values, d[label].values, monotonic_cst=cst)
    return m, s, cols


def _cv(d, label, group, cols, cst, k=5):
    p = pd.Series(np.nan, index=d.index)
    g = pd.factorize(group)[0] % k
    for f in range(k):
        tr, te = g != f, g == f
        m, s, c = _fit(d[tr], label, cols, cst)
        p[te] = m.predict_proba(s.transform(d.loc[te, c].values))[:, 1]
    return p


def input_effects(d, label, cols, cst):
    m, s, cols = _fit(d, label, cols, cst)
    ref = d[cols].sample(min(400, len(d)), random_state=0).values
    out = {}
    for i, c in enumerate(cols):
        lo, hi = np.nanpercentile(d[c], [5, 95])
        r = []
        for v in np.linspace(lo, hi, 15):
            z = ref.copy()
            z[:, i] = v
            r.append(m.predict_proba(s.transform(z))[:, 1].mean())
        out[c] = round(float(max(r) - min(r)), 3)
    return out


def _refs(d):
    return {n: sgn * _num(d, c).values for n, c, sgn in
            (("USFS ERC", "bench_erc", 1), ("USFS Burning Index", "bench_bi", 1),
             ("100h fuel moisture", "gridmet_fm100", -1), ("1000h fuel moisture", "gridmet_fm1000", -1)) if c in d}


def _report(d, p, label, cols):
    y = d[label].values
    ref_auroc = {}
    for n, sc in _refs(d).items():
        ok = ~np.isnan(sc)
        if ok.sum() > 200 and len(np.unique(y[ok])) == 2:
            ref_auroc[n] = round(roc_auc_score(y[ok], sc[ok]), 3)
    fails = (gates.calibration_failures(p.values, y)
             + gates.region_failures(p.values, y, d["lat"].values, pd.to_numeric(d["elevation"]).values)
             + gates.beats_reference_failures(p.values, y, _refs(d))
             + tier_failures(tier_report(p.values, y)))
    return {**_metrics(y, p), "references": ref_auroc, "failures": fails}


def evaluate_a(d, burnable_only=True):
    a = model_a_rows(d, burnable_only).copy()
    year = a["acq_date"].str[:4].astype(int)
    pair_year = np.where(a["control_type"].eq("same_place"), year + 1, year)
    region = gates.region_of(a["lat"].values, pd.to_numeric(a["elevation"]).values)
    locked = np.isin(pair_year, LOCKED_YEARS) | (region == LOCKED_REGION)
    dev, test = a[~locked], a[locked]
    place = (dev["lat"] // 1).astype(int).astype(str) + "," + (dev["lon"] // 1).astype(int).astype(str)
    out = {"rows": {"dev": int(len(dev)), "locked test": int(len(test)),
                    "confirmed fires": int(a["fire"].sum()), "controls": int((a["fire"] == 0).sum())}}
    for name, cols, cst in (("base 6", FEATURE_COLS, CST[:6]), ("all inputs", COLS, CST)):
        r = {}
        for split, grp in (("held-out years", pair_year[~locked] // 5), ("held-out places", place)):
            r[split] = _report(dev, _cv(dev, "fire", grp, cols, cst), "fire", cols)
        r["dead inputs"] = gates.missing_failures(dev, [c for c in cols if c not in ERA]) + \
            gates.missing_failures(dev, [c for c in cols if c in ERA], ERA)
        eff = input_effects(dev, "fire", cols, cst)
        r["input effects (each alone)"] = eff
        m_, s_, c_ = _fit(dev, "fire", cols, cst)
        geff = gates.group_effects(m_, s_, dev[c_].sample(min(400, len(dev)), random_state=0).values, c_,
                                   [cst[cols.index(c)] for c in c_])
        r["group effects (gate 5)"] = geff
        r["weak groups (< %s)" % MIN_INPUT_EFFECT] = [g for g, e in geff.items() if e < MIN_INPUT_EFFECT]
        m, s, c = _fit(dev, "fire", cols, cst)
        r["LOCKED TEST"] = _report(test, pd.Series(m.predict_proba(s.transform(test[c].values))[:, 1], index=test.index),
                                   "fire", c)
        out[name] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=WEATHERAPP)
    ap.add_argument("--growth", default=GROWTH)
    ap.add_argument("--weights")
    a = ap.parse_args()
    weights = json.load(open(a.weights))["weights"] if a.weights else None
    d = pd.read_csv(a.data, low_memory=False, dtype={"lat": str, "lon": str})
    d["pair_key"] = d["lat"] + "," + d["lon"] + "," + d["acq_date"]  # same text key the spatial step wrote
    d["lat"], d["lon"] = pd.to_numeric(d["lat"]), pd.to_numeric(d["lon"])
    d = build_inputs(d, weights)
    out = {"weights": "station-measured" if weights else "PROVISIONAL equal weights (not promotable)",
           "Model A (burnable random spots)": evaluate_a(d, True),
           "Model A (all random spots)": evaluate_a(d, False)}
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
