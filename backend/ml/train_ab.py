"""Models A and B (Weather App Plan, 2026-11-26). Evaluation only: writes nothing.

  A, "will a fire start here today": fires vs three kinds of control (same place a
     year earlier, same place in the wet season, a random California spot on the
     same date), so it learns both WHEN and WHERE.
  B, "will this fire grow large": each mapped fire at its first detection
     (ml.backfill_matched growth-set) plus small unmapped detections; label =
     final size >= LARGE_ACRES. Wind acts here.
  Risk of a dangerous wildfire = P(A) x P(B).

Event-day inputs are blended from the raw per-source columns at load time: the
mean of the sources that answered, needing >= 2 (else blank). Gust is blank
before its two-source era, for both rows of a same-place pair.

  python -m ml.train_ab [--weights calibration.json]
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

from ml.train_monotonic import train_monotonic, FEATURE_COLS, MIN_INPUT_EFFECT, tier_report, tier_failures
from routes.ml_ingest import EVENT_KEYS, WEATHER_SOURCES, RTMA_GUST_FROM

WEATHERAPP = "ml/training_data/weather_app_plan/california_weatherapp.csv"
GROWTH = "ml/training_data/weather_app_plan/california_growth.csv"
LARGE_ACRES = 1000
GUST_PAIRS_FROM = "2015-02-01"  # a fire's year-earlier control must also be in the gust era
COLS = FEATURE_COLS + list(EVENT_KEYS)
# +1 risk never falls as the input rises, -1 never rises, 0 free (base six as in production)
CST = [0, 1, 1, -1, 0, 1] + [1, 1, -1, 1, -1]  # gust, offshore, rh_min, vpd_max, rain_7d


def blend_events(d, weights=None):
    """Event inputs from raw columns. weights: {key: {source: w}} from station skill;
    without it, equal weights (fine for evaluation, never for promotion)."""
    for k in EVENT_KEYS:
        cols = {s: f"{s}_{k}" for s in WEATHER_SOURCES if f"{s}_{k}" in d}
        if not cols:
            d[k] = np.nan
            continue
        vals = pd.DataFrame({s: pd.to_numeric(d[c], errors="coerce") for s, c in cols.items()})
        w = pd.Series({s: (weights or {}).get(k, {}).get(s, 1.0) for s in vals})
        ok = vals.notna()
        d[k] = (vals.fillna(0) * w).sum(axis=1) / (ok * w).sum(axis=1).replace(0, np.nan)
        d.loc[ok.sum(axis=1) < 2, k] = np.nan  # skill-weighted rule: two sources or blank
    pair_day = pd.to_datetime(d["acq_date"])
    same_place = d["control_type"].eq("same_place")
    pair_day[same_place] = pair_day[same_place] + pd.DateOffset(years=1)
    d.loc[pair_day < GUST_PAIRS_FROM, "gust_event"] = np.nan
    return d


def _metrics(y, p):
    return {"auroc": round(roc_auc_score(y, p), 3), "brier": round(brier_score_loss(y, p), 3), "n": int(len(y))}


def _cv(d, label, group, cols, cst, k=5):
    """Out-of-fold predictions with whole groups held out (years or 1-degree places)."""
    assert not any(c.startswith("bench_") for c in cols), "reference systems are benchmarks, never inputs"
    keep = [i for i, c in enumerate(cols) if d[c].notna().any()]  # an all-blank input can't be learned
    cols, cst = [cols[i] for i in keep], [cst[i] for i in keep]
    p = pd.Series(np.nan, index=d.index)
    g = pd.factorize(group)[0] % k
    for f in range(k):
        tr, te = g != f, g == f
        m, s = train_monotonic(d.loc[tr, cols].values, d.loc[tr, label].values, monotonic_cst=cst)
        p[te] = m.predict_proba(s.transform(d.loc[te, cols].values))[:, 1]
    return p


def input_effects(d, label, cols, cst):
    """Every-input-matters check on a model fit to all rows: average risk swept
    over each input's 5th-95th percentile on real rows (ml.train_monotonic gate)."""
    keep = [i for i, c in enumerate(cols) if d[c].notna().any()]
    cols, cst = [cols[i] for i in keep], [cst[i] for i in keep]
    m, s = train_monotonic(d[cols].values, d[label].values, monotonic_cst=cst)
    ref = d[cols].sample(min(400, len(d)), random_state=0).values
    out = {}
    for i, c in enumerate(cols):
        lo, hi = np.nanpercentile(d[c], [5, 95])
        r = []
        for v in np.linspace(lo, hi, 15):
            z = ref.copy(); z[:, i] = v
            r.append(m.predict_proba(s.transform(z))[:, 1].mean())
        out[c] = round(float(max(r) - min(r)), 3)
    return {"effects": out, "weak (< %s)" % MIN_INPUT_EFFECT: [c for c, e in out.items() if e < MIN_INPUT_EFFECT]}


def _gate_report(d, p, label, cols):
    """ml.gates on out-of-fold predictions: calibration, regional floor, leakage,
    dead inputs, and beating the reference systems present in the data."""
    from ml import gates
    y = d[label].values
    refs = {n: sgn * pd.to_numeric(d[c], errors="coerce").values
            for n, c, sgn in (("USFS ERC", "bench_erc", 1), ("USFS Burning Index", "bench_bi", 1),
                              ("100h fuel moisture", "gridmet_fm100", -1), ("1000h fuel moisture", "gridmet_fm1000", -1))
            if c in d}
    ref_auroc = {}
    for n, sc in refs.items():
        ok = ~np.isnan(sc)
        if ok.sum() > 200 and len(np.unique(y[ok])) == 2:
            ref_auroc[n] = round(roc_auc_score(y[ok], sc[ok]), 3)
    return {"references_auroc": ref_auroc,
            "failures": (gates.calibration_failures(p.values, y)
                         + gates.region_failures(p.values, y, d["lat"].values, d["elevation"].values)
                         + gates.leakage_failures(cols)
                         + gates.missing_failures(d, [c for c in cols if c != "gust_event"])
                         + gates.missing_failures(d, [c for c in cols if c == "gust_event"], {"gust_event": GUST_PAIRS_FROM})
                         + (gates.beats_reference_failures(p.values, y, refs) if refs else
                            ["reference comparison: no benchmark columns in this data"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--data", default=WEATHERAPP)
    ap.add_argument("--growth", default=GROWTH)
    a = ap.parse_args()
    weights = json.load(open(a.weights))["weights"] if a.weights else None
    d = blend_events(pd.read_csv(a.data, low_memory=False), weights)
    year = d["acq_date"].str[:4].astype(int)
    pair_year = np.where(d["control_type"].eq("same_place"), year + 1, year)  # keep pairs together
    place = (d["lat"] // 1).astype(int).astype(str) + "," + (d["lon"] // 1).astype(int).astype(str)
    out = {"rows": {k: int(v) for k, v in d.groupby("control_type", dropna=False).size().items()}}
    for name, cols, cst in (("A base 6", FEATURE_COLS, CST[:6]), ("A + events", COLS, CST)):
        out[name] = {}
        for split, grp in (("held-out years", pair_year // 5), ("held-out places", place)):
            ps = _cv(d, "fire", grp, cols, cst)
            out[name][split] = {**_metrics(d["fire"], ps), "gates": _gate_report(d, ps, "fire", cols)}
        sp = d[d["control_type"].isin(["", "spatial"]) | d["fire"].eq(1)]
        tp = d[d["control_type"].isin(["same_place", "same_season"]) | d["fire"].eq(1)]
        p = _cv(d, "fire", pair_year // 5, cols, cst)
        out[name]["where (vs random spots)"] = _metrics(sp["fire"], p[sp.index])
        out[name]["when (vs same spot)"] = _metrics(tp["fire"], p[tp.index])
        tiers = tier_report(p.values, d["fire"].values)
        out[name]["tiers (held-out years)"] = {"report": tiers, "failures": tier_failures(tiers)}
        out[name]["inputs"] = input_effects(d, "fire", cols, cst)
    try:
        g = blend_events(pd.read_csv(a.growth, low_memory=False), weights)
    except FileNotFoundError:
        g = None
    if g is not None and len(g):
        small = d[(d["fire"] == 1) & d["fire_id"].isna() & ~d["source"].eq("perimeter_start")]
        b = pd.concat([g, small], ignore_index=True)
        b["large"] = (pd.to_numeric(b["fire_acres"], errors="coerce").fillna(0) >= LARGE_ACRES).astype(int)
        by = b["acq_date"].str[:4].astype(int) // 5
        out["B rows"] = {"large": int(b["large"].sum()), "not large": int((1 - b["large"]).sum())}
        for name, cols, cst in (("B base 6", FEATURE_COLS, CST[:6]), ("B + events", COLS, CST)):
            pb = _cv(b, "large", by, cols, cst)
            out[name] = {**_metrics(b["large"], pb), "inputs": input_effects(b, "large", cols, cst)}
        gust = b[b["gust_event"].notna()]
        if gust["large"].nunique() == 2:
            out["B + events, gust era only"] = _metrics(gust["large"], _cv(gust, "large", gust["acq_date"].str[:4], COLS, CST))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
