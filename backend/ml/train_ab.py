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
ERA = {"gust_event": GUST_PAIRS_FROM}  # soil has 2 sources from 2001 with GLDAS (G-SM)


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
    d["soil"] = _blend(d, ["era5l_sm1", "smap_sm", "gldas_sm"])  # GLDAS gives 2 sources from 2001 (G-SM)
    if "gldas_sm" not in d:
        d.loc[pair_day < SOIL_FROM, "soil"] = np.nan
    d["slope"] = _num(d, "slope")
    d["southness"] = -np.cos(np.radians(_num(d, "aspect")))  # +1 south-facing (sunnier, drier)
    return d


def model_a_rows(d, burnable_only=True):
    """Confirmed wildfires and the controls paired with them."""
    f = d[(d["fire"] == 1) & (d["label_class"] == "wildfire_confirmed")]
    keys = set(f["lat"].round(5).astype(str) + "," + f["lon"].round(5).astype(str))
    fire_keys = set(f["pair_key"]) | set(f["pair_of"].dropna()) if "pair_of" in f else set(f["pair_key"])  # FPA groups link by fpa:<id>
    c = d[d["fire"] == 0]
    loc = c["lat"].round(5).astype(str) + "," + c["lon"].round(5).astype(str)
    same = c[c["control_type"].isin(["same_place", "same_season"]) & loc.isin(keys)]
    sp = c[c["control_type"].eq("spatial") & c["pair_of"].isin(fire_keys)]
    if burnable_only:
        sp = sp[sp["burnable"].astype(str).isin(["1", "1.0"])]
    return pd.concat([f, same, sp])


class StackedGroups:
    """Sub-models (Ido, 2026-11-27): one monotonic model per factor group (wind, heat,
    moisture, vegetation, terrain), combined by a logistic whose weights must stay
    positive. Every group has a measurable weight, and every input is judged inside
    its own group's model, so overlapping inputs no longer hide each other."""

    def __init__(self, cols, cst):
        self.cols, self.cst = cols, cst
        self.groups = {g: [c for c in m if c in cols] for g, m in gates.INPUT_GROUPS.items()}
        self.groups = {g: m for g, m in self.groups.items() if m}
        for c in cols:
            if not any(c in m for m in self.groups.values()):
                self.groups[c] = [c]

    def _score(self, models, X):
        return np.column_stack([models[g][0].decision_function(models[g][1].transform(X[:, [self.cols.index(c) for c in m]]))
                                for g, m in self.groups.items()])

    def fit(self, X, y, grp, k=5):
        from sklearn.linear_model import LogisticRegression
        X, y = np.asarray(X, float), np.asarray(y)
        fold = np.unique(np.asarray(grp).astype(str), return_inverse=True)[1] % k
        fit_g = lambda Xg, yg, m: train_monotonic(Xg, yg, calibrate=False,
                                                  monotonic_cst=[self.cst[self.cols.index(c)] for c in m])
        oof = np.full((len(y), len(self.groups)), np.nan)
        for f in range(k):
            tr, te = fold != f, fold == f
            if te.any():
                ms = {}
                for g, m in self.groups.items():
                    Xg = X[tr][:, [self.cols.index(c) for c in m]].copy()
                    Xg[:, np.isnan(Xg).all(axis=0)] = 0.0  # input absent from this fold's years
                    ms[g] = fit_g(Xg, y[tr], m)
                oof[te] = self._score(ms, X[te])
        self.models = {g: fit_g(X[:, [self.cols.index(c) for c in m]], y, m) for g, m in self.groups.items()}
        keep = list(range(len(self.groups)))
        while True:  # positive weights only: a group the combiner would weigh negatively carries no weight
            meta = LogisticRegression().fit(oof[:, keep], y)
            neg = [keep[i] for i, w in enumerate(meta.coef_[0]) if w <= 0]
            if not neg or len(keep) == 1:
                break
            keep = [i for i in keep if i not in neg]
        self.keep, self.meta = keep, meta
        names = list(self.groups)
        sd = oof[:, keep].std(axis=0)
        w = meta.coef_[0] * sd
        self.weights = {names[i]: round(float(v / w.sum()), 3) for i, v in zip(keep, w)}
        self.dropped = [names[i] for i in range(len(names)) if i not in keep]
        return self

    def predict_proba(self, X):
        return self.meta.predict_proba(self._score(self.models, np.asarray(X, float))[:, self.keep])


class _Identity:
    def transform(self, X):
        return np.asarray(X, float)


def _metrics(y, p):
    return {"auroc": round(roc_auc_score(y, p), 3), "brier": round(brier_score_loss(y, p), 3), "n": int(len(y))}


def _fit(d, label, cols, cst, stacked=False):
    keep = [i for i, c in enumerate(cols) if d[c].notna().any()]
    cols, cst = [cols[i] for i in keep], [cst[i] for i in keep]
    assert not gates.leakage_failures(cols), gates.leakage_failures(cols)
    if stacked:
        m = StackedGroups(cols, cst).fit(d[cols].values, d[label].values, d["_grp"].values)
        return m, _Identity(), cols
    m, s = train_monotonic(d[cols].values, d[label].values, monotonic_cst=cst,
                           groups=d["_grp"].values if "_grp" in d else None)
    return m, s, cols


def _cv(d, label, group, cols, cst, k=5, stacked=False):
    p = pd.Series(np.nan, index=d.index)
    g = pd.factorize(group)[0] % k
    for f in range(k):
        tr, te = g != f, g == f
        if not te.any() or d.loc[tr, label].nunique() < 2:
            continue
        m, s, c = _fit(d[tr], label, cols, cst, stacked)
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
             ("100h fuel moisture", "gridmet_fm100", -1), ("1000h fuel moisture", "gridmet_fm1000", -1),
             ("Fosberg FFWI", "bench_ffwi", 1), ("Hot-Dry-Windy", "bench_hdw", 1),
             ("NWS Red Flag Warning", "bench_rfw", 1)) if c in d and _num(d, c).notna().any()}


def _report(d, p, label, cols, ref_train=None, w=None, w_train=None):
    """ref_train: (training rows, label) to calibrate the reference systems on, as the model was.
    w: explicit per-row design weights (overrides the training-share reweighting)."""
    y = d[label].values
    ref_auroc = {}
    for n, sc in _refs(d).items():
        ok = ~np.isnan(sc)
        if ok.sum() > 200 and len(np.unique(y[ok])) == 2:
            ref_auroc[n] = round(roc_auc_score(y[ok], sc[ok]), 3)
    rp = gates.fit_reference_probs(_refs(ref_train[0]), ref_train[0][ref_train[1]].values, _refs(d), w_train) if ref_train else None
    share = float(ref_train[0][ref_train[1]].mean()) if ref_train else None  # judge under the training design
    share = w if w is not None else share
    board = gates.scoreboard(p.values, y, _refs(d), d["_grp"].values if "_grp" in d else None, rp, share)
    fails = (gates.operational_failures(board) + gates.calibration_failures(p.values, y, w=gates.design_weights(y, share))
             + gates.region_failures(p.values, y, d["lat"].values, pd.to_numeric(d["elevation"]).values)
             + gates.beats_reference_failures(p.values, y, _refs(d), d["_grp"].values if "_grp" in d else None,
                                              ref_probs=rp, target_share=share)
             + tier_failures(tier_report(p.values, y)))
    return {**_metrics(y, p), "references": ref_auroc, "scoreboard": board, "failures": fails}


def evaluate_a(d, burnable_only=True):
    a = model_a_rows(d, burnable_only).copy()
    year = a["acq_date"].str[:4].astype(int)
    pair_year = np.where(a["control_type"].eq("same_place"), year + 1, year)
    region = gates.region_of(a["lat"].values, pd.to_numeric(a["elevation"]).values)
    locked = np.isin(pair_year, LOCKED_YEARS) | (region == LOCKED_REGION)
    a["_grp"] = pair_year // 5  # calibration folds hold out whole year blocks
    dev, test = a[~locked], a[locked]
    place = (dev["lat"] // 1).astype(int).astype(str) + "," + (dev["lon"] // 1).astype(int).astype(str)
    out = {"rows": {"dev": int(len(dev)), "locked test": int(len(test)),
                    "confirmed fires": int(a["fire"].sum()), "controls": int((a["fire"] == 0).sum())}}
    for name, cols, cst, stk in (("base 6", FEATURE_COLS, CST[:6], False), ("all inputs", COLS, CST, False),
                                 ("stacked groups", COLS, CST, True)):
        r = {}
        for split, grp in (("held-out years", pair_year[~locked] // 5), ("held-out places", place)):
            r[split] = _report(dev, _cv(dev, "fire", grp, cols, cst, stacked=stk), "fire", cols)
        r["dead inputs"] = gates.missing_failures(dev, [c for c in cols if c not in ERA]) + \
            gates.missing_failures(dev, [c for c in cols if c in ERA], ERA)
        eff = input_effects(dev, "fire", cols, cst)
        r["input effects (each alone)"] = eff
        m_, s_, c_ = _fit(dev, "fire", cols, cst, stk)
        geff = gates.group_effects(m_, s_, dev[c_].sample(min(400, len(dev)), random_state=0).values, c_,
                                   [cst[cols.index(c)] for c in c_])
        r["group effects (gate 5)"] = geff
        r["weak groups (< %s)" % MIN_INPUT_EFFECT] = [g for g, e in geff.items() if e < MIN_INPUT_EFFECT]
        if stk:
            r["group weights in the combiner"] = m_.weights
            r["groups with no weight (gate 5 fail)"] = m_.dropped
        m, s, c = m_, s_, c_
        r["LOCKED TEST"] = _report(test, pd.Series(m.predict_proba(s.transform(test[c].values))[:, 1], index=test.index),
                                   "fire", c, ref_train=(dev, "fire"))
        out[name] = r
    return out


def evaluate_b(g, d):
    """Model B: growth-set fires (first detection, start-day weather) plus confirmed
    wildfires with no mapped perimeter (they stayed small); large = >= LARGE_ACRES."""
    small = d[(d["fire"] == 1) & (d["label_class"] == "wildfire_confirmed") & d["fire_id"].isna()]
    b = pd.concat([g, small], ignore_index=True)
    b["large"] = (pd.to_numeric(b["fire_acres"], errors="coerce").fillna(0) >= LARGE_ACRES).astype(int)
    year = b["acq_date"].str[:4].astype(int)
    region = gates.region_of(b["lat"].values, pd.to_numeric(b["elevation"]).values)
    locked = np.isin(year, LOCKED_YEARS) | (region == LOCKED_REGION)
    b["_grp"] = year // 5
    dev, test = b[~locked], b[locked]
    out = {"rows": {"dev": int(len(dev)), "locked test": int(len(test)), "large": int(b["large"].sum()),
                    "not large": int((1 - b["large"]).sum())}}
    for name, cols, cst in (("base 6", FEATURE_COLS, CST[:6]), ("all inputs", COLS, CST)):
        r = {"held-out years": _report(dev, _cv(dev, "large", dev["_grp"], cols, cst), "large", cols)}
        m, s, c = _fit(dev, "large", cols, cst)
        r["group effects (gate 5)"] = gates.group_effects(m, s, dev[c].sample(min(400, len(dev)), random_state=0).values, c,
                                                          [cst[cols.index(x)] for x in c])
        r["LOCKED TEST"] = _report(test, pd.Series(m.predict_proba(s.transform(test[c].values))[:, 1], index=test.index),
                                   "large", c, ref_train=(dev, "large"))
        out[name] = r
    return out


class _CrossFit:
    """Average of the fold models whose out-of-fold scores trained the calibration layer
    (cross-fitting), so test predictions and their calibration come from the same models.
    Root-cause fix 2026-11-27: a full refit was sharper than the out-of-fold scores it was
    calibrated on, leaving probabilities overconfident (ECE 0.11)."""

    def __init__(self, fits):
        self.fits = fits

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(s.transform(X))[:, 1] for m, s in self.fits], axis=0)
        return np.c_[1 - p, p]


def _cv_fits(d, label, group, cols, cst, k=5):
    """Out-of-fold predictions plus the fold models (for cross-fitting)."""
    keep = [c for c in cols if d[c].notna().any()]
    p = pd.Series(np.nan, index=d.index)
    g = pd.factorize(group)[0] % k
    fits = []
    for f in range(k):
        tr, te = g != f, g == f
        if not te.any() or d.loc[tr, label].nunique() < 2:
            continue
        m, s, c = _fit(d[tr], label, keep, [cst[cols.index(x)] for x in keep])
        if c != keep:
            continue
        p[te] = m.predict_proba(s.transform(d.loc[te, c].values))[:, 1]
        fits.append((m, s))
    return p, _CrossFit(fits), keep


class _Combined:
    """Live risk = P_A(start) x P_B(large | start), Architecture v2. Both models read the
    same column list, so group effects sweep an input through both at once."""

    def __init__(self, ma, sa, ca, mb, sb, cb, cols, cal=None):
        self.a, self.b, self.cols, self.cal = (ma, sa, ca), (mb, sb, cb), cols, cal

    def raw(self, X):
        X = pd.DataFrame(np.asarray(X, float), columns=self.cols)
        pa = self.a[0].predict_proba(self.a[1].transform(X[self.a[2]].values))[:, 1]
        pb = self.b[0].predict_proba(self.b[1].transform(X[self.b[2]].values))[:, 1]
        return pa * pb

    def predict_proba(self, X):
        p = self.raw(X)
        if self.cal is not None:  # monotonic in the product, so ranking is unchanged
            p = self.cal.predict_proba(np.log(np.clip(p, 1e-6, 1)).reshape(-1, 1))[:, 1]
        return np.c_[1 - p, p]


# Real share of California wildfire starts that grow >= 1,000 acres: 1,185 of 200,113 FPA FOD
# ignitions, development years 2001-2023 (root cause 2026-11-28: the case-control test had 48%).
LARGE_SHARE_OF_STARTS = 1185 / 200113


def _population_weights(rows):
    """Combined-test design weights: fire vs control share as sampled (Model A's design), but
    large vs small fires at the real FPA FOD ratio, so the A x B score is judged on the true mix."""
    fire = rows["fire"].values == 1
    large = rows["large_start"].values == 1
    w = np.ones(len(rows))
    nf, nl = fire.sum(), (fire & large).sum()
    if nl and nf > nl:
        w[fire & large] = LARGE_SHARE_OF_STARTS * nf / nl
        w[fire & ~large] = (1 - LARGE_SHARE_OF_STARTS) * nf / (nf - nl)
    return w


def evaluate_combined(d, g):
    """A5c: the score the map shows, judged on 'a wildfire that grows >= LARGE_ACRES starts
    here today' (confirmed fires that grew large vs all Model A controls)."""
    a = model_a_rows(d, False).copy()
    year = a["acq_date"].str[:4].astype(int)
    pair_year = np.where(a["control_type"].eq("same_place"), year + 1, year)
    region = gates.region_of(a["lat"].values, pd.to_numeric(a["elevation"]).values)
    locked = np.isin(pair_year, LOCKED_YEARS) | (region == LOCKED_REGION)
    a["_grp"] = pair_year // 5
    a["large_start"] = ((a["fire"] == 1) & (pd.to_numeric(a["fire_acres"], errors="coerce").fillna(0) >= LARGE_ACRES)).astype(int)
    small = d[(d["fire"] == 1) & (d["label_class"] == "wildfire_confirmed") & d["fire_id"].isna()]
    b = pd.concat([g, small], ignore_index=True)
    b["large"] = (pd.to_numeric(b["fire_acres"], errors="coerce").fillna(0) >= LARGE_ACRES).astype(int)
    by = b["acq_date"].str[:4].astype(int)
    b["_grp"] = by // 5
    b_dev = b[~(np.isin(by, LOCKED_YEARS) | (gates.region_of(b["lat"].values, pd.to_numeric(b["elevation"]).values) == LOCKED_REGION))]
    dev, test = a[~locked], a[locked]
    pa_oof, ma, ca = _cv_fits(dev, "fire", dev["_grp"], COLS, CST)
    sa = _Identity()
    mb, sb, cb = _fit(b_dev, "large", COLS, CST)
    cols = sorted(set(ca) | set(cb), key=COLS.index)
    # Calibration layer (root-cause fix 2026-11-27: the raw product P_A x P_B answers no single
    # question, Brier 0.247 vs ERC 0.208): out-of-fold P_A on dev (year blocks) times P_B, then
    # a logistic on log(product) for the large-start target. Monotonic, so ranking is unchanged.
    from sklearn.linear_model import LogisticRegression
    pb_dev = mb.predict_proba(sb.transform(dev[cb].values))[:, 1]
    ok = pa_oof.notna().values
    w_dev = _population_weights(dev)
    cal = LogisticRegression().fit(np.log(np.clip(pa_oof.values[ok] * pb_dev[ok], 1e-6, 1)).reshape(-1, 1),
                                   dev["large_start"].values[ok], sample_weight=w_dev[ok])
    comb = _Combined(ma, sa, ca, mb, sb, cb, cols, cal)
    out = {"rows": {"dev": int(len(dev)), "locked test": int(len(test)), "large starts (locked)": int(test["large_start"].sum())}}
    for name, part in (("dev (in-sample for A, B trained without locked)", dev), ("LOCKED TEST", test)):
        p = pd.Series(comb.predict_proba(part[cols].values)[:, 1], index=part.index)
        out[name] = _report(part, p, "large_start", cols, ref_train=(dev, "large_start") if part is test else None,
                            w=_population_weights(part), w_train=w_dev)
    # root-cause diagnostic (2026-11-28): which part loses ranking on the large-start target?
    from sklearn.metrics import roc_auc_score
    ya = test["large_start"].values
    pa_t = ma.predict_proba(sa.transform(test[ca].values))[:, 1]
    pb_t = mb.predict_proba(sb.transform(test[cb].values))[:, 1]
    sub = test["fire"].values == 1
    out["parts on LOCKED TEST (AUROC)"] = {
        "P_A alone": round(roc_auc_score(ya, pa_t), 3), "P_B alone": round(roc_auc_score(ya, pb_t), 3),
        "P_A x P_B": round(roc_auc_score(ya, pa_t * pb_t), 3),
        "P_B among fires only": round(roc_auc_score(ya[sub], pb_t[sub]), 3),
        "P_A fires vs controls": round(roc_auc_score(test["fire"].values, pa_t), 3),
        "small fires / large / controls": [int((sub & (ya == 0)).sum()), int(ya.sum()), int((~sub).sum())]}
    out["group effects on A x B (gate 5)"] = gates.group_effects(
        comb, _Identity(), dev[cols].sample(min(400, len(dev)), random_state=0).values, cols,
        [CST[COLS.index(c)] for c in cols])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=WEATHERAPP)
    ap.add_argument("--growth", default=GROWTH)
    ap.add_argument("--weights")
    ap.add_argument("--only", choices=["combined"], help="run only the combined A x B evaluation")
    a = ap.parse_args()
    weights = json.load(open(a.weights))["weights"] if a.weights else None
    d = pd.read_csv(a.data, low_memory=False, dtype={"lat": str, "lon": str})
    d["pair_key"] = d["lat"] + "," + d["lon"] + "," + d["acq_date"]  # same text key the spatial step wrote
    d["lat"], d["lon"] = pd.to_numeric(d["lat"]), pd.to_numeric(d["lon"])
    d = build_inputs(d, weights)
    out = {"weights": "station-measured" if weights else "PROVISIONAL equal weights (not promotable)"}
    if a.only != "combined":
        out["Model A (burnable random spots)"] = evaluate_a(d, True)
        out["Model A (all random spots)"] = evaluate_a(d, False)
    try:
        g = pd.read_csv(a.growth, low_memory=False)
    except FileNotFoundError:
        g = None
    if g is not None and len(g):
        g["control_type"] = ""
        # Model B also learns from every FPA FOD fire: exact discovery point and date, final size
        fpa = d[d["source"].eq("fpa_fod")]
        g = pd.concat([build_inputs(g, weights), fpa], ignore_index=True)
        if a.only != "combined":
            out["Model B (grows >= %d acres)" % LARGE_ACRES] = evaluate_b(g, d)
        out["Combined A x B (large wildfire starts)"] = evaluate_combined(d, g)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
