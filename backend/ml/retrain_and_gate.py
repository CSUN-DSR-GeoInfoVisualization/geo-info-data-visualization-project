"""Gated retrain: train the monotonic candidate on the rolling dataset, evaluate
it, and promote it to production ONLY if it clears the gate.

Gate (in order):
  1. PHYSICS (hard, non-negotiable): every constrained feature must point the
     right way (ml.train_monotonic.validate_monotonicity). A physically-wrong
     model never ships, no matter how good its accuracy looks.
  2. METRICS vs the current production model on a held-out split:
       - AUROC must not regress by more than AUROC_TOL
       - Brier must not regress by more than BRIER_TOL
     BUT: if the current production model itself FAILS physics, metric
     regression is waived for the first physics-passing candidate — we refuse to
     keep a physically-broken model on accuracy grounds.

On promotion: archive the current model to models/archive/, write the new model
+ scaler + metadata, and append a RETRAIN_LOG.md entry. This module makes no
network calls and never auto-commits — the caller (CI workflow) decides that.

Usage:
    python -m ml.retrain_and_gate            # evaluate + promote if it passes
    python -m ml.retrain_and_gate --dry-run  # evaluate only, never write
"""
import os
import sys
import json
import shutil
import argparse

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score

from ml.train_monotonic import (
    train_monotonic, validate_monotonicity, gate_passes, dead_constrained_features,
    weak_features, MIN_INPUT_EFFECT, tier_report, tier_failures, FEATURE_COLS, MONOTONIC_CST,
)
from ml.geo_checks import on_ca_land

_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_DIR, "models")
_ARCHIVE_DIR = os.path.join(_MODELS_DIR, "archive")
_TRAIN_DIR = os.path.join(_DIR, "training_data")
_MODEL_OUT = os.path.join(_MODELS_DIR, "wildfire_model_predictive.pkl")
_SCALER_OUT = os.path.join(_MODELS_DIR, "wildfire_scaler_predictive.pkl")
_META_OUT = os.path.join(_MODELS_DIR, "model_metadata.json")
_LOG = os.path.join(_MODELS_DIR, "RETRAIN_LOG.md")

# Both committed to the repo (zero DB cost-risk). The frozen 2020 base never
# changes; the daily file is appended by the ingest cron. _load_dataset() unions
# them and de-dups.
_BASE = os.path.join(_TRAIN_DIR, "california_2020_kbdi.csv")
_DAILY = os.path.join(_TRAIN_DIR, "california_daily.csv")

# Matched-control training set (built 2026-07-28). Every positive is paired with
# controls at the SAME lat/lon — one in the wet season and one on the same
# calendar date a year earlier — so terrain, vegetation type and elevation cancel
# and what remains is weather and drought.
#
# It replaces _BASE/_DAILY rather than adding to them because those two encode a
# different EVI quantity: build_dataset.py stores the SPRING (May 1) composite as
# a pre-season fuel-load proxy, while the live path (data.live_evi_gee) serves the
# CURRENT composite. Mixing the two would put two different variables in one
# column. The matched set uses date-matched EVI throughout, which is the quantity
# inference actually receives.
_MATCHED = os.path.join(_TRAIN_DIR, "california_matched.csv")

# Weather App Plan set (2026-11-26): the matched design from 2001 on, every row's
# weather from the skill-weighted blend, with each source's raw reading kept for
# recalibration. Preferred over _MATCHED when present, and gated by
# check_skill_weighted: no amount of data justifies an unweighted row.
_WEATHERAPP = os.path.join(_TRAIN_DIR, "weather_app_plan", "california_weatherapp.csv")
SKILL_WEIGHTED_METHODS = {"skill_weighted_v0"}


def check_skill_weighted(d):
    """Raise ValueError unless every row carries a known skill-weighted method tag
    and >= 2 weather sources, and the raw-reading columns exist. Returns the number
    of rows still waiting for their raw readings (allowed: the blend is what trains)."""
    from routes.ml_ingest import RAW_WEATHER_COLS
    missing = [c for c in ["weather_method", "n_weather_sources"] + RAW_WEATHER_COLS if c not in d.columns]
    if missing:
        raise ValueError(f"training file lacks column(s) {missing}")
    bad_method = ~d["weather_method"].isin(SKILL_WEIGHTED_METHODS)
    bad_sources = ~(pd.to_numeric(d["n_weather_sources"], errors="coerce") >= 2)
    if bad_method.any() or bad_sources.any():
        raise ValueError(f"{int(bad_method.sum())} row(s) without a skill-weighted method tag and "
                         f"{int(bad_sources.sum())} with < 2 weather sources; refusing to train")
    return int(d[RAW_WEATHER_COLS].isna().all(axis=1).sum())

AUROC_TOL = 0.005
BRIER_TOL = 0.005

# The comparison is made on rows collected AFTER the production model was
# trained — the only rows it provably never saw. Below these sizes the holdout
# can't support a decision, so the gate HOLDS (promotes nothing, rejects
# nothing) instead of pretending to have an answer.
MIN_HOLDOUT_ROWS = 120
MIN_HOLDOUT_PER_CLASS = 30


def holdout_sufficient(rows: int, positives: int, negatives: int) -> bool:
    """Whether a temporal holdout can support a promote/reject decision at all.
    Both classes must be represented: AUROC on a holdout with a handful of one
    label is noise, and shipping a model on that basis is worse than holding."""
    return rows >= MIN_HOLDOUT_ROWS and min(positives, negatives) >= MIN_HOLDOUT_PER_CLASS

_DB_COLS = FEATURE_COLS + ["fire", "acq_date"]
_GEO_COLS = ["lat", "lon"]


def _load_dataset():
    """Union the frozen 2020 base CSV with the daily-appended CSV (both committed
    to the repo). De-dups identical rows so a re-ingested point can't double-count.
    Returns (df, description)."""
    frames, parts = [], []
    want = _DB_COLS + _GEO_COLS
    sources = next((s for s in ((_WEATHERAPP,), (_MATCHED,)) if os.path.exists(s[0])), (_BASE, _DAILY))
    for path in sources:
        if os.path.exists(path):
            d = pd.read_csv(path)
            if path == _WEATHERAPP:
                no_raw = check_skill_weighted(d)
                parts.append(f"{os.path.basename(path)}({len(d)}, skill-weighted"
                             + (f", {no_raw} awaiting raw readings)" if no_raw else ")"))
                frames.append(d[want])
            elif set(want).issubset(d.columns) and len(d):
                frames.append(d[want])
                parts.append(f"{os.path.basename(path)}({len(d)})")

    if not frames:
        raise FileNotFoundError(f"no training data found at {_BASE} or {_DAILY}")
    df = pd.concat(frames, ignore_index=True).dropna(subset=want).drop_duplicates()

    # Wind sanity check. A provider field-name bug recorded wind == 0 on a block of
    # ingested rows; being in-range AND constant, it evades the statistical outlier
    # monitor, so we exclude it here before training/evaluation. Real wind (archive
    # daily-max or live) is always > 0, so this drops exactly the corrupted rows.
    before = len(df)
    df = df[df["wind"] > 0].reset_index(drop=True)
    dropped_wind = before - len(df)

    # ONE geographic rule for BOTH classes (audit 2026-07-28). FIRMS positives are
    # collected by bounding box and CA_BBOX is a rectangle covering large parts of
    # Nevada, Arizona and the Pacific, while no-fire points are CA-land-sampled.
    # That made "off CA land" a near-perfect predictor — of NO-FIRE in the 2020
    # base (74% of its negatives are off-land) and of FIRE in the daily feed (56%
    # of its positives were). The two halves taught opposite labels for the same
    # terrain, and the model could not discriminate on new data. Restricting both
    # classes to CA land removed the contradiction: on a July CA-land holdout,
    # AUROC 0.516 -> 0.828 and Brier 0.229 -> 0.169 despite 32% fewer rows.
    # Applied here, at load, so the committed CSVs stay an unedited record.
    before = len(df)
    keep = [on_ca_land(la, lo) for la, lo in zip(df["lat"], df["lon"])]
    df = df[keep].reset_index(drop=True)
    dropped_geo = before - len(df)

    desc = " + ".join(parts)
    if dropped_wind:
        desc += f" -{dropped_wind} wind<=0"
    if dropped_geo:
        desc += f" -{dropped_geo} off-CA-land"
    # lat/lon are kept: the holdout is split by LOCATION, because a fire and its
    # matched controls share a point and must never straddle the split.
    return df, desc


def _physics_failures(report):
    """Feature names that fail physics: wrong direction, or (every input must
    matter) an average effect below MIN_INPUT_EFFECT, named with its effect."""
    wrong = [n for n, v in report.items() if not v["ok"]]
    weak = [f"{n} (effect {report[n]['range']} < {MIN_INPUT_EFFECT})" for n in weak_features(report)]
    return wrong + weak


def decide(cand_m, cand_phys_ok, cand_bad_features, prod_m, prod_phys_ok):
    """Pure gate decision. Returns (promote: bool, reasons: list[str]).

    Order: physics is the hard gate; metric regression only matters when the
    current production model is itself physically sound.
    """
    if not cand_phys_ok:
        return False, [f"candidate FAILS physics ({', '.join(cand_bad_features)})"]
    if prod_m is None:
        return True, ["no production model present — promoting first physics-passing candidate"]
    if prod_phys_ok is False:
        return True, ["production model FAILS physics — metric-regression waived, "
                      "promoting physically-correct candidate"]
    auroc_ok = cand_m["auroc"] >= prod_m["auroc"] - AUROC_TOL
    brier_ok = cand_m["brier"] <= prod_m["brier"] + BRIER_TOL
    reasons = []
    if not auroc_ok:
        reasons.append(f"AUROC regressed {prod_m['auroc']:.4f} -> {cand_m['auroc']:.4f}")
    if not brier_ok:
        reasons.append(f"Brier regressed {prod_m['brier']:.4f} -> {cand_m['brier']:.4f}")
    if auroc_ok and brier_ok:
        reasons.append("candidate clears physics + metric gate")
    return (auroc_ok and brier_ok), reasons


def _metrics(model, scaler, X, y):
    p = model.predict_proba(scaler.transform(X))[:, 1]
    fires = p[y == 1]
    return {
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "acc": float(accuracy_score(y, (p >= 0.5))),
        # Reported, not gated: the fraction of real fires the model would have
        # called High+ (>=0.40, ml.inference.risk_label). This is the number the
        # weekly backtest alerts on, so it belongs in the retrain log next to it.
        "recall_at_high": float((fires >= 0.40).mean()) if len(fires) else None,
    }


def _prod_trained_at():
    """The production model's training timestamp, or None if there is no usable
    production model. Rows newer than this are the only ones it provably never
    trained on — promote() refits on ALL rows, so anything older is memorised."""
    if not (os.path.exists(_MODEL_OUT) and os.path.exists(_SCALER_OUT)):
        return None
    try:
        with open(_META_OUT) as f:
            return str(json.load(f)["trained_at"])[:10]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def evaluate(now_iso):
    """Train a candidate and return a decision dict. Pure — writes nothing.

    Both models are scored on a TEMPORAL holdout: rows whose acq_date is newer
    than the production model's trained_at. That is the only split that is
    honest for both sides. The previous version compared production against the
    candidate on a random split of the whole dataset — but promote() refits the
    production model on every row, so production had memorised ~89% of that test
    set and won every comparison (measured 2026-07-28: AUROC 0.989 on rows it
    trained on vs 0.550 on rows it had never seen). That is what rejected six
    consecutive retrains and left the model unchanged for eight weeks.
    """
    df, dataset_desc = _load_dataset()
    X = df[FEATURE_COLS].values
    y = df["fire"].values

    cutoff = _prod_trained_at()
    if cutoff is None:
        # No production model to beat: train on everything, promote on physics.
        cand, cand_scaler = train_monotonic(X, y)
        cand_phys = validate_monotonicity(cand, cand_scaler, X_ref=X)
        cand_phys_ok = gate_passes(cand_phys, MIN_INPUT_EFFECT)
        bad = _physics_failures(cand_phys)
        promote, reasons = decide(None, cand_phys_ok, bad, None, None)
        return {
            "when": now_iso, "dataset": dataset_desc, "rows": int(len(df)),
            "holdout": {"cutoff": None, "rows": 0, "positives": 0, "negatives": 0},
            "candidate": {"metrics": None, "physics_ok": cand_phys_ok, "physics": cand_phys},
            "production": {"metrics": None, "physics_ok": None},
            "promote": promote, "held": False, "reasons": reasons,
            "_full_fit": (X, y),
        }

    # Split by FIRE EVENT, not by row date. A control carries a historical
    # acq_date (the same point a year earlier, or the preceding wet season), so
    # a plain date filter would put every fire on the new side and every control
    # on the old side — the holdout would be single-class and the model could
    # also see a location on both sides of the split. Group each location by the
    # date of its most recent fire and move the whole group together.
    loc = (df["lat"].round(5).astype(str) + "," + df["lon"].round(5).astype(str))
    fire_dates = (df.assign(_loc=loc)[df["fire"] == 1]
                    .groupby("_loc")["acq_date"].max().astype(str))
    loc_is_new = loc.map(fire_dates).fillna("") > cutoff
    tr, te = df[~loc_is_new], df[loc_is_new]
    n_pos, n_neg = int((te["fire"] == 1).sum()), int((te["fire"] == 0).sum())
    holdout = {"cutoff": cutoff, "rows": int(len(te)),
               "positives": n_pos, "negatives": n_neg}

    # Candidate is trained ONLY on rows at or before the cutoff, so the holdout
    # is unseen by both models and the comparison is symmetric.
    cand, cand_scaler = train_monotonic(tr[FEATURE_COLS].values, tr["fire"].values)
    cand_phys = validate_monotonicity(cand, cand_scaler, X_ref=tr[FEATURE_COLS].values)
    cand_phys_ok = gate_passes(cand_phys, MIN_INPUT_EFFECT)
    bad = _physics_failures(cand_phys)
    dead = dead_constrained_features(cand_phys)

    # Not enough new data to judge: hold. Never silently reject on thin evidence
    # — a hold is visible in the log and trips the staleness alarm if it repeats.
    if not holdout_sufficient(len(te), n_pos, n_neg):
        return {
            "when": now_iso, "dataset": dataset_desc, "rows": int(len(df)),
            "holdout": holdout,
            "candidate": {"metrics": None, "physics_ok": cand_phys_ok, "physics": cand_phys},
            "production": {"metrics": None, "physics_ok": None},
            "promote": False, "held": True,
            "reasons": [f"insufficient holdout since {cutoff}: {len(te)} rows "
                        f"({n_pos} fire / {n_neg} no-fire); need >={MIN_HOLDOUT_ROWS} "
                        f"and >={MIN_HOLDOUT_PER_CLASS} of each"],
            "_full_fit": (X, y),
        }

    Xte, yte = te[FEATURE_COLS].values, te["fire"].values
    cand_m = _metrics(cand, cand_scaler, Xte, yte)
    # The five tiers must be used, judged on all rows (the holdout is one season).
    tiers = tier_report(cand.predict_proba(cand_scaler.transform(X))[:, 1], y)
    # Gates review 2026-11-26 (ml.gates): calibrated, no region at chance, no dead
    # input, no benchmark/label column as an input.
    from ml import gates
    p_te = cand.predict_proba(cand_scaler.transform(Xte))[:, 1]
    extra = ([f"tiers: {r}" for r in tier_failures(tiers)]
             + gates.calibration_failures(p_te, yte)
             + gates.region_failures(p_te, yte, te["lat"].values, te["elevation"].values)
             + gates.missing_failures(df, FEATURE_COLS) + gates.leakage_failures(FEATURE_COLS))
    if extra:
        cand_phys_ok = False
        bad = bad + extra
    pm, ps = joblib.load(_MODEL_OUT), joblib.load(_SCALER_OUT)
    prod_m = _metrics(pm, ps, Xte, yte)
    prod_phys_ok = gate_passes(validate_monotonicity(pm, ps, X_ref=tr[FEATURE_COLS].values), MIN_INPUT_EFFECT)

    promote, reasons = decide(cand_m, cand_phys_ok, bad, prod_m, prod_phys_ok)
    if dead:
        # Not a veto — a dead feature means the DATA carries no usable signal in
        # the physically-correct direction, and blocking would keep a worse model
        # live without fixing that. But it must never pass silently again.
        reasons.append(f"WARNING: constrained feature(s) with no effect: {', '.join(dead)}")

    return {
        "when": now_iso,
        "dataset": dataset_desc,
        "rows": int(len(df)),
        "holdout": holdout,
        "dead_features": dead,
        "candidate": {"metrics": cand_m, "physics_ok": cand_phys_ok, "physics": cand_phys},
        "production": {"metrics": prod_m, "physics_ok": prod_phys_ok},
        "promote": promote,
        "held": False,
        "reasons": reasons,
        # carry the data for promotion (final model re-fits on ALL rows; that is
        # safe because the NEXT run's holdout is defined by the new trained_at)
        "_full_fit": (X, y),
    }


def _fmt(m, phys_ok):
    if not m:
        return "not scored"
    rec = "n/a" if m.get("recall_at_high") is None else f"{m['recall_at_high']:.3f}"
    return (f"AUROC={m['auroc']:.4f} Brier={m['brier']:.4f} "
            f"recall@High={rec} physics_ok={phys_ok}")


def _append_log(decision):
    head = "# Retrain Log\n\n" if not os.path.exists(_LOG) else ""
    if decision["promote"]:
        verdict = "PROMOTED"
    elif decision.get("held"):
        verdict = "HELD"
    else:
        verdict = "REJECTED"
    h = decision.get("holdout") or {}
    holdout_line = (
        f"- holdout: {h.get('rows', 0)} rows newer than {h.get('cutoff')} "
        f"({h.get('positives', 0)} fire / {h.get('negatives', 0)} no-fire)\n"
        if h.get("cutoff") else "- holdout: none (no production model)\n"
    )
    line = (
        f"## {decision['when']} — {verdict}\n"
        f"- dataset: {decision['dataset']} ({decision['rows']} rows)\n"
        + holdout_line +
        f"- candidate: {_fmt(decision['candidate']['metrics'], decision['candidate']['physics_ok'])}\n"
        f"- production: {_fmt(decision['production']['metrics'], decision['production']['physics_ok'])}\n"
        f"- reasons: {'; '.join(decision['reasons'])}\n\n"
    )
    with open(_LOG, "a") as f:
        f.write(head + line)


def promote(decision, now_iso):
    """Archive current production, refit the candidate on ALL rows, write it."""
    os.makedirs(_ARCHIVE_DIR, exist_ok=True)
    stamp = now_iso.replace(":", "").replace("-", "")[:13]
    if os.path.exists(_MODEL_OUT):
        shutil.copy2(_MODEL_OUT, os.path.join(_ARCHIVE_DIR, f"model_{stamp}.pkl"))
        shutil.copy2(_SCALER_OUT, os.path.join(_ARCHIVE_DIR, f"scaler_{stamp}.pkl"))
        # keep only the newest 3 archived models
        archived = sorted(f for f in os.listdir(_ARCHIVE_DIR) if f.startswith("model_"))
        for old in archived[:-3]:
            os.remove(os.path.join(_ARCHIVE_DIR, old))
            sc = old.replace("model_", "scaler_")
            if os.path.exists(os.path.join(_ARCHIVE_DIR, sc)):
                os.remove(os.path.join(_ARCHIVE_DIR, sc))

    X, y = decision["_full_fit"]
    model, scaler = train_monotonic(X, y)  # final model uses ALL rows
    joblib.dump(model, _MODEL_OUT)
    joblib.dump(scaler, _SCALER_OUT)
    cand_m = decision["candidate"]["metrics"]
    meta = {
        "model_type": "monotonic_hgb_isotonic",
        "feature_cols": FEATURE_COLS,
        "monotonic_cst": MONOTONIC_CST,
        "trained_at": now_iso,
        "dataset": decision["dataset"],
        "rows": decision["rows"],
        # None on a re-baseline or a first model: there was no holdout large
        # enough to score against, and the reason below says why it shipped.
        "heldout_auroc": cand_m["auroc"] if cand_m else None,
        "holdout": decision.get("holdout"),
        "promoted_because": "; ".join(decision["reasons"]),
    }
    with open(_META_OUT, "w") as f:
        json.dump(meta, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="evaluate only; never write")
    ap.add_argument("--now", default=None, help="ISO timestamp (CI passes this; scripts can't call datetime here)")
    ap.add_argument("--rebaseline", metavar="REASON",
                    help="Promote the physics-passing candidate WITHOUT comparing it to "
                         "the incumbent, recording REASON in the log. For use only after a "
                         "correction that invalidates the incumbent as a baseline (e.g. it "
                         "was trained on data since found to be corrupt) — comparing against "
                         "a model trained on bad data measures nothing. Never used by the "
                         "weekly cron; a human runs it deliberately and the reason is logged.")
    args = ap.parse_args()
    now_iso = args.now or "1970-01-01T00:00:00"

    d = evaluate(now_iso)
    if args.rebaseline and d["candidate"]["physics_ok"]:
        d["promote"], d["held"] = True, False
        d["reasons"] = [f"RE-BASELINE (manual): {args.rebaseline}"]
    elif args.rebaseline:
        d["reasons"] = ["re-baseline refused: candidate FAILS physics"] + d["reasons"]
    h = d.get("holdout") or {}
    print(f"dataset={d['dataset']} rows={d['rows']}")
    if h.get("cutoff"):
        print(f"holdout: {h['rows']} rows newer than {h['cutoff']} "
              f"({h['positives']} fire / {h['negatives']} no-fire)")
    print(f"candidate: {_fmt(d['candidate']['metrics'], d['candidate']['physics_ok'])}")
    print(f"production: {_fmt(d['production']['metrics'], d['production']['physics_ok'])}")
    verdict = "PROMOTE" if d["promote"] else ("HOLD" if d.get("held") else "REJECT")
    print("DECISION:", verdict, "—", "; ".join(d["reasons"]))

    if args.dry_run:
        print("(dry-run — no files written)")
        return 0
    _append_log(d)
    if d["promote"]:
        promote(d, now_iso)
        print("Promoted candidate to production + archived previous + logged.")
    else:
        print("Kept production. Logged rejection.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
