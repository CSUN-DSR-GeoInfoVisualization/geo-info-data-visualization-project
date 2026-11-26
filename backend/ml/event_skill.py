"""Skill of each source's event-day readings against station truth (Gust Backfill
Plan G1). Truth: ml.station_truth rows; event = the fire-day definitions (max of
two days, min RH, 7-day rain). Weights = inverse squared mean absolute error,
the rule the existing blend uses. Writes the weights file ml.train_ab reads.

  python -m ml.event_skill --truth stations/asos_daily_v2.csv --n 800 --out event_weights_v0.json
"""
import argparse
import json
import statistics as st
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from routes.ml_ingest import EVENT_KEYS, RTMA_GUST_FROM, _event_sources

TRUTH = {"gust_event": ("gust_max", max, 2), "offshore_event": ("offshore_max", max, 2),
         "rh_min_event": ("rh_min", min, 2), "vpd_max_event": ("vpd_max", max, 2), "rain_7d": ("precip_mm", sum, 7)}


def truth_for(t, station, day):
    rows = t.get(station, {})
    out = {}
    for k, (col, agg, ndays) in TRUTH.items():
        vals = [rows.get(str(day - pd.Timedelta(days=i))[:10], {}).get(col) for i in range(ndays)]
        if all(v is not None and v == v for v in vals):
            out[k] = agg(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.truth)
    meta = df.groupby("station")[["lat", "lon"]].first().to_dict("index")
    t = defaultdict(dict)
    for r in df.to_dict("records"):
        t[r["station"]][r["date"]] = r
    # half the sample from the gust era, so RTMA's gust gets enough samples to be scored
    sample = pd.concat([df[df["date"] >= RTMA_GUST_FROM].sample(a.n // 2, random_state=0),
                        df.sample(a.n - a.n // 2, random_state=1)])

    def job(r):
        tr = truth_for(t, r.station, pd.Timestamp(r.date))
        if not tr:
            return None
        m = meta[r.station]
        try:
            return tr, _event_sources(m["lat"], m["lon"], r.date)
        except Exception:
            return None
    with ThreadPoolExecutor(4) as ex:
        res = [x for x in ex.map(job, sample.itertuples()) if x]
    err = defaultdict(lambda: defaultdict(list))
    for tr, src in res:
        for k, v in tr.items():
            for s, vals in src.items():
                if k in vals:
                    err[k][s].append(abs(vals[k] - v))
    mae = {k: {s: round(st.mean(e), 3) for s, e in by.items() if len(e) >= 30} for k, by in err.items()}
    weights = {k: {s: round((1 / m ** 2) / sum(1 / x ** 2 for x in by.values()), 3) for s, m in by.items()}
               for k, by in mae.items() if len(by) >= 2}
    out = {"method": "event_skill_v0", "samples": len(res), "mae": mae,
           "n": {k: {s: len(e) for s, e in by.items()} for k, by in err.items()}, "weights": weights}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
