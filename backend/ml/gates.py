"""Promotion gates shared by ml.retrain_and_gate and ml.train_ab (Weather App Plan,
"Gates", reviewed 2026-11-26). Physics direction, every-input-matters and the
five-tier check live in ml.train_monotonic; these add the gates that review
found missing. Each returns a list of failure reasons ([] = pass).
"""
import numpy as np
from sklearn.metrics import roc_auc_score

MAX_ECE = 0.05            # calibration: mean gap between predicted and observed fire rate, by decile
REGION_MIN_AUROC = 0.52   # no region may sit at chance while the state average looks good
MIN_REGION_ROWS = 200
MAX_MISSING = 0.20        # an input blank on > 20% of its era's rows means a source died
BOOT = 300                # bootstrap draws for "the gain is real, not noise"
FORBIDDEN_PREFIXES = ("bench_", "rfw", "firms_", "frp", "ta_")   # benchmarks and label-derived data


def calibration_failures(p, y, bins=10):
    """Expected calibration error over probability deciles, plus observed fire rate
    must not fall from one decile to the next."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    ece, rates = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
            rates.append(y[m].mean())
    out = []
    if ece > MAX_ECE:
        out.append(f"calibration error {ece:.3f} > {MAX_ECE}")
    if any(b < a - 0.02 for a, b in zip(rates, rates[1:])):
        out.append("observed fire rate falls between probability deciles")
    return out


def region_of(lat, elevation):
    """Six coarse regions: north/south of 37 N x elevation band (<300 m, 300-1500, >1500)."""
    band = np.where(elevation < 300, "low", np.where(elevation < 1500, "mid", "high"))
    return np.char.add(np.where(np.asarray(lat) >= 37, "north-", "south-"), band.astype(str))


def region_failures(p, y, lat, elevation):
    out = []
    reg = region_of(np.asarray(lat), np.asarray(elevation, float))
    p, y = np.asarray(p), np.asarray(y)
    for r in np.unique(reg):
        m = reg == r
        if m.sum() >= MIN_REGION_ROWS and len(np.unique(y[m])) == 2:
            a = roc_auc_score(y[m], p[m])
            if a < REGION_MIN_AUROC:
                out.append(f"region {r}: AUROC {a:.3f} < {REGION_MIN_AUROC} (n={int(m.sum())})")
    return out


def _calibrated_reference(score, y, groups, k=5):
    """A reference index as a probability, fairly: a one-input logistic fitted out of
    fold (whole groups held out), so its Brier is comparable to the model's."""
    from sklearn.linear_model import LogisticRegression
    g = np.unique(np.asarray(groups).astype(str), return_inverse=True)[1] % k
    p = np.full(len(y), np.nan)
    for f in range(k):
        tr, te = g != f, g == f
        if te.any() and len(np.unique(y[tr])) == 2:
            p[te] = LogisticRegression().fit(score[tr].reshape(-1, 1), y[tr]).predict_proba(score[te].reshape(-1, 1))[:, 1]
    return p


def beats_reference_failures(p, y, references, groups=None, seed=0, ref_probs=None):
    """Gate 10 (Ido, 2026-11-27): beat EVERY reference system on AUROC and on Brier,
    each gain above 0 at the bootstrap 5th percentile. references: {name: score,
    higher = riskier}; each is calibrated out of fold before its Brier is compared."""
    from sklearn.metrics import brier_score_loss
    p, y = np.asarray(p, float), np.asarray(y)
    groups = np.zeros(len(y)) if groups is None else np.asarray(groups)
    out, rng = [], np.random.default_rng(seed)
    if not references:
        return ["reference comparison: no reference systems in this data"]
    for name, sc in references.items():
        sc = np.asarray(sc, float)
        ok = ~np.isnan(p) & ~np.isnan(sc)
        if ok.sum() < 200 or len(np.unique(y[ok])) < 2:
            out.append(f"{name}: too few rows to compare")
            continue
        pp, yy, ss = p[ok], y[ok], sc[ok]
        # Same treatment as the model: if the caller supplies reference probabilities fitted on
        # the training rows (ref_probs), use them; calibrating on the test rows themselves would
        # tune the reference to the test's own base rate (2026-11-27 fairness fix).
        cal = (np.asarray(ref_probs[name], float)[ok] if ref_probs and name in ref_probs
               else _calibrated_reference(ss, yy, groups[ok]))
        m = ~np.isnan(cal)
        da, db = [], []
        for _ in range(BOOT):
            i = rng.integers(0, len(yy), len(yy))
            if len(np.unique(yy[i])) == 2:
                da.append(roc_auc_score(yy[i], pp[i]) - roc_auc_score(yy[i], ss[i]))
                j = i[m[i]]
                if len(np.unique(yy[j])) == 2:
                    db.append(brier_score_loss(yy[j], cal[j]) - brier_score_loss(yy[j], pp[j]))
        lo_a, lo_b = float(np.percentile(da, 5)), float(np.percentile(db, 5)) if db else -1.0
        if lo_a <= 0:
            out.append(f"does not reliably beat {name} on AUROC (gain 5th pct {lo_a:+.3f})")
        if lo_b <= 0:
            out.append(f"does not reliably beat {name} on Brier (gain 5th pct {lo_b:+.4f})")
    return out

def missing_failures(df, cols, era_start=None):
    """An input blank on more than MAX_MISSING of the rows where it should exist.
    era_start: {col: 'YYYY-MM-DD'} for inputs that only exist from a date (gust 2015-02)."""
    out = []
    for c in cols:
        rows = df if not era_start or c not in era_start else df[df["acq_date"] >= era_start[c]]
        if len(rows) and rows[c].isna().mean() > MAX_MISSING:
            out.append(f"{c} blank on {rows[c].isna().mean():.0%} of its rows")
    return out


def leakage_failures(cols):
    return [f"{c} is a benchmark or label-derived column, never an input"
            for c in cols if c.lower().startswith(FORBIDDEN_PREFIXES)]


# Gate 5, per input group (Ido, 2026-11-26): overlapping inputs share their effect,
# so each PHYSICAL FACTOR must matter, swept as a group. Members move together
# toward "riskier": inputs whose constraint is -1 (wetter = safer) are swept in
# reverse. Inputs outside every group are their own group.
INPUT_GROUPS = {
    "wind": ["wind", "gust_event", "offshore_event"],
    "heat": ["air_temp_encoded", "temp_anom", "vpd_max_event"],
    "moisture": ["humidity", "rh_min_event", "rh_anom", "fm100", "fm1000", "rain_7d", "soil", "kbdi"],
    "vegetation": ["evi"],
    "terrain": ["elevation", "slope", "southness"],
}


def group_effects(model, scaler, X_ref, cols, cst, n=15):
    """{group: average-risk range when the group's inputs sweep their 5th-95th
    percentile together}, on real rows X_ref (columns = cols)."""
    X_ref = np.asarray(X_ref, float)
    groups = {g: [c for c in m if c in cols] for g, m in INPUT_GROUPS.items()}
    groups = {g: m for g, m in groups.items() if m}
    for c in cols:
        if not any(c in m for m in groups.values()):
            groups[c] = [c]
    out = {}
    for g, members in groups.items():
        idx = [cols.index(c) for c in members]
        sign = [-1 if cst[i] < 0 else 1 for i in idx]
        risks = []
        for q in np.linspace(5, 95, n):
            z = X_ref.copy()
            for i, sg in zip(idx, sign):
                col = X_ref[:, i][~np.isnan(X_ref[:, i])]
                if len(col):
                    z[:, i] = np.percentile(col, q if sg > 0 else 100 - q)
            risks.append(model.predict_proba(scaler.transform(z))[:, 1].mean())
        out[g] = round(float(max(risks) - min(risks)), 3)
    return out


def _skill(score, y, prob=None):
    """Standard fire-danger / forecast verification measures for one score."""
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
    score, y = np.asarray(score, float), np.asarray(y)
    base = y.mean()
    out = {"AUROC": roc_auc_score(y, score), "PR-AUC": average_precision_score(y, score)}
    if prob is not None:
        prob = np.clip(np.asarray(prob, float), 1e-6, 1 - 1e-6)
        b = brier_score_loss(y, prob)
        out.update({"Brier": b, "Brier skill": 1 - b / (base * (1 - base)), "log loss": log_loss(y, prob)})
    for top in (0.20, 0.10, 0.03):  # alarm levels: the riskiest 20% / 10% / 3% of rows
        cut = np.quantile(score, 1 - top)
        alarm = score >= cut
        hits, fa, miss = (alarm & (y == 1)).sum(), (alarm & (y == 0)).sum(), (~alarm & (y == 1)).sum()
        cn = (~alarm & (y == 0)).sum()
        n = len(y)
        exp = ((hits + miss) * (hits + fa) + (cn + miss) * (cn + fa)) / n
        out[f"fire capture top {int(top*100)}%"] = hits / max(hits + miss, 1)          # POD
        if top == 0.10:
            out["FAR top 10%"] = fa / max(hits + fa, 1)
            out["CSI top 10%"] = hits / max(hits + fa + miss, 1)
            out["Heidke top 10%"] = (hits + cn - exp) / max(n - exp, 1)
            out["lift top 10%"] = (hits / max(hits + fa, 1)) / max(base, 1e-9)
    return {k: round(float(v), 4) for k, v in out.items()}


def scoreboard(p, y, references, groups=None, ref_probs=None):
    """Every verification measure for the model and each reference system on the same
    rows (references calibrated out of fold for the probability measures)."""
    p, y = np.asarray(p, float), np.asarray(y)
    groups = np.zeros(len(y)) if groups is None else np.asarray(groups)
    board = {"FireScope": _skill(p, y, p)}
    for name, sc in references.items():
        sc = np.asarray(sc, float)
        ok = ~np.isnan(sc) & ~np.isnan(p)
        if ok.sum() >= 200 and len(np.unique(y[ok])) == 2:
            cal = (np.asarray(ref_probs[name], float)[ok] if ref_probs and name in ref_probs
                   else _calibrated_reference(sc[ok], y[ok], groups[ok]))
            m = ~np.isnan(cal)
            board[name] = _skill(sc[ok][m], y[ok][m], cal[m])
            board[name]["_FireScope_same_rows"] = _skill(p[ok][m], y[ok][m], p[ok][m])
    return board


def operational_failures(board):
    """Gate 10b (2026-11-27): on the same rows as each reference, FireScope must capture
    more fires in its riskiest 10% and have a higher CSI at that alarm level."""
    out = []
    for name, r in board.items():
        if name == "FireScope":
            continue
        f = r["_FireScope_same_rows"]
        for k in ("fire capture top 10%", "CSI top 10%"):
            if f[k] <= r[k]:
                out.append(f"{name}: {k} {f[k]:.3f} <= reference {r[k]:.3f}")
    return out


def fit_reference_probs(train_refs, train_y, test_refs):
    """Reference systems calibrated on the TRAINING rows (one-input logistic each) and
    applied to the test rows: the same data the model's own calibration saw."""
    from sklearn.linear_model import LogisticRegression
    out = {}
    for name, sc in train_refs.items():
        sc, y = np.asarray(sc, float), np.asarray(train_y)
        ok = ~np.isnan(sc)
        if ok.sum() >= 200 and len(np.unique(y[ok])) == 2 and name in test_refs:
            lr = LogisticRegression().fit(sc[ok].reshape(-1, 1), y[ok])
            t = np.asarray(test_refs[name], float)
            pr = np.full(len(t), np.nan)
            tok = ~np.isnan(t)
            pr[tok] = lr.predict_proba(t[tok].reshape(-1, 1))[:, 1]
            out[name] = pr
    return out
