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


def beats_reference_failures(p, y, references, seed=0):
    """The model must beat the best reference system by AUROC, and the bootstrap 5th
    percentile of the gain must be above 0 (a gain noise could reverse is not a gain).
    references: {name: scores where higher = riskier} on the same rows."""
    p, y = np.asarray(p), np.asarray(y)
    refs = {n: np.asarray(s, float) for n, s in references.items()}
    ok = ~np.isnan(p)
    for s in refs.values():
        ok &= ~np.isnan(s)
    if ok.sum() < 200 or not refs:
        return ["reference comparison: too few rows with every reference"]
    p, y = p[ok], y[ok]
    refs = {n: s[ok] for n, s in refs.items()}
    best = max(refs, key=lambda n: roc_auc_score(y, refs[n]))
    rng, gains = np.random.default_rng(seed), []
    for _ in range(BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            gains.append(roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], refs[best][i]))
    lo = float(np.percentile(gains, 5))
    if lo <= 0:
        return [f"does not reliably beat {best}: AUROC gain 5th percentile {lo:+.3f}"]
    return []


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
