"""Monotonic-constrained wildfire-risk trainer + physical-direction gate.

Why this exists (audit 2026-06-03): the shipped CalibratedRF, trained on ~1,022
rows, learned physically-backwards relationships — wind ANTI-correlated with
risk, humidity inverted at the dry end, KBDI saturating with a low-end dip,
elevation effectively dead. Root cause: no physical priors + sparse data.

This trainer encodes fire physics as MONOTONIC CONSTRAINTS so the model can't
learn a backwards relationship no matter how noisy/sparse the data is, and ships
a validation gate that REFUSES any model whose constrained features point the
wrong way.

Feature order matches ml/inference.py:
    [evi, air_temp_encoded, wind, humidity, elevation, kbdi]
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ["evi", "air_temp_encoded", "wind", "humidity", "elevation", "kbdi"]

# Physical priors. +1: risk must not DECREASE as the feature rises.
#                  -1: risk must not INCREASE as the feature rises.
#                   0: unconstrained (direction genuinely ambiguous).
#   evi               0  more vegetation = more fuel, but greener also = wetter
#   air_temp_encoded +1  hotter -> drier fuels -> more risk
#   wind             +1  wind drives ignition/spread -> more risk
#   humidity         -1  more atmospheric moisture -> less risk
#   elevation         0  higher can be cooler/wetter OR more exposed
#   kbdi             +1  drought index -> more risk
MONOTONIC_CST = [0, 1, 1, -1, 0, 1]

# HGB extrapolates flat past the training range (like RF) but, unlike RF, the
# monotonic constraint guarantees direction throughout. Tuned modestly; the real
# retrain can grid-search these.
HGB_PARAMS = dict(
    loss="log_loss",
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=42,
)


def train_monotonic(X, y, calibrate=True, monotonic_cst=None, groups=None):
    """Fit StandardScaler + monotonic HGB, optionally isotonic-calibrated.

    StandardScaler has a strictly-positive scale_, so the monotonic direction is
    preserved between raw and scaled space — the constraint indices still apply.
    Isotonic calibration is monotonic in the score, so it preserves direction too
    (and, unlike sigmoid/Platt, does not compress the top end into saturation).

    Returns (model, scaler).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    base = HistGradientBoostingClassifier(
        monotonic_cst=MONOTONIC_CST if monotonic_cst is None else monotonic_cst, **HGB_PARAMS)
    if not calibrate:
        base.fit(Xs, y)
        return base, scaler
    if groups is not None:
        # Grouped calibration (2026-11-27): random calibration folds mixed years and
        # fire/control pairs and fit each model on 80% of rows, which flattened the
        # probabilities (held-out AUROC 0.661 vs 0.698 uncalibrated). Fit the model on
        # every row, then one sigmoid on out-of-fold scores with whole groups held out.
        return GroupCalibrated.fit(base, Xs, y, groups), scaler
    model = CalibratedClassifierCV(estimator=base, method="isotonic", cv=5)
    model.fit(Xs, y)
    return model, scaler


class GroupCalibrated:
    """A monotonic HGB plus a sigmoid fitted on group-held-out scores. Sigmoid is
    monotonic, so the physics direction is preserved."""

    def __init__(self, model, calibrator):
        self.model, self.calibrator = model, calibrator

    @classmethod
    def fit(cls, base, Xs, y, groups, k=5):
        from sklearn.base import clone
        from sklearn.linear_model import LogisticRegression
        y = np.asarray(y)
        fold = pd_factorize(groups) % k
        oof = np.full(len(y), np.nan)
        for f in range(k):
            tr, te = fold != f, fold == f
            if te.any() and len(np.unique(y[tr])) == 2:
                Xtr = Xs[tr].copy()
                dead = np.isnan(Xtr).all(axis=0)  # an input absent from this fold's years (e.g. gust before 2014)
                Xtr[:, dead] = 0.0                  # constant: the fold model cannot split on it
                oof[te] = clone(base).fit(Xtr, y[tr]).decision_function(Xs[te])
        ok = ~np.isnan(oof)
        cal = LogisticRegression().fit(oof[ok].reshape(-1, 1), y[ok])
        return cls(clone(base).fit(Xs, y), cal)

    def predict_proba(self, X):
        return self.calibrator.predict_proba(self.model.decision_function(X).reshape(-1, 1))


def pd_factorize(groups):
    _, codes = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    return codes


def validate_monotonicity(model, scaler, n=25, tol=1e-6, X_ref=None):
    """PDP-style gate. For each constrained feature, sweep it across the training
    range (others held at the training mean) and check risk moves in the required
    direction the whole way. The retrain pipeline must REFUSE to promote a
    candidate if any constrained feature fails — that's how an inverted/dead
    feature can never silently ship again.

    X_ref (real rows): sweep each feature across its 5th-95th percentile and
    average the risk over those rows (a true partial dependence), for EVERY
    feature, constrained or not. That is what the map experiences; a sweep at the
    single mean point misses effects that only appear with other conditions
    (e.g. heat on dry days) and reports dead features as alive (audit 2026-11-26).

    Returns {feature: {expected, ok, max_violation, range}}.
    """
    mean = scaler.mean_
    sd = scaler.scale_
    ref = None if X_ref is None else np.asarray(X_ref, dtype=float)
    if ref is not None and len(ref) > 400:
        ref = ref[np.random.default_rng(0).choice(len(ref), 400, replace=False)]
    report = {}
    for i, name in enumerate(FEATURE_COLS):
        want = MONOTONIC_CST[i]
        if want == 0 and ref is None:
            report[name] = {"expected": "free", "ok": True, "max_violation": 0.0, "range": None}
            continue
        if ref is None:
            lo, hi = mean[i] - 2 * sd[i], mean[i] + 2 * sd[i]
        else:
            lo, hi = np.percentile(ref[:, i], [5, 95])
        pts = np.linspace(lo, hi, n)
        risks = []
        for p in pts:
            if ref is None:
                x = mean.copy()
                x[i] = p
                risks.append(float(model.predict_proba(scaler.transform([x]))[0][1]))
            else:
                z = ref.copy()
                z[:, i] = p
                risks.append(float(model.predict_proba(scaler.transform(z))[:, 1].mean()))
        if want == 0:
            report[name] = {"expected": "free", "ok": True, "max_violation": 0.0,
                            "range": round(max(risks) - min(risks), 4)}
            continue
        diffs = np.diff(risks)
        # A step in the WRONG direction is a positive violation; want=+1 means
        # diffs should be >=0, so a violation is a negative diff (and vice versa).
        max_violation = float(max((-want * d) for d in diffs))
        report[name] = {
            "expected": "increasing" if want > 0 else "decreasing",
            "ok": max_violation <= tol,
            "max_violation": max_violation,
            "range": round(max(risks) - min(risks), 4),
        }
    return report


def gate_passes(report, min_effect=None):
    """True iff every constrained feature respects its physical direction and,
    with min_effect, every feature moves risk by at least that much."""
    return all(v["ok"] for v in report.values()) and (
        min_effect is None or not weak_features(report, min_effect))


# Every input must matter (Ido, 2026-11-26): across its 5th-95th percentile, on
# real rows, each feature must move the average risk by at least a quarter of a
# risk tier (tiers are 0.20 wide). A model that ignores temperature or wind is
# not predicting wildfire risk, however good its AUROC.
MIN_INPUT_EFFECT = 0.05


def weak_features(report, min_effect=MIN_INPUT_EFFECT):
    """Features whose measured effect is below min_effect (unmeasured counts as weak)."""
    return [n for n, v in report.items() if v.get("range") is None or v["range"] < min_effect]


# A constrained feature whose risk range over the swept interval is below this
# is not being used at all. Direction alone cannot catch that: a perfectly flat
# response satisfies both "non-decreasing" and "non-increasing", so a dead
# feature passes the physics gate silently. Measured 2026-07-28: humidity had
# range 0.0085 in the shipped model and exactly 0.0 in the candidate, i.e. the
# model ignored atmospheric moisture entirely while the gate reported OK.
MIN_EFFECT = 0.01


def dead_constrained_features(report, min_effect=MIN_EFFECT):
    """Constrained features the model has effectively flattened to nothing.

    Reported rather than hard-gated: a dead feature usually means the DATA has
    no usable signal in the physically-correct direction (for humidity it was
    an inverted correlation caused by the negative-sampling design), and
    blocking promotion would keep a worse model in production without fixing
    the cause. The caller surfaces this so it cannot pass unnoticed.
    """
    return [name for name, v in report.items()
            if v["range"] is not None and v["range"] < min_effect]

# The five NFDRS tiers must be used (Ido, 2026-11-26): a model that puts nearly
# every place in one tier cannot tell a safe place from a dangerous one, however
# well it ranks. Measured: the 2026-11-26 model put 88.7% of rows in "High".
TIERS = ("Low", "Moderate", "High", "Very High", "Extreme")
MAX_TIER_SHARE = 0.50   # no tier holds more than half the rows
MIN_TIERS_USED = 3      # at least this many tiers each hold >= MIN_TIER_SHARE
MIN_TIER_SHARE = 0.05


def tier_report(probs, y):
    """{tier: {share, fire_rate}} with the live cutoffs (ml.inference.risk_label)."""
    from ml.inference import risk_label
    probs, y = np.asarray(probs), np.asarray(y)
    labels = np.array([risk_label(p) for p in probs])
    return {t: {"share": round(float((labels == t).mean()), 3),
                "fire_rate": round(float(y[labels == t].mean()), 3) if (labels == t).any() else None}
            for t in TIERS}


def tier_failures(report):
    """Reasons the tiers fail: one tier dominates, too few tiers used, or the fire
    rate falls from a used tier to the next (or never rises at all)."""
    out = []
    top = max(report, key=lambda t: report[t]["share"])
    if report[top]["share"] > MAX_TIER_SHARE:
        out.append(f"{report[top]['share']:.0%} of rows in {top} (max {MAX_TIER_SHARE:.0%})")
    used = [t for t in TIERS if report[t]["share"] >= MIN_TIER_SHARE]
    if len(used) < MIN_TIERS_USED:
        out.append(f"only {len(used)} tier(s) hold >= {MIN_TIER_SHARE:.0%} of rows (need {MIN_TIERS_USED})")
    rates = [report[t]["fire_rate"] for t in TIERS if report[t]["share"] >= 0.01]
    if any(b < a for a, b in zip(rates, rates[1:])) or (rates and rates[-1] <= rates[0]):
        out.append(f"fire rate does not rise with tier ({rates})")
    return out
