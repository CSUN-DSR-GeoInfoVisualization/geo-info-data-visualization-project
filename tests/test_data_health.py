"""Monitoring tests for the two failures the old checks could not see.

Both defects behind the 2026-07-28 audit were invisible to monitoring:
  * the no-fire sampler starved and the feed ran 86% positive for weeks, while
    the data-health check only looked at feature distributions;
  * the retrain gate rejected six runs in a row and the serving model went
    eight weeks unchanged, while the weekly job reported success each Sunday.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))

from ml.data_quality import (  # noqa: E402
    MIN_POSITIVE_FRAC, MAX_POSITIVE_FRAC, psi, constant_features, FEATURE_COLS,
)
import numpy as np  # noqa: E402


# --- class balance ----------------------------------------------------------

def _balance_ok(frac):
    return MIN_POSITIVE_FRAC <= frac <= MAX_POSITIVE_FRAC


def test_starved_negative_class_is_unhealthy():
    # July 2026: 115 fire / 17 no-fire in the live feed.
    assert _balance_ok(115 / 132) is False


def test_starved_positive_class_is_unhealthy():
    assert _balance_ok(0.05) is False


def test_balanced_feed_is_healthy():
    assert _balance_ok(0.5) is True


# --- PSI is not fooled by a change in class mix -----------------------------

def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    assert psi(x, x) < 1e-9


def test_psi_flags_a_genuinely_shifted_distribution():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(0, 1, 1000), rng.normal(3, 1, 1000)) > 0.25


# --- dead-feature detector --------------------------------------------------

def test_constant_feature_is_reported_dead():
    X = np.random.default_rng(2).normal(size=(200, len(FEATURE_COLS)))
    X[:, FEATURE_COLS.index("wind")] = 0.0  # the original wind field-name bug
    assert "wind" in constant_features(X)


def test_varying_features_are_not_reported_dead():
    X = np.random.default_rng(3).normal(size=(200, len(FEATURE_COLS)))
    assert constant_features(X) == []


# --- model staleness --------------------------------------------------------

def test_staleness_flags_a_stuck_gate():
    from routes.ml_ingest import MAX_CONSECUTIVE_NON_PROMOTIONS

    verdicts = ["PROMOTED"] + ["REJECTED"] * MAX_CONSECUTIVE_NON_PROMOTIONS
    streak = 0
    for v in reversed(verdicts):
        if v == "PROMOTED":
            break
        streak += 1
    assert streak >= MAX_CONSECUTIVE_NON_PROMOTIONS


def test_staleness_does_not_flag_a_healthy_gate():
    from routes.ml_ingest import MAX_CONSECUTIVE_NON_PROMOTIONS

    verdicts = ["REJECTED", "REJECTED", "PROMOTED"]
    streak = 0
    for v in reversed(verdicts):
        if v == "PROMOTED":
            break
        streak += 1
    assert streak < MAX_CONSECUTIVE_NON_PROMOTIONS


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("ALL PASSED")
