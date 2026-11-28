import numpy as np
import pandas as pd
from ml import gates


def _data(n=2000, signal=1.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = 1 / (1 + np.exp(-(signal * (y - 0.5) * 2 + rng.normal(0, 1, n))))
    return p, y


def test_calibration():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 5000); y = (rng.uniform(0, 1, 5000) < p).astype(int)   # calibrated
    assert gates.calibration_failures(p, y) == []
    assert gates.calibration_failures(np.clip(p * 0.3 + 0.6, 0, 1), y)          # squeezed + shifted


def test_region_floor():
    p, y = _data()
    lat = np.where(np.arange(len(y)) % 2, 38.0, 34.0)
    elev = np.full(len(y), 100.0)
    p2 = p.copy(); south = lat < 37; p2[south] = 0.5  # no skill at all in the south
    assert gates.region_failures(p, y, lat, elev) == []
    assert any("south-low" in r for r in gates.region_failures(p2, y, lat, elev))


def test_beats_reference_needs_a_real_gain_on_auroc_and_brier():
    p, y = _data(signal=1.5)
    weak, _ = _data(signal=0.2, seed=5)
    groups = np.arange(len(y)) % 10
    assert gates.beats_reference_failures(p, y, {"weak": weak}, groups) == []
    same = gates.beats_reference_failures(p, y, {"same": p.copy()}, groups)
    assert any("AUROC" in f for f in same)                 # equal skill is not a win
    assert gates.beats_reference_failures(p, y, {}) == ["reference comparison: no reference systems in this data"]


def test_missing_and_leakage():
    df = pd.DataFrame({"acq_date": ["2010-01-01", "2016-01-01", "2017-01-01"], "gust_event": [None, 5.0, 6.0]})
    assert gates.missing_failures(df, ["gust_event"]) == ["gust_event blank on 33% of its rows"]
    assert gates.missing_failures(df, ["gust_event"], {"gust_event": "2015-02-01"}) == []
    assert gates.leakage_failures(["evi", "bench_erc", "rfw"]) == [
        "bench_erc is a benchmark or label-derived column, never an input",
        "rfw is a benchmark or label-derived column, never an input"]


def test_group_effects_sum_overlapping_inputs():
    # two copies of one signal: each alone moves risk a little, together a lot
    from ml.train_monotonic import train_monotonic
    rng = np.random.default_rng(0); n = 3000
    x = rng.normal(0, 1, n); y = (rng.uniform(size=n) < 1 / (1 + np.exp(-2 * x))).astype(int)
    X = np.c_[x + rng.normal(0, .1, n), x + rng.normal(0, .1, n)]
    m, s = train_monotonic(X, y, monotonic_cst=[1, 1])
    g = gates.group_effects(m, s, X[:300], ["wind", "gust_event"], [1, 1])
    assert set(g) == {"wind"} and g["wind"] >= 0.3


def test_group_calibration_is_monotonic_and_calibrated():
    from ml.train_monotonic import train_monotonic
    rng = np.random.default_rng(4); n = 4000
    x = rng.normal(0, 1, n); y = (rng.uniform(size=n) < 1 / (1 + np.exp(-1.5 * x))).astype(int)
    groups = rng.integers(0, 10, n)
    m, s = train_monotonic(x.reshape(-1, 1), y, monotonic_cst=[1], groups=groups)
    p = m.predict_proba(s.transform(np.linspace(-2, 2, 9).reshape(-1, 1)))[:, 1]
    assert np.all(np.diff(p) >= -1e-9)                                  # direction preserved
    assert gates.calibration_failures(m.predict_proba(s.transform(x.reshape(-1, 1)))[:, 1], y) == []


def test_scoreboard_and_operational_gate():
    p, y = _data(signal=1.5)
    weak, _ = _data(signal=0.2, seed=5)
    board = gates.scoreboard(p, y, {"weak": weak}, np.arange(len(y)) % 10)
    for k in ("AUROC", "PR-AUC", "Brier", "Brier skill", "log loss", "fire capture top 10%", "CSI top 10%",
              "Heidke top 10%", "FAR top 10%", "lift top 10%"):
        assert k in board["FireScope"] and k in board["weak"]
    assert gates.operational_failures(board) == []
    assert gates.operational_failures(gates.scoreboard(weak, y, {"strong": p}, np.arange(len(y)) % 10))
