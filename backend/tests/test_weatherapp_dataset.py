"""Weather App Plan dataset: the trainer refuses rows not built by the skill-weighted
blend, and the merge keeps pairs whole and fills raw readings without overwriting."""
import argparse
import csv

import pandas as pd
import pytest

from ml import backfill_matched as bm
from ml.retrain_and_gate import check_skill_weighted
from routes.ml_ingest import RAW_WEATHER_COLS, WEATHERAPP_COLS


def _df(**over):
    row = {c: None for c in WEATHERAPP_COLS}
    row.update(weather_method="skill_weighted_v0", n_weather_sources=3, rtma_humidity=40.0)
    row.update(over)
    return pd.DataFrame([row])


def test_gate_accepts_skill_weighted_rows():
    assert check_skill_weighted(_df()) == 0
    assert check_skill_weighted(_df(rtma_humidity=None, n_weather_sources=2)) == 1  # awaiting raw readings


@pytest.mark.parametrize("over", [{"weather_method": None}, {"weather_method": "unweighted"},
                                  {"n_weather_sources": 1}, {"n_weather_sources": None}])
def test_gate_refuses_unweighted_or_single_source(over):
    with pytest.raises(ValueError, match="refusing to train"):
        check_skill_weighted(_df(**over))


def test_gate_refuses_missing_columns():
    with pytest.raises(ValueError, match="lacks column"):
        check_skill_weighted(_df().drop(columns=["weather_method"]))


def _write(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def test_merge(tmp_path):
    base = {"evi": 0.3, "air_temp_encoded": 14000, "wind": 3, "humidity": 40, "elevation": 10, "kbdi": 1,
            "n_weather_sources": 2}
    a = [dict(base, lat=34.1, lon=-118.1, acq_date="2005-07-01", fire=1, source="firms_modis"),
         dict(base, lat=34.1, lon=-118.1, acq_date="2004-07-01", fire=0, source="matched_control"),
         dict(base, lat=35.0, lon=-119.0, acq_date="2013-08-01", fire=0, source="matched_control")]  # no fire
    b = [dict(base, lat=34.1, lon=-118.1, acq_date="2005-07-01", fire=1, source="firms_modis", humidity=99)]
    cols = ["lat", "lon", "acq_date", "evi", "air_temp_encoded", "wind", "humidity", "elevation", "kbdi",
            "fire", "source", "n_weather_sources"]
    _write(tmp_path / "a.csv", a, cols)
    _write(tmp_path / "b.csv", b, cols)
    _write(tmp_path / "raw.csv", [{"lat": 34.1, "lon": -118.1, "acq_date": "2004-07-01", "fire": 0,
                                   "reblend_ok": 1, "era5_humidity": 41}],
           ["lat", "lon", "acq_date", "fire", "reblend_ok"] + RAW_WEATHER_COLS)
    out = tmp_path / "out.csv"
    bm.merge(argparse.Namespace(inp=[tmp_path / "a.csv", tmp_path / "b.csv"], raw=[tmp_path / "raw.csv"], out=out))
    d = pd.read_csv(out)
    assert len(d) == 3                                    # duplicate dropped, unpaired control kept
    assert (d.humidity == 40).all()                       # earlier file wins
    by_date = d.set_index("acq_date")
    assert by_date.loc["2004-07-01", "sensor"] == "MODIS"  # control inherits its fire's sensor
    assert by_date.loc["2013-08-01", "sensor"] == "VIIRS" and by_date.loc["2013-08-01", "pixel_m"] == 375
    assert set(d.weather_method) == {"skill_weighted_v0"}
    assert by_date.loc["2004-07-01", "era5_humidity"] == 41
    check_skill_weighted(d)


def test_merge_refuses_single_source(tmp_path):
    cols = ["lat", "lon", "acq_date", "evi", "air_temp_encoded", "wind", "humidity", "elevation", "kbdi",
            "fire", "source", "n_weather_sources"]
    rows = [{"lat": 1, "lon": 1, "acq_date": "2005-01-01", "fire": f, "source": "firms_viirs", "n_weather_sources": 1}
            for f in (0, 1)]
    _write(tmp_path / "a.csv", rows, cols)
    with pytest.raises(SystemExit, match="refusing"):
        bm.merge(argparse.Namespace(inp=[tmp_path / "a.csv"], raw=None, out=tmp_path / "o.csv"))


def test_every_input_must_matter():
    from ml.train_monotonic import gate_passes, weak_features
    rep = {"evi": {"ok": True, "range": 0.15}, "air_temp_encoded": {"ok": True, "range": 0.013},
           "wind": {"ok": True, "range": None}}
    assert weak_features(rep, 0.05) == ["air_temp_encoded", "wind"]  # unmeasured counts as weak
    assert gate_passes(rep) is True                 # direction-only (old behaviour) still available
    assert gate_passes(rep, 0.05) is False
    rep["wind"]["ok"] = False
    assert gate_passes(rep) is False


def test_tier_gate():
    from ml.train_monotonic import tier_report, tier_failures
    import numpy as np
    y = np.r_[np.zeros(50), np.ones(50)]
    spread = tier_report(np.r_[np.linspace(0.05, 0.55, 50), np.linspace(0.45, 0.95, 50)], y)
    assert tier_failures(spread) == []
    squeezed = tier_report(np.full(100, 0.47), y)          # everything "High"
    assert any("in High" in r for r in tier_failures(squeezed))


def test_open_append_upgrades_an_old_header(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n")
    f, w = bm._open_append(str(p), ["a", "b", "c"])
    w.writerow({"a": 3, "b": 4, "c": 5})
    f.close()
    d = pd.read_csv(p)
    assert list(d.columns) == ["a", "b", "c"]
    assert d.loc[0, "a"] == 1 and pd.isna(d.loc[0, "c"])        # old row kept, by name
    assert d.loc[1].tolist() == [3, 4, 5]                        # new row aligned
