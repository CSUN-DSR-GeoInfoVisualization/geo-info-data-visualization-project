import datetime as dt

from ml import v35


def test_display_score_maps_percentile_classes_onto_the_site_bands():
    from ml.inference import risk_label
    assert risk_label(v35.display_score(59.9)) == "Low"
    assert risk_label(v35.display_score(60)) == "Moderate"
    assert risk_label(v35.display_score(80)) == "High"
    assert risk_label(v35.display_score(90)) == "Very High"
    assert risk_label(v35.display_score(97)) == "Extreme"


def test_lookup_finds_nearest_tile_within_three_tenths():
    g = {"tiles": {"34.0,-118.0": {"label": "High"}}}
    assert v35.lookup(g, 34.04, -118.03)["label"] == "High"
    assert v35.lookup(g, 34.2, -118.2)["label"] == "High"
    assert v35.lookup(g, 35.0, -118.0) is None


def test_county_risk_serves_v35_grid_and_falls_back_when_stale(monkeypatch):
    from routes import research as R
    tiles = {f"{la / 10:.1f},{lo / 10:.1f}": {"risk_score": 0.5, "label": "High", "features": {"kbdi": 300.0}}
             for la in range(320, 425) for lo in range(-1245, -1135)}
    fresh = {"date": "2026-11-28", "built_at": dt.datetime.now(dt.timezone.utc).isoformat(), "tiles": tiles}
    monkeypatch.setattr(R, "_load_cache_from_db", lambda k: {"data": fresh})
    R._V35_MEM.clear()
    out = R._compute_county_risk()
    assert out["model"] == "v3.5" and all(c["label"] == "High" for c in out["counties"].values())
    stale = {**fresh, "built_at": (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=40)).isoformat()}
    monkeypatch.setattr(R, "_load_cache_from_db", lambda k: {"data": stale})
    R._V35_MEM.clear()
    assert R._v35_grid() is None
