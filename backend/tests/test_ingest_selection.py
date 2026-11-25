from flask import Flask

from routes import ml_ingest as mi


def test_thinning_keeps_one_detection_per_cell_per_day_and_is_deterministic():
    fires = [(34.001, -118.001, "2026-11-10"), (34.002, -118.002, "2026-11-10"),  # same fire
             (34.001, -118.001, "2026-11-11"),                                     # same spot, next day
             (36.5, -119.5, "2026-11-10")]
    a = mi._thin_detections(fires, 10, seed="2026-11-10")
    assert len(a) == 3
    assert a == mi._thin_detections(fires, 10, seed="2026-11-10")


def test_static_source_screen_includes_neighbour_cells(monkeypatch):
    monkeypatch.setattr(mi, "_STATIC_SOURCES", {(33.77, -118.24)})
    assert mi._is_static_source(33.771, -118.239)
    assert mi._is_static_source(33.781, -118.249)   # neighbouring cell
    assert not mi._is_static_source(34.5, -118.24)


def test_ingest_keeps_complete_pairs_only(monkeypatch):
    monkeypatch.setenv("INTERNAL_CRON_TOKEN", "t")
    fires = [(36.1, -119.1, "2026-11-10"), (37.2, -120.2, "2026-11-10")]
    monkeypatch.setattr(mi, "_fetch_firms", lambda days, date="": (fires, 0))
    monkeypatch.setattr(mi, "on_ca_land", lambda la, lo: True)
    monkeypatch.setattr(mi, "_county_features", lambda: [1])
    monkeypatch.setattr(mi, "_STATIC_SOURCES", set())
    feats = {"evi": 0.3, "air_temp_encoded": 14600.0, "wind": 3.0, "humidity": 40.0, "elevation": 100.0, "kbdi": 50.0}

    def fake(lat, lon, date):
        if lat == 37.2 and date.startswith("2025"):
            raise RuntimeError("control failed")  # this fire's control can't be built
        return dict(feats), 3
    monkeypatch.setattr(mi, "_features_for_historical", fake)
    app = Flask(__name__)
    app.register_blueprint(mi.ml_ingest_bp, url_prefix="/api")
    d = app.test_client().post("/api/internal/ml/ingest?days=1&date=2026-11-10", headers={"X-Internal-Token": "t"}).get_json()
    assert d["fire_rows"] == 1 and d["nofire_rows"] == 1
    assert {(r["lat"], r["fire"]) for r in d["rows"]} == {(36.1, 1), (36.1, 0)}


def test_dataset_gate_flags_unpaired_recent_fires(tmp_path):
    import datetime as dt
    from ml.weather_skill import dataset_report
    d = (dt.date.today() - dt.timedelta(days=8)).isoformat()
    prior = (dt.date.today() - dt.timedelta(days=8)).replace(year=dt.date.today().year - 1).isoformat()
    p = tmp_path / "m.csv"
    lines = ["lat,lon,acq_date,evi,air_temp_encoded,wind,humidity,elevation,kbdi,fire,source"]
    lines += [f"3{i}.0,-118.0,{d},0,0,0,0,0,0,1,firms_viirs" for i in range(5)]
    lines += [f"30.0,-118.0,{prior},0,0,0,0,0,0,0,matched_control"]   # only 1 of 5 fires paired
    p.write_text("\n".join(lines) + "\n")
    r = dataset_report(str(p))
    assert r["recent_fires"] == 5 and r["unpaired_fires"] == 4 and r["problems"]
