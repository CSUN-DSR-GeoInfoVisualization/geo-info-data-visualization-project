import gzip
import json

from flask import Flask

from data import dins_archive as da

GOOD = {"HAZARDTYPE": "Fire", "DAMAGE": "Destroyed (>50%)", "LATITUDE": 34.05, "LONGITUDE": -118.5,
        "INCIDENTNAME": "Palisades", "INCIDENTNUM": "CALFD 000001", "INCIDENTSTARTDATE": 1736208000000}


def test_validation_keeps_only_fire_damage_in_california_with_an_incident():
    assert da._valid(GOOD) is None
    assert da._valid({**GOOD, "HAZARDTYPE": "Flood"}) == "not a fire"
    assert da._valid({**GOOD, "DAMAGE": "No Damage"}) == "no damage recorded"
    assert da._valid({**GOOD, "DAMAGE": "Inaccessible"}) == "no damage recorded"
    assert da._valid({**GOOD, "LATITUDE": 0, "LONGITUDE": 0}) == "coordinates outside California"
    assert da._valid({**GOOD, "INCIDENTNUM": None}).startswith("missing incident")


def _app(tmp_path, monkeypatch, years):
    monkeypatch.setattr(da, "ARCHIVE_DIR", str(tmp_path))
    for y, feats in years.items():
        with gzip.open(tmp_path / f"dins_{y}.json.gz", "wt") as f:
            json.dump({"type": "FeatureCollection", "year": y, "inspection_summary": {"X": {"name": "Palisades", "counts": {"No Damage": 5}}},
                       "features": feats}, f)
    from routes import history
    history._DINS_MEMO.clear()
    app = Flask(__name__)
    app.register_blueprint(history.history_bp, url_prefix="/api/history")
    return app.test_client()


def test_endpoint_serves_the_archive_and_explains_pre_2013(tmp_path, monkeypatch):
    c = _app(tmp_path, monkeypatch, {2025: [{"type": "Feature", "properties": GOOD}]})
    d = c.get("/api/history/dins?year=2025").get_json()
    assert len(d["features"]) == 1 and d["inspection_summary"]["X"]["counts"]["No Damage"] == 5
    assert c.get("/api/history/dins?year=2025").get_json() == d          # served from memory
    old = c.get("/api/history/dins?year=2010").get_json()
    assert old["features"] == [] and "2013" in old["note"]
    assert c.get("/api/history/dins").status_code == 400
