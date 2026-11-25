import pytest
from routes import ml_ingest


class _Resp:
    text = "latitude,longitude,acq_date,confidence\n37.8,-122.3,2026-11-06,n\n"

    def raise_for_status(self):
        pass


def test_fetch_firms_appends_backfill_date(monkeypatch):
    urls = []
    monkeypatch.setattr(ml_ingest, "FIRMS_MAP_KEY", "k")
    monkeypatch.setattr(ml_ingest.requests, "get", lambda url, timeout: urls.append(url) or _Resp())
    assert ml_ingest._fetch_firms(1, "2026-11-06") == ([(37.8, -122.3, "2026-11-06")], 0)
    assert ml_ingest._fetch_firms(2) == ([(37.8, -122.3, "2026-11-06")], 0)
    assert urls[0].endswith("/VIIRS_NOAA20_NRT/-124.0,32.0,-114.0,42.0/1/2026-11-06")
    assert urls[1].endswith("/VIIRS_NOAA20_NRT/-124.0,32.0,-114.0,42.0/2")


def _stub_static(monkeypatch, gee_fails=False):
    import data.live_evi_gee as gee, data.live_evi as ornl, data.live_elevation as el, data.live_kbdi as kb
    def gee_evi(*a):
        if gee_fails:
            raise RuntimeError("image is null")
        return 0.30
    monkeypatch.setattr(gee, "get_evi_at", gee_evi)
    monkeypatch.setattr(ornl, "get_evi_at_ornl", lambda *a: 0.25)
    monkeypatch.setattr(el, "get_elevation", lambda *a: 100.0)
    monkeypatch.setattr(kb, "get_kbdi_at", lambda *a: 50.0)


ARCHIVE = {"air_temp_encoded": (20 + 273.15) / 0.02, "wind": 3.0, "humidity": 40.0}
POWER = {"temperature_celsius": 21.0, "humidity": 45.0, "wind_speed": 3.5}


def test_historical_never_invents_weather_when_both_sources_fail(monkeypatch):
    _stub_static(monkeypatch)
    monkeypatch.setattr(ml_ingest, "_weather_for_date", lambda *a: None)
    monkeypatch.setattr(ml_ingest, "_power_weather_for_date", lambda *a: None)
    with pytest.raises(RuntimeError):
        ml_ingest._features_for_historical(37.8, -122.3, "2026-11-06")


def test_historical_uses_power_backup_and_skips_cross_reference(monkeypatch):
    _stub_static(monkeypatch)
    monkeypatch.setattr(ml_ingest, "_weather_for_date", lambda *a: None)
    monkeypatch.setattr(ml_ingest, "_power_weather_for_date", lambda *a: POWER)
    feats, second = ml_ingest._features_for_historical(37.8, -122.3, "2026-11-06")
    assert feats["humidity"] == 45.0 and feats["wind"] == 3.5 and second is None


def test_historical_cross_references_both_sources_and_falls_back_to_ornl_evi(monkeypatch):
    _stub_static(monkeypatch, gee_fails=True)
    monkeypatch.setattr(ml_ingest, "_weather_for_date", lambda *a: ARCHIVE)
    monkeypatch.setattr(ml_ingest, "_power_weather_for_date", lambda *a: POWER)
    feats, second = ml_ingest._features_for_historical(37.8, -122.3, "2026-11-06")
    assert feats["humidity"] == 40.0 and feats["evi"] == 0.25 and second is POWER
    assert ml_ingest._weather_mismatch(feats, second) is None
    assert "humidity" in ml_ingest._weather_mismatch(feats, {**POWER, "humidity": 75.0})


def test_archive_window_never_requests_future_dates(monkeypatch):
    # Open-Meteo's archive rejects end_date > today, which used to fail the whole
    # lookup for any date in the last 3 days.
    import datetime as dt
    seen = {}

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"daily": {}}

    monkeypatch.setattr(ml_ingest.requests, "get", lambda url, params, timeout: seen.update(params) or R())
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    ml_ingest._weather_for_date(37.8, -122.3, today)
    assert seen["end_date"] < today
