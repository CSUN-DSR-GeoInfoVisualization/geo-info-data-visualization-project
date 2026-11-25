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


ERA5 = {"air_temp_encoded": (20 + 273.15) / 0.02, "wind": 3.0, "humidity": 40.0}
POWER = {"temperature_celsius": 21.0, "humidity": 45.0, "wind_speed": 3.5}
RTMA = {"temperature_celsius": 19.0, "humidity": 50.0, "wind_speed": 2.0}


def _sources(monkeypatch, era5=ERA5, power=POWER, rtma=RTMA):
    monkeypatch.setattr(ml_ingest, "_weather_for_date", lambda *a: era5)
    monkeypatch.setattr(ml_ingest, "_power_weather_for_date", lambda *a: power)
    monkeypatch.setattr(ml_ingest, "_rtma_weather_for_date", lambda *a: rtma)


def test_historical_never_invents_weather_when_every_source_fails(monkeypatch):
    _stub_static(monkeypatch)
    _sources(monkeypatch, None, None, None)
    with pytest.raises(RuntimeError):
        ml_ingest._features_for_historical(37.8, -122.3, "2026-11-06")


def test_single_source_is_used_and_counted(monkeypatch):
    _stub_static(monkeypatch)
    _sources(monkeypatch, None, POWER, None)
    feats, n = ml_ingest._features_for_historical(37.8, -122.3, "2026-11-06")
    assert n == 1 and feats["humidity"] == 45.0 and feats["wind"] == 3.5


def test_three_sources_use_measured_weights_and_ornl_backup(monkeypatch):
    _stub_static(monkeypatch, gee_fails=True)
    _sources(monkeypatch)
    feats, n = ml_ingest._features_for_historical(37.8, -122.3, "2026-11-06")
    w = ml_ingest.WEATHER_WEIGHTS["humidity"]
    expect_h = (w["rtma"] * 50 + w["era5"] * 40 + w["merra2"] * 45) / sum(w.values())
    assert n == 3 and feats["evi"] == 0.25
    assert abs(feats["humidity"] - expect_h) < 1e-9          # humidity: skill-weighted blend
    assert feats["wind"] == 2.0                                # wind: RTMA (best measured)
    assert abs(feats["air_temp_encoded"] * 0.02 - 273.15 - 19.0) < 1e-6  # temperature: RTMA


def test_outlier_source_is_dropped_but_row_kept():
    blended, notes = ml_ingest._blend_weather({
        "rtma": RTMA, "merra2": POWER,
        "era5": {"temperature_celsius": 20.0, "humidity": 95.0, "wind_speed": 3.0}})
    assert any("era5 humidity outlier" in n for n in notes)
    w = ml_ingest.WEATHER_WEIGHTS["humidity"]
    assert abs(blended["humidity"] - (w["rtma"] * 50 + w["merra2"] * 45) / (w["rtma"] + w["merra2"])) < 1e-9


def test_two_disagreeing_sources_have_no_tiebreak():
    with pytest.raises(ml_ingest.CrossSourceMismatch):
        ml_ingest._blend_weather({"rtma": RTMA, "era5": {"temperature_celsius": 19.5, "humidity": 90.0, "wind_speed": 2.2}})


def test_fallback_order_when_rtma_missing():
    blended, _ = ml_ingest._blend_weather({"era5": {"temperature_celsius": 20.0, "humidity": 40.0, "wind_speed": 3.0},
                                           "merra2": POWER})
    assert blended["temperature_celsius"] == 20.0 and blended["wind_speed"] == 3.0  # ERA5 before MERRA-2


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
