from unittest import mock

import pytest

from data import open_meteo


def _resp(status, text=""):
    r = mock.Mock(status_code=status, text=text)
    r.raise_for_status = mock.Mock()
    return r


def test_429_raises_rate_limited(monkeypatch):
    monkeypatch.delenv("OPEN_METEO_API_KEY", raising=False)
    with mock.patch("data.open_meteo.requests.get", return_value=_resp(429, "Daily API request limit exceeded")):
        with pytest.raises(open_meteo.RateLimited):
            open_meteo.archive({"latitude": 1})


def test_key_uses_customer_endpoint(monkeypatch):
    monkeypatch.setenv("OPEN_METEO_API_KEY", "k")
    with mock.patch("data.open_meteo.requests.get", return_value=_resp(200)) as g:
        open_meteo.archive({"latitude": 1})
    assert g.call_args.args[0].startswith("https://customer-archive-api.open-meteo.com/")
    assert g.call_args.kwargs["params"]["apikey"] == "k"


def test_weather_for_date_does_not_swallow_rate_limit():
    from routes import ml_ingest as mi
    with mock.patch("data.open_meteo.archive", side_effect=open_meteo.RateLimited("429")):
        with pytest.raises(open_meteo.RateLimited):
            mi._weather_for_date(37.0, -120.0, "2010-06-01")


def test_one_weather_source_is_not_enough():
    from routes import ml_ingest as mi
    one = {"wind": 3.0, "air_temp_encoded": 14650.0, "humidity": 40.0}
    with mock.patch.object(mi, "_weather_for_date", return_value=one), \
         mock.patch.object(mi, "_power_weather_for_date", return_value=None), \
         mock.patch.object(mi, "_rtma_weather_for_date", return_value=None):
        with pytest.raises(RuntimeError, match="need 2"):
            mi._features_for_historical(37.0, -120.0, "2005-06-01")


def test_power_429_backs_off_then_raises_rate_limited(monkeypatch):
    monkeypatch.setenv("POWER_MIN_INTERVAL", "0.001")
    from data import nasa_power
    nasa_power.window.cache_clear()
    with mock.patch("data.nasa_power.requests.get", return_value=_resp(429)) as g, \
         mock.patch("data.nasa_power.time.sleep") as sl:
        with pytest.raises(open_meteo.RateLimited):
            nasa_power.window(37.0, -120.0, "2010-06-01")
    assert g.call_count == 4 and [c.args[0] for c in sl.call_args_list if c.args[0] >= 1] == [60, 120, 240]


def test_power_throttle_does_not_switch_kbdi_to_open_meteo():
    from data import live_kbdi
    with mock.patch("ml.enrich_kbdi.fetch_spinup_window", side_effect=open_meteo.RateLimited("429")), \
         mock.patch.object(live_kbdi, "_spinup_open_meteo") as backup:
        with pytest.raises(open_meteo.RateLimited):
            live_kbdi.get_kbdi_at(37.0, -120.0, "2010-06-01")
    backup.assert_not_called()


def test_power_429_in_production_fails_fast(monkeypatch):
    monkeypatch.delenv("POWER_MIN_INTERVAL", raising=False)
    from data import nasa_power
    nasa_power.window.cache_clear()
    with mock.patch("data.nasa_power.requests.get", return_value=_resp(429)) as g, \
         mock.patch("data.nasa_power.time.sleep") as sl:
        with pytest.raises(open_meteo.RateLimited):
            nasa_power.window(37.0, -120.0, "2010-06-01")
    assert g.call_count == 2 and [c.args[0] for c in sl.call_args_list] == [5]
