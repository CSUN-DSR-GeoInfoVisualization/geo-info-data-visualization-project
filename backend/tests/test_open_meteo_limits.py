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
