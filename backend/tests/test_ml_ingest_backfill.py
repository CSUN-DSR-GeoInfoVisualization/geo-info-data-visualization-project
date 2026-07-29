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


def test_backfill_features_never_fall_back_to_current_weather(monkeypatch):
    # Backfilled fire rows use _features_for_historical, which must raise, not
    # silently substitute today's weather, when the archive has no data.
    monkeypatch.setattr(ml_ingest, "_weather_for_date", lambda *a: None)
    with pytest.raises(RuntimeError):
        ml_ingest._features_for_historical(37.8, -122.3, "2026-11-06")
