import datetime as dt

import pytest


def test_kbdi_cached_per_quarter_degree_and_stale_fallback(app, monkeypatch):
    import data.live_kbdi as lk
    from data import live_kbdi_cached as c
    from models import db, FeatureCacheKbdi
    calls = []
    monkeypatch.setattr(lk, "get_kbdi", lambda la, lo: calls.append((la, lo)) or 300.0)
    with app.app_context():
        assert c.get_kbdi_cached(34.01, -118.02) == 300.0
        assert c.get_kbdi_cached(34.10, -117.90) == 300.0      # same 0.25 deg cell: cached
        assert calls == [(34.0, -118.0)]                       # computed once, at the cell centre
        row = FeatureCacheKbdi.query.first()
        row.fetched_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=30)
        db.session.commit()
        monkeypatch.setattr(lk, "get_kbdi", lambda la, lo: (_ for _ in ()).throw(RuntimeError("POWER 429")))
        assert c.get_kbdi_cached(34.0, -118.0) == 300.0        # stale but < 72 h beats failing
        row.fetched_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=100)
        db.session.commit()
        with pytest.raises(RuntimeError):
            c.get_kbdi_cached(34.0, -118.0)
