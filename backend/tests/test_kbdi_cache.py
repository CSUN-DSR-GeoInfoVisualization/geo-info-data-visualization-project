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


def test_concurrent_insert_race_is_retried_as_update(app, monkeypatch):
    """Two threads warming one tile: the loser's insert hits the primary key; merge_commit
    retries and the write lands instead of logging 'KBDI cache write failed'."""
    from sqlalchemy.exc import IntegrityError
    from models import db, merge_commit, FeatureCacheKbdi
    from decimal import Decimal
    with app.app_context():
        real, calls = db.session.merge, {"n": 0}

        def racy(row):
            calls["n"] += 1
            if calls["n"] == 1:
                real(FeatureCacheKbdi(tile_lat=row.tile_lat, tile_lon=row.tile_lon, kbdi=1.0, source="x"))
                db.session.commit()  # the other thread's row lands first
                raise IntegrityError("INSERT", {}, Exception("duplicate key"))
            return real(row)
        monkeypatch.setattr(db.session, "merge", racy)
        merge_commit(FeatureCacheKbdi(tile_lat=Decimal("32.7500"), tile_lon=Decimal("-116.2500"), kbdi=161.6, source="nasa_power"))
        got = FeatureCacheKbdi.query.filter_by(tile_lat=Decimal("32.7500"), tile_lon=Decimal("-116.2500")).one()
        assert float(got.kbdi) == 161.6 and calls["n"] == 2
