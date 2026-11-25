import time

from flask import Flask

from routes import history
from services import cache


def test_past_years_never_expire_current_year_refreshes():
    this_year = time.gmtime().tm_year
    assert history._freshness(this_year - 1) > 50 * 365 * 86400
    assert history._freshness(this_year) == 30 * 86400
    assert history._freshness(None) == 30 * 86400


def test_old_past_year_row_is_served_without_recompute(monkeypatch):
    # A 200-day-old row used to trip the age backstop, recompute, and rewrite a
    # multi-MB row, which OOM-killed the 256 MB database.
    body = b'{"type":"FeatureCollection","features":[]}'
    monkeypatch.setattr(cache, "_load_from_db", lambda key: {
        "body": body, "body_br": None, "etag": '"e"', "content_type": "application/json",
        "computed_at": time.time() - 200 * 86400})
    cache._mem.pop("t-old", None)

    def must_not_recompute():
        raise AssertionError("recomputed an immutable past year")

    app = Flask(__name__)

    @app.route("/t")
    def t():
        return cache.serve_cached(cache_key="t-old", ttl_seconds=60, compute_fn=must_not_recompute,
                                  db_freshness_seconds=history._freshness(time.gmtime().tm_year - 3))

    r = app.test_client().get("/t")
    assert r.status_code == 200 and r.data == body
