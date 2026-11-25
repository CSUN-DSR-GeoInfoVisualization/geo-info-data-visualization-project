from flask import Flask

from services import rate_limit


def test_limiter_blocks_after_limit_and_counts_cost():
    with Flask(__name__).app_context():  # a non-testing app: limits enforced
        assert all(rate_limit.allow("t:a", 3, 60) for _ in range(3))
        assert not rate_limit.allow("t:a", 3, 60)
        assert rate_limit.allow("t:b", 10, 60, cost=10)
        assert not rate_limit.allow("t:b", 10, 60, cost=1)


def test_decorator_returns_429_and_trusts_only_last_forwarded_ip():
    app = Flask(__name__)

    @app.route("/x", methods=["POST"])
    @rate_limit.rate_limited("t-dec", 2, 60)
    def x():
        return "ok"

    c = app.test_client()
    # A client-supplied first XFF entry must not create a fresh bucket.
    codes = [c.post("/x", headers={"X-Forwarded-For": f"9.9.9.{i}, 1.2.3.4"}).status_code for i in range(3)]
    assert codes == [200, 200, 429]


def test_admin_cron_routes_reject_missing_token(monkeypatch):
    monkeypatch.setenv("INTERNAL_CRON_TOKEN", "secret-token")
    from routes.history import history_bp
    from routes.research import research_bp
    app = Flask(__name__)
    app.register_blueprint(history_bp, url_prefix="/api/history")
    app.register_blueprint(research_bp, url_prefix="/api/research")
    c = app.test_client()
    for path in ("/api/history/admin/backfill-years", "/api/research/admin/prewarm-tiles"):
        assert c.post(path).status_code == 401
        assert c.post(path, headers={"X-Internal-Token": "wrong"}).status_code == 401
