import logging

from services import failure_digest


def test_digest_groups_repeats_and_emails_once(monkeypatch):
    sent = []
    import routes.ml_ingest as ml
    monkeypatch.setattr(ml, "_send_alert_email", lambda subject, body: sent.append((subject, body)) or ("id", None))
    failure_digest.install()
    failure_digest.install()  # idempotent: a second call must not double-count
    with failure_digest._lock:
        failure_digest._counts.clear()
        failure_digest._samples.clear()
    log = logging.getLogger("data.live_weather")
    for lat in (34.1, 35.2, 36.3):
        log.warning("live wind lookup failed at %s,-118: 429 Too Many Requests", lat)
    log.error("NASA POWER fetch failed: timeout")
    log.info("not a failure")
    failure_digest.flush()
    assert len(sent) == 1
    subject, body = sent[0]
    assert subject.startswith("4 failures") and "(2 kinds)" in subject and "since the last report" in subject
    assert "x3  WARNING [data.live_weather] live wind lookup failed" in body
    assert "NASA POWER fetch failed" in body
    failure_digest.flush()
    assert len(sent) == 1  # nothing new -> no email
