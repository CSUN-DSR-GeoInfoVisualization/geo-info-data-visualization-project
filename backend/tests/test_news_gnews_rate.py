import requests

from services.fire_news import aggregator as ag


class _R:
    def __init__(self, status):
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} for url: https://gnews.io/?apikey=FAKE", response=self)

    def json(self):
        return {"articles": [{"title": "Wildfire near Ojai", "description": "brush fire burns 200 acres",
                              "url": "https://example.com/a", "publishedAt": "2026-11-25T10:00:00Z", "source": {"name": "X"}}]}


def test_gnews_called_at_most_hourly_and_backs_off_after_429(monkeypatch):
    monkeypatch.setenv("GNEWS_API_KEY", "fake")
    calls = []
    monkeypatch.setattr(ag.requests, "get", lambda *a, **k: calls.append(1) or _R(200 if len(calls) == 1 else 429))
    now = [1_000_000.0]
    monkeypatch.setattr(ag.time, "time", lambda: now[0])
    ag._gnews.update(at=0.0, items=[])
    first = ag._fetch_gnews()
    assert len(calls) == 1 and first
    now[0] += 600
    assert ag._fetch_gnews() == first and len(calls) == 1     # within the hour: cached
    now[0] += 3600
    assert ag._fetch_gnews() == first and len(calls) == 2     # 429: keeps last results
    now[0] += 60
    ag._fetch_gnews()
    assert len(calls) == 2                                    # and waits an hour to retry
