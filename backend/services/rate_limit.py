"""Per-client sliding-window limits for the public endpoints an attacker could use
to brute-force logins or burn the shared upstream quotas (each prediction fetches
live weather/elevation/EVI). In-memory is enough: gunicorn runs one process.
shortcut: limits reset on redeploy and keys for idle IPs are pruned only when that
IP returns; move to Redis if the API ever runs more than one process.
"""
import collections
import threading
import time
from functools import wraps

from flask import current_app, has_app_context, jsonify, request

_lock = threading.Lock()
_hits = collections.defaultdict(collections.deque)


def client_ip() -> str:
    # Render's proxy appends the address it saw; earlier entries are client-supplied
    # and spoofable, so only the last one is trusted.
    xff = request.headers.get("X-Forwarded-For", "")
    return xff.split(",")[-1].strip() if xff else (request.remote_addr or "unknown")


def allow(key: str, limit: int, window_s: float, cost: int = 1) -> bool:
    if has_app_context() and current_app.testing:
        return True  # test suites log in hundreds of times from one address
    now = time.monotonic()
    with _lock:
        q = _hits[key]
        while q and q[0] <= now - window_s:
            q.popleft()
        if len(q) + cost > limit:
            return False
        q.extend([now] * cost)
        return True


def rate_limited(name: str, limit: int, window_s: float, cost=lambda: 1):
    """Decorator: 429 once a client IP exceeds `limit` units per `window_s`."""
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if not allow(f"{name}:{client_ip()}", limit, window_s, max(1, int(cost()))):
                return jsonify({"error": "too many requests, slow down"}), 429
            return fn(*args, **kwargs)
        return wrapped
    return deco
