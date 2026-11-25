"""Email the owner every failure the API logs, as one digest per hour.

Every data-source failure, fallback (backup source, IDW interpolation), rate limit
and unhandled exception in this codebase is already logged at WARNING or ERROR.
This collects those records, groups repeats (numbers normalised, so 96 identical
429s become one "x96" line), and emails the group once an hour, plus once more on
shutdown so a redeploy doesn't lose the buffer. One process (gunicorn runs a
single worker with threads), so one flusher.
"""
import atexit
import collections
import logging
import re
import threading
import time

DIGEST_SECONDS = 3600

_lock = threading.Lock()
_counts = collections.Counter()
_samples = {}
_installed = False


# Credentials that upstream error messages embed in URLs (query keys, FIRMS'
# 32-character path key) must never reach an email.
_SECRET = re.compile(r"((?:api_?key|key|token|access_token|map_key)=)[^&\s'\"]+", re.I)
_PATH_KEY = re.compile(r"/[0-9a-f]{32}/")


def redact(msg):
    return _PATH_KEY.sub("/***/", _SECRET.sub(r"\1***", msg))


class _Collector(logging.Handler):
    def emit(self, record):
        if record.name == __name__:
            return  # never digest our own send failures (would loop)
        if record.name == "googleapiclient.http" and "before retry" in record.getMessage():
            return  # a retry in progress; if every retry fails, our own code logs the failure
        try:
            msg = redact(record.getMessage())
        except Exception:
            msg = redact(str(record.msg))
        key = f"{record.levelname} [{record.name}] " + re.sub(r"-?\d+(\.\d+)?", "N", msg)[:180]
        with _lock:
            _counts[key] += 1
            _samples.setdefault(key, msg[:500])


def flush():
    """Send whatever has been collected; no-op when nothing failed."""
    with _lock:
        items = _counts.most_common()
        samples = dict(_samples)
        _counts.clear()
        _samples.clear()
    if not items:
        return
    total = sum(n for _, n in items)
    body = "\n\n".join(f"x{n}  {k}\n    e.g. {samples[k]}" for k, n in items[:60])
    if len(items) > 60:
        body += f"\n\n... and {len(items) - 60} more kinds"
    from routes.ml_ingest import _send_alert_email
    _, err = _send_alert_email(f"{total} failures/fallbacks ({len(items)} kinds) since the last report", body)
    if err:
        logging.getLogger(__name__).error("failure digest email failed: %s", err)


def _loop():
    while True:
        time.sleep(DIGEST_SECONDS)
        try:
            flush()
        except Exception as e:  # the digest itself must never die silently
            logging.getLogger(__name__).error("failure digest flush crashed: %s", e)


def install():
    global _installed
    if _installed:
        return
    _installed = True
    logging.getLogger().addHandler(_Collector(level=logging.WARNING))
    threading.Thread(target=_loop, daemon=True, name="failure-digest").start()
    atexit.register(flush)
