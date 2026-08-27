"""
Retry-with-backoff for transient infrastructure failures (rate limits,
timeouts), deliberately NOT for genuine model behavior failures like
GraphRecursionError, an agent that loops too long is a real finding about
that provider, not something to paper over with a retry.
"""
import time

# Substrings that indicate a transient, worth-retrying failure. Deliberately
# narrow: anything not matching here (GraphRecursionError, a genuine 400
# validation error, etc.) fails immediately and gets recorded as real data,
# not silently retried into a different outcome.
_RETRYABLE_MARKERS = (
    "RateLimitError",
    "rate_limit",
    "429",
    "Too Many Requests",
    "APITimeoutError",
    "ConnectTimeout",
    "ReadTimeout",
    "ServiceUnavailable",
    "503",
)


def is_retryable(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def retry_with_backoff(fn, max_retries=5, base_delay=3, on_retry=None):
    """
    Calls fn() and returns its result. On a retryable exception, waits with
    exponential backoff (base_delay * 2^attempt, so 3s, 6s, 12s, 24s, 48s by
    default) and tries again, up to max_retries times. A non-retryable
    exception, or exhausting all retries, re-raises immediately, the caller's
    existing try/except still catches it and records a real failure.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if not is_retryable(e) or attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            if on_retry:
                on_retry(attempt + 1, max_retries, delay, e)
            time.sleep(delay)
    raise last_exc  # unreachable, but keeps type checkers happy
