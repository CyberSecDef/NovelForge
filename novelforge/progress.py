"""In-memory progress store for chapter generation, shared across modules."""

import logging
import threading

_progress_store: dict[str, dict] = {}
_progress_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Correlation ID for background generation threads
# ---------------------------------------------------------------------------

_correlation = threading.local()


def set_correlation_token(token: str) -> None:
    """Set the correlation token for the current thread."""
    _correlation.token = token


def get_correlation_token() -> str:
    """Return the correlation token for the current thread, or empty string."""
    return getattr(_correlation, "token", "")


def clear_correlation_token() -> None:
    """Clear the correlation token for the current thread."""
    _correlation.token = ""


class CorrelationFilter(logging.Filter):
    """Logging filter that prepends the correlation token to log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        token = get_correlation_token()
        if token:
            record.msg = f"[token={token}] {record.msg}"
        return True
