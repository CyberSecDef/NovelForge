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
    """Logging filter that injects ``correlation_token`` into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_token = get_correlation_token()  # type: ignore[attr-defined]
        return True
