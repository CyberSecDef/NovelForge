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
    """Logging filter that attaches the correlation token to log records as structured metadata.

    Sets ``record.correlation_token`` to the current thread's correlation token
    (empty string when none is set).  The original log message is never mutated,
    so multiple handlers and formatter-based rendering both work correctly.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_token = get_correlation_token()  # type: ignore[attr-defined]
        return True


class CorrelationFormatter(logging.Formatter):
    """Formatter that renders ``correlation_token`` before the log message when present.

    Reads ``record.correlation_token`` (set by :class:`CorrelationFilter`) and
    prepends ``[token=<value>]`` to the formatted output.  The original
    ``record.msg`` is never modified; the prefix is injected only during
    formatting.
    """

    def format(self, record: logging.LogRecord) -> str:
        token = getattr(record, "correlation_token", "")
        if token:
            original_msg = record.msg
            record.msg = f"[token={token}] {record.msg}"
            try:
                result = super().format(record)
            finally:
                record.msg = original_msg
            return result
        return super().format(record)
