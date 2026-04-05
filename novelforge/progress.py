"""In-memory progress store for chapter generation, shared across modules."""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from typing import TypedDict


class ProgressState(TypedDict, total=False):
    """Typed schema for a single progress entry.

    All keys are optional so that callers can supply only the fields they
    care about when constructing patches.  The six fields below are
    required when creating a brand-new entry via :meth:`ProgressManager.create`.
    """

    # Core lifecycle fields (required at creation)
    status: str                      # "running" | "done" | "error"
    current: int                     # chapters completed so far
    total: int                       # total chapters to generate
    step: str                        # human-readable description of current step
    chapters_done: list[dict[str, Any]]  # completed chapter payloads
    error: Optional[str]             # error message, or None

    # Supplementary lifecycle fields
    error_code: str                  # machine-readable error code
    _live: bool                      # True while a generation thread is active
    snapshot: dict[str, Any]        # session snapshot captured at generation start

    # Per-chapter accumulation
    character_state_log: list[str]

    # Degraded-pass tracking: list of {pass_name, chapter_num, failure_summary} dicts
    # populated whenever an optional chapter-agent pass fails but execution continues.
    degraded_passes: list[dict[str, Any]]

    # Post-generation quality-audit reports.
    # These may be set to None to signal that the report is stale and must be
    # regenerated (e.g. after a chapter revision invalidates old analysis).
    consistency: dict[str, Any] | None
    global_continuity_audit: dict[str, Any] | None
    narrative_compression_report: dict[str, Any] | None
    character_resolution_report: dict[str, Any] | None
    thematic_payoff_report: dict[str, Any] | None
    climax_integrity_report: dict[str, Any] | None
    loose_thread_report: dict[str, Any] | None
    reader_immersion_report: dict[str, Any] | None
    pacing_heatmap: dict[str, Any] | None
    character_relationship_map: dict[str, Any] | None

    # Illustration results
    illustrations: list[dict[str, Any]]

    # Token linking a novel entry to its illustration job
    illustration_token: str


_REQUIRED_CREATION_KEYS: frozenset[str] = frozenset(
    {"status", "current", "total", "step", "chapters_done", "error"}
)
_VALID_STATUSES: frozenset[str] = frozenset({"running", "done", "error"})


class ProgressManager:
    """Thread-safe, typed repository for in-progress generation state.

    All public methods acquire an internal lock, so callers never need to
    manage synchronisation themselves.

    Typical lifecycle::

        progress_manager.create(token, {
            "status": "running", "current": 0, "total": 10,
            "step": "Preparing…", "chapters_done": [], "error": None,
        })

        progress_manager.update(token, {"current": 1, "step": "Chapter 1: complete"})

        data = progress_manager.get(token)   # returns a shallow copy

        progress_manager.delete(token)
    """

    def __init__(self) -> None:
        self._store: dict[str, ProgressState] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core CRUD operations
    # ------------------------------------------------------------------

    def create(self, token: str, initial_state: dict[str, Any]) -> None:
        """Create (or overwrite) a progress entry for *token*.

        Raises :exc:`ValueError` if *status* is absent or not one of the
        recognised values.
        """
        status = initial_state.get("status", "")
        if status not in _VALID_STATUSES:
            raise ValueError(
                f"Invalid status {status!r}; must be one of {sorted(_VALID_STATUSES)}"
            )
        with self._lock:
            self._store[token] = dict(initial_state)  # type: ignore[assignment]

    def get(self, token: str) -> ProgressState | None:
        """Return a shallow copy of the progress state, or *None* if unknown.

        .. warning::

           Nested structures (lists, dicts) are shared references.  Callers
           that need to **mutate** nested data (e.g. updating a chapter inside
           ``chapters_done``) must use :meth:`get_deep_copy` instead, or the
           internal store will be corrupted.
        """
        with self._lock:
            data = self._store.get(token)
            return dict(data) if data is not None else None  # type: ignore[return-value]

    def get_deep_copy(self, token: str) -> ProgressState | None:
        """Return a deep copy of the progress state, or *None* if unknown.

        Use this when the caller needs to mutate nested structures (e.g.
        modifying chapter content during revision) without affecting the
        internal store.
        """
        import copy
        with self._lock:
            data = self._store.get(token)
            return copy.deepcopy(data) if data is not None else None  # type: ignore[return-value]

    def update(self, token: str, patch: dict[str, Any]) -> None:
        """Atomically apply *patch* fields to an existing entry.

        Raises :exc:`KeyError` if *token* is not found.
        Raises :exc:`ValueError` if *patch* contains an invalid ``status``.
        """
        if "status" in patch and patch["status"] not in _VALID_STATUSES:
            raise ValueError(
                f"Invalid status {patch['status']!r}; must be one of {sorted(_VALID_STATUSES)}"
            )
        with self._lock:
            if token not in self._store:
                raise KeyError(f"Unknown progress token: {token!r}")
            self._store[token].update(patch)  # type: ignore[typeddict-item]

    def delete(self, token: str) -> None:
        """Remove a progress entry; no-op if *token* is not found."""
        with self._lock:
            self._store.pop(token, None)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def list_active(self) -> list[str]:
        """Return tokens for all entries whose *status* is ``"running"``."""
        with self._lock:
            return [t for t, s in self._store.items() if s.get("status") == "running"]

    def keys(self) -> list[str]:
        """Return a snapshot of all current token keys."""
        with self._lock:
            return list(self._store.keys())

    def snapshot(self) -> dict[str, ProgressState]:
        """Return shallow copies of every entry, keyed by token.

        Primarily intended for diagnostics and test assertions.
        """
        with self._lock:
            return {k: dict(v) for k, v in self._store.items()}  # type: ignore[misc]

    def clear(self) -> None:
        """Remove all entries (primarily for test teardown)."""
        with self._lock:
            self._store.clear()


#: Module-level singleton — import and use this object instead of the
#: former ``_progress_store`` / ``_progress_lock`` globals.
progress_manager: ProgressManager = ProgressManager()

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
