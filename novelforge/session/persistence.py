"""Session state schema validation and persistence for crash recovery."""

import json
import logging
import os
import re
import tempfile
import threading
import uuid
from pathlib import Path

from flask import session

import novelforge.config as config
from novelforge.progress import progress_manager

logger = logging.getLogger(__name__)


def _atomic_write(filepath: Path, content: str) -> None:
    """Write content to a file atomically using write-to-temp-then-rename.

    The temp file is created with mode 0o600 by ``tempfile.mkstemp``; the
    explicit ``os.chmod`` call below enforces this regardless of the process
    umask before the atomic rename.
    """
    fd, tmp_path = tempfile.mkstemp(
        dir=str(filepath.parent),
        prefix=f".{filepath.stem}_",
        suffix=".tmp",
    )
    try:
        os.chmod(tmp_path, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, str(filepath))
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

# ---------------------------------------------------------------------------
# Per-session persistence lock
# ---------------------------------------------------------------------------

# One lock per session ID.  Serialises concurrent read-modify-write cycles on
# the same session file so that last-writer-wins overwrites cannot silently
# drop chapters or progress data written by the other side.
_session_locks: dict[str, threading.Lock] = {}
_session_locks_lock = threading.Lock()  # protects _session_locks itself


def _get_session_lock(session_id: str) -> threading.Lock:
    """Return the per-session :class:`threading.Lock` for *session_id*.

    Creates a new lock on first access.  The registry lock ensures that two
    threads racing to create the same entry always receive the same object.
    """
    with _session_locks_lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = threading.Lock()
        return _session_locks[session_id]


# ---------------------------------------------------------------------------
# Session ID validation and safe path resolution
# ---------------------------------------------------------------------------

# Session IDs must be standard UUIDs: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
_SESSION_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def resolve_session_path(session_id: str) -> Path:
    """Validate *session_id* and return the absolute path to its JSON file.

    Session IDs must be well-formed UUIDs
    (``xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx``).  Any other value raises
    :exc:`ValueError` so callers can return a 4xx error before touching the
    filesystem.
    """
    if not isinstance(session_id, str) or not _SESSION_ID_RE.match(session_id):
        raise ValueError(f"Invalid session_id: {session_id!r}")
    return Path(config.NOVELS_DIR) / f"{session_id}.json"


# ---------------------------------------------------------------------------
# Session state schema validation
# ---------------------------------------------------------------------------

# Define expected types and defaults for all session state fields.
# Each entry: (expected_type, default_value)
_SESSION_SCHEMA: dict[str, tuple[type, object]] = {
    "session_id":               (str,   ""),
    "premise":                  (str,   ""),
    "genre":                    (str,   ""),
    "chapters":                 (int,   0),
    "word_count":               (int,   0),
    "special_events":           (str,   ""),
    "special_instructions":     (str,   ""),
    "title":                    (str,   ""),
    "chapter_list":             (list,  []),
    "character_list":           (list,  []),
    "story_architecture":       (dict,  {}),
    "master_timeline":          (dict,  {}),
    "character_fate_registry":  (dict,  {}),
    "character_arc_plan":       (dict,  {}),
    "antagonist_motivation_plan": (dict, {}),
    "technology_rules":         (dict,  {}),
    "theme_reinforcement":      (dict,  {}),
    "pov_focal_character_plan": (dict,  {}),
    "narrative_perspective":    (str,   "third_person"),
    "progress_token":           (str,   ""),
    "completed_chapters":       (list,  []),
    "illustrations":            (list,  []),
    "voice_seed":               (dict,  {}),
}


def validate_session_state(state: dict) -> dict:
    """
    Validate and coerce a session state dict against the expected schema.

    - Missing keys are filled with defaults.
    - Wrong types are coerced when possible, or replaced with defaults.
    - Extra keys are preserved (e.g. progress_data).
    - Issues are logged as warnings, never hard-fail.

    Returns the cleaned state dict.
    """
    for key, (expected_type, default) in _SESSION_SCHEMA.items():
        value = state.get(key)

        # Missing key — fill default
        if value is None:
            state[key] = default if not isinstance(default, (list, dict)) else type(default)()
            continue

        # Correct type — no action needed
        if isinstance(value, expected_type):
            continue

        # Attempt coercion
        try:
            if expected_type is int:
                state[key] = int(value)
                logger.warning(
                    "Session field '%s' coerced from %s to int",
                    key, type(value).__name__,
                )
            elif expected_type is str:
                state[key] = str(value)
                logger.warning(
                    "Session field '%s' coerced from %s to str",
                    key, type(value).__name__,
                )
            else:
                raise TypeError(f"cannot coerce {type(value).__name__} to {expected_type.__name__}")
        except (TypeError, ValueError):
            logger.warning(
                "Session field '%s' has wrong type %s (expected %s) — reset to default",
                key, type(value).__name__, expected_type.__name__,
            )
            state[key] = default if not isinstance(default, (list, dict)) else type(default)()

    # Validate chapter_list entries
    chapter_list = state.get("chapter_list", [])
    if isinstance(chapter_list, list):
        for i, ch in enumerate(chapter_list):
            if not isinstance(ch, dict):
                logger.warning("chapter_list[%d] is not a dict — replacing with empty", i)
                chapter_list[i] = {"number": i + 1, "title": f"Chapter {i + 1}", "summary": ""}

    # Validate character_list entries
    character_list = state.get("character_list", [])
    if isinstance(character_list, list):
        for i, ch in enumerate(character_list):
            if not isinstance(ch, dict):
                logger.warning("character_list[%d] is not a dict — replacing with empty", i)
                character_list[i] = {"name": "", "age": "", "role": "", "background": "", "arc": ""}

    # Validate completed_chapters entries
    completed = state.get("completed_chapters", [])
    if isinstance(completed, list):
        for i, ch in enumerate(completed):
            if not isinstance(ch, dict):
                logger.warning("completed_chapters[%d] is not a dict — removing", i)
                completed[i] = None
        state["completed_chapters"] = [ch for ch in completed if ch is not None]

    return state


# ---------------------------------------------------------------------------
# Session persistence for crash recovery
# ---------------------------------------------------------------------------

def get_session_id() -> str:
    """Get or create a unique session ID for this user session."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return str(session["session_id"])


def get_session_file_path() -> Path:
    """Get the file path for the current session's persistence data."""
    session_id = get_session_id()
    return Path(config.NOVELS_DIR) / f"{session_id}.json"


def save_session_state() -> bool:
    """
    Save current session state and generation progress to disk.
    Called after each significant step to enable crash recovery.

    The write is serialised against :func:`persist_completed_chapters` via a
    per-session lock so that concurrent background updates cannot be silently
    overwritten by a request-thread save (or vice versa).

    Returns ``True`` on success, ``False`` if the file could not be written
    due to an OS-level error (e.g. permission denied, disk full).  Programming
    errors such as missing Flask context or non-serialisable session data
    propagate immediately so callers cannot silently ignore them.
    """
    session_id = get_session_id()
    session_file = Path(config.NOVELS_DIR) / f"{session_id}.json"

    # Gather all session data
    state = {
        "session_id": session_id,
        "premise": session.get("premise", ""),
        "genre": session.get("genre", ""),
        "chapters": session.get("chapters", 0),
        "word_count": session.get("word_count", 0),
        "special_events": session.get("special_events", ""),
        "special_instructions": session.get("special_instructions", ""),
        "title": session.get("title", ""),
        "chapter_list": session.get("chapter_list", []),
        "character_list": session.get("character_list", []),
        "story_architecture": session.get("story_architecture", {}),
        "master_timeline": session.get("master_timeline", {}),
        "character_fate_registry": session.get("character_fate_registry", {}),
        "character_arc_plan": session.get("character_arc_plan", {}),
        "antagonist_motivation_plan": session.get("antagonist_motivation_plan", {}),
        "technology_rules": session.get("technology_rules", {}),
        "theme_reinforcement": session.get("theme_reinforcement", {}),
        "pov_focal_character_plan": session.get("pov_focal_character_plan", {}),
        "narrative_perspective": session.get("narrative_perspective", "third_person"),
        "progress_token": session.get("progress_token", ""),
        "completed_chapters": session.get("completed_chapters", []),
        "illustrations": session.get("illustrations", []),
        "voice_seed": session.get("voice_seed", {}),
    }

    # Add progress store data if available
    token = session.get("progress_token")
    if token:
        progress = progress_manager.get(token)
        if progress is not None:
            state["progress_data"] = progress
            # Keep completed_chapters in sync with progress data
            done = progress.get("chapters_done", [])
            if done:
                state["completed_chapters"] = list(done)

    # Validate before writing
    state = validate_session_state(state)

    try:
        # Serialise writes to this session file against concurrent background
        # updates so that a last-writer-wins overwrite cannot silently drop
        # completed chapters or progress data.
        with _get_session_lock(session_id):
            _atomic_write(session_file, json.dumps(state, indent=2))
        logger.info("Saved session state to %s", session_file)
        return True
    except OSError as e:
        logger.error(
            "Failed to save session state for %s: %s",
            session_id, e, exc_info=True,
        )
        return False


def load_session_state() -> dict | None:
    """
    Load session state from disk if it exists.
    Returns the state dict or None if no saved state exists or if the file
    cannot be read or parsed.  Only :exc:`OSError` and
    :exc:`json.JSONDecodeError` are caught; programming errors propagate.
    """
    try:
        session_file = get_session_file_path()
        if not session_file.exists():
            return None

        state = json.loads(session_file.read_text(encoding="utf-8"))
        state = validate_session_state(state)
        logger.info("Loaded session state from %s", session_file)
        return state
    except (OSError, json.JSONDecodeError) as e:
        logger.error(
            "Failed to load session state: %s",
            e, exc_info=True,
        )
        return None


def rebuild_stale_progress(
    completed_chapters: list,
    total_chapters: int,
    progress_token: str,
) -> dict:
    """Rebuild a completed-generation progress snapshot and persist it to the
    in-memory store.

    Called from the session-restore path so that recovery happens once, in a
    dedicated lifecycle hook, rather than on every GET /.

    Returns the rebuilt progress dict.
    """
    rebuilt = {
        "status": "done",
        "current": len(completed_chapters),
        "total": total_chapters or len(completed_chapters),
        "step": "Complete",
        "chapters_done": completed_chapters,
        "error": None,
    }
    progress_manager.create(progress_token, rebuilt)
    logger.info(
        "Rebuilt stale progress snapshot for token %s (%d chapters)",
        progress_token,
        len(completed_chapters),
    )
    return rebuilt


def restore_session_from_state(state: dict) -> None:
    """
    Restore session and progress store from saved state dict.
    """
    state = validate_session_state(state)

    # Clean up the old session's progress entry before loading the new state.
    # This prevents stale entries when switching sessions (token mismatch).
    old_token = session.get("progress_token", "")
    new_token = state.get("progress_token", "")
    if old_token and old_token != new_token:
        progress_manager.delete(old_token)
        logger.debug(
            "Removed stale progress entry for token %s (replaced by session load)", old_token
        )

    # Restore session variables
    session["premise"] = state.get("premise", "")
    session["genre"] = state.get("genre", "")
    session["chapters"] = state.get("chapters", 0)
    session["word_count"] = state.get("word_count", 0)
    session["special_events"] = state.get("special_events", "")
    session["special_instructions"] = state.get("special_instructions", "")
    session["title"] = state.get("title", "")
    session["chapter_list"] = state.get("chapter_list", [])
    session["character_list"] = state.get("character_list", [])
    session["story_architecture"] = state.get("story_architecture", {})
    session["master_timeline"] = state.get("master_timeline", {})
    session["character_fate_registry"] = state.get("character_fate_registry", {})
    session["character_arc_plan"] = state.get("character_arc_plan", {})
    session["antagonist_motivation_plan"] = state.get("antagonist_motivation_plan", {})
    session["technology_rules"] = state.get("technology_rules", {})
    session["theme_reinforcement"] = state.get("theme_reinforcement", {})
    session["pov_focal_character_plan"] = state.get("pov_focal_character_plan", {})
    session["narrative_perspective"] = state.get("narrative_perspective", "third_person")
    session["progress_token"] = state.get("progress_token", "")
    session["completed_chapters"] = state.get("completed_chapters", [])
    session["illustrations"] = state.get("illustrations", [])
    session["voice_seed"] = state.get("voice_seed", {})

    # Restore progress store if available, rebuilding stale snapshots in one place
    token = state.get("progress_token", "")
    pd = state.get("progress_data")
    completed = state.get("completed_chapters", [])
    if token:
        if completed and (
            not pd
            or (pd.get("status") == "running" and not pd.get("_live"))
        ):
            rebuild_stale_progress(completed, state.get("chapters", 0), token)
        elif pd:
            progress_manager.create(token, pd)

    logger.info("Restored session from saved state")


def list_session_summaries() -> list[dict]:
    """Return a list of all saved sessions that have a book title.

    Each entry contains ``"session_id"`` and ``"title"``.  Corrupt, unreadable,
    or untitled session files are logged as warnings and silently skipped so
    that one bad file never breaks the listing for valid sessions.
    """
    sessions_dir = Path(config.NOVELS_DIR)
    result = []
    for session_file in sessions_dir.glob("*.json"):
        if session_file.name.endswith("_progress.json"):
            continue
        try:
            raw = session_file.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("Skipping unreadable session file %s: %s", session_file.name, e)
            continue
        try:
            state = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning("Skipping corrupt session file %s: %s", session_file.name, e)
            continue
        try:
            state = validate_session_state(state)
        except Exception as e:
            logger.warning("Skipping invalid session file %s: %s", session_file.name, e)
            continue
        title = (state.get("title") or "").strip()
        if title:
            result.append({
                "session_id": state.get("session_id") or session_file.stem,
                "title": title,
            })
    result.sort(key=lambda s: s["title"].lower())
    return result


def load_session_by_id(session_id: str) -> dict | None:
    """Load and validate a session state by its ID.

    Raises :exc:`ValueError` if *session_id* is not a valid UUID (call site
    should return HTTP 400).  Returns ``None`` if the session file does not
    exist (call site should return HTTP 404).  Raises
    :exc:`json.JSONDecodeError` or :exc:`OSError` if the file is corrupt or
    unreadable (call site should return HTTP 500).  On success, returns the
    validated state dict ready for :func:`restore_session_from_state`.
    """
    session_file = resolve_session_path(session_id)  # raises ValueError if invalid
    # Defense-in-depth: ensure the resolved path is inside NOVELS_DIR even
    # though resolve_session_path() has already validated the UUID format.
    expected_dir = Path(config.NOVELS_DIR).resolve()
    if session_file.resolve().parent != expected_dir:
        raise ValueError(f"Session file path outside sessions directory: {session_id!r}")
    if not session_file.exists():
        return None
    state = json.loads(session_file.read_text(encoding="utf-8"))
    state = validate_session_state(state)
    logger.info("Loaded session %s from %s", session_id, session_file)
    return state


def persist_completed_chapters(
    session_id: str,
    chapters_done: list[dict],
    progress_token: str = "",
) -> bool:
    """
    Persist completed chapters and progress data to the session JSON file.

    Called from the background generation thread (no Flask request context).
    If *progress_token* is provided, the current in-memory progress snapshot
    is written into the ``progress_data`` field so that audit reports,
    status, and chapter data stay in sync on disk.

    Returns ``True`` on success, ``False`` if the session file does not exist
    or cannot be read/written.  :exc:`ValueError` from an invalid
    *session_id* propagates immediately (programming error).
    :exc:`OSError` and :exc:`json.JSONDecodeError` are caught and logged.
    """
    try:
        session_file = resolve_session_path(session_id)
        # Serialise this read-modify-write against save_session_state() so that
        # a concurrent request-thread write cannot be silently overwritten by an
        # older snapshot read by this background thread (or vice versa).
        with _get_session_lock(session_id):
            if not session_file.exists():
                return False
            state = json.loads(session_file.read_text(encoding="utf-8"))
            state["completed_chapters"] = list(chapters_done)

            if progress_token:
                progress = progress_manager.get(progress_token)
                if progress is not None:
                    state["progress_data"] = progress

            _atomic_write(session_file, json.dumps(state, indent=2))
        return True
    except (OSError, json.JSONDecodeError) as e:
        logger.error(
            "Failed to persist completed chapters for %s: %s",
            session_id, e, exc_info=True,
        )
        return False


def clear_session_state() -> bool:
    """
    Clear the current session's saved state file.

    Returns ``True`` on success (including when the file did not exist),
    ``False`` if the file could not be removed due to an OS-level error.
    """
    try:
        session_file = get_session_file_path()
        if session_file.exists():
            session_file.unlink()
            logger.info("Cleared session state file %s", session_file)
        return True
    except OSError as e:
        logger.error(
            "Failed to clear session state: %s",
            e, exc_info=True,
        )
        return False
