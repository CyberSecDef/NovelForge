"""Session state schema validation and persistence for crash recovery."""

import json
import logging
import os
import tempfile
import uuid
from pathlib import Path

from flask import session

import novelforge.config as config
from novelforge.progress import _progress_store, _progress_lock

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


def save_session_state() -> None:
    """
    Save current session state and generation progress to disk.
    Called after each significant step to enable crash recovery.
    """
    try:
        session_file = get_session_file_path()

        # Gather all session data
        state = {
            "session_id": get_session_id(),
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
            "progress_token": session.get("progress_token", ""),
            "completed_chapters": session.get("completed_chapters", []),
            "illustrations": session.get("illustrations", []),
            "voice_seed": session.get("voice_seed", {}),
        }

        # Add progress store data if available
        token = session.get("progress_token")
        if token:
            with _progress_lock:
                if token in _progress_store:
                    state["progress_data"] = _progress_store[token]
                    # Keep completed_chapters in sync with progress data
                    done = _progress_store[token].get("chapters_done", [])
                    if done:
                        state["completed_chapters"] = list(done)

        # Validate before writing
        state = validate_session_state(state)

        # Write atomically (temp file + rename) to prevent corruption on crash
        _atomic_write(session_file, json.dumps(state, indent=2))
        logger.info(f"Saved session state to {session_file}")
    except Exception as e:
        logger.error(f"Failed to save session state: {e}")


def load_session_state() -> dict | None:
    """
    Load session state from disk if it exists.
    Returns the state dict or None if no saved state exists.
    """
    try:
        session_file = get_session_file_path()
        if not session_file.exists():
            return None

        state = json.loads(session_file.read_text(encoding="utf-8"))
        state = validate_session_state(state)
        logger.info(f"Loaded session state from {session_file}")
        return state
    except Exception as e:
        logger.error(f"Failed to load session state: {e}")
        return None


def restore_session_from_state(state: dict) -> None:
    """
    Restore session and progress store from saved state dict.
    """
    state = validate_session_state(state)

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
    session["progress_token"] = state.get("progress_token", "")
    session["completed_chapters"] = state.get("completed_chapters", [])
    session["illustrations"] = state.get("illustrations", [])
    session["voice_seed"] = state.get("voice_seed", {})

    # Restore progress store if available
    if "progress_data" in state and state.get("progress_token"):
        token = state["progress_token"]
        with _progress_lock:
            _progress_store[token] = state["progress_data"]

    logger.info("Restored session from saved state")


def _persist_completed_chapters(
    session_id: str,
    chapters_done: list[dict],
    progress_token: str = "",
) -> None:
    """
    Persist completed chapters and progress data to the session JSON file.

    Called from the background generation thread (no Flask request context).
    If *progress_token* is provided, the current in-memory progress snapshot
    is written into the ``progress_data`` field so that audit reports,
    status, and chapter data stay in sync on disk.
    """
    try:
        session_file = Path(config.NOVELS_DIR) / f"{session_id}.json"
        if not session_file.exists():
            return
        state = json.loads(session_file.read_text(encoding="utf-8"))
        state["completed_chapters"] = list(chapters_done)

        if progress_token:
            with _progress_lock:
                progress = _progress_store.get(progress_token)
                if progress is not None:
                    state["progress_data"] = dict(progress)

        _atomic_write(session_file, json.dumps(state, indent=2))
    except Exception as e:
        logger.error(f"Failed to persist completed chapters: {e}")


def clear_session_state() -> None:
    """
    Clear the current session's saved state file.
    """
    try:
        session_file = get_session_file_path()
        if session_file.exists():
            session_file.unlink()
            logger.info(f"Cleared session state file {session_file}")
    except Exception as e:
        logger.error(f"Failed to clear session state: {e}")
