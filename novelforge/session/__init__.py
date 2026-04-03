"""Session persistence for crash recovery."""

from novelforge.session.persistence import (
    validate_session_state,
    get_session_id,
    get_session_file_path,
    save_session_state,
    load_session_state,
    restore_session_from_state,
    persist_completed_chapters,
    clear_session_state,
)

__all__ = [
    "validate_session_state",
    "get_session_id",
    "get_session_file_path",
    "save_session_state",
    "load_session_state",
    "restore_session_from_state",
    "persist_completed_chapters",
    "clear_session_state",
]
