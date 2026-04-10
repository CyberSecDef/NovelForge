"""Session management routes: list, load, delete, new."""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

from flask import Blueprint, Response, jsonify, request, session

import novelforge.config as config
from novelforge.progress import progress_manager
from novelforge.session.persistence import (
    get_session_file_path,
    list_session_summaries,
    load_session_by_id,
    release_session_lock,
    restore_session_from_state,
    save_session_state,
)

logger = logging.getLogger(__name__)

sessions_bp = Blueprint("sessions", __name__)


@sessions_bp.route("/list_sessions")
def list_sessions() -> Response:
    """Return a list of all saved sessions that have a book title."""
    return jsonify({"sessions": list_session_summaries()})


@sessions_bp.route("/load_session", methods=["POST"])
def load_session() -> Response | tuple[Response, int]:
    """Load a specific session by its session_id."""
    data = request.get_json(silent=True) or {}
    target_id = data.get("session_id", "").strip()
    if not target_id:
        return jsonify({"error": "session_id is required."}), 400

    try:
        state = load_session_by_id(target_id)
    except (json.JSONDecodeError, OSError) as e:
        # File-level errors (corrupt JSON or unreadable file) → 500
        logger.error("Failed to read session %s: %s", target_id, e)
        return jsonify({"error": "Failed to read session data."}), 500
    except ValueError:
        # Invalid UUID format → 400 (note: must come after JSONDecodeError
        # because JSONDecodeError is a subclass of ValueError)
        return jsonify({"error": "Invalid session_id."}), 400
    except Exception as e:
        logger.error("Failed to read session %s: %s", target_id, e)
        return jsonify({"error": "Failed to read session data."}), 500

    if state is None:
        return jsonify({"error": "Session not found."}), 404

    restore_session_from_state(state)

    return jsonify({"status": "loaded", "title": state.get("title", "")})


@sessions_bp.route("/delete_session", methods=["POST"])
def delete_session() -> Response:
    """Delete the currently active session's JSON file and clear session data."""
    session_id = session.get("session_id", "")

    try:
        session_file = get_session_file_path()
        if session_file.exists():
            session_file.unlink()
            logger.info("Deleted session file %s", session_file)
    except Exception as e:
        logger.error("Failed to delete session file: %s", e)

    token = session.get("progress_token", "")
    if token:
        progress_manager.delete(token)
        logger.debug("Removed progress entry for token %s (session deleted)", token)

    session.clear()

    # Clean up the per-session persistence lock so it doesn't leak memory
    if session_id:
        release_session_lock(session_id)

    return jsonify({"status": "success", "message": "Session deleted"})


@sessions_bp.route("/new_session", methods=["POST"])
def new_session() -> Response:
    """Archive the current LLM log file and start a new session."""
    llm_log = Path(config.LOGS_DIR) / "llm.log"
    if llm_log.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"llm_{timestamp}.log"
        archive_path = Path(config.LOGS_DIR) / archive_name
        try:
            shutil.copy2(llm_log, archive_path)
            llm_log.write_text("", encoding="utf-8")
            logger.info(f"Archived LLM log to {archive_path}")
        except Exception as e:
            logger.error(f"Failed to archive LLM log: {e}")

    token = session.get("progress_token", "")
    if token:
        progress_manager.delete(token)
        logger.debug("Removed progress entry for token %s (new session started)", token)

    session.clear()

    return jsonify({"status": "success", "message": "New session started"})


@sessions_bp.route("/save_session_state", methods=["POST"])
def save_session_state_route() -> Response | tuple[Response, int]:
    """Force a full rewrite of the current session JSON file to disk.

    Captures everything currently in flask.session and progress_manager
    (including audit reports, character relationships, illustrations, etc.)
    and writes it to ``sessions/novels/<session_id>.json`` atomically.
    """
    if not session.get("title"):
        return jsonify({"error": "No active session to save."}), 400
    try:
        ok = save_session_state()
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to save session state: %s", e, exc_info=True)
        return jsonify({"error": "Failed to save session state."}), 500
    if not ok:
        return jsonify({"error": "Failed to save session state."}), 500
    return jsonify({"status": "saved"})
