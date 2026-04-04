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
    get_session_file_path, restore_session_from_state, resolve_session_path,
)

logger = logging.getLogger(__name__)

sessions_bp = Blueprint("sessions", __name__)


@sessions_bp.route("/list_sessions")
def list_sessions() -> Response:
    """Return a list of all saved sessions that have a book title."""
    sessions_dir = Path(config.NOVELS_DIR)
    result = []
    for session_file in sessions_dir.glob("*.json"):
        if session_file.name.endswith("_progress.json"):
            continue
        try:
            state = json.loads(session_file.read_text(encoding="utf-8"))
            title = (state.get("title") or "").strip()
            if title:
                result.append({
                    "session_id": state.get("session_id", session_file.stem),
                    "title": title,
                })
        except Exception:
            continue
    result.sort(key=lambda s: s["title"].lower())
    return jsonify({"sessions": result})


@sessions_bp.route("/load_session", methods=["POST"])
def load_session() -> Response | tuple[Response, int]:
    """Load a specific session by its session_id."""
    data = request.get_json(silent=True) or {}
    target_id = data.get("session_id", "").strip()
    if not target_id:
        return jsonify({"error": "session_id is required."}), 400

    try:
        session_file = resolve_session_path(target_id)
    except ValueError:
        return jsonify({"error": "Invalid session_id."}), 400

    if not session_file.exists():
        return jsonify({"error": "Session not found."}), 404

    try:
        state = json.loads(session_file.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to read session file {session_file}: {e}")
        return jsonify({"error": "Failed to read session data."}), 500

    restore_session_from_state(state)

    token = state.get("progress_token")
    if token and "progress_data" in state:
        progress_manager.create(token, state["progress_data"])

    return jsonify({"status": "loaded", "title": state.get("title", "")})


@sessions_bp.route("/delete_session", methods=["POST"])
def delete_session() -> Response:
    """Delete the currently active session's JSON file and clear session data."""
    try:
        session_file = get_session_file_path()
        if session_file.exists():
            session_file.unlink()
            logger.info(f"Deleted session file {session_file}")
    except Exception as e:
        logger.error(f"Failed to delete session file: {e}")

    session.clear()
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

    session.clear()

    return jsonify({"status": "success", "message": "New session started"})
