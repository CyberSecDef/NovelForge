"""NovelForge – Flask application factory."""

import json
import logging
from pathlib import Path

import flask
from flask import Flask, Response, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_session import Session
from flask_wtf.csrf import CSRFProtect, generate_csrf

import novelforge.config as config
from novelforge.progress import _progress_store, _progress_lock, CorrelationFilter
from novelforge.routes import register_blueprints

# Module-level limiter, initialised without app (attached in create_app)
limiter = Limiter(get_remote_address, default_limits=["60 per minute"])


def create_app(*, testing: bool = False) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parent.parent / "templates"),
        static_folder=str(Path(__file__).resolve().parent.parent / "static"),
    )

    # Configure secret key FIRST - required for sessions
    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    # CSRF settings: 7-day token lifetime (long-running local generation process)
    app.config["WTF_CSRF_TIME_LIMIT"] = 604800  # 7 days in seconds
    app.config["WTF_CSRF_SSL_STRICT_MODE"] = False

    # Ensure directories exist BEFORE initializing sessions
    Path(config.SESSION_FILE_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.EXPORT_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.EXPORT_DIR, "illustrations").mkdir(parents=True, exist_ok=True)
    Path("./logs").mkdir(parents=True, exist_ok=True)
    Path("./sessions/novels").mkdir(parents=True, exist_ok=True)

    # Configure filesystem-based sessions
    app.config["SESSION_TYPE"] = "filesystem"
    app.config["SESSION_FILE_DIR"] = config.SESSION_FILE_DIR
    app.config["SESSION_PERMANENT"] = False

    Session(app)
    CSRFProtect(app)
    limiter.init_app(app)

    # Set up logging
    # Set up logging with correlation ID support
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] [token=%(correlation_token)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Install correlation filter on the root logger so all modules get it
    logging.getLogger().addFilter(CorrelationFilter())
    logger = logging.getLogger(__name__)

    if testing:
        app.config["TESTING"] = True

    # Validate configuration – warns in debug/testing, exits in production
    config.validate_config(debug=app.debug or testing)

    # Register route blueprints
    register_blueprints(app)

    @app.after_request
    def set_csrf_cookie(response: Response) -> Response:
        """Set CSRF token in a cookie so JS can read it after page refresh."""
        csrf_token = generate_csrf()
        response.set_cookie(
            "csrf_token",
            csrf_token,
            max_age=604800,
            samesite="Lax",
            httponly=False,
        )
        return response

    @app.route("/")
    def index() -> str:
        """Main single-page application view."""
        sess = flask.session
        session_data = None
        if sess.get("title"):
            session_data = {
                "premise": sess.get("premise", ""),
                "genre": sess.get("genre", ""),
                "chapters": sess.get("chapters", 0),
                "word_count": sess.get("word_count", 0),
                "special_events": sess.get("special_events", ""),
                "special_instructions": sess.get("special_instructions", ""),
                "title": sess.get("title", ""),
                "chapter_list": sess.get("chapter_list", []),
                "character_list": sess.get("character_list", []),
            }
            token = sess.get("progress_token", "")
            if token:
                session_data["progress_token"] = token
                with _progress_lock:
                    progress = _progress_store.get(token)
                if progress:
                    session_data["progress_data"] = progress

            completed_chapters = sess.get("completed_chapters", [])
            if not completed_chapters:
                pd = session_data.get("progress_data") or sess.get("progress_data") or {}
                completed_chapters = pd.get("chapters_done", [])
            if completed_chapters:
                session_data["completed_chapters"] = completed_chapters

            existing_pd = session_data.get("progress_data")
            if completed_chapters and (
                not existing_pd
                or (existing_pd.get("status") == "running" and not existing_pd.get("_live"))
            ):
                rebuilt = {
                    "status": "done",
                    "current": len(completed_chapters),
                    "total": sess.get("chapters", len(completed_chapters)),
                    "step": "Complete",
                    "chapters_done": completed_chapters,
                    "error": None,
                }
                session_data["progress_data"] = rebuilt
                if token:
                    with _progress_lock:
                        _progress_store[token] = rebuilt

            illustrations = sess.get("illustrations", [])
            if illustrations:
                session_data["illustrations"] = illustrations

        return render_template("index.html", session_data=session_data)

    @app.route("/llm_log")
    def get_llm_log() -> Response:
        """Return recent LLM log entries for the chat display. Debug mode only."""
        if not app.debug:
            from flask import abort
            abort(404)
        log_path = Path(__file__).resolve().parent.parent / "logs" / "llm.log"

        if not log_path.exists():
            logger.warning(f"LLM log file not found at {log_path}")
            return jsonify({"entries": []})

        try:
            entries = []
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()

            json_objects = []
            current_obj = ""
            brace_count = 0

            for line in content.split('\n'):
                if line.strip().startswith('{') and brace_count == 0:
                    if current_obj:
                        json_objects.append(current_obj)
                    current_obj = line + '\n'
                    brace_count = line.count('{') - line.count('}')
                elif brace_count > 0:
                    current_obj += line + '\n'
                    brace_count += line.count('{') - line.count('}')
                    if brace_count == 0:
                        json_objects.append(current_obj)
                        current_obj = ""

            if current_obj:
                json_objects.append(current_obj)

            for obj_str in json_objects[-10:]:
                try:
                    entry = json.loads(obj_str)
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue

            return jsonify({"entries": entries})
        except Exception as e:
            logger.error(f"Error reading LLM log: {e}")
            return jsonify({"entries": [], "error": str(e)})

    @app.route("/clear_log", methods=["POST"])
    def clear_log() -> Response | tuple[Response, int]:
        """Clear the LLM log file. Debug mode only."""
        if not app.debug:
            from flask import abort
            abort(404)
        log_path = Path(__file__).resolve().parent.parent / "logs" / "llm.log"
        try:
            log_path.write_text("", encoding="utf-8")
            logger.info("LLM log cleared by user")
            return jsonify({"status": "ok"})
        except Exception as e:
            logger.error("Failed to clear LLM log: %s", e)
            return jsonify({"error": str(e)}), 500

    return app
