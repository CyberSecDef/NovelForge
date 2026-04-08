"""NovelForge – Flask application factory."""

import collections
import json
import logging
import sys
import threading
from pathlib import Path

import flask
from flask import Flask, Response, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_session import Session
from flask_wtf.csrf import CSRFProtect, generate_csrf

import novelforge.config as config
from novelforge.progress import progress_manager, CorrelationFilter, CorrelationFormatter
from novelforge.routes import register_blueprints

# Module-level limiter, initialised without app (attached in create_app)
limiter = Limiter(get_remote_address, default_limits=["60 per minute"])

# Sentinel flag: ensures _bootstrap_logging() is fully idempotent across
# repeated create_app() calls (e.g. in tests or multi-app scenarios).
_logging_bootstrapped: bool = False
_logging_bootstrap_lock = threading.Lock()


def _bootstrap_logging() -> None:
    """Configure root logging once; subsequent calls are safe no-ops.

    Configures the root logger directly rather than relying on
    ``logging.basicConfig()``, which becomes a no-op once any handler has
    already been attached to the root logger (regardless of whether *our*
    handler was the one that did it).  Attaches ``CorrelationFilter`` to the
    root logger so background-thread messages are tagged with a correlation
    token.

    This function is thread-safe: a lock ensures that concurrent
    ``create_app()`` calls cannot both execute the setup body simultaneously.
    """
    global _logging_bootstrapped
    # Fast path: no locking overhead after first successful bootstrap.
    if _logging_bootstrapped:
        return
    with _logging_bootstrap_lock:
        # Re-check inside the lock to handle concurrent first callers.
        if _logging_bootstrapped:
            return

        root_logger = logging.getLogger()

        # Set level unconditionally – idempotent and cheap.
        root_logger.setLevel(logging.INFO)

        # Only add our StreamHandler when the root logger has no handlers yet,
        # to avoid duplicating output when a test framework (or the host process)
        # already installed its own handler.
        if not root_logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            handler.setFormatter(
                CorrelationFormatter(
                    fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            root_logger.addHandler(handler)

        # Attach the correlation filter exactly once.
        if not any(isinstance(f, CorrelationFilter) for f in root_logger.filters):
            root_logger.addFilter(CorrelationFilter())

        _logging_bootstrapped = True


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

    # Ensure all required directories exist and are writeable BEFORE initializing
    # sessions or attaching any logging file handlers.
    config.ensure_app_dirs()

    # Set up the dedicated LLM file logger now that the log directory exists.
    # setup_llm_logger() is idempotent: repeated create_app() calls are safe.
    from novelforge.llm.client import setup_llm_logger
    setup_llm_logger()

    # Configure filesystem-based sessions using CacheLib directly
    # (avoids deprecated SESSION_FILE_DIR / FileSystemSessionInterface)
    from cachelib import FileSystemCache
    app.config["SESSION_TYPE"] = "cachelib"
    app.config["SESSION_CACHELIB"] = FileSystemCache(
        cache_dir=config.SESSION_FILE_DIR,
        threshold=500,
    )
    app.config["SESSION_PERMANENT"] = False

    Session(app)
    CSRFProtect(app)
    limiter.init_app(app)

    # Bootstrap logging once, idempotently (safe for repeated create_app calls).
    _bootstrap_logging()
    logger = logging.getLogger(__name__)

    if testing:
        app.config["TESTING"] = True

    # Validate configuration – warns in debug/testing, exits in production
    try:
        config.validate_config(debug=app.debug or testing)
    except config.ConfigurationError as exc:
        # Errors were already logged individually inside validate_config();
        # re-log the summary here so the startup failure is visible in one line.
        logging.getLogger(__name__).critical("Startup aborted: %s", exc)
        sys.exit(1)

    # Register route blueprints
    register_blueprints(app)

    @app.after_request
    def set_security_headers(response: Response) -> Response:
        """Set CSRF cookie and standard security headers on every response."""
        # --- CSRF token cookie (JS-readable for SPA AJAX calls) ---
        csrf_token = generate_csrf()
        response.set_cookie(
            "csrf_token",
            csrf_token,
            max_age=604800,
            samesite="Lax",
            httponly=False,
        )

        # --- Security headers ---
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        # Content-Security-Policy: allowlist the CDN sources used in index.html.
        # 'unsafe-inline' is required for the session-data <script> block and
        # the mermaid.initialize() call, plus inline style="" attributes.
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://code.jquery.com; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "font-src 'self' https://cdn.jsdelivr.net; "
            "img-src 'self' data:; "
            "connect-src 'self' https://cdn.jsdelivr.net"
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
                "narrative_perspective": sess.get("narrative_perspective", "third_person"),
            }
            token = sess.get("progress_token", "")
            if token:
                session_data["progress_token"] = token
                progress = progress_manager.get(token)
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
                # Progress-store recovery is handled in restore_session_from_state(),
                # not here, so that GET / remains free of write side-effects.

            illustrations = sess.get("illustrations", [])
            if illustrations:
                session_data["illustrations"] = illustrations

        from novelforge.validation import ALLOWED_GENRES
        return render_template(
            "index.html",
            session_data=session_data,
            allowed_genres=sorted(ALLOWED_GENRES),
        )

    @app.route("/llm_log")
    def get_llm_log() -> Response:
        """Return recent LLM log entries for the chat display."""
        log_path = Path(config.LOGS_DIR) / "llm.log"

        if not log_path.exists():
            logger.warning(f"LLM log file not found at {log_path}")
            return jsonify({"entries": []})

        try:
            window: collections.deque = collections.deque(maxlen=10)
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue

                    try:
                        window.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        # llm.log is treated as JSONL: one complete JSON object per
                        # non-empty line. Skip malformed/incomplete lines rather than
                        # trying to resynchronise based on "{" prefixes, which can
                        # corrupt pretty-printed nested JSON content.
                        continue
            return jsonify({"entries": list(window)})
        except Exception as e:
            logger.error(f"Error reading LLM log: {e}")
            return jsonify({"entries": [], "error": str(e)})

    @app.route("/clear_log", methods=["POST"])
    def clear_log() -> Response | tuple[Response, int]:
        """Clear the LLM log file."""
        log_path = Path(config.LOGS_DIR) / "llm.log"
        try:
            log_path.write_text("", encoding="utf-8")
            logger.info("LLM log cleared by user")
            return jsonify({"status": "ok"})
        except Exception as e:
            logger.error("Failed to clear LLM log: %s", e)
            return jsonify({"error": str(e)}), 500

    return app
