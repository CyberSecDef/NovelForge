"""
NovelForge – Flask backend for AI-powered novel generation.

Run with:
    python app.py

Requires environment variables (see .env.example / config.py):
    LLM_API_URL  – LLM API endpoint
    LLM_API_KEY  – API key for the LLM provider
    SECRET_KEY   – Flask secret key (change in production)
"""

import os
import json
import time
import logging
import threading
import uuid
from pathlib import Path

import requests
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    send_file,
    abort,
)
from flask_session import Session
import markupsafe

import config

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# Use cachelib-based filesystem sessions (avoids deprecated flask-session keys)
from cachelib.file import FileSystemCache  # noqa: E402

_session_cache = FileSystemCache(config.SESSION_FILE_DIR, threshold=500, mode=0o600)
app.config["SESSION_TYPE"] = "cachelib"
app.config["SESSION_CACHELIB"] = _session_cache
app.config["SESSION_PERMANENT"] = False

Session(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure directories exist
Path(config.SESSION_FILE_DIR).mkdir(parents=True, exist_ok=True)
Path(config.EXPORT_DIR).mkdir(parents=True, exist_ok=True)

# In-memory store for chapter-generation progress keyed by session token
_progress_store: dict[str, dict] = {}
_progress_lock = threading.Lock()

# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

# Words indicative of LLM-generated text that the anti-LLM agent should remove
_FORBIDDEN_WORDS = [
    "embark", "delve", "realm", "tapestry", "testament", "nuance",
    "beacon", "uncharted", "multifaceted", "leverage", "synergy",
    "pivotal", "groundbreaking", "commendable", "meticulous",
]

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def call_llm(messages: list[dict], *, json_mode: bool = False) -> str:
    """
    Call the configured LLM API and return the assistant message content.

    Retries up to MAX_RETRIES times on transient errors (429, 5xx).
    Raises RuntimeError on persistent failure.
    """
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: dict = {
        "model": config.LLM_MODEL,
        "messages": messages,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                config.LLM_API_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = RETRY_DELAY * attempt
                logger.warning(
                    "LLM API returned %s – retry %d/%d in %ds",
                    resp.status_code, attempt, MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            logger.warning("LLM request timed out (attempt %d/%d)", attempt, MAX_RETRIES)
            if attempt == MAX_RETRIES:
                raise RuntimeError("LLM API timed out after multiple retries.")
            time.sleep(RETRY_DELAY * attempt)
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"LLM API request failed: {exc}") from exc

    raise RuntimeError("LLM API failed after maximum retries.")


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_system_prompt(role: str) -> dict:
    return {"role": "system", "content": role}


def build_title_prompt(premise: str, genre: str) -> list[dict]:
    return [
        _build_system_prompt(
            "You are a bestselling author skilled at crafting memorable book titles."
        ),
        {
            "role": "user",
            "content": (
                f"Generate a single catchy, original title for a {genre} novel based on this premise:\n\n"
                f"{premise}\n\n"
                "Return ONLY the title text, nothing else."
            ),
        },
    ]


def build_outline_prompt(
    premise: str,
    genre: str,
    chapters: int,
    word_count: int,
    special_events: str,
    special_instructions: str,
) -> list[dict]:
    arch = (
        "Novel architecture:\n"
        "1. Hook (opening tension/question)\n"
        "2. Setup (protagonist, world, tone, conflict)\n"
        "3. Inciting incident (disruptive event)\n"
        "4. Rising action (escalating obstacles)\n"
        "5. Midpoint shift (major revelation)\n"
        "6. Complications (worsening problems)\n"
        "7. Crisis (hardest decision)\n"
        "8. Climax (decisive confrontation)\n"
        "9. Resolution (aftermath, character changes)\n\n"
        "Structure: Beginning 25% (Hook→Setup→Inciting), "
        "Middle 50% (Rising→Midpoint→Complications), "
        "End 25% (Crisis→Climax→Resolution).\n"
        "Scene pattern: Goal → Obstacle → Outcome → New problem."
    )
    events_section = (
        f"\nSpecial events to incorporate:\n{special_events}" if special_events.strip() else ""
    )
    instructions_section = (
        f"\nSpecial instructions:\n{special_instructions}" if special_instructions.strip() else ""
    )
    return [
        _build_system_prompt(
            "You are a master story architect who creates detailed, compelling novel outlines."
        ),
        {
            "role": "user",
            "content": (
                f"Create a detailed chapter-by-chapter outline for a {genre} novel.\n\n"
                f"Premise: {premise}\n"
                f"Total chapters: {chapters}\n"
                f"Target word count: {word_count:,}\n"
                f"{arch}"
                f"{events_section}"
                f"{instructions_section}\n\n"
                "Return a JSON object with this structure:\n"
                '{"chapters": [{"number": 1, "title": "...", "summary": "..."}, ...]}\n'
                "Each chapter summary should be 2-4 sentences describing key events and purpose."
            ),
        },
    ]


def build_characters_prompt(
    premise: str, genre: str, outline_text: str
) -> list[dict]:
    return [
        _build_system_prompt(
            "You are a character development expert who creates vivid, memorable fictional characters."
        ),
        {
            "role": "user",
            "content": (
                f"Based on this {genre} novel premise and outline, create 3-7 main characters.\n\n"
                f"Premise: {premise}\n\n"
                f"Outline:\n{outline_text}\n\n"
                "Return a JSON object with this structure:\n"
                '{"characters": [{"name": "...", "age": "...", "background": "...", '
                '"role": "...", "arc": "..."}, ...]}'
            ),
        },
    ]


def build_chapter_draft_prompt(
    premise: str,
    genre: str,
    title: str,
    chapter_num: int,
    chapter_title: str,
    chapter_summary: str,
    characters_text: str,
    previous_summaries: str,
    target_words: int,
    special_instructions: str,
) -> list[dict]:
    instructions_section = (
        f"\nSpecial instructions: {special_instructions}" if special_instructions.strip() else ""
    )
    prev_section = (
        f"\nPrevious chapter summaries (for continuity):\n{previous_summaries}"
        if previous_summaries.strip()
        else ""
    )
    return [
        _build_system_prompt(
            "You are a skilled novelist writing in a natural, human voice. "
            "Vary sentence length and structure. Use vivid, specific detail. "
            "Avoid overused phrases, clichés, and robotic transitions. "
            "Do NOT use the words: " + ", ".join(_FORBIDDEN_WORDS) + "."
        ),
        {
            "role": "user",
            "content": (
                f"Write Chapter {chapter_num}: {chapter_title} of '{title}'.\n\n"
                f"Genre: {genre}\n"
                f"Premise: {premise}\n"
                f"Chapter summary: {chapter_summary}\n\n"
                f"Main characters:\n{characters_text}\n"
                f"{prev_section}"
                f"{instructions_section}\n\n"
                f"Target: approximately {target_words:,} words. "
                "Write immersive, human-sounding prose. "
                "Follow the scene pattern: Goal → Obstacle → Outcome → New problem."
            ),
        },
    ]


def build_editing_agent_prompt(chapter_text: str, chapter_summary: str) -> list[dict]:
    return [
        _build_system_prompt(
            "You are a professional fiction editor specialising in plot, pacing, and "
            "character consistency. Your job is to refine, not rewrite wholesale."
        ),
        {
            "role": "user",
            "content": (
                f"Chapter summary (what should happen):\n{chapter_summary}\n\n"
                f"Chapter draft:\n{chapter_text}\n\n"
                "Identify and fix: plot holes, pacing issues, character inconsistencies, "
                "unclear motivations. Return the improved chapter text only."
            ),
        },
    ]


def build_polish_agent_prompt(chapter_text: str) -> list[dict]:
    forbidden = ", ".join(_FORBIDDEN_WORDS)
    return [
        _build_system_prompt(
            f"You are a literary polisher. Elevate grammar, style, and language quality. "
            f"Ensure varied sentence structure and vivid prose. "
            f"Remove any of these overused words: {forbidden}. "
            "Return only the polished chapter text."
        ),
        {
            "role": "user",
            "content": chapter_text,
        },
    ]


def build_chapter_summary_prompt(chapter_text: str) -> list[dict]:
    return [
        _build_system_prompt("You are a precise summariser of fiction."),
        {
            "role": "user",
            "content": (
                f"Write a 100-200 word summary of this chapter for continuity tracking:\n\n{chapter_text}"
            ),
        },
    ]


def build_consistency_pass_prompt(
    title: str, all_summaries: list[str], special_instructions: str
) -> list[dict]:
    summaries_text = "\n\n".join(
        f"Chapter {i+1}:\n{s}" for i, s in enumerate(all_summaries)
    )
    return [
        _build_system_prompt(
            "You are a senior editor reviewing the full arc of a novel for consistency."
        ),
        {
            "role": "user",
            "content": (
                f"Novel title: {title}\n\n"
                f"Chapter summaries:\n{summaries_text}\n\n"
                "Review for: plot holes, character arc completion, unresolved threads, "
                "world-building inconsistencies, thematic payoff.\n"
                f"Special instructions: {special_instructions}\n\n"
                "Return a JSON object: "
                '{"issues": ["..."], "overall_assessment": "..."}'
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

ALLOWED_GENRES = {"Fantasy", "Sci-Fi", "Mystery", "Romance", "Horror", "Thriller", "Historical"}


def validate_outline_input(data: dict) -> tuple[bool, str]:
    """Validate the /generate_outline form data. Returns (ok, error_message)."""
    premise = data.get("premise", "").strip()
    if not premise:
        return False, "Story premise is required."
    if len(premise) > 1200:
        return False, "Story premise must be 1200 characters or fewer."

    genre = data.get("genre", "").strip()
    if genre not in ALLOWED_GENRES:
        return False, f"Invalid genre. Choose from: {', '.join(sorted(ALLOWED_GENRES))}."

    try:
        chapters = int(data.get("chapters", 0))
        if chapters < 3:
            return False, "Number of chapters must be at least 3."
    except (ValueError, TypeError):
        return False, "Chapters must be a valid number."

    try:
        word_count = int(data.get("word_count", 0))
        if word_count < 1000:
            return False, "Word count must be at least 1000."
    except (ValueError, TypeError):
        return False, "Word count must be a valid number."

    return True, ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Main single-page application view."""
    return render_template("index.html")


@app.route("/generate_outline", methods=["POST"])
def generate_outline():
    """
    Phase 1: Generate title, chapter outline, and main characters.

    Expects JSON body with: premise, genre, chapters, word_count,
    special_events, special_instructions.
    """
    data = request.get_json(silent=True) or {}

    ok, err = validate_outline_input(data)
    if not ok:
        return jsonify({"error": err}), 400

    premise = markupsafe.escape(data["premise"].strip())
    genre = data["genre"].strip()
    chapters = int(data["chapters"])
    word_count = int(data["word_count"])
    special_events = data.get("special_events", "").strip()
    special_instructions = data.get("special_instructions", "").strip()

    # Store sanitised inputs in session for later phases
    session["premise"] = str(premise)
    session["genre"] = genre
    session["chapters"] = chapters
    session["word_count"] = word_count
    session["special_events"] = special_events
    session["special_instructions"] = special_instructions

    try:
        # 1. Generate title
        title = call_llm(build_title_prompt(str(premise), genre)).strip().strip('"')

        # 2. Generate outline
        outline_raw = call_llm(
            build_outline_prompt(
                str(premise), genre, chapters, word_count, special_events, special_instructions
            ),
            json_mode=True,
        )
        try:
            outline_data = json.loads(outline_raw)
            chapter_list = outline_data.get("chapters", [])
        except json.JSONDecodeError:
            # Fallback: wrap in minimal structure
            chapter_list = [
                {"number": i + 1, "title": f"Chapter {i+1}", "summary": ""}
                for i in range(chapters)
            ]

        outline_text = "\n".join(
            f"Chapter {c['number']}: {c['title']} – {c['summary']}"
            for c in chapter_list
        )

        # 3. Generate characters
        characters_raw = call_llm(
            build_characters_prompt(str(premise), genre, outline_text),
            json_mode=True,
        )
        try:
            characters_data = json.loads(characters_raw)
            character_list = characters_data.get("characters", [])
        except json.JSONDecodeError:
            character_list = []

        # Store outline data in session
        session["title"] = title
        session["chapter_list"] = chapter_list
        session["character_list"] = character_list

        return jsonify(
            {
                "title": title,
                "chapters": chapter_list,
                "characters": character_list,
            }
        )

    except RuntimeError as exc:
        logger.error("Outline generation failed: %s", exc)
        return jsonify({"error": str(exc)}), 502


@app.route("/approve_outline", methods=["POST"])
def approve_outline():
    """
    Save user-edited outline and characters back to the session.
    Expects JSON with: title, chapters (list), characters (list).
    """
    data = request.get_json(silent=True) or {}

    title = data.get("title", "").strip()
    if not title:
        return jsonify({"error": "Title is required."}), 400

    chapter_list = data.get("chapters", [])
    if not isinstance(chapter_list, list) or len(chapter_list) == 0:
        return jsonify({"error": "Chapter list is required."}), 400

    character_list = data.get("characters", [])

    # Sanitise string fields to prevent XSS leaking into stored session data
    def sanitise_str(v):
        return str(markupsafe.escape(v)) if isinstance(v, str) else v

    session["title"] = sanitise_str(title)
    session["chapter_list"] = [
        {k: sanitise_str(v) for k, v in ch.items()} for ch in chapter_list
    ]
    session["character_list"] = [
        {k: sanitise_str(v) for k, v in ch.items()} for ch in character_list
    ]

    return jsonify({"status": "approved"})


@app.route("/generate_chapters", methods=["POST"])
def generate_chapters():
    """
    Phase 2: Start async chapter generation.
    Returns a progress token to poll via /progress/<token>.
    """
    # Verify required session data exists
    for key in ("premise", "genre", "chapters", "word_count", "title", "chapter_list"):
        if key not in session:
            return jsonify({"error": "Session data missing. Please start over."}), 400

    token = str(uuid.uuid4())
    with _progress_lock:
        _progress_store[token] = {
            "status": "running",
            "current": 0,
            "total": session["chapters"],
            "chapters_done": [],
            "error": None,
        }

    # Snapshot session data for the background thread
    snapshot = {
        "premise": session["premise"],
        "genre": session["genre"],
        "chapters": session["chapters"],
        "word_count": session["word_count"],
        "special_instructions": session.get("special_instructions", ""),
        "title": session["title"],
        "chapter_list": session["chapter_list"],
        "character_list": session.get("character_list", []),
    }

    thread = threading.Thread(
        target=_run_chapter_generation,
        args=(token, snapshot),
        daemon=True,
    )
    thread.start()

    session["progress_token"] = token
    return jsonify({"token": token})


def _run_chapter_generation(token: str, snap: dict) -> None:
    """Background worker: generate all chapters sequentially."""
    premise = snap["premise"]
    genre = snap["genre"]
    total_chapters = snap["chapters"]
    word_count = snap["word_count"]
    special_instructions = snap["special_instructions"]
    title = snap["title"]
    chapter_list = snap["chapter_list"]
    character_list = snap["character_list"]

    target_per_chapter = max(500, word_count // total_chapters)
    characters_text = _format_characters(character_list)

    chapters_done: list[dict] = []
    summaries: list[str] = []

    try:
        for idx, ch in enumerate(chapter_list):
            chapter_num = ch.get("number", idx + 1)
            chapter_title = ch.get("title", f"Chapter {chapter_num}")
            chapter_outline_summary = ch.get("summary", "")

            previous_summaries = "\n\n".join(
                f"Chapter {i+1}: {s}" for i, s in enumerate(summaries)
            )

            # --- Draft ---
            draft = call_llm(
                build_chapter_draft_prompt(
                    premise, genre, title, chapter_num, chapter_title,
                    chapter_outline_summary, characters_text,
                    previous_summaries, target_per_chapter, special_instructions,
                )
            )

            # --- Editing agent ---
            edited = call_llm(
                build_editing_agent_prompt(draft, chapter_outline_summary)
            )

            # --- Polish agent ---
            polished = call_llm(build_polish_agent_prompt(edited))

            # --- Chapter summary for continuity ---
            summary = call_llm(build_chapter_summary_prompt(polished))
            summaries.append(summary)

            chapters_done.append({
                "number": chapter_num,
                "title": chapter_title,
                "content": polished,
                "summary": summary,
            })

            with _progress_lock:
                _progress_store[token]["current"] = idx + 1
                _progress_store[token]["chapters_done"] = list(chapters_done)

        # --- Final consistency pass ---
        consistency_raw = call_llm(
            build_consistency_pass_prompt(title, summaries, special_instructions),
            json_mode=True,
        )
        try:
            consistency = json.loads(consistency_raw)
        except json.JSONDecodeError:
            consistency = {"issues": [], "overall_assessment": ""}

        with _progress_lock:
            _progress_store[token]["status"] = "done"
            _progress_store[token]["consistency"] = consistency

    except (RuntimeError, requests.exceptions.RequestException, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.error("Chapter generation failed for token %s: %s", token, exc)
        with _progress_lock:
            _progress_store[token]["status"] = "error"
            _progress_store[token]["error"] = str(exc)


def _format_characters(character_list: list[dict]) -> str:
    if not character_list:
        return "No characters defined."
    lines = []
    for ch in character_list:
        lines.append(
            f"- {ch.get('name','?')} (age {ch.get('age','?')}): "
            f"{ch.get('role','?')}. Background: {ch.get('background','')}. "
            f"Arc: {ch.get('arc','')}."
        )
    return "\n".join(lines)


@app.route("/progress/<token>")
def progress(token: str):
    """Poll endpoint for chapter generation progress."""
    with _progress_lock:
        data = _progress_store.get(token)
    if data is None:
        return jsonify({"error": "Unknown token"}), 404
    return jsonify(data)


@app.route("/export", methods=["POST"])
def export_novel():
    """
    Compile the completed novel into a Markdown file and return a download URL.
    Expects JSON: { "token": "<progress_token>" }
    """
    data = request.get_json(silent=True) or {}
    token = data.get("token", "")

    with _progress_lock:
        progress_data = _progress_store.get(token)

    if not progress_data or progress_data.get("status") != "done":
        return jsonify({"error": "Novel generation not complete."}), 400

    title = session.get("title", "Novel")
    chapters_done = progress_data.get("chapters_done", [])
    consistency = progress_data.get("consistency", {})

    # Build Markdown
    lines = [f"# {title}\n"]
    for ch in chapters_done:
        lines.append(f"\n## Chapter {ch['number']}: {ch['title']}\n")
        lines.append(f"*Summary: {ch['summary']}*\n")
        lines.append(f"\n{ch['content']}\n")

    if consistency.get("overall_assessment"):
        lines.append("\n---\n")
        lines.append("## Editor's Notes\n")
        lines.append(f"{consistency['overall_assessment']}\n")

    markdown_content = "\n".join(lines)

    # Safe filename
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:80]
    filename = f"{safe_title}.md"
    export_path = Path(config.EXPORT_DIR) / filename

    export_path.write_text(markdown_content, encoding="utf-8")

    return jsonify({"download_url": f"/download/{filename}"})


@app.route("/download/<path:filename>")
def download_file(filename: str):
    """Serve a generated export file."""
    # Prevent directory traversal
    safe_filename = Path(filename).name
    export_path = Path(config.EXPORT_DIR) / safe_filename
    if not export_path.exists():
        abort(404)
    return send_file(str(export_path), as_attachment=True, download_name=safe_filename)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Bind to 127.0.0.1 for local development.
    # In production, use a WSGI server (e.g. gunicorn) behind a reverse proxy.
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    app.run(debug=False, host=host, port=port)
