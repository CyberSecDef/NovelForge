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
import pprint

from dotenv import load_dotenv
load_dotenv() 
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

# Configure secret key FIRST - required for sessions
app.config["SECRET_KEY"] = config.SECRET_KEY

# Ensure directories exist BEFORE initializing sessions
Path(config.SESSION_FILE_DIR).mkdir(parents=True, exist_ok=True)
Path(config.EXPORT_DIR).mkdir(parents=True, exist_ok=True)
Path("./logs").mkdir(parents=True, exist_ok=True)
Path("./sessions").mkdir(parents=True, exist_ok=True)

# Configure filesystem-based sessions
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = config.SESSION_FILE_DIR
app.config["SESSION_PERMANENT"] = False

Session(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up dedicated LLM logger that writes JSON objects to logs/llm.log
llm_logger = logging.getLogger("llm_requests")
llm_logger.setLevel(logging.INFO)
llm_handler = logging.FileHandler("./logs/llm.log")
llm_handler.setFormatter(logging.Formatter("%(message)s"))
llm_logger.addHandler(llm_handler)
llm_logger.propagate = False

# In-memory store for chapter-generation progress keyed by session token
_progress_store: dict[str, dict] = {}
_progress_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Session persistence for crash recovery
# ---------------------------------------------------------------------------

def get_session_id() -> str:
    """Get or create a unique session ID for this user session."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]


def get_session_file_path() -> Path:
    """Get the file path for the current session's persistence data."""
    session_id = get_session_id()
    return Path("./sessions") / f"{session_id}.json"


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
            "progress_token": session.get("progress_token", ""),
        }
        
        # Add progress store data if available
        token = session.get("progress_token")
        if token:
            with _progress_lock:
                if token in _progress_store:
                    state["progress_data"] = _progress_store[token]
        
        # Write to file
        session_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
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
        logger.info(f"Loaded session state from {session_file}")
        return state
    except Exception as e:
        logger.error(f"Failed to load session state: {e}")
        return None


def restore_session_from_state(state: dict) -> None:
    """
    Restore session and progress store from saved state dict.
    """
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
    session["progress_token"] = state.get("progress_token", "")
    
    # Restore progress store if available
    if "progress_data" in state and state.get("progress_token"):
        token = state["progress_token"]
        with _progress_lock:
            _progress_store[token] = state["progress_data"]
    
    logger.info("Restored session from saved state")


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

# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

# Words indicative of LLM-generated text that the anti-LLM agent should remove
_FORBIDDEN_WORDS = [
    "embark", "delve", "realm", "tapestry", "testament", "nuance",
    "beacon", "uncharted", "multifaceted", "leverage", "synergy",
    "pivotal", "groundbreaking", "commendable", "meticulous",
]

MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds


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

    # Log the request (sanitize API key)
    request_log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": "request",
        "url": config.LLM_API_URL,
        "headers": {
            "Authorization": f"Bearer {config.LLM_API_KEY[:8]}..." if config.LLM_API_KEY else "None",
            "Content-Type": "application/json",
        },
        "payload": payload,
    }
    llm_logger.info(json.dumps(request_log, indent=2))

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                config.LLM_API_URL,
                headers=headers,
                json=payload,
                timeout=240,
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
            
            # Log the response
            response_log = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": "response",
                "status_code": resp.status_code,
                "headers": dict(resp.headers),
                "response": data,
            }
            llm_logger.info(json.dumps(response_log, indent=2))
            
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            logger.warning("LLM request timed out (attempt %d/%d)", attempt, MAX_RETRIES)
            if attempt == MAX_RETRIES:
                raise RuntimeError("LLM API timed out after multiple retries.")
            time.sleep(RETRY_DELAY * attempt)
        except requests.exceptions.RequestException as exc:
            # Log the error
            error_log = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": "error",
                "error": str(exc),
            }
            llm_logger.info(json.dumps(error_log))
            raise RuntimeError(f"LLM API request failed: {exc}") from exc

    raise RuntimeError("LLM API failed after maximum retries.")


def parse_llm_json(response: str) -> dict:
    """
    Parse JSON from LLM response, handling common irregularities:
    - Markdown code fences (```json, ```)
    - Extra whitespace
    - BOM characters
    - Text before/after the JSON object
    
    Raises json.JSONDecodeError if parsing fails.
    """
    # Strip BOM if present
    if response.startswith('\ufeff'):
        response = response[1:]
    
    # Remove markdown code fences
    response = response.strip()
    
    # Handle ```json\n...\n``` or ```\n...\n```
    if response.startswith('```'):
        # Find the first newline after opening fence
        first_newline = response.find('\n')
        if first_newline != -1:
            response = response[first_newline + 1:]
        
        # Remove closing fence
        if response.endswith('```'):
            response = response[:-3]
    
    response = response.strip()
    
    # Try to extract JSON by finding first { or [ and last } or ]
    # This handles cases where there's explanatory text before/after
    start_brace = response.find('{')
    start_bracket = response.find('[')
    
    # Determine which comes first (or if only one exists)
    if start_brace == -1 and start_bracket == -1:
        # No JSON structure found, try parsing as-is
        return json.loads(response)
    
    if start_brace == -1:
        start = start_bracket
        end_char = ']'
    elif start_bracket == -1:
        start = start_brace
        end_char = '}'
    else:
        start = min(start_brace, start_bracket)
        end_char = '}' if start == start_brace else ']'
    
    # Find the matching closing character from the end
    end = response.rfind(end_char)
    
    if end != -1 and end > start:
        response = response[start:end + 1]
    
    return json.loads(response)


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
                "***Return ONLY the title text, nothing else.***"
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
                "***CRITICAL: Return ONLY a valid JSON object with NO markdown code blocks, NO introduction, NO explanation.***\n"
                "Required structure:\n"
                '{"chapters": [{"number": 1, "title": "...", "summary": "..."}, ...]}\n\n'
                "***Each chapter summary should be 2-4 sentences describing key events and purpose.***"
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
                "***CRITICAL: Return ONLY a valid JSON object with NO markdown code blocks, NO introduction, NO explanation.***\n"
                "Required structure:\n"
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
                f"***CRITICAL: You are writing CHAPTER {chapter_num}. Keep this chapter number in mind throughout.***\n"
                f"Target: approximately {target_words:,} words. "
                "Write immersive, human-sounding prose. "
                "Follow the scene pattern: Goal → Obstacle → Outcome → New problem.\n\n"
                "***Return ONLY the chapter text with NO introduction, NO title header, NO explanation.***"
            ),
        },
    ]


def build_dialog_agent_prompt(chapter_text: str, chapter_num: int, title: str) -> list[dict]:
    """
    Dialog agent: refines all dialogue in the chapter for naturalism, voice
    distinction, and subtext.  Returns only the revised chapter text.
    """
    return [
        _build_system_prompt(
            "You are a dialogue specialist for literary fiction. "
            "Your task is to make every line of dialogue feel genuinely human: "
            "distinct voices per character, natural rhythm, subtext beneath the surface, "
            "and beats of action/reaction woven into dialogue scenes. "
            "Do not change narration that contains no dialogue. "
            "Return the full chapter text with improved dialogue only."
        ),
        {
            "role": "user",
            "content": (
                f"Novel: '{title}' - Chapter {chapter_num}\n\n"
                f"Refine the dialogue in this chapter. "
                "Ensure each character speaks distinctly, conversations feel natural "
                "and purposeful, and subtext is present where appropriate.\n\n"
                "***CRITICAL: Return ONLY the complete revised chapter text with NO introduction, NO explanation, NO markdown.***\n\n"
                f"{chapter_text}"
            ),
        },
    ]


def build_scene_agent_prompt(chapter_text: str, chapter_num: int, title: str) -> list[dict]:
    """
    Scene agent: ensures every scene follows the Goal → Obstacle → Outcome →
    New Problem pattern.  Returns only the revised chapter text.
    """
    return [
        _build_system_prompt(
            "You are a scene architect for commercial fiction. "
            "Every scene must have a clear character goal, a concrete obstacle, "
            "an outcome (success, failure, or complication), and a new problem "
            "that propels the story forward. "
            "Identify any scenes that lack this structure and rewrite them accordingly. "
            "Return the full chapter text."
        ),
        {
            "role": "user",
            "content": (
                f"Novel: '{title}' - Chapter {chapter_num}\n\n"
                "Review and revise this chapter so that every scene follows the "
                "Goal → Obstacle → Outcome → New Problem pattern.\n\n"
                "***CRITICAL: Return ONLY the complete revised chapter text with NO introduction, NO explanation, NO markdown.***\n\n"
                f"{chapter_text}"
            ),
        },
    ]


def build_structure_agent_prompt(
    chapter_text: str,
    chapter_num: int,
    total_chapters: int,
    outline_summary: str,
) -> list[dict]:
    """
    Structure agent: ensures the chapter fits its designated position in the
    nine-phase novel architecture.  Returns only the revised chapter text.
    """
    # Determine which story phase this chapter should serve
    position_pct = (chapter_num / total_chapters * 100) if total_chapters > 0 else 50
    if position_pct <= 25:
        phase_hint = "Beginning (Hook / Setup / Inciting Incident)"
    elif position_pct <= 75:
        phase_hint = "Middle (Rising Action / Midpoint Shift / Complications)"
    else:
        phase_hint = "End (Crisis / Climax / Resolution)"

    return [
        _build_system_prompt(
            "You are a structural editor who ensures every chapter fulfils its "
            "designated role in the novel's nine-phase story architecture. "
            "You do not change style or prose quality—only structural fit. "
            "Return the full revised chapter text."
        ),
        {
            "role": "user",
            "content": (
                f"Chapter {chapter_num} of {total_chapters}.\n"
                f"Structural position: {phase_hint}.\n"
                f"Outline summary for this chapter:\n{outline_summary}\n\n"
                "Ensure this chapter delivers the tension, revelations, and story movement "
                f"appropriate for a {phase_hint} chapter. "
                "Adjust pacing, scene order, or emphasis as needed.\n\n"
                "***CRITICAL: Return ONLY the complete revised chapter text with NO introduction, NO explanation, NO markdown.***\n\n"
                f"{chapter_text}"
            ),
        },
    ]


def build_character_agent_prompt(
    chapter_text: str,
    characters_text: str,
    chapter_num: int,
    title: str,
) -> list[dict]:
    """
    Character agent: checks and deepens character arcs and consistency.
    Returns only the revised chapter text.
    """
    return [
        _build_system_prompt(
            "You are a character development specialist. "
            "Your role is to ensure every character acts according to their established "
            "background, motivations, and arc trajectory. "
            "Deepen emotional authenticity, eliminate out-of-character moments, "
            "and advance each key character's arc. "
            "Return the full revised chapter text."
        ),
        {
            "role": "user",
            "content": (
                f"Novel: '{title}' - Chapter {chapter_num}\n\n"
                f"Character profiles:\n{characters_text}\n\n"
                "Review this chapter for character consistency and arc progression. "
                "Fix any moments where a character acts against their established nature. "
                "Deepen internal thought and emotional response where thin.\n\n"
                "***CRITICAL: Return ONLY the complete revised chapter text with NO introduction, NO explanation, NO markdown.***\n\n"
                f"{chapter_text}"
            ),
        },
    ]


def build_context_analyzer_prompt(
    chapter_text: str,
    previous_summaries: str,
    chapter_num: int,
    title: str,
) -> list[dict]:
    """
    Context analyzer: verifies world-building details are consistent with
    what has already been established in prior chapters.
    Returns only the revised chapter text.
    """
    prev_section = (
        f"Previous chapter summaries (Chapters 1-{chapter_num-1}):\n{previous_summaries}\n\n"
        if previous_summaries.strip()
        else "This is Chapter 1 (first chapter).\n\n"
    )
    return [
        _build_system_prompt(
            "You are a world-building consistency analyst. "
            "You compare the current chapter against all previously established facts "
            "(locations, character names, rules of the world, timeline) and correct "
            "any contradictions. You do not change plot or style. "
            "Return the full revised chapter text."
        ),
        {
            "role": "user",
            "content": (
                f"Novel: '{title}' - Working on Chapter {chapter_num}\n\n"
                f"{prev_section}"
                "Identify and correct any world-building inconsistencies in this chapter "
                "(wrong character names, contradicted geography, timeline errors, etc.).\n\n"
                "***CRITICAL: Return ONLY the complete revised chapter text with NO introduction, NO explanation, NO markdown.***\n\n"
                f"{chapter_text}"
            ),
        },
    ]


def build_synthesizer_prompt(
    chapter_text: str,
    chapter_num: int,
    title: str,
    genre: str,
) -> list[dict]:
    """
    Synthesizer agent: integrates all craft elements (plot, character, dialogue,
    scene, structure, theme) into a cohesive whole.
    Returns only the revised chapter text.
    """
    return [
        _build_system_prompt(
            "You are a master novelist acting as a synthesizer. "
            "Your task is to read a chapter that has been through multiple specialist passes "
            "and unify it: smooth seams between revised passages, ensure a consistent "
            "narrative voice, reinforce the thematic thread, and guarantee the chapter "
            "reads as a single, coherent piece of literary fiction. "
            "Do not add new plot events. Return the full synthesized chapter text."
        ),
        {
            "role": "user",
            "content": (
                f"Novel: '{title}' (Genre: {genre})\n"
                f"Chapter {chapter_num}\n\n"
                "Synthesize this chapter into a seamless whole. Unify voice, smooth "
                "any jarring transitions between sections, and ensure the chapter "
                "contributes meaningfully to the novel's theme.\n\n"
                "***CRITICAL: Return ONLY the complete revised chapter text with NO introduction, NO explanation, NO markdown.***\n\n"
                f"{chapter_text}"
            ),
        },
    ]


def build_quality_controller_prompt(chapter_text: str, chapter_num: int, title: str) -> list[dict]:
    """
    Quality controller: assesses the chapter for reader engagement, pacing,
    and narrative flow, then applies targeted improvements.
    Returns only the revised chapter text.
    """
    return [
        _build_system_prompt(
            "You are a senior quality controller for commercial and literary fiction. "
            "You evaluate chapters on: reader engagement (does every paragraph earn its place?), "
            "pacing (are slow passages dragging?), tension (is there always something at stake?), "
            "and hook strength (does the chapter end on a compelling note?). "
            "Apply targeted edits to raise quality. Do not rewrite entire sections unless "
            "necessary. Return the full revised chapter text."
        ),
        {
            "role": "user",
            "content": (
                f"Novel: '{title}' - Chapter {chapter_num}\n\n"
                "Evaluate and improve this chapter for engagement, pacing, tension, "
                "and the strength of its opening and closing hooks.\n\n"
                "***CRITICAL: Return ONLY the complete revised chapter text with NO introduction, NO explanation, NO markdown.***\n\n"
                f"{chapter_text}"
            ),
        },
    ]


def build_editing_agent_prompt(chapter_text: str, chapter_summary: str, chapter_num: int, title: str) -> list[dict]:
    """
    Editing agent: refines draft for plot holes, pacing, and character
    consistency.  Returns only the revised chapter text.
    """
    return [
        _build_system_prompt(
            "You are a professional fiction editor specialising in plot, pacing, and "
            "character consistency. Your job is to refine, not rewrite wholesale."
        ),
        {
            "role": "user",
            "content": (
                f"Novel: '{title}' - Chapter {chapter_num}\n\n"
                f"Chapter summary (what should happen):\n{chapter_summary}\n\n"
                f"Chapter draft:\n{chapter_text}\n\n"
                "Identify and fix: plot holes, pacing issues, character inconsistencies, "
                "unclear motivations.\n\n"
                "***CRITICAL: Return ONLY the complete revised chapter text with NO introduction, NO explanation, NO markdown.***"
            ),
        },
    ]


def build_polish_agent_prompt(chapter_text: str, chapter_num: int, title: str, genre: str) -> list[dict]:
    """
    Polish agent: elevates grammar, style, and vivid language.
    Returns only the polished chapter text.
    """
    return [
        _build_system_prompt(
            "You are a literary polisher. Elevate grammar, style, and language quality. "
            "Ensure varied sentence structure and vivid prose. "
            "Return only the polished chapter text."
        ),
        {
            "role": "user",
            "content": (
                f"Novel: '{title}' ({genre}) - Chapter {chapter_num}\n\n"
                "Polish this chapter for grammar, style, and language quality. "
                "Ensure varied sentence structure and vivid prose.\n\n"
                "***CRITICAL: Return ONLY the complete polished chapter text with NO introduction, NO explanation, NO markdown.***\n\n"
                f"{chapter_text}"
            ),
        },
    ]


def build_anti_llm_agent_prompt(chapter_text: str, chapter_num: int, title: str) -> list[dict]:
    """
    Anti-LLM agent: dedicated pass to strip robotic language patterns, overused
    phrases, and other LLM hallmarks.  Returns only the revised chapter text.

    Note: ``_FORBIDDEN_WORDS`` contains overused individual words that are easy
    to track and reference throughout the codebase.  The robotic *transition
    phrases* listed in the prompt below are multi-word patterns that only make
    sense as inline prose instructions rather than a simple word list.
    """
    forbidden = ", ".join(_FORBIDDEN_WORDS)
    return [
        _build_system_prompt(
            "You are an anti-LLM specialist. Your only job is to make AI-generated "
            "text sound genuinely human-written. Remove or replace: "
            f"overused words ({forbidden}), "
            "robotic transition phrases ('In conclusion', 'It is worth noting', "
            "'As a result of this'), "
            "unnecessary hedging, repetitive sentence openings, "
            "unnatural summarising, and any phrasing that feels mechanical or generic. "
            "Introduce subtle human imperfections: varied sentence length, "
            "occasional fragments for emphasis, colloquial rhythms where appropriate. "
            "Do NOT change plot, characters, or factual content. "
            "Return the full revised chapter text."
        ),
        {
            "role": "user",
            "content": (
                f"Novel: '{title}' - Chapter {chapter_num}\n\n"
                "Strip all LLM-sounding patterns from this chapter and make it read "
                "as naturally human-written literary fiction.\n\n"
                "***CRITICAL: Return ONLY the complete revised chapter text with NO introduction, NO explanation, NO markdown.***\n\n"
                f"{chapter_text}"
            ),
        },
    ]


def build_chapter_summary_prompt(chapter_text: str, chapter_num: int) -> list[dict]:
    return [
        _build_system_prompt("You are a precise summariser of fiction."),
        {
            "role": "user",
            "content": (
                f"Write a 100-200 word summary of Chapter {chapter_num} for continuity tracking.\n\n"
                "***CRITICAL: Return ONLY the summary text with NO introduction, NO chapter number header, NO explanation.***\n\n"
                f"{chapter_text}"
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
                "***CRITICAL: Return ONLY a valid JSON object with NO markdown code blocks, NO introduction, NO explanation.***\n"
                "Required structure:\n"
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
            outline_data = parse_llm_json(outline_raw)
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
            characters_data = parse_llm_json(characters_raw)
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

    # Auto-save session state after outline approval
    save_session_state()

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
            "step": "Preparing…",
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
    
    # Auto-save session state after starting generation
    save_session_state()
    
    return jsonify({"token": token})


def _resume_chapter_generation(
    token: str, 
    snap: dict, 
    chapters_done: list[dict], 
    summaries: list[str], 
    start_idx: int
) -> None:
    """
    Resume chapter generation from a specific chapter index after a crash.
    Uses the same pipeline as _run_chapter_generation but starts from start_idx.
    """
    # Call the main generation function but with pre-populated chapters_done and summaries
    _run_chapter_generation_internal(token, snap, chapters_done, summaries, start_idx)


def _run_chapter_generation(token: str, snap: dict) -> None:
    """
    Background worker: generate all chapters sequentially using the full
    twelve-step agent pipeline.

    Per-chapter pipeline:
        1. Draft           – initial prose (Novelist)
        2. Dialog agent    – naturalise dialogue
        3. Scene agent     – enforce Goal→Obstacle→Outcome→New Problem
        4. Context analyzer– fix world-building continuity errors
        5. Editing agent   – plot holes, pacing, character consistency
        6. Structure agent – confirm chapter fits story architecture
        7. Character agent – deepen arcs and fix out-of-character moments
        8. Synthesizer     – unify voice and theme after multi-pass edits
        9. Polish agent    – grammar, style, vivid language
       10. Anti-LLM agent  – strip robotic patterns and forbidden words
       11. Quality control – engagement, tension, pacing check
       12. Summary         – 100-200 word continuity summary
    """
    _run_chapter_generation_internal(token, snap, [], [], 0)


def _run_chapter_generation_internal(
    token: str, 
    snap: dict, 
    chapters_done: list[dict], 
    summaries: list[str], 
    start_idx: int
) -> None:
    """
    Internal function that performs the actual chapter generation.
    Can start from any chapter index to support resume functionality.
    """
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

    def _set_step(step_label: str) -> None:
        with _progress_lock:
            _progress_store[token]["step"] = step_label
        # Auto-save after each step
        try:
            # We need to save from the main session context, but we're in a background thread
            # So we save the progress_store data directly to a file named by token
            save_file = Path("./sessions") / f"{token}_progress.json"
            with _progress_lock:
                progress_data = dict(_progress_store[token])
            save_data = {
                "token": token,
                "snapshot": snap,
                "chapters_done": chapters_done,
                "summaries": summaries,
                "progress": progress_data,
            }
            save_file.write_text(json.dumps(save_data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to auto-save progress: {e}")

    try:
        for idx, ch in enumerate(chapter_list[start_idx:], start=start_idx):
            chapter_num = ch.get("number", idx + 1)
            chapter_title = ch.get("title", f"Chapter {chapter_num}")
            chapter_outline_summary = ch.get("summary", "")

            previous_summaries = "\n\n".join(
                f"Chapter {i+1}: {s}" for i, s in enumerate(summaries)
            )

            # Update progress with current agent step label
            # 1. Draft
            _set_step(f"Chapter {chapter_num}: drafting")
            text = call_llm(
                build_chapter_draft_prompt(
                    premise, genre, title, chapter_num, chapter_title,
                    chapter_outline_summary, characters_text,
                    previous_summaries, target_per_chapter, special_instructions,
                )
            )

            # 2. Dialog agent
            _set_step(f"Chapter {chapter_num}: refining dialogue")
            text = call_llm(build_dialog_agent_prompt(text, chapter_num, title))

            # 3. Scene agent
            _set_step(f"Chapter {chapter_num}: structuring scenes")
            text = call_llm(build_scene_agent_prompt(text, chapter_num, title))

            # 4. Context analyzer
            _set_step(f"Chapter {chapter_num}: verifying continuity")
            text = call_llm(build_context_analyzer_prompt(text, previous_summaries, chapter_num, title))

            # 5. Editing agent
            _set_step(f"Chapter {chapter_num}: editing")
            text = call_llm(build_editing_agent_prompt(text, chapter_outline_summary, chapter_num, title))

            # 6. Structure agent
            _set_step(f"Chapter {chapter_num}: checking structure")
            text = call_llm(
                build_structure_agent_prompt(
                    text, chapter_num, total_chapters, chapter_outline_summary
                )
            )

            # 7. Character agent
            _set_step(f"Chapter {chapter_num}: deepening characters")
            text = call_llm(build_character_agent_prompt(text, characters_text, chapter_num, title))

            # 8. Synthesizer
            _set_step(f"Chapter {chapter_num}: synthesizing")
            text = call_llm(build_synthesizer_prompt(text, chapter_num, title, genre))

            # 9. Polish agent
            _set_step(f"Chapter {chapter_num}: polishing")
            text = call_llm(build_polish_agent_prompt(text, chapter_num, title, genre))

            # 10. Anti-LLM agent
            _set_step(f"Chapter {chapter_num}: anti-LLM pass")
            text = call_llm(build_anti_llm_agent_prompt(text, chapter_num, title))

            # 11. Quality controller
            _set_step(f"Chapter {chapter_num}: quality control")
            text = call_llm(build_quality_controller_prompt(text, chapter_num, title))

            # 12. Summary for continuity
            _set_step(f"Chapter {chapter_num}: summarising")
            summary = call_llm(build_chapter_summary_prompt(text, chapter_num))
            summaries.append(summary)

            chapters_done.append({
                "number": chapter_num,
                "title": chapter_title,
                "content": text,
                "summary": summary,
            })

            with _progress_lock:
                _progress_store[token]["current"] = idx + 1
                _progress_store[token]["step"] = f"Chapter {chapter_num}: complete"
                _progress_store[token]["chapters_done"] = list(chapters_done)

        # --- Final consistency pass (context analyzer at novel level) ---
        with _progress_lock:
            _progress_store[token]["step"] = "Final consistency pass"
        consistency_raw = call_llm(
            build_consistency_pass_prompt(title, summaries, special_instructions),
            json_mode=True,
        )
        try:
            consistency = parse_llm_json(consistency_raw)
        except json.JSONDecodeError:
            consistency = {"issues": [], "overall_assessment": ""}

        with _progress_lock:
            _progress_store[token]["status"] = "done"
            _progress_store[token]["consistency"] = consistency
        
        # Auto-save final state
        _set_step("Complete")

    except (RuntimeError, requests.exceptions.RequestException, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.error("Chapter generation failed for token %s: %s", token, exc)
        with _progress_lock:
            _progress_store[token]["status"] = "error"
            _progress_store[token]["error"] = str(exc)
        
        # Auto-save error state
        _set_step(f"Error: {str(exc)}")


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


@app.route("/check_saved_state")
def check_saved_state():
    """
    Check if there's a saved session state that can be resumed.
    Returns information about the saved state if it exists.
    """
    state = load_session_state()
    if not state:
        return jsonify({"has_saved_state": False})
    
    # Check if there's progress data
    token = state.get("progress_token")
    progress_file = Path("./sessions") / f"{token}_progress.json" if token else None
    has_progress = progress_file and progress_file.exists() if token else False
    
    progress_info = None
    if has_progress and progress_file:
        try:
            progress_data = json.loads(progress_file.read_text(encoding="utf-8"))
            progress = progress_data.get("progress", {})
            progress_info = {
                "current": progress.get("current", 0),
                "total": progress.get("total", 0),
                "step": progress.get("step", ""),
                "status": progress.get("status", ""),
            }
        except Exception as e:
            logger.error(f"Failed to read progress file: {e}")
    
    return jsonify({
        "has_saved_state": True,
        "title": state.get("title", "Untitled"),
        "chapters": state.get("chapters", 0),
        "has_progress": has_progress,
        "progress_info": progress_info,
    })


@app.route("/resume_session", methods=["POST"])
def resume_session():
    """
    Restore the saved session state and optionally resume chapter generation.
    """
    state = load_session_state()
    if not state:
        return jsonify({"error": "No saved state found"}), 404
    
    # Restore session
    restore_session_from_state(state)
    
    # Check if there's progress to resume
    token = state.get("progress_token")
    progress_file = Path("./sessions") / f"{token}_progress.json" if token else None
    
    if token and progress_file and progress_file.exists():
        try:
            # Load progress data
            progress_data = json.loads(progress_file.read_text(encoding="utf-8"))
            
            # Restore progress store
            with _progress_lock:
                if token not in _progress_store:
                    _progress_store[token] = progress_data.get("progress", {})
            
            # Check if we should resume generation
            progress = progress_data.get("progress", {})
            if progress.get("status") == "running":
                # Resume generation from where it left off
                snapshot = progress_data.get("snapshot", {})
                chapters_done = progress_data.get("chapters_done", [])
                summaries = progress_data.get("summaries", [])
                current_chapter = progress.get("current", 0)
                
                # Start a new thread to continue generation
                thread = threading.Thread(
                    target=_resume_chapter_generation,
                    args=(token, snapshot, chapters_done, summaries, current_chapter),
                    daemon=True,
                )
                thread.start()
                
                return jsonify({
                    "status": "resumed",
                    "token": token,
                    "message": f"Resuming from chapter {current_chapter + 1}"
                })
        except Exception as e:
            logger.error(f"Failed to resume generation: {e}")
    
    return jsonify({"status": "restored", "message": "Session restored successfully"})


@app.route("/new_session", methods=["POST"])
def new_session():
    """
    Archive the current LLM log file and start a new session.
    Clears all session data and saved state.
    """
    import shutil
    from datetime import datetime
    
    # Archive the current LLM log file
    llm_log = Path("./logs/llm.log")
    if llm_log.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"llm_{timestamp}.log"
        archive_path = Path("./logs") / archive_name
        try:
            shutil.copy2(llm_log, archive_path)
            # Clear the original log file
            llm_log.write_text("", encoding="utf-8")
            logger.info(f"Archived LLM log to {archive_path}")
        except Exception as e:
            logger.error(f"Failed to archive LLM log: {e}")
    
    # Clear session state file
    clear_session_state()
    
    # Clear current session data
    session.clear()
    
    # Clear progress token files
    try:
        sessions_dir = Path("./sessions")
        for progress_file in sessions_dir.glob("*_progress.json"):
            progress_file.unlink()
    except Exception as e:
        logger.error(f"Failed to clear progress files: {e}")
    
    return jsonify({"status": "success", "message": "New session started"})


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


@app.route("/llm_log")
def get_llm_log():
    """Return recent LLM log entries for the chat display."""
    # Use absolute path based on app location
    log_path = Path(__file__).parent / "logs" / "llm.log"
    
    if not log_path.exists():
        logger.warning(f"LLM log file not found at {log_path}")
        return jsonify({"entries": []})
    
    try:
        entries = []
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split by lines that start with { (beginning of a JSON object)
        # This handles pretty-printed JSON where each object spans multiple lines
        json_objects = []
        current_obj = ""
        brace_count = 0
        
        for line in content.split('\n'):
            if line.strip().startswith('{') and brace_count == 0:
                # Start of a new JSON object
                if current_obj:
                    json_objects.append(current_obj)
                current_obj = line + '\n'
                brace_count = line.count('{') - line.count('}')
            elif brace_count > 0:
                # Continuation of current object
                current_obj += line + '\n'
                brace_count += line.count('{') - line.count('}')
                
                if brace_count == 0:
                    # End of current object
                    json_objects.append(current_obj)
                    current_obj = ""
        
        # Don't forget the last object if any
        if current_obj:
            json_objects.append(current_obj)
        
        # Parse each JSON object
        for obj_str in json_objects:
            try:
                entry = json.loads(obj_str)
                entries.append(entry)
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse log entry: {e}")
                continue
        
        logger.info(f"Returning {len(entries)} log entries")
        return jsonify({"entries": entries})
    except Exception as e:
        logger.error(f"Error reading LLM log: {e}")
        return jsonify({"entries": [], "error": str(e)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # In production, use a WSGI server (e.g. gunicorn) behind a reverse proxy.
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    app.run(debug=False, host=host, port=port)
