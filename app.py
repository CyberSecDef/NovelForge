"""
NovelForge – Flask backend for AI-powered novel generation.

Run with:
    python app.py

Requires environment variables (see .env.example / config.py):
    LLM_API_URL  – LLM API endpoint
    LLM_API_KEY  – API key for the LLM provider
    SECRET_KEY   – Flask secret key (change in production)
"""

# Standard library
import json
import logging
import os
import pprint
import threading
import time
import uuid
from pathlib import Path

# Third-party
import markupsafe
import requests
import yaml
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, render_template, request, send_file, session
from flask_session import Session
from jinja2 import Template

# Load environment variables before local imports so config picks them up
load_dotenv()

# Local
import config

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Configure secret key FIRST - required for sessions
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["TEMPLATES_AUTO_RELOAD"] = True


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
            "story_architecture": session.get("story_architecture", {}),
            "master_timeline": session.get("master_timeline", {}),
            "character_fate_registry": session.get("character_fate_registry", {}),
            "character_arc_plan": session.get("character_arc_plan", {}),
            "antagonist_motivation_plan": session.get("antagonist_motivation_plan", {}),
            "technology_rules": session.get("technology_rules", {}),
            "theme_reinforcement": session.get("theme_reinforcement", {}),
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
    session["story_architecture"] = state.get("story_architecture", {})
    session["master_timeline"] = state.get("master_timeline", {})
    session["character_fate_registry"] = state.get("character_fate_registry", {})
    session["character_arc_plan"] = state.get("character_arc_plan", {})
    session["antagonist_motivation_plan"] = state.get("antagonist_motivation_plan", {})
    session["technology_rules"] = state.get("technology_rules", {})
    session["theme_reinforcement"] = state.get("theme_reinforcement", {})
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

def load_prompt_by_name(prompt_name, filename='prompts.yaml'):
    """
    Loads a specific prompt by name from a YAML file.
    
    Args:
        prompt_name (str): The name (key) of the prompt to retrieve.
        filename (str): The path to the YAML file.
        
    Returns:
        dict or None: The prompt dictionary if found, otherwise None.
    """
    # Ensure the file path is correct
    filepath = os.path.join(os.getcwd(), filename)
    if not os.path.exists(filepath):
        print(f"Error: {filename} not found at {filepath}")
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            # Use safe_load to avoid potential security issues from untrusted sources
            prompts_data = yaml.safe_load(file)
            
            if prompts_data and prompt_name in prompts_data:
                return prompts_data[prompt_name]
            else:
                print(f"Error: Prompt '{prompt_name}' not found in {filename}")
                return None
                
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None


_prompts_cache: dict | None = None


def _load_prompts() -> dict:
    """Load and cache prompts from prompts.yml, keyed by prompt name."""
    global _prompts_cache
    if _prompts_cache is None:
        filepath = os.path.join(os.path.dirname(__file__), "prompts.yml")
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        _prompts_cache = {p["name"]: p for p in data.get("prompts", [])}
    return _prompts_cache


def render_prompt(name: str, **context) -> list[dict]:
    """Render a named prompt from prompts.yml using Jinja2 and return a message list."""
    prompts = _load_prompts()
    if name not in prompts:
        raise KeyError(f"Prompt '{name}' not found in prompts.yml")
    prompt = prompts[name]
    system_text = Template(prompt["system"]).render(**context)
    user_text = Template(prompt["user"]).render(**context)
    return [
        {"role": "system", "content": system_text.strip()},
        {"role": "user", "content": user_text.strip()},
    ]


def call_llm(messages: list[dict], *, action: str = "", json_mode: bool = False) -> str:
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
    action = action if action else "Updating Content"

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    # Log the request (sanitize API key)
    request_log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": "request",
        "action": action,
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


def choose_story_architecture_mode(total_chapters: int) -> str:
    """Choose a strict act model based on project size."""
    return "four-act" if total_chapters >= 16 else "three-act"


def build_story_architecture_prompt(
    title: str,
    premise: str,
    genre: str,
    chapter_list: list[dict],
    special_instructions: str = "",
) -> list[dict]:
    """
    Story Architecture Planner: converts a chapter outline into a strict
    three-act or four-act architecture with explicit escalation targets.
    """
    total_chapters = max(1, len(chapter_list))
    architecture_mode = choose_story_architecture_mode(total_chapters)
    outline_text = "\n".join(
        f"Chapter {ch.get('number', i + 1)}: {ch.get('title', f'Chapter {i + 1}')} – {ch.get('summary', '')}"
        for i, ch in enumerate(chapter_list)
    )
    return render_prompt(
        "story_architecture_planning",
        title=title,
        premise=premise,
        genre=genre,
        total_chapters=total_chapters,
        architecture_mode=architecture_mode,
        outline_text=outline_text,
        special_instructions=special_instructions or "",
    )


def _coerce_positive_int(value, default: int) -> int:
    try:
        coerced = int(value)
        return coerced if coerced > 0 else default
    except (TypeError, ValueError):
        return default


def _build_fallback_story_architecture(chapter_list: list[dict], total_chapters: int) -> dict:
    """Create a deterministic fallback architecture if the planner fails."""
    total_chapters = max(1, total_chapters)
    architecture_type = choose_story_architecture_mode(total_chapters)

    if architecture_type == "four-act":
        split_one = max(1, round(total_chapters * 0.25))
        split_two = max(split_one + 1, round(total_chapters * 0.50)) if total_chapters > 2 else total_chapters
        split_three = max(split_two + 1, round(total_chapters * 0.75)) if total_chapters > 3 else total_chapters
        split_two = min(split_two, total_chapters)
        split_three = min(split_three, total_chapters)
        acts = [
            {
                "act": 1,
                "label": "Act I",
                "chapter_start": 1,
                "chapter_end": split_one,
                "purpose": "Setup, hook, and inciting pressure.",
                "escalation": "Introduce the core conflict with tightly limited operations.",
            },
            {
                "act": 2,
                "label": "Act II-A",
                "chapter_start": min(split_one + 1, total_chapters),
                "chapter_end": split_two,
                "purpose": "First wave of pursuit, resistance, and complications.",
                "escalation": "Increase pressure while keeping each chapter focused on one main turn.",
            },
            {
                "act": 3,
                "label": "Act II-B",
                "chapter_start": min(split_two + 1, total_chapters),
                "chapter_end": split_three,
                "purpose": "After the midpoint reversal, consequences intensify and options narrow.",
                "escalation": "Escalate consequences and compress the room for recovery.",
            },
            {
                "act": 4,
                "label": "Act III",
                "chapter_start": min(split_three + 1, total_chapters),
                "chapter_end": total_chapters,
                "purpose": "Crisis, climax, and resolution.",
                "escalation": "Deliver the decisive confrontation, then resolve remaining tension.",
            },
        ]
    else:
        split_one = max(1, round(total_chapters * 0.25))
        split_two = max(split_one + 1, round(total_chapters * 0.75)) if total_chapters > 2 else total_chapters
        split_two = min(split_two, total_chapters)
        acts = [
            {
                "act": 1,
                "label": "Act I",
                "chapter_start": 1,
                "chapter_end": split_one,
                "purpose": "Setup, hook, and inciting disruption.",
                "escalation": "Constrain the chapter actions and establish the conflict cleanly.",
            },
            {
                "act": 2,
                "label": "Act II",
                "chapter_start": min(split_one + 1, total_chapters),
                "chapter_end": split_two,
                "purpose": "Escalation, midpoint reversal, and mounting complications.",
                "escalation": "Increase pressure in measured steps, with each chapter creating a sharper problem.",
            },
            {
                "act": 3,
                "label": "Act III",
                "chapter_start": min(split_two + 1, total_chapters),
                "chapter_end": total_chapters,
                "purpose": "Crisis, climax, and resolution.",
                "escalation": "Convert accumulated pressure into decisive action and payoff.",
            },
        ]

    midpoint_chapter = max(1, min(total_chapters, round((total_chapters + 1) / 2)))
    climax_chapter = max(1, total_chapters - 1) if total_chapters > 1 else 1
    resolution_chapter = total_chapters
    inciting_chapter = 2 if total_chapters >= 4 else 1

    def _act_for_chapter(chapter_num: int) -> dict:
        for act in acts:
            if act["chapter_start"] <= chapter_num <= act["chapter_end"]:
                return act
        return acts[-1]

    chapter_plan = []
    for idx, chapter in enumerate(chapter_list or [{"number": 1, "title": "Chapter 1", "summary": ""}], start=1):
        chapter_num = _coerce_positive_int(chapter.get("number", idx), idx)
        act = _act_for_chapter(chapter_num)

        if chapter_num == inciting_chapter:
            phase = "Inciting Incident"
            required_turn = "Inciting incident"
            operation_limit = 1
            escalation = "Disrupt the status quo with one irreversible turn."
        elif chapter_num == midpoint_chapter:
            phase = "Midpoint Reversal"
            required_turn = "Midpoint reversal"
            operation_limit = 2
            escalation = "Deliver a reversal that changes the protagonist's understanding or options."
        elif chapter_num == climax_chapter:
            phase = "Climax Build"
            required_turn = "Climax setup"
            operation_limit = 2
            escalation = "Narrow choices and force commitment to the final confrontation."
        elif chapter_num == resolution_chapter:
            phase = "Resolution"
            required_turn = "Resolution"
            operation_limit = 1
            escalation = "Resolve consequences and land the emotional aftermath."
        elif chapter_num < midpoint_chapter:
            phase = "Escalation"
            required_turn = "None"
            operation_limit = 1
            escalation = "Advance the conflict through one clear pressure increase."
        else:
            phase = "Complication"
            required_turn = "None"
            operation_limit = 2 if chapter_num >= max(2, total_chapters - 2) else 1
            escalation = "Tighten consequences and hand a harder problem into the next chapter."

        chapter_plan.append(
            {
                "number": chapter_num,
                "title": chapter.get("title", f"Chapter {chapter_num}"),
                "act": act["label"],
                "phase": phase,
                "purpose": chapter.get("summary", "") or act["purpose"],
                "escalation": escalation,
                "operation_limit": operation_limit,
                "required_turn": required_turn,
                "carry_forward": "End by handing the protagonist a sharper next problem.",
            }
        )

    return {
        "architecture_type": architecture_type,
        "acts": acts,
        "global_turns": {
            "inciting_incident": {
                "chapter": inciting_chapter,
                "detail": "The core conflict becomes unavoidable.",
            },
            "midpoint_reversal": {
                "chapter": midpoint_chapter,
                "detail": "A major revelation or reversal changes the trajectory.",
            },
            "climax": {
                "chapter": climax_chapter,
                "detail": "The decisive confrontation reaches its peak.",
            },
            "resolution": {
                "chapter": resolution_chapter,
                "detail": "Aftermath and payoff settle the story.",
            },
        },
        "chapter_plan": sorted(chapter_plan, key=lambda item: item["number"]),
    }


def normalise_story_architecture(
    architecture_data: dict,
    chapter_list: list[dict],
    total_chapters: int,
) -> dict:
    """Merge planner output with a deterministic fallback shape."""
    fallback = _build_fallback_story_architecture(chapter_list, total_chapters)
    if not isinstance(architecture_data, dict):
        return fallback

    architecture_type = architecture_data.get("architecture_type")
    if architecture_type not in {"three-act", "four-act"}:
        architecture_type = fallback["architecture_type"]

    acts = architecture_data.get("acts")
    if not isinstance(acts, list) or not acts:
        acts = fallback["acts"]

    global_turns = architecture_data.get("global_turns")
    if not isinstance(global_turns, dict):
        global_turns = fallback["global_turns"]

    raw_chapter_plan = architecture_data.get("chapter_plan")
    if not isinstance(raw_chapter_plan, list):
        raw_chapter_plan = architecture_data.get("chapters", [])
    raw_map = {
        _coerce_positive_int(item.get("number"), idx + 1): item
        for idx, item in enumerate(raw_chapter_plan)
        if isinstance(item, dict)
    }
    fallback_map = {item["number"]: item for item in fallback["chapter_plan"]}

    merged_plan = []
    safe_chapter_list = chapter_list or [{"number": 1, "title": "Chapter 1", "summary": ""}]
    for idx, chapter in enumerate(safe_chapter_list, start=1):
        chapter_num = _coerce_positive_int(chapter.get("number", idx), idx)
        fallback_item = fallback_map.get(chapter_num, fallback["chapter_plan"][0])
        planner_item = raw_map.get(chapter_num, {})

        merged_plan.append(
            {
                "number": chapter_num,
                "title": chapter.get("title", fallback_item.get("title", f"Chapter {chapter_num}")),
                "act": str(planner_item.get("act") or fallback_item["act"]),
                "phase": str(planner_item.get("phase") or fallback_item["phase"]),
                "purpose": str(
                    planner_item.get("purpose")
                    or planner_item.get("summary")
                    or chapter.get("summary", "")
                    or fallback_item["purpose"]
                ),
                "escalation": str(
                    planner_item.get("escalation")
                    or planner_item.get("escalation_target")
                    or fallback_item["escalation"]
                ),
                "operation_limit": _coerce_positive_int(
                    planner_item.get("operation_limit"), fallback_item["operation_limit"]
                ),
                "required_turn": str(
                    planner_item.get("required_turn")
                    or planner_item.get("turn")
                    or fallback_item["required_turn"]
                ),
                "carry_forward": str(
                    planner_item.get("carry_forward")
                    or planner_item.get("handoff")
                    or fallback_item["carry_forward"]
                ),
            }
        )

    return {
        "architecture_type": architecture_type,
        "acts": acts,
        "global_turns": global_turns,
        "chapter_plan": sorted(merged_plan, key=lambda item: item["number"]),
    }


def plan_story_architecture(
    title: str,
    premise: str,
    genre: str,
    chapter_list: list[dict],
    special_instructions: str = "",
) -> dict:
    """Run the Story Architecture Planner with safe fallback behavior."""
    total_chapters = max(1, len(chapter_list))
    try:
        raw = call_llm(
            build_story_architecture_prompt(
                title=title,
                premise=premise,
                genre=genre,
                chapter_list=chapter_list,
                special_instructions=special_instructions,
            ),
            action="Planning Story Architecture",
            json_mode=True,
        )
        return normalise_story_architecture(parse_llm_json(raw), chapter_list, total_chapters)
    except (RuntimeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Story architecture planner failed, using fallback: %s", exc)
        return _build_fallback_story_architecture(chapter_list, total_chapters)


def get_chapter_architecture_context(story_architecture: dict, chapter_num: int) -> str:
    """Format the planner output needed by the current chapter generation pass."""
    if not isinstance(story_architecture, dict):
        return ""

    chapter_entry = next(
        (
            item for item in story_architecture.get("chapter_plan", [])
            if _coerce_positive_int(item.get("number"), -1) == chapter_num
        ),
        None,
    )
    if not chapter_entry:
        return ""

    lines = [
        "Story Architecture Planner output for this chapter:",
        f"- Architecture model: {story_architecture.get('architecture_type', 'three-act')}",
        f"- Act: {chapter_entry.get('act', '')}",
        f"- Phase: {chapter_entry.get('phase', '')}",
        f"- Chapter purpose: {chapter_entry.get('purpose', '')}",
        f"- Escalation target: {chapter_entry.get('escalation', '')}",
        f"- Major operations limit: {chapter_entry.get('operation_limit', 1)}",
        f"- Required turning point: {chapter_entry.get('required_turn', 'None')}",
        f"- Carry forward: {chapter_entry.get('carry_forward', '')}",
    ]

    global_turns = story_architecture.get("global_turns", {})
    for turn_key, label in (
        ("inciting_incident", "Inciting incident"),
        ("midpoint_reversal", "Midpoint reversal"),
        ("climax", "Climax"),
        ("resolution", "Resolution"),
    ):
        turn = global_turns.get(turn_key)
        if isinstance(turn, dict):
            turn_chapter = _coerce_positive_int(turn.get("chapter"), 0)
            turn_detail = str(turn.get("detail", "")).strip()
            if turn_chapter:
                suffix = f" – {turn_detail}" if turn_detail else ""
                lines.append(f"- {label}: Chapter {turn_chapter}{suffix}")

    return "\n".join(lines)


def build_master_timeline_prompt(
    title: str,
    premise: str,
    genre: str,
    chapter_list: list[dict],
    character_list: list[dict],
    special_instructions: str = "",
) -> list[dict]:
    """
    Master Timeline Builder: creates a chronological event ledger with explicit
    state transitions to prevent contradictory character conditions.
    """
    chapter_text = "\n".join(
        f"Chapter {ch.get('number', i + 1)}: {ch.get('title', f'Chapter {i + 1}')} – {ch.get('summary', '')}"
        for i, ch in enumerate(chapter_list)
    )
    characters_text = "\n".join(
        f"- {c.get('name', '?')}: role={c.get('role', '')}; arc={c.get('arc', '')}; background={c.get('background', '')}"
        for c in character_list
    )
    if not characters_text.strip():
        characters_text = "- No explicit characters provided. Infer conservatively from outline."

    return render_prompt(
        "master_timeline_building",
        title=title,
        premise=premise,
        genre=genre,
        chapter_text=chapter_text,
        characters_text=characters_text,
        special_instructions=special_instructions or "",
    )


def _build_fallback_master_timeline(chapter_list: list[dict], character_list: list[dict]) -> dict:
    """Deterministic fallback timeline when planner output is unavailable."""
    safe_chapters = chapter_list or [{"number": 1, "title": "Chapter 1", "summary": ""}]
    safe_characters = character_list or []

    ledger = []
    chapter_constraints = []
    for idx, chapter in enumerate(safe_chapters, start=1):
        chapter_num = _coerce_positive_int(chapter.get("number", idx), idx)
        summary = str(chapter.get("summary", "")).strip()
        ledger.append(
            {
                "index": idx,
                "chapter": chapter_num,
                "event_type": "operation",
                "event": summary or f"Primary operation for Chapter {chapter_num}",
                "actors": [],
                "targets": [],
                "state_changes": [],
                "continuity_note": "Advance one major operation and preserve prior state continuity.",
            }
        )
        chapter_constraints.append(
            {
                "chapter": chapter_num,
                "must_include": ["At least one continuity-consistent consequence from prior events."],
                "must_avoid": ["Contradicting established character status without explicit transition."],
            }
        )

    character_states = []
    for character in safe_characters:
        name = str(character.get("name", "")).strip()
        if not name:
            continue
        character_states.append(
            {
                "character": name,
                "status": "active",
                "location": "unknown",
                "last_event_index": 0,
                "notes": "Baseline state before chapter drafting.",
            }
        )

    return {
        "ledger": ledger,
        "character_states": character_states,
        "chapter_constraints": chapter_constraints,
        "continuity_risks": [],
    }


def normalise_master_timeline(
    timeline_data: dict,
    chapter_list: list[dict],
    character_list: list[dict],
) -> dict:
    """Normalize timeline planner output into a stable schema."""
    fallback = _build_fallback_master_timeline(chapter_list, character_list)
    if not isinstance(timeline_data, dict):
        return fallback

    ledger = timeline_data.get("ledger", [])
    if not isinstance(ledger, list):
        ledger = []

    normalised_ledger = []
    for idx, event in enumerate(ledger, start=1):
        if not isinstance(event, dict):
            continue
        chapter_num = _coerce_positive_int(event.get("chapter"), 1)
        state_changes = event.get("state_changes", [])
        if not isinstance(state_changes, list):
            state_changes = []
        normalised_ledger.append(
            {
                "index": _coerce_positive_int(event.get("index"), idx),
                "chapter": chapter_num,
                "event_type": str(event.get("event_type", "other")),
                "event": str(event.get("event", "")).strip(),
                "actors": [str(a) for a in event.get("actors", []) if str(a).strip()],
                "targets": [str(t) for t in event.get("targets", []) if str(t).strip()],
                "state_changes": [sc for sc in state_changes if isinstance(sc, dict)],
                "continuity_note": str(event.get("continuity_note", "")).strip(),
            }
        )

    character_states = timeline_data.get("character_states", [])
    if not isinstance(character_states, list):
        character_states = []
    normalised_character_states = []
    for state in character_states:
        if not isinstance(state, dict):
            continue
        name = str(state.get("character", "")).strip()
        if not name:
            continue
        normalised_character_states.append(
            {
                "character": name,
                "status": str(state.get("status", "active")),
                "location": str(state.get("location", "unknown")),
                "last_event_index": _coerce_positive_int(state.get("last_event_index"), 0),
                "notes": str(state.get("notes", "")).strip(),
            }
        )

    chapter_constraints = timeline_data.get("chapter_constraints", [])
    if not isinstance(chapter_constraints, list):
        chapter_constraints = []
    normalised_constraints = []
    for idx, constraint in enumerate(chapter_constraints, start=1):
        if not isinstance(constraint, dict):
            continue
        normalised_constraints.append(
            {
                "chapter": _coerce_positive_int(constraint.get("chapter"), idx),
                "must_include": [str(x) for x in constraint.get("must_include", []) if str(x).strip()],
                "must_avoid": [str(x) for x in constraint.get("must_avoid", []) if str(x).strip()],
            }
        )

    continuity_risks = timeline_data.get("continuity_risks", [])
    if not isinstance(continuity_risks, list):
        continuity_risks = []

    merged = {
        "ledger": normalised_ledger or fallback["ledger"],
        "character_states": normalised_character_states or fallback["character_states"],
        "chapter_constraints": normalised_constraints or fallback["chapter_constraints"],
        "continuity_risks": [str(r) for r in continuity_risks if str(r).strip()],
    }
    return merged


def plan_master_timeline(
    title: str,
    premise: str,
    genre: str,
    chapter_list: list[dict],
    character_list: list[dict],
    special_instructions: str = "",
) -> dict:
    """Run Master Timeline Builder and return normalized timeline output."""
    try:
        raw = call_llm(
            build_master_timeline_prompt(
                title=title,
                premise=premise,
                genre=genre,
                chapter_list=chapter_list,
                character_list=character_list,
                special_instructions=special_instructions,
            ),
            action="Planning Master Timeline",
            json_mode=True,
        )
        parsed = parse_llm_json(raw)
        return normalise_master_timeline(parsed, chapter_list, character_list)
    except (RuntimeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Master Timeline Builder failed, using fallback: %s", exc)
        return _build_fallback_master_timeline(chapter_list, character_list)


def get_chapter_timeline_context(master_timeline: dict, chapter_num: int) -> str:
    """Return chapter-relevant timeline constraints and recent events as prompt text."""
    if not isinstance(master_timeline, dict):
        return ""

    ledger = master_timeline.get("ledger", [])
    if not isinstance(ledger, list):
        ledger = []
    chapter_events = [
        event for event in ledger
        if isinstance(event, dict) and _coerce_positive_int(event.get("chapter"), 0) == chapter_num
    ]
    recent_events = [
        event for event in ledger
        if isinstance(event, dict) and _coerce_positive_int(event.get("chapter"), 0) < chapter_num
    ]
    recent_events = recent_events[-3:]

    constraints = master_timeline.get("chapter_constraints", [])
    if not isinstance(constraints, list):
        constraints = []
    chapter_constraint = next(
        (
            c for c in constraints
            if isinstance(c, dict) and _coerce_positive_int(c.get("chapter"), 0) == chapter_num
        ),
        None,
    )

    states = master_timeline.get("character_states", [])
    if not isinstance(states, list):
        states = []

    lines = ["Master Timeline Builder output for this chapter:"]
    for event in recent_events:
        lines.append(
            f"- Prior event (Ch {event.get('chapter')}): {event.get('event', '')} "
            f"[{event.get('event_type', 'other')}]"
        )
    for event in chapter_events:
        lines.append(
            f"- Planned chapter event: {event.get('event', '')} "
            f"[{event.get('event_type', 'other')}]"
        )

    if chapter_constraint:
        must_include = chapter_constraint.get("must_include", [])
        must_avoid = chapter_constraint.get("must_avoid", [])
        if must_include:
            lines.append("- Must include: " + "; ".join(str(x) for x in must_include))
        if must_avoid:
            lines.append("- Must avoid: " + "; ".join(str(x) for x in must_avoid))

    if states:
        lines.append("- Character state ledger (current):")
        for state in states[:12]:
            lines.append(
                f"  - {state.get('character', '?')}: status={state.get('status', 'active')}, "
                f"location={state.get('location', 'unknown')}, notes={state.get('notes', '')}"
            )

    risks = master_timeline.get("continuity_risks", [])
    if isinstance(risks, list) and risks:
        lines.append("- Continuity risks: " + "; ".join(str(r) for r in risks[:5]))

    return "\n".join(lines)


def build_character_fate_registry_prompt(
    title: str,
    premise: str,
    genre: str,
    character_list: list[dict],
    chapter_list: list[dict],
    master_timeline: dict | None = None,
    special_instructions: str = "",
) -> list[dict]:
    """
    Character Fate Registry: defines and tracks each character's life/death,
    injury, capture, and narrative state with non-contradiction constraints.
    """
    characters_text = "\n".join(
        f"- {c.get('name', '?')}: role={c.get('role', '')}; arc={c.get('arc', '')}; "
        f"background={c.get('background', '')}"
        for c in character_list
    )
    if not characters_text.strip():
        characters_text = "- No explicit character list provided. Infer only from outline."

    chapter_text = "\n".join(
        f"Chapter {ch.get('number', i + 1)}: {ch.get('title', f'Chapter {i + 1}')} – {ch.get('summary', '')}"
        for i, ch in enumerate(chapter_list)
    )

    timeline_lines = []
    if isinstance(master_timeline, dict):
        for event in master_timeline.get("ledger", [])[:60]:
            if isinstance(event, dict):
                timeline_lines.append(
                    f"- Ch {event.get('chapter')}: {event.get('event', '')} [{event.get('event_type', 'other')}]"
                )
    timeline_text = "\n".join(timeline_lines)

    return render_prompt(
        "character_fate_registry",
        title=title,
        premise=premise,
        genre=genre,
        characters_text=characters_text,
        chapter_text=chapter_text,
        timeline_text=timeline_text,
        special_instructions=special_instructions or "",
    )


def _build_fallback_character_fate_registry(character_list: list[dict], total_chapters: int) -> dict:
    """Deterministic fallback registry if planning fails."""
    total_chapters = max(1, total_chapters)
    safe_characters = character_list or []

    registry = []
    for ch in safe_characters:
        name = str(ch.get("name", "")).strip()
        if not name:
            continue
        registry.append(
            {
                "character": name,
                "current_status": "alive",
                "capture_state": "free",
                "injuries": [],
                "narrative_status": "active",
                "definitive_outcome_required": False,
                "definitive_outcome": "unknown",
                "outcome_locked": False,
                "single_death_rule": True,
                "death_chapter": None,
                "recovery_conditions": [],
                "state_constraints": ["Do not contradict established status transitions."],
                "pivotal_chapters": [1, total_chapters],
            }
        )

    chapter_constraints = []
    for chapter in range(1, total_chapters + 1):
        chapter_constraints.append(
            {
                "chapter": chapter,
                "must_track": ["Character status continuity from prior chapters."],
                "must_not_contradict": ["No dead character appears active without explicit recovery mechanism."],
            }
        )

    return {
        "registry": registry,
        "chapter_constraints": chapter_constraints,
        "conflict_checks": [],
    }


def normalise_character_fate_registry(
    registry_data: dict,
    character_list: list[dict],
    total_chapters: int,
) -> dict:
    """Normalize fate registry output into a stable schema."""
    fallback = _build_fallback_character_fate_registry(character_list, total_chapters)
    if not isinstance(registry_data, dict):
        return fallback

    allowed_statuses = {"alive", "captured", "injured", "deceased", "missing", "recovered", "unknown"}
    allowed_capture = {"free", "captured", "escaped", "unknown"}
    allowed_narrative = {"active", "inactive", "resolved", "deceased"}
    allowed_outcome = {"unknown", "survival", "death", "redemption", "exile", "betrayal"}

    raw_registry = registry_data.get("registry", [])
    if not isinstance(raw_registry, list):
        raw_registry = []

    normalised_registry = []
    seen_names = set()
    for item in raw_registry:
        if not isinstance(item, dict):
            continue
        name = str(item.get("character", "")).strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)

        current_status = str(item.get("current_status", "alive")).strip().lower()
        if current_status not in allowed_statuses:
            current_status = "unknown"

        capture_state = str(item.get("capture_state", "unknown")).strip().lower()
        if capture_state not in allowed_capture:
            capture_state = "unknown"

        narrative_status = str(item.get("narrative_status", "active")).strip().lower()
        if narrative_status not in allowed_narrative:
            narrative_status = "active"

        definitive_outcome = str(item.get("definitive_outcome", "unknown")).strip().lower()
        if definitive_outcome not in allowed_outcome:
            definitive_outcome = "unknown"

        death_chapter_raw = item.get("death_chapter")
        death_chapter = None
        if death_chapter_raw not in (None, ""):
            death_chapter = _coerce_positive_int(death_chapter_raw, 0) or None
            if death_chapter and death_chapter > total_chapters:
                death_chapter = total_chapters

        entry = {
            "character": name,
            "current_status": "deceased" if definitive_outcome == "death" else current_status,
            "capture_state": capture_state,
            "injuries": [str(x) for x in item.get("injuries", []) if str(x).strip()],
            "narrative_status": "deceased" if definitive_outcome == "death" else narrative_status,
            "definitive_outcome_required": bool(item.get("definitive_outcome_required", False)),
            "definitive_outcome": definitive_outcome,
            "outcome_locked": bool(item.get("outcome_locked", False)),
            "single_death_rule": bool(item.get("single_death_rule", True)),
            "death_chapter": death_chapter,
            "recovery_conditions": [str(x) for x in item.get("recovery_conditions", []) if str(x).strip()],
            "state_constraints": [str(x) for x in item.get("state_constraints", []) if str(x).strip()],
            "pivotal_chapters": [
                _coerce_positive_int(ch, 1)
                for ch in item.get("pivotal_chapters", [])
                if _coerce_positive_int(ch, 0) > 0
            ],
        }
        if entry["definitive_outcome"] == "death" and entry["death_chapter"] is None:
            entry["death_chapter"] = max(1, total_chapters - 1)
        normalised_registry.append(entry)

    chapter_constraints = registry_data.get("chapter_constraints", [])
    if not isinstance(chapter_constraints, list):
        chapter_constraints = []
    normalised_chapter_constraints = []
    for idx, item in enumerate(chapter_constraints, start=1):
        if not isinstance(item, dict):
            continue
        chapter = _coerce_positive_int(item.get("chapter"), idx)
        if chapter > total_chapters:
            chapter = total_chapters
        normalised_chapter_constraints.append(
            {
                "chapter": chapter,
                "must_track": [str(x) for x in item.get("must_track", []) if str(x).strip()],
                "must_not_contradict": [str(x) for x in item.get("must_not_contradict", []) if str(x).strip()],
            }
        )

    conflict_checks = registry_data.get("conflict_checks", [])
    if not isinstance(conflict_checks, list):
        conflict_checks = []

    merged = {
        "registry": normalised_registry or fallback["registry"],
        "chapter_constraints": normalised_chapter_constraints or fallback["chapter_constraints"],
        "conflict_checks": [str(x) for x in conflict_checks if str(x).strip()],
    }
    return merged


def plan_character_fate_registry(
    title: str,
    premise: str,
    genre: str,
    character_list: list[dict],
    chapter_list: list[dict],
    master_timeline: dict | None = None,
    special_instructions: str = "",
) -> dict:
    """Run Character Fate Registry planner and return normalized output."""
    total_chapters = max(1, len(chapter_list))
    try:
        raw = call_llm(
            build_character_fate_registry_prompt(
                title=title,
                premise=premise,
                genre=genre,
                character_list=character_list,
                chapter_list=chapter_list,
                master_timeline=master_timeline,
                special_instructions=special_instructions,
            ),
            action="Planning Character Fate Registry",
            json_mode=True,
        )
        parsed = parse_llm_json(raw)
        return normalise_character_fate_registry(parsed, character_list, total_chapters)
    except (RuntimeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Character Fate Registry planner failed, using fallback: %s", exc)
        return _build_fallback_character_fate_registry(character_list, total_chapters)


def get_chapter_fate_context(character_fate_registry: dict, chapter_num: int) -> str:
    """Format chapter-relevant character fate constraints for prompt consumption."""
    if not isinstance(character_fate_registry, dict):
        return ""

    registry = character_fate_registry.get("registry", [])
    if not isinstance(registry, list):
        registry = []

    lines = ["Character Fate Registry output for this chapter:"]
    for entry in registry:
        if not isinstance(entry, dict):
            continue
        pivotal_chapters = entry.get("pivotal_chapters", [])
        chapter_in_scope = False
        if isinstance(pivotal_chapters, list):
            chapter_in_scope = any(_coerce_positive_int(ch, 0) == chapter_num for ch in pivotal_chapters)

        should_include = chapter_in_scope or entry.get("outcome_locked") or entry.get("definitive_outcome_required")
        if should_include:
            lines.append(
                f"- {entry.get('character', '?')}: status={entry.get('current_status', 'unknown')}, "
                f"capture={entry.get('capture_state', 'unknown')}, narrative={entry.get('narrative_status', 'active')}, "
                f"outcome={entry.get('definitive_outcome', 'unknown')}, locked={entry.get('outcome_locked', False)}"
            )
            if entry.get("state_constraints"):
                lines.append(
                    "  - State constraints: " + "; ".join(str(x) for x in entry.get("state_constraints", [])[:4])
                )
            if entry.get("injuries"):
                lines.append("  - Injuries: " + "; ".join(str(x) for x in entry.get("injuries", [])[:4]))
            if entry.get("recovery_conditions"):
                lines.append(
                    "  - Recovery conditions: " + "; ".join(str(x) for x in entry.get("recovery_conditions", [])[:3])
                )

    chapter_constraints = character_fate_registry.get("chapter_constraints", [])
    if isinstance(chapter_constraints, list):
        chapter_constraint = next(
            (
                c for c in chapter_constraints
                if isinstance(c, dict) and _coerce_positive_int(c.get("chapter"), 0) == chapter_num
            ),
            None,
        )
        if chapter_constraint:
            must_track = chapter_constraint.get("must_track", [])
            must_not = chapter_constraint.get("must_not_contradict", [])
            if must_track:
                lines.append("- Must track: " + "; ".join(str(x) for x in must_track[:6]))
            if must_not:
                lines.append("- Must not contradict: " + "; ".join(str(x) for x in must_not[:6]))

    checks = character_fate_registry.get("conflict_checks", [])
    if isinstance(checks, list) and checks:
        lines.append("- Conflict checks: " + "; ".join(str(x) for x in checks[:5]))

    return "\n".join(lines)


def build_character_arc_planner_prompt(
    title: str,
    premise: str,
    genre: str,
    character_list: list[dict],
    chapter_list: list[dict],
    special_instructions: str = "",
) -> list[dict]:
    """
    Character Arc Planner: defines start state, midpoint transformation,
    crisis point, and final moral choice for each primary character.
    """
    characters_text = "\n".join(
        f"- {c.get('name', '?')}: role={c.get('role', '')}; arc={c.get('arc', '')}; background={c.get('background', '')}"
        for c in character_list
    )
    if not characters_text.strip():
        characters_text = "- No explicit characters provided. Infer primary characters conservatively."

    chapters_text = "\n".join(
        f"Chapter {ch.get('number', i + 1)}: {ch.get('title', f'Chapter {i + 1}')} – {ch.get('summary', '')}"
        for i, ch in enumerate(chapter_list)
    )

    return render_prompt(
        "character_arc_planner",
        title=title,
        premise=premise,
        genre=genre,
        characters_text=characters_text,
        chapters_text=chapters_text,
        special_instructions=special_instructions or "",
    )


def _build_fallback_character_arc_plan(
    character_list: list[dict],
    chapter_list: list[dict],
) -> dict:
    """Deterministic fallback character arc plan."""
    total_chapters = max(1, len(chapter_list))
    midpoint_chapter = max(1, min(total_chapters, round((total_chapters + 1) / 2)))
    crisis_chapter = max(1, total_chapters - 1 if total_chapters > 1 else 1)
    final_chapter = total_chapters

    arcs = []
    for idx, character in enumerate(character_list or []):
        name = str(character.get("name", "")).strip()
        if not name:
            continue
        role = str(character.get("role", "")).strip().lower()
        role_class = "primary" if idx < 3 or "protagonist" in role or "antagonist" in role else "secondary"
        if role_class != "primary":
            continue

        arcs.append(
            {
                "character": name,
                "role": role_class,
                "start_state": "Begins with a constrained worldview and unresolved internal tension.",
                "midpoint_transformation": "Confronts disconfirming evidence that shifts priorities.",
                "crisis_point": "Must choose between self-protection and core values.",
                "final_moral_choice": "Makes a definitive ethical choice that resolves the arc.",
                "arc_theme": str(character.get("arc", "Identity under pressure") or "Identity under pressure"),
                "chapter_beats": [
                    {"chapter": 1, "phase": "start", "beat": "Establishes baseline motivations and flaws."},
                    {
                        "chapter": midpoint_chapter,
                        "phase": "midpoint",
                        "beat": "Midpoint shift challenges assumptions and role.",
                    },
                    {
                        "chapter": crisis_chapter,
                        "phase": "crisis",
                        "beat": "Faces hardest internal/external decision.",
                    },
                    {
                        "chapter": final_chapter,
                        "phase": "final",
                        "beat": "Commits to final moral choice and consequence.",
                    },
                ],
                "consistency_rules": [
                    "Arc must move forward each appearance.",
                    "No regression to start state after midpoint without explicit cause.",
                ],
            }
        )

    chapter_constraints = []
    for idx, chapter in enumerate(chapter_list or [{"number": 1}], start=1):
        chapter_num = _coerce_positive_int(chapter.get("number", idx), idx)
        chapter_constraints.append(
            {
                "chapter": chapter_num,
                "must_advance": ["At least one active arc beat or consequence must progress."],
                "must_not_undo": ["Do not reset established character growth without explicit trigger."],
            }
        )

    return {
        "arcs": arcs,
        "chapter_constraints": chapter_constraints,
        "global_arc_risks": [],
    }


def normalise_character_arc_plan(
    arc_data: dict,
    character_list: list[dict],
    chapter_list: list[dict],
) -> dict:
    """Normalize Character Arc Planner output into stable schema."""
    fallback = _build_fallback_character_arc_plan(character_list, chapter_list)
    if not isinstance(arc_data, dict):
        return fallback

    raw_arcs = arc_data.get("arcs", [])
    if not isinstance(raw_arcs, list):
        raw_arcs = []

    normalised_arcs = []
    seen_names = set()
    for item in raw_arcs:
        if not isinstance(item, dict):
            continue
        name = str(item.get("character", "")).strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)

        chapter_beats = item.get("chapter_beats", [])
        if not isinstance(chapter_beats, list):
            chapter_beats = []
        consistency_rules = item.get("consistency_rules", [])
        if not isinstance(consistency_rules, list):
            consistency_rules = []

        normalised_beats = []
        for beat in chapter_beats:
            if not isinstance(beat, dict):
                continue
            normalised_beats.append(
                {
                    "chapter": _coerce_positive_int(beat.get("chapter"), 1),
                    "phase": str(beat.get("phase", "start")),
                    "beat": str(beat.get("beat", "")).strip(),
                }
            )

        normalised_arcs.append(
            {
                "character": name,
                "role": str(item.get("role", "primary") or "primary"),
                "start_state": str(item.get("start_state", "")).strip(),
                "midpoint_transformation": str(item.get("midpoint_transformation", "")).strip(),
                "crisis_point": str(item.get("crisis_point", "")).strip(),
                "final_moral_choice": str(item.get("final_moral_choice", "")).strip(),
                "arc_theme": str(item.get("arc_theme", "")).strip(),
                "chapter_beats": normalised_beats,
                "consistency_rules": [str(x) for x in consistency_rules if str(x).strip()],
            }
        )

    raw_constraints = arc_data.get("chapter_constraints", [])
    if not isinstance(raw_constraints, list):
        raw_constraints = []
    normalised_constraints = []
    for idx, item in enumerate(raw_constraints, start=1):
        if not isinstance(item, dict):
            continue
        normalised_constraints.append(
            {
                "chapter": _coerce_positive_int(item.get("chapter"), idx),
                "must_advance": [str(x) for x in item.get("must_advance", []) if str(x).strip()],
                "must_not_undo": [str(x) for x in item.get("must_not_undo", []) if str(x).strip()],
            }
        )

    global_arc_risks = arc_data.get("global_arc_risks", [])
    if not isinstance(global_arc_risks, list):
        global_arc_risks = []

    return {
        "arcs": normalised_arcs or fallback["arcs"],
        "chapter_constraints": normalised_constraints or fallback["chapter_constraints"],
        "global_arc_risks": [str(x) for x in global_arc_risks if str(x).strip()],
    }


def plan_character_arc_plan(
    title: str,
    premise: str,
    genre: str,
    character_list: list[dict],
    chapter_list: list[dict],
    special_instructions: str = "",
) -> dict:
    """Run Character Arc Planner and return normalized arc plan."""
    try:
        raw = call_llm(
            build_character_arc_planner_prompt(
                title=title,
                premise=premise,
                genre=genre,
                character_list=character_list,
                chapter_list=chapter_list,
                special_instructions=special_instructions,
            ),
            action="Planning Character Arcs",
            json_mode=True,
        )
        parsed = parse_llm_json(raw)
        return normalise_character_arc_plan(parsed, character_list, chapter_list)
    except (RuntimeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Character Arc Planner failed, using fallback: %s", exc)
        return _build_fallback_character_arc_plan(character_list, chapter_list)


def get_chapter_arc_context(character_arc_plan: dict, chapter_num: int) -> str:
    """Format chapter-relevant character arc targets for prompt injection."""
    if not isinstance(character_arc_plan, dict):
        return ""

    arcs = character_arc_plan.get("arcs", [])
    if not isinstance(arcs, list):
        arcs = []
    constraints = character_arc_plan.get("chapter_constraints", [])
    if not isinstance(constraints, list):
        constraints = []

    lines = ["Character Arc Planner output for this chapter:"]
    for arc in arcs:
        if not isinstance(arc, dict):
            continue
        beats = arc.get("chapter_beats", [])
        if not isinstance(beats, list):
            beats = []
        matching_beats = [
            beat for beat in beats
            if isinstance(beat, dict) and _coerce_positive_int(beat.get("chapter"), 0) == chapter_num
        ]
        if matching_beats:
            lines.append(
                f"- {arc.get('character', '?')}: start={arc.get('start_state', '')}; midpoint={arc.get('midpoint_transformation', '')}; "
                f"crisis={arc.get('crisis_point', '')}; final_choice={arc.get('final_moral_choice', '')}"
            )
            for beat in matching_beats[:3]:
                lines.append(f"  - Beat ({beat.get('phase', 'arc')}): {beat.get('beat', '')}")
            rules = arc.get("consistency_rules", [])
            if isinstance(rules, list) and rules:
                lines.append("  - Arc rules: " + "; ".join(str(x) for x in rules[:4]))

    chapter_constraint = next(
        (
            item for item in constraints
            if isinstance(item, dict) and _coerce_positive_int(item.get("chapter"), 0) == chapter_num
        ),
        None,
    )
    if chapter_constraint:
        must_advance = chapter_constraint.get("must_advance", [])
        must_not_undo = chapter_constraint.get("must_not_undo", [])
        if must_advance:
            lines.append("- Must advance: " + "; ".join(str(x) for x in must_advance))
        if must_not_undo:
            lines.append("- Must not undo: " + "; ".join(str(x) for x in must_not_undo))

    risks = character_arc_plan.get("global_arc_risks", [])
    if isinstance(risks, list) and risks:
        lines.append("- Global arc risks: " + "; ".join(str(x) for x in risks[:5]))

    return "\n".join(lines)


def build_antagonist_motivation_prompt(
    title: str,
    premise: str,
    genre: str,
    character_list: list[dict],
    chapter_list: list[dict],
    master_timeline: dict | None = None,
    special_instructions: str = "",
) -> list[dict]:
    """
    Antagonist Motivation Architect: maps antagonist goals, pressure points,
    and chapter-by-chapter escalation logic to preserve coherent opposition.
    """
    characters_text = "\n".join(
        f"- {c.get('name', '?')}: role={c.get('role', '')}; arc={c.get('arc', '')}; background={c.get('background', '')}"
        for c in character_list
    )
    if not characters_text.strip():
        characters_text = "- No explicit characters provided. Infer conservatively from outline."

    chapters_text = "\n".join(
        f"Chapter {ch.get('number', i + 1)}: {ch.get('title', f'Chapter {i + 1}')} – {ch.get('summary', '')}"
        for i, ch in enumerate(chapter_list)
    )

    timeline_lines = []
    if isinstance(master_timeline, dict):
        for event in master_timeline.get("ledger", [])[:60]:
            if isinstance(event, dict):
                timeline_lines.append(
                    f"- Ch {event.get('chapter')}: {event.get('event', '')} [{event.get('event_type', 'other')}]"
                )
    timeline_text = "\n".join(timeline_lines)

    return render_prompt(
        "antagonist_motivation",
        title=title,
        premise=premise,
        genre=genre,
        characters_text=characters_text,
        chapters_text=chapters_text,
        timeline_text=timeline_text,
        special_instructions=special_instructions or "",
    )


def _build_fallback_antagonist_motivation_plan(
    character_list: list[dict],
    chapter_list: list[dict],
) -> dict:
    """Deterministic fallback antagonist motivation plan."""
    total_chapters = max(1, len(chapter_list))

    antagonists = []
    for character in character_list or []:
        name = str(character.get("name", "")).strip()
        if not name:
            continue
        role = str(character.get("role", "")).strip().lower()
        is_antagonist = any(tag in role for tag in ("antagonist", "villain", "rival", "enemy", "opposition"))
        if not is_antagonist:
            continue
        antagonists.append(
            {
                "character": name,
                "motivation_core": "Preserve control in response to a perceived existential threat.",
                "external_goal": "Block protagonist progress toward decisive objective.",
                "internal_need": "Avoid vulnerability and loss of authority.",
                "fear_trigger": "Loss of control or public exposure of weakness.",
                "moral_line": "Will escalate harm strategically but avoids indiscriminate destruction.",
                "pressure_points": ["Public legitimacy", "Trusted lieutenant", "Resource access"],
                "escalation_plan": [
                    {
                        "chapter": 1,
                        "action": "Signals opposition through indirect interference.",
                        "tactic": "Plausible deniability",
                        "motivation_link": "Tests threat level while preserving cover.",
                    },
                    {
                        "chapter": max(1, min(total_chapters, round((total_chapters + 1) / 2))),
                        "action": "Commits to direct pressure after setbacks.",
                        "tactic": "Targeted retaliation",
                        "motivation_link": "Escalates to restore control.",
                    },
                    {
                        "chapter": total_chapters,
                        "action": "Makes final high-risk move aligned with core fear.",
                        "tactic": "All-in confrontation",
                        "motivation_link": "Chooses decisive action over gradual containment.",
                    },
                ],
                "consistency_rules": [
                    "Escalation must track rising pressure; no random reversals.",
                    "Tactics should follow established risk tolerance and moral line.",
                ],
            }
        )

    chapter_constraints = []
    for idx, chapter in enumerate(chapter_list or [{"number": 1}], start=1):
        chapter_num = _coerce_positive_int(chapter.get("number"), idx)
        chapter_constraints.append(
            {
                "chapter": chapter_num,
                "must_show": ["Antagonist pressure should have clear motivation and objective."],
                "must_not_break": ["Do not use antagonist tactics that contradict prior moral line or incentives."],
            }
        )

    return {
        "antagonists": antagonists,
        "chapter_constraints": chapter_constraints,
        "global_risks": [],
    }


def normalise_antagonist_motivation_plan(
    plan_data: dict,
    character_list: list[dict],
    chapter_list: list[dict],
) -> dict:
    """Normalize Antagonist Motivation Architect output into stable schema."""
    fallback = _build_fallback_antagonist_motivation_plan(character_list, chapter_list)
    if not isinstance(plan_data, dict):
        return fallback

    total_chapters = max(1, len(chapter_list))
    raw_antagonists = plan_data.get("antagonists", [])
    if not isinstance(raw_antagonists, list):
        raw_antagonists = []

    normalised_antagonists = []
    seen_names = set()
    for item in raw_antagonists:
        if not isinstance(item, dict):
            continue
        name = str(item.get("character", "")).strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)

        escalation_plan = item.get("escalation_plan", [])
        if not isinstance(escalation_plan, list):
            escalation_plan = []
        normalised_escalation = []
        for step in escalation_plan:
            if not isinstance(step, dict):
                continue
            chapter = _coerce_positive_int(step.get("chapter"), 1)
            chapter = min(chapter, total_chapters)
            normalised_escalation.append(
                {
                    "chapter": chapter,
                    "action": str(step.get("action", "")).strip(),
                    "tactic": str(step.get("tactic", "")).strip(),
                    "motivation_link": str(step.get("motivation_link", "")).strip(),
                }
            )

        normalised_antagonists.append(
            {
                "character": name,
                "motivation_core": str(item.get("motivation_core", "")).strip(),
                "external_goal": str(item.get("external_goal", "")).strip(),
                "internal_need": str(item.get("internal_need", "")).strip(),
                "fear_trigger": str(item.get("fear_trigger", "")).strip(),
                "moral_line": str(item.get("moral_line", "")).strip(),
                "pressure_points": [str(x) for x in item.get("pressure_points", []) if str(x).strip()],
                "escalation_plan": normalised_escalation,
                "consistency_rules": [str(x) for x in item.get("consistency_rules", []) if str(x).strip()],
            }
        )

    raw_constraints = plan_data.get("chapter_constraints", [])
    if not isinstance(raw_constraints, list):
        raw_constraints = []
    normalised_constraints = []
    for idx, item in enumerate(raw_constraints, start=1):
        if not isinstance(item, dict):
            continue
        chapter = _coerce_positive_int(item.get("chapter"), idx)
        chapter = min(chapter, total_chapters)
        normalised_constraints.append(
            {
                "chapter": chapter,
                "must_show": [str(x) for x in item.get("must_show", []) if str(x).strip()],
                "must_not_break": [str(x) for x in item.get("must_not_break", []) if str(x).strip()],
            }
        )

    global_risks = plan_data.get("global_risks", [])
    if not isinstance(global_risks, list):
        global_risks = []

    return {
        "antagonists": normalised_antagonists or fallback["antagonists"],
        "chapter_constraints": normalised_constraints or fallback["chapter_constraints"],
        "global_risks": [str(x) for x in global_risks if str(x).strip()],
    }


def plan_antagonist_motivation_plan(
    title: str,
    premise: str,
    genre: str,
    character_list: list[dict],
    chapter_list: list[dict],
    master_timeline: dict | None = None,
    special_instructions: str = "",
) -> dict:
    """Run Antagonist Motivation Architect and return normalized plan."""
    try:
        raw = call_llm(
            build_antagonist_motivation_prompt(
                title=title,
                premise=premise,
                genre=genre,
                character_list=character_list,
                chapter_list=chapter_list,
                master_timeline=master_timeline,
                special_instructions=special_instructions,
            ),
            action="Planning Antagonist Motivation",
            json_mode=True,
        )
        parsed = parse_llm_json(raw)
        return normalise_antagonist_motivation_plan(parsed, character_list, chapter_list)
    except (RuntimeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Antagonist Motivation Architect failed, using fallback: %s", exc)
        return _build_fallback_antagonist_motivation_plan(character_list, chapter_list)


def get_chapter_antagonist_context(antagonist_motivation_plan: dict, chapter_num: int) -> str:
    """Format chapter-specific antagonist motivation and escalation constraints."""
    if not isinstance(antagonist_motivation_plan, dict):
        return ""

    antagonists = antagonist_motivation_plan.get("antagonists", [])
    if not isinstance(antagonists, list):
        antagonists = []
    constraints = antagonist_motivation_plan.get("chapter_constraints", [])
    if not isinstance(constraints, list):
        constraints = []

    lines = ["Antagonist Motivation Architect output for this chapter:"]
    for antagonist in antagonists:
        if not isinstance(antagonist, dict):
            continue
        escalation_plan = antagonist.get("escalation_plan", [])
        if not isinstance(escalation_plan, list):
            escalation_plan = []
        matching_steps = [
            step for step in escalation_plan
            if isinstance(step, dict) and _coerce_positive_int(step.get("chapter"), 0) == chapter_num
        ]
        if matching_steps:
            lines.append(
                f"- {antagonist.get('character', '?')}: core={antagonist.get('motivation_core', '')}; "
                f"goal={antagonist.get('external_goal', '')}; fear={antagonist.get('fear_trigger', '')}; "
                f"moral_line={antagonist.get('moral_line', '')}"
            )
            pressure_points = antagonist.get("pressure_points", [])
            if isinstance(pressure_points, list) and pressure_points:
                lines.append("  - Pressure points: " + "; ".join(str(x) for x in pressure_points[:4]))
            for step in matching_steps[:3]:
                lines.append(
                    f"  - Escalation: action={step.get('action', '')}; tactic={step.get('tactic', '')}; "
                    f"motivation_link={step.get('motivation_link', '')}"
                )
            rules = antagonist.get("consistency_rules", [])
            if isinstance(rules, list) and rules:
                lines.append("  - Consistency rules: " + "; ".join(str(x) for x in rules[:4]))

    chapter_constraint = next(
        (
            item for item in constraints
            if isinstance(item, dict) and _coerce_positive_int(item.get("chapter"), 0) == chapter_num
        ),
        None,
    )
    if chapter_constraint:
        must_show = chapter_constraint.get("must_show", [])
        must_not_break = chapter_constraint.get("must_not_break", [])
        if must_show:
            lines.append("- Must show: " + "; ".join(str(x) for x in must_show[:6]))
        if must_not_break:
            lines.append("- Must not break: " + "; ".join(str(x) for x in must_not_break[:6]))

    risks = antagonist_motivation_plan.get("global_risks", [])
    if isinstance(risks, list) and risks:
        lines.append("- Global motivation risks: " + "; ".join(str(x) for x in risks[:5]))

    return "\n".join(lines)


def build_technology_rules_prompt(
    title: str,
    premise: str,
    genre: str,
    chapter_list: list[dict],
    special_instructions: str = "",
) -> list[dict]:
    """
    Technology Rules Designer: defines fixed operational limits for fictional
    technology to avoid omnipotent behavior and continuity drift.
    """
    chapters_text = "\n".join(
        f"Chapter {ch.get('number', i + 1)}: {ch.get('title', f'Chapter {i + 1}')} – {ch.get('summary', '')}"
        for i, ch in enumerate(chapter_list)
    )

    return render_prompt(
        "technology_rules",
        title=title,
        premise=premise,
        genre=genre,
        chapters_text=chapters_text,
        special_instructions=special_instructions or "",
    )


def _build_fallback_technology_rules(chapter_list: list[dict]) -> dict:
    """Deterministic fallback technology rules when planner output is unavailable."""
    total_chapters = max(1, len(chapter_list))
    systems = [
        {
            "name": "Primary Surveillance Grid",
            "purpose": "Detect movements and anomalies across controlled zones.",
            "latency_ms": 1800,
            "detection_methods": ["Pattern analysis", "Fixed checkpoint scans"],
            "detection_blind_spots": ["Sensor dead angles", "Heavy weather or smoke"],
            "resource_constraints": ["Finite compute budget", "Nightly recalibration windows"],
            "operational_limits": ["Cannot track all targets in real time", "False positives under crowd density"],
            "failure_modes": ["Overload under coordinated decoys", "Delayed alert propagation"],
            "countermeasures": ["Manual review queue", "Tiered alert escalation"],
            "forbidden_capabilities": ["Instant omniscient tracking", "Retroactive perfect reconstruction"],
        }
    ]

    chapter_constraints = []
    for idx, chapter in enumerate(chapter_list or [{"number": 1}], start=1):
        chapter_num = _coerce_positive_int(chapter.get("number"), idx)
        chapter_constraints.append(
            {
                "chapter": chapter_num,
                "must_respect": [
                    "Technology outcomes must follow stated latency and operational limits.",
                ],
                "must_not_allow": [
                    "Do not grant instant detection or infinite processing without explicit setup.",
                ],
            }
        )

    return {
        "systems": systems,
        "global_constraints": [
            "Every tech action has delay, uncertainty, or resource cost.",
            "Capabilities cannot exceed declared operational limits.",
        ],
        "chapter_constraints": chapter_constraints,
        "continuity_risks": [],
    }


def normalise_technology_rules(technology_data: dict, chapter_list: list[dict]) -> dict:
    """Normalize Technology Rules Designer output into stable schema."""
    fallback = _build_fallback_technology_rules(chapter_list)
    if not isinstance(technology_data, dict):
        return fallback

    raw_systems = technology_data.get("systems", [])
    if not isinstance(raw_systems, list):
        raw_systems = []

    normalised_systems = []
    seen_names = set()
    for item in raw_systems:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        normalised_systems.append(
            {
                "name": name,
                "purpose": str(item.get("purpose", "")).strip(),
                "latency_ms": _coerce_positive_int(item.get("latency_ms"), 1000),
                "detection_methods": [str(x) for x in item.get("detection_methods", []) if str(x).strip()],
                "detection_blind_spots": [str(x) for x in item.get("detection_blind_spots", []) if str(x).strip()],
                "resource_constraints": [str(x) for x in item.get("resource_constraints", []) if str(x).strip()],
                "operational_limits": [str(x) for x in item.get("operational_limits", []) if str(x).strip()],
                "failure_modes": [str(x) for x in item.get("failure_modes", []) if str(x).strip()],
                "countermeasures": [str(x) for x in item.get("countermeasures", []) if str(x).strip()],
                "forbidden_capabilities": [str(x) for x in item.get("forbidden_capabilities", []) if str(x).strip()],
            }
        )

    global_constraints = technology_data.get("global_constraints", [])
    if not isinstance(global_constraints, list):
        global_constraints = []

    raw_constraints = technology_data.get("chapter_constraints", [])
    if not isinstance(raw_constraints, list):
        raw_constraints = []
    total_chapters = max(1, len(chapter_list))
    normalised_constraints = []
    for idx, item in enumerate(raw_constraints, start=1):
        if not isinstance(item, dict):
            continue
        chapter = _coerce_positive_int(item.get("chapter"), idx)
        chapter = min(chapter, total_chapters)
        normalised_constraints.append(
            {
                "chapter": chapter,
                "must_respect": [str(x) for x in item.get("must_respect", []) if str(x).strip()],
                "must_not_allow": [str(x) for x in item.get("must_not_allow", []) if str(x).strip()],
            }
        )

    continuity_risks = technology_data.get("continuity_risks", [])
    if not isinstance(continuity_risks, list):
        continuity_risks = []

    return {
        "systems": normalised_systems or fallback["systems"],
        "global_constraints": [str(x) for x in global_constraints if str(x).strip()] or fallback["global_constraints"],
        "chapter_constraints": normalised_constraints or fallback["chapter_constraints"],
        "continuity_risks": [str(x) for x in continuity_risks if str(x).strip()],
    }


def plan_technology_rules(
    title: str,
    premise: str,
    genre: str,
    chapter_list: list[dict],
    special_instructions: str = "",
) -> dict:
    """Run Technology Rules Designer and return normalized output."""
    try:
        raw = call_llm(
            build_technology_rules_prompt(
                title=title,
                premise=premise,
                genre=genre,
                chapter_list=chapter_list,
                special_instructions=special_instructions,
            ),
            action="Planning Technology Rules",
            json_mode=True,
        )
        parsed = parse_llm_json(raw)
        return normalise_technology_rules(parsed, chapter_list)
    except (RuntimeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Technology Rules Designer failed, using fallback: %s", exc)
        return _build_fallback_technology_rules(chapter_list)


def get_chapter_technology_context(technology_rules: dict, chapter_num: int) -> str:
    """Format chapter-relevant technology constraints for prompt injection."""
    if not isinstance(technology_rules, dict):
        return ""

    systems = technology_rules.get("systems", [])
    if not isinstance(systems, list):
        systems = []

    lines = ["Technology Rules Designer output for this chapter:"]
    for system in systems[:6]:
        if not isinstance(system, dict):
            continue
        lines.append(
            f"- {system.get('name', '?')}: latency={system.get('latency_ms', 0)}ms; purpose={system.get('purpose', '')}"
        )
        for key, label in (
            ("operational_limits", "Operational limits"),
            ("resource_constraints", "Resource constraints"),
            ("detection_blind_spots", "Detection blind spots"),
            ("failure_modes", "Failure modes"),
            ("forbidden_capabilities", "Forbidden capabilities"),
        ):
            values = system.get(key, [])
            if isinstance(values, list) and values:
                lines.append(f"  - {label}: " + "; ".join(str(x) for x in values[:4]))

    global_constraints = technology_rules.get("global_constraints", [])
    if isinstance(global_constraints, list) and global_constraints:
        lines.append("- Global constraints: " + "; ".join(str(x) for x in global_constraints[:6]))

    chapter_constraints = technology_rules.get("chapter_constraints", [])
    if isinstance(chapter_constraints, list):
        chapter_constraint = next(
            (
                item for item in chapter_constraints
                if isinstance(item, dict) and _coerce_positive_int(item.get("chapter"), 0) == chapter_num
            ),
            None,
        )
        if chapter_constraint:
            must_respect = chapter_constraint.get("must_respect", [])
            must_not_allow = chapter_constraint.get("must_not_allow", [])
            if must_respect:
                lines.append("- Must respect: " + "; ".join(str(x) for x in must_respect[:6]))
            if must_not_allow:
                lines.append("- Must not allow: " + "; ".join(str(x) for x in must_not_allow[:6]))

    risks = technology_rules.get("continuity_risks", [])
    if isinstance(risks, list) and risks:
        lines.append("- Technology continuity risks: " + "; ".join(str(x) for x in risks[:5]))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Theme Reinforcement Planner
# ---------------------------------------------------------------------------

def build_theme_reinforcement_prompt(
    title: str,
    premise: str,
    genre: str,
    chapter_list: list[dict],
    special_instructions: str = "",
) -> list[dict]:
    chapters_text = "\n".join(
        f"Chapter {c.get('number', i+1)}: {c.get('title', '')} – {c.get('summary', '')}"
        for i, c in enumerate(chapter_list)
    )
    return render_prompt(
        "theme_reinforcement",
        title=title,
        premise=premise,
        genre=genre,
        chapters_text=chapters_text,
        special_instructions=special_instructions or "",
    )


def _build_fallback_theme_reinforcement(chapter_list: list[dict]) -> dict:
    fallback_themes = [
        {
            "name": "Identity Under Pressure",
            "description": "How characters maintain or lose their sense of self under systemic pressure.",
            "motifs": ["mirrors", "names", "documents"],
            "pillar_moments": ["Inciting incident", "Midpoint crisis", "Final moral choice"],
            "chapter_appearances": [
                {"chapter": c.get("number", i + 1), "role": "background", "guidance": "Show character making a small compromise."}
                for i, c in enumerate(chapter_list)
            ],
        },
        {
            "name": "Moral Compromise",
            "description": "The cost of choosing safety over principle.",
            "motifs": ["closed doors", "silence", "small betrayals"],
            "pillar_moments": ["First compromise", "Point of no return", "Reckoning"],
            "chapter_appearances": [
                {"chapter": c.get("number", i + 1), "role": "background", "guidance": "Show institutional pressure shaping a decision."}
                for i, c in enumerate(chapter_list)
            ],
        },
    ]
    chapter_constraints = [
        {
            "chapter": c.get("number", i + 1),
            "themes_present": ["Identity Under Pressure"],
            "thematic_guidance": "Reinforce the protagonist's internal conflict quietly.",
        }
        for i, c in enumerate(chapter_list)
    ]
    return {
        "themes": fallback_themes,
        "global_thematic_arcs": [
            "Individual identity erodes under systemic control.",
            "Moral compromise accumulates until a breaking point forces reckoning.",
        ],
        "chapter_constraints": chapter_constraints,
        "continuity_risks": [
            "Theme abandoned mid-story without resolution.",
            "Motifs introduced but never paid off.",
        ],
    }


def normalise_theme_reinforcement(theme_data: dict, chapter_list: list[dict]) -> dict:
    fallback = _build_fallback_theme_reinforcement(chapter_list)
    if not isinstance(theme_data, dict):
        return fallback

    themes = theme_data.get("themes", [])
    if not isinstance(themes, list) or len(themes) == 0:
        themes = fallback["themes"]
    else:
        valid = []
        for t in themes:
            if not isinstance(t, dict):
                continue
            valid.append({
                "name": str(t.get("name", "Theme")),
                "description": str(t.get("description", "")),
                "motifs": t.get("motifs", []) if isinstance(t.get("motifs"), list) else [],
                "pillar_moments": t.get("pillar_moments", []) if isinstance(t.get("pillar_moments"), list) else [],
                "chapter_appearances": t.get("chapter_appearances", []) if isinstance(t.get("chapter_appearances"), list) else [],
            })
        themes = valid if valid else fallback["themes"]

    global_arcs = theme_data.get("global_thematic_arcs", [])
    if not isinstance(global_arcs, list):
        global_arcs = fallback["global_thematic_arcs"]

    chapter_constraints = theme_data.get("chapter_constraints", [])
    if not isinstance(chapter_constraints, list) or len(chapter_constraints) == 0:
        chapter_constraints = fallback["chapter_constraints"]
    else:
        valid_cc = []
        for cc in chapter_constraints:
            if not isinstance(cc, dict):
                continue
            try:
                ch_num = int(cc.get("chapter", 0))
            except (TypeError, ValueError):
                ch_num = 0
            valid_cc.append({
                "chapter": ch_num,
                "themes_present": cc.get("themes_present", []) if isinstance(cc.get("themes_present"), list) else [],
                "thematic_guidance": str(cc.get("thematic_guidance", "")),
            })
        chapter_constraints = valid_cc if valid_cc else fallback["chapter_constraints"]

    continuity_risks = theme_data.get("continuity_risks", [])
    if not isinstance(continuity_risks, list):
        continuity_risks = fallback["continuity_risks"]

    return {
        "themes": themes,
        "global_thematic_arcs": global_arcs,
        "chapter_constraints": chapter_constraints,
        "continuity_risks": continuity_risks,
    }


def plan_theme_reinforcement(
    title: str,
    premise: str,
    genre: str,
    chapter_list: list[dict],
    special_instructions: str = "",
) -> dict:
    try:
        raw = call_llm(
            build_theme_reinforcement_prompt(
                title, premise, genre, chapter_list, special_instructions
            ),
            action="Planning Theme Reinforcement",
            json_mode=True,
        )
        parsed = parse_llm_json(raw)
        return normalise_theme_reinforcement(parsed, chapter_list)
    except Exception:
        return _build_fallback_theme_reinforcement(chapter_list)


def get_chapter_theme_context(theme_reinforcement: dict, chapter_num: int) -> str:
    if not isinstance(theme_reinforcement, dict):
        return ""

    lines = ["Theme Reinforcement Planner – Chapter guidance:"]

    themes = theme_reinforcement.get("themes", [])
    for theme in themes[:4]:
        if not isinstance(theme, dict):
            continue
        name = theme.get("name", "")
        desc = theme.get("description", "")
        if name:
            lines.append(f"- Theme '{name}': {desc}")
        appearances = theme.get("chapter_appearances", [])
        for ap in appearances:
            if not isinstance(ap, dict):
                continue
            try:
                if int(ap.get("chapter", -1)) == chapter_num:
                    role = ap.get("role", "")
                    guidance = ap.get("guidance", "")
                    lines.append(f"  ▸ Role in this chapter: {role}. {guidance}")
                    break
            except (TypeError, ValueError):
                continue

    global_arcs = theme_reinforcement.get("global_thematic_arcs", [])
    if isinstance(global_arcs, list) and global_arcs:
        lines.append("- Global thematic arcs: " + "; ".join(str(a) for a in global_arcs[:3]))

    chapter_constraints = theme_reinforcement.get("chapter_constraints", [])
    for cc in chapter_constraints:
        if not isinstance(cc, dict):
            continue
        try:
            if int(cc.get("chapter", -1)) == chapter_num:
                themes_present = cc.get("themes_present", [])
                if isinstance(themes_present, list) and themes_present:
                    lines.append("- Themes active this chapter: " + ", ".join(str(t) for t in themes_present))
                thematic_guidance = cc.get("thematic_guidance", "")
                if thematic_guidance:
                    lines.append(f"- Thematic guidance: {thematic_guidance}")
                break
        except (TypeError, ValueError):
            continue

    risks = theme_reinforcement.get("continuity_risks", [])
    if isinstance(risks, list) and risks:
        lines.append("- Thematic continuity risks: " + "; ".join(str(r) for r in risks[:3]))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Continuity Gatekeeper
# ---------------------------------------------------------------------------

def build_continuity_gatekeeper_prompt(
    chapter_num: int,
    chapter_title: str,
    chapter_summary: str,
    previous_summaries: str,
    chapter_timeline_context: str = "",
    chapter_fate_context: str = "",
    chapter_arc_context: str = "",
    character_state_log: str = "",
) -> list[dict]:
    """
    Continuity Gatekeeper: validates the upcoming chapter against the master
    timeline and character state registry. Produces a validation brief listing
    forbidden scenarios, locked events, and required character states.
    Runs immediately before the Draft agent for every chapter.
    """
    return render_prompt(
        "continuity_gatekeeper",
        chapter_num=chapter_num,
        chapter_title=chapter_title,
        chapter_summary=chapter_summary,
        previous_summaries=previous_summaries or "",
        chapter_timeline_context=chapter_timeline_context or "",
        chapter_fate_context=chapter_fate_context or "",
        chapter_arc_context=chapter_arc_context or "",
        character_state_log=character_state_log or "",
    )


def run_continuity_gatekeeper(
    chapter_num: int,
    chapter_title: str,
    chapter_summary: str,
    previous_summaries: str,
    chapter_timeline_context: str = "",
    chapter_fate_context: str = "",
    chapter_arc_context: str = "",
    character_state_log: str = "",
) -> str:
    """
    Run the Continuity Gatekeeper LLM pass and return a validation brief string.
    Falls back to an empty string on error so generation is never blocked.
    """
    try:
        return call_llm(
            build_continuity_gatekeeper_prompt(
                chapter_num=chapter_num,
                chapter_title=chapter_title,
                chapter_summary=chapter_summary,
                previous_summaries=previous_summaries,
                chapter_timeline_context=chapter_timeline_context,
                chapter_fate_context=chapter_fate_context,
                chapter_arc_context=chapter_arc_context,
                character_state_log=character_state_log,
            ),
            action=f"Running Continuity Gatekeeper for Chapter {chapter_num}",
        )
    except Exception:
        return ""


def build_title_prompt(premise: str, genre: str) -> list[dict]:
    return render_prompt("title", premise=premise, genre=genre)


def build_outline_prompt(
    premise: str,
    genre: str,
    chapters: int,
    word_count: int,
    special_events: str,
    special_instructions: str,
) -> list[dict]:
    return render_prompt(
        "outline",
        premise=premise,
        genre=genre,
        chapters=chapters,
        word_count=f"{word_count:,}",
        special_events=special_events or "",
        special_instructions=special_instructions or "",
    )


def build_characters_prompt(
    premise: str, genre: str, outline_text: str
) -> list[dict]:
    return render_prompt("characters", premise=premise, genre=genre, outline_text=outline_text)


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
    chapter_architecture_context: str = "",
    chapter_timeline_context: str = "",
    chapter_fate_context: str = "",
    chapter_arc_context: str = "",
    chapter_antagonist_context: str = "",
    chapter_technology_context: str = "",
    chapter_theme_context: str = "",
    gatekeeper_brief: str = "",
    compression_guidance: str = "",
) -> list[dict]:
    return render_prompt(
        "chapter_draft",
        title=title,
        genre=genre,
        premise=premise,
        chapter_num=chapter_num,
        chapter_title=chapter_title,
        chapter_summary=chapter_summary,
        characters_text=characters_text,
        previous_summaries=previous_summaries or "",
        target_words=f"{target_words:,}",
        special_instructions=special_instructions or "",
        chapter_architecture_context=chapter_architecture_context or "",
        chapter_timeline_context=chapter_timeline_context or "",
        chapter_fate_context=chapter_fate_context or "",
        chapter_arc_context=chapter_arc_context or "",
        chapter_antagonist_context=chapter_antagonist_context or "",
        chapter_technology_context=chapter_technology_context or "",
        chapter_theme_context=chapter_theme_context or "",
        gatekeeper_brief=gatekeeper_brief or "",
        compression_guidance=compression_guidance or "",
        forbidden_words=", ".join(_FORBIDDEN_WORDS),
    )


def build_dialog_agent_prompt(chapter_text: str, chapter_num: int, title: str) -> list[dict]:
    """
    Dialog agent: refines all dialogue in the chapter for naturalism, voice
    distinction, and subtext.  Returns only the revised chapter text.
    """
    return render_prompt("dialog_agent", title=title, chapter_num=chapter_num, chapter_text=chapter_text)


def build_scene_agent_prompt(chapter_text: str, chapter_num: int, title: str) -> list[dict]:
    """
    Scene agent: ensures every scene follows the Goal → Obstacle → Outcome →
    New Problem pattern.  Returns only the revised chapter text.
    """
    return render_prompt("scene_agent", title=title, chapter_num=chapter_num, chapter_text=chapter_text)


def build_structure_agent_prompt(
    chapter_text: str,
    chapter_num: int,
    total_chapters: int,
    outline_summary: str,
    chapter_architecture_context: str = "",
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

    return render_prompt(
        "structure_agent",
        chapter_num=chapter_num,
        total_chapters=total_chapters,
        phase_hint=phase_hint,
        outline_summary=outline_summary,
        chapter_architecture_context=chapter_architecture_context or "",
        chapter_text=chapter_text,
    )


def build_character_agent_prompt(
    chapter_text: str,
    characters_text: str,
    chapter_num: int,
    title: str,
    chapter_fate_context: str = "",
    chapter_arc_context: str = "",
    chapter_antagonist_context: str = "",
) -> list[dict]:
    """
    Character agent: checks and deepens character arcs and consistency.
    Returns only the revised chapter text.
    """
    return render_prompt(
        "character_agent",
        title=title,
        chapter_num=chapter_num,
        characters_text=characters_text,
        chapter_fate_context=chapter_fate_context or "",
        chapter_arc_context=chapter_arc_context or "",
        chapter_antagonist_context=chapter_antagonist_context or "",
        chapter_text=chapter_text,
    )


def build_character_thread_tracker_prompt(
    chapter_text: str,
    characters_text: str,
    chapter_num: int,
    title: str,
    chapter_arc_context: str = "",
) -> list[dict]:
    """
    Character Thread Tracker: ensures every named character who appears in this
    chapter receives meaningful forward movement in their arc, relationship, or
    situation. Flags and repairs any character who is dropped, sidelined without
    purpose, or left in narrative stasis.
    Returns only the revised chapter text.
    """
    return render_prompt(
        "character_thread_tracker",
        title=title,
        chapter_num=chapter_num,
        characters_text=characters_text,
        chapter_arc_context=chapter_arc_context or "",
        chapter_text=chapter_text,
    )


def build_context_analyzer_prompt(
    chapter_text: str,
    previous_summaries: str,
    chapter_num: int,
    title: str,
    chapter_timeline_context: str = "",
    chapter_technology_context: str = "",
    chapter_theme_context: str = "",
    gatekeeper_brief: str = "",
) -> list[dict]:
    """
    Context analyzer: verifies world-building details are consistent with
    what has already been established in prior chapters.
    Returns only the revised chapter text.
    """
    return render_prompt(
        "context_analyzer",
        title=title,
        chapter_num=chapter_num,
        previous_summaries=previous_summaries or "",
        chapter_timeline_context=chapter_timeline_context or "",
        chapter_technology_context=chapter_technology_context or "",
        chapter_theme_context=chapter_theme_context or "",
        gatekeeper_brief=gatekeeper_brief or "",
        chapter_text=chapter_text,
    )


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
    return render_prompt("synthesizer", title=title, genre=genre, chapter_num=chapter_num, chapter_text=chapter_text)


def build_quality_controller_prompt(chapter_text: str, chapter_num: int, title: str) -> list[dict]:
    """
    Quality controller: assesses the chapter for reader engagement, pacing,
    and narrative flow, then applies targeted improvements.
    Returns only the revised chapter text.
    """
    return render_prompt("quality_controller", title=title, chapter_num=chapter_num, chapter_text=chapter_text)


def build_editing_agent_prompt(chapter_text: str, chapter_summary: str, chapter_num: int, title: str) -> list[dict]:
    """
    Editing agent: refines draft for plot holes, pacing, and character
    consistency.  Returns only the revised chapter text.
    """
    return render_prompt(
        "editing_agent",
        title=title,
        chapter_num=chapter_num,
        chapter_summary=chapter_summary,
        chapter_text=chapter_text,
    )


def build_narrative_redundancy_detector_prompt(
    chapter_text: str,
    previous_summaries: str,
    chapter_summary: str,
    chapter_num: int,
    title: str,
) -> list[dict]:
    """
    Narrative Redundancy Detector: removes repeated operations, duplicate
    sacrificial beats, and over-familiar scene logic so each chapter earns its
    place with distinct narrative movement. Returns only the revised chapter.
    """
    return render_prompt(
        "narrative_redundancy_detector",
        title=title,
        chapter_num=chapter_num,
        chapter_summary=chapter_summary,
        previous_summaries=previous_summaries or "",
        chapter_text=chapter_text,
    )


def build_operational_distinctiveness_prompt(
    chapter_text: str,
    previous_summaries: str,
    chapter_summary: str,
    chapter_num: int,
    title: str,
) -> list[dict]:
    """
    Operational Distinctiveness Agent: ensures each major operation in this chapter
    differs in strategy, environment, and stakes from operations in prior chapters.
    Returns only the revised chapter text.
    """
    return render_prompt(
        "operational_distinctiveness",
        title=title,
        chapter_num=chapter_num,
        chapter_summary=chapter_summary,
        previous_summaries=previous_summaries or "",
        chapter_text=chapter_text,
    )


def build_polish_agent_prompt(chapter_text: str, chapter_num: int, title: str, genre: str) -> list[dict]:
    """
    Polish agent: elevates grammar, style, and vivid language.
    Returns only the polished chapter text.
    """
    return render_prompt("polish_agent", title=title, genre=genre, chapter_num=chapter_num, chapter_text=chapter_text)


def build_anti_llm_agent_prompt(chapter_text: str, chapter_num: int, title: str) -> list[dict]:
    """
    Anti-LLM agent: dedicated pass to strip robotic language patterns, overused
    phrases, and other LLM hallmarks.  Returns only the revised chapter text.

    Note: ``_FORBIDDEN_WORDS`` contains overused individual words that are easy
    to track and reference throughout the codebase.  The robotic *transition
    phrases* listed in the prompt below are multi-word patterns that only make
    sense as inline prose instructions rather than a simple word list.
    """
    return render_prompt(
        "anti_llm_agent",
        title=title,
        chapter_num=chapter_num,
        chapter_text=chapter_text,
        forbidden_words=", ".join(_FORBIDDEN_WORDS),
    )


def build_story_momentum_tracker_prompt(
    chapter_text: str,
    previous_summaries: str,
    chapter_num: int,
    title: str,
    total_chapters: int,
) -> list[dict]:
    """
    Story Momentum Tracker: verifies that the stakes in this chapter are higher
    than those in previous chapters and that the narrative is escalating toward
    the climax. Revises the chapter to raise urgency, deepen consequence, or
    tighten tension wherever the momentum has stalled or regressed.
    Returns only the revised chapter text.
    """
    position_pct = (chapter_num / total_chapters * 100) if total_chapters > 0 else 50
    if position_pct <= 25:
        escalation_target = "establish foundational threat and personal stakes"
    elif position_pct <= 50:
        escalation_target = "deepen the cost of failure and raise the personal price"
    elif position_pct <= 75:
        escalation_target = "force irreversible decisions and close off safe options"
    else:
        escalation_target = "push stakes to maximum – survival, identity, or irreversible loss"

    return render_prompt(
        "story_momentum_tracker",
        title=title,
        chapter_num=chapter_num,
        total_chapters=total_chapters,
        escalation_target=escalation_target,
        previous_summaries=previous_summaries or "",
        chapter_text=chapter_text,
    )


def build_chapter_summary_prompt(chapter_text: str, chapter_num: int) -> list[dict]:
    return render_prompt("chapter_summary", chapter_num=chapter_num, chapter_text=chapter_text)


def build_per_chapter_compression_check_prompt(
    chapter_num: int,
    chapter_summary: str,
    previous_summaries: str,
    title: str,
) -> list[dict]:
    """
    Per-chapter compression check: analyzes the just-completed chapter against
    previous chapters to identify redundancy patterns. Returns guidance for the
    NEXT chapter to avoid repeating similar operations, emotional beats, or
    structural patterns.
    """
    return render_prompt(
        "per_chapter_compression_check",
        title=title,
        chapter_num=chapter_num,
        chapter_summary=chapter_summary,
        previous_summaries=previous_summaries,
    )


def run_per_chapter_compression_check(
    chapter_num: int,
    chapter_summary: str,
    previous_summaries: str,
    title: str,
) -> str:
    """
    Run the per-chapter compression check and return guidance for the next chapter.
    Falls back to empty string on error so generation is never blocked.
    """
    # Skip for chapter 1 - no previous chapters to compare against
    if chapter_num <= 1 or not previous_summaries.strip():
        return ""
    try:
        return call_llm(
            build_per_chapter_compression_check_prompt(
                chapter_num=chapter_num,
                chapter_summary=chapter_summary,
                previous_summaries=previous_summaries,
                title=title,
            ),
            action=f"Running Per-Chapter Compression Check for Chapter {chapter_num}"
        )
    except Exception:
        return ""


def build_character_state_updater_prompt(
    chapter_text: str,
    chapter_summary: str,
    characters_text: str,
    chapter_num: int,
    title: str,
) -> list[dict]:
    """
    Character State Updater: reads the completed chapter and its summary, then
    produces a concise log entry recording each named character's current state
    (alive, dead, captured, injured, escaped, psychologically changed) as it
    stands at the end of this chapter. This log is fed to the Continuity
    Gatekeeper before every subsequent chapter to prevent contradictions.
    Returns only the character state log text – not chapter prose.
    """
    return render_prompt(
        "character_state_updater",
        title=title,
        chapter_num=chapter_num,
        characters_text=characters_text,
        chapter_summary=chapter_summary,
        chapter_text=chapter_text,
    )


def run_character_state_updater(
    chapter_text: str,
    chapter_summary: str,
    characters_text: str,
    chapter_num: int,
    title: str,
) -> str:
    """
    Run the Character State Updater LLM pass and return the state log string.
    Falls back to an empty string on error so generation is never blocked.
    """
    try:
        return call_llm(
            build_character_state_updater_prompt(
                chapter_text=chapter_text,
                chapter_summary=chapter_summary,
                characters_text=characters_text,
                chapter_num=chapter_num,
                title=title,
            ),
            action=f"Running Character State Updater for Chapter {chapter_num}"
        )
    except Exception:
        return ""


def build_chapter_revision_prompt(
    chapter_text: str,
    chapter_num: int,
    title: str,
    chapter_outline_summary: str,
    revision_instructions: str,
    chapter_architecture_context: str = "",
    chapter_timeline_context: str = "",
    chapter_fate_context: str = "",
    chapter_arc_context: str = "",
    chapter_antagonist_context: str = "",
    chapter_technology_context: str = "",
    chapter_theme_context: str = "",
    gatekeeper_brief: str = "",
) -> list[dict]:
    """
    Revision prompt for incorporating editor instructions into an already-generated
    chapter before running the full agent pipeline again.
    """
    return render_prompt(
        "chapter_revision",
        title=title,
        chapter_num=chapter_num,
        chapter_outline_summary=chapter_outline_summary,
        revision_instructions=revision_instructions,
        chapter_architecture_context=chapter_architecture_context or "",
        chapter_timeline_context=chapter_timeline_context or "",
        chapter_fate_context=chapter_fate_context or "",
        chapter_arc_context=chapter_arc_context or "",
        chapter_antagonist_context=chapter_antagonist_context or "",
        chapter_technology_context=chapter_technology_context or "",
        chapter_theme_context=chapter_theme_context or "",
        gatekeeper_brief=gatekeeper_brief or "",
        chapter_text=chapter_text,
    )


def build_consistency_pass_prompt(
    title: str, all_summaries: list[str], special_instructions: str
) -> list[dict]:
    summaries_text = "\n\n".join(
        f"Chapter {i+1}:\n{s}" for i, s in enumerate(all_summaries)
    )
    return render_prompt(
        "consistency_pass",
        title=title,
        summaries_text=summaries_text,
        special_instructions=special_instructions,
    )


def build_global_continuity_auditor_prompt(
    title: str,
    all_summaries: list[str],
    character_state_log: list[str],
    master_timeline: dict,
    character_fate_registry: dict,
) -> list[dict]:
    """
    Global Continuity Auditor: cross-references the complete set of chapter
    summaries, the character state log, the master timeline ledger, and the
    character fate registry to detect contradictions across the full manuscript.
    Returns a structured JSON audit report.
    """
    summaries_text = "\n\n".join(
        f"Chapter {i + 1}:\n{s}" for i, s in enumerate(all_summaries)
    )

    state_log_text = (
        "\n\n".join(character_state_log)
        if character_state_log
        else "No character state log available."
    )

    # Condense master timeline ledger into a readable list
    timeline_lines: list[str] = []
    if isinstance(master_timeline, dict):
        for event in master_timeline.get("ledger", []):
            if isinstance(event, dict):
                timeline_lines.append(
                    f"  Ch {event.get('chapter', '?')}: {event.get('event', '')} "
                    f"[{event.get('event_type', 'other')}]"
                )
    timeline_text = (
        "\n".join(timeline_lines) if timeline_lines else "No master timeline available."
    )

    # Condense fate registry into a readable list
    registry_lines: list[str] = []
    if isinstance(character_fate_registry, dict):
        for entry in character_fate_registry.get("registry", []):
            if isinstance(entry, dict):
                name = entry.get("character", "?")
                status = entry.get("current_status", "unknown")
                outcome = entry.get("definitive_outcome", "unknown")
                death_ch = entry.get("death_chapter")
                registry_lines.append(
                    f"  {name}: status={status}, outcome={outcome}"
                    + (f", death_chapter={death_ch}" if death_ch else "")
                )
    registry_text = (
        "\n".join(registry_lines) if registry_lines else "No fate registry available."
    )

    return render_prompt(
        "global_continuity_auditor",
        title=title,
        summaries_text=summaries_text,
        state_log_text=state_log_text,
        timeline_text=timeline_text,
        registry_text=registry_text,
    )


def build_narrative_compression_editor_prompt(
    title: str,
    all_summaries: list[str],
    continuity_audit: dict | None = None,
) -> list[dict]:
    """
    Narrative Compression Editor: reads all chapter summaries and the continuity
    audit (if available) to identify repetitive operations, duplicate sacrifice
    beats, and sequences that can be consolidated into fewer, stronger events
    across the full manuscript. Returns a JSON report with specific consolidation
    recommendations keyed to chapter numbers.
    """
    summaries_text = "\n\n".join(
        f"Chapter {i + 1}:\n{s}" for i, s in enumerate(all_summaries)
    )

    audit_section = ""
    if continuity_audit and isinstance(continuity_audit, dict):
        contradictions = continuity_audit.get("contradictions", [])
        if contradictions:
            audit_lines = [
                f"  - Chapters {c.get('chapters', [])}: {c.get('description', '')}"
                for c in contradictions
                if isinstance(c, dict)
            ]
            if audit_lines:
                audit_section = (
                    "\n=== CONTINUITY AUDIT FLAGS (already identified) ===\n"
                    + "\n".join(audit_lines)
                    + "\n"
                )

    return render_prompt(
        "narrative_compression_editor",
        title=title,
        summaries_text=summaries_text,
        audit_section=audit_section,
    )


def build_character_resolution_validator_prompt(
    title: str,
    all_summaries: list[str],
    character_arc_plan: dict,
    character_fate_registry: dict,
    character_state_log: list[str],
) -> list[dict]:
    """
    Character Resolution Validator: cross-references the character arc plan,
    fate registry, and final chapter summaries to confirm that every major
    character receives deliberate narrative closure or a clearly intentional
    open ending. Reports unresolved characters and missing arc conclusions.
    Returns a JSON report.
    """
    summaries_text = "\n\n".join(
        f"Chapter {i + 1}:\n{s}" for i, s in enumerate(all_summaries)
    )

    # Condense arc plan
    arc_lines: list[str] = []
    if isinstance(character_arc_plan, dict):
        for arc in character_arc_plan.get("arcs", []):
            if isinstance(arc, dict):
                arc_lines.append(
                    f"  {arc.get('character', '?')}: "
                    f"start='{arc.get('start_state', '')}' → "
                    f"midpoint='{arc.get('midpoint_transformation', '')}' → "
                    f"crisis='{arc.get('crisis_point', '')}' → "
                    f"final_choice='{arc.get('final_moral_choice', '')}'"
                )
    arc_text = (
        "\n".join(arc_lines) if arc_lines else "No character arc plan available."
    )

    # Condense fate registry
    registry_lines: list[str] = []
    if isinstance(character_fate_registry, dict):
        for entry in character_fate_registry.get("registry", []):
            if isinstance(entry, dict):
                name = entry.get("character", "?")
                outcome = entry.get("definitive_outcome", "unknown")
                locked = entry.get("outcome_locked", False)
                registry_lines.append(
                    f"  {name}: required_outcome={outcome}, locked={locked}"
                )
    registry_text = (
        "\n".join(registry_lines) if registry_lines else "No fate registry available."
    )

    state_log_text = (
        "\n\n".join(character_state_log)
        if character_state_log
        else "No character state log available."
    )

    return render_prompt(
        "character_resolution_validator",
        title=title,
        arc_text=arc_text,
        registry_text=registry_text,
        state_log_text=state_log_text,
        summaries_text=summaries_text,
    )


def build_thematic_payoff_analyzer_prompt(
    title: str,
    all_summaries: list[str],
    theme_reinforcement: dict,
    total_chapters: int,
) -> list[dict]:
    """
    Thematic Payoff Analyzer: verifies that every major thematic pillar
    established in the Theme Reinforcement Plan culminates meaningfully
    in the final quarter of the manuscript. Reports themes that are
    introduced but abandoned, themes that peak too early without resolution,
    and themes whose planned final-chapter appearances are absent.
    Returns a JSON report.
    """
    summaries_text = "\n\n".join(
        f"Chapter {i + 1}:\n{s}" for i, s in enumerate(all_summaries)
    )

    # Condense theme plan into readable form
    theme_lines: list[str] = []
    if isinstance(theme_reinforcement, dict):
        for theme in theme_reinforcement.get("themes", []):
            if not isinstance(theme, dict):
                continue
            name = theme.get("name", "?")
            desc = theme.get("description", "")
            appearances = theme.get("chapter_appearances", [])
            final_chapters = [
                ap.get("chapter")
                for ap in appearances
                if isinstance(ap, dict)
            ]
            theme_lines.append(
                f"  Theme '{name}': {desc} | planned appearances: chapters {final_chapters}"
            )
        global_arcs = theme_reinforcement.get("global_thematic_arcs", [])
        if isinstance(global_arcs, list) and global_arcs:
            theme_lines.append(
                "  Global thematic arcs: " + "; ".join(str(a) for a in global_arcs)
            )
    theme_text = (
        "\n".join(theme_lines) if theme_lines else "No theme reinforcement plan available."
    )

    final_quarter_start = max(1, round(total_chapters * 0.75))

    return render_prompt(
        "thematic_payoff_analyzer",
        title=title,
        total_chapters=total_chapters,
        final_quarter_start=final_quarter_start,
        theme_text=theme_text,
        summaries_text=summaries_text,
    )


def build_climax_integrity_checker_prompt(
    title: str,
    all_summaries: list[str],
    character_arc_plan: dict,
    total_chapters: int,
) -> list[dict]:
    """
    Climax Integrity Checker: verifies that the protagonist makes a definitive,
    active, and morally meaningful final decision in the climax that genuinely
    resolves their arc. Returns a JSON report.
    """
    summaries_text = "\n\n".join(
        f"Chapter {i + 1}:\n{s}" for i, s in enumerate(all_summaries)
    )

    arc_lines: list[str] = []
    if isinstance(character_arc_plan, dict):
        for arc in character_arc_plan.get("arcs", []):
            if not isinstance(arc, dict):
                continue
            role = str(arc.get("role", "")).lower()
            if "protagonist" in role or "lead" in role or not role:
                arc_lines.append(
                    f"  {arc.get('character', '?')}: "
                    f"start='{arc.get('start_state', '')}' -> "
                    f"final_moral_choice='{arc.get('final_moral_choice', '')}'"
                )
    if not arc_lines and isinstance(character_arc_plan, dict):
        for arc in character_arc_plan.get("arcs", [])[:2]:
            if isinstance(arc, dict):
                arc_lines.append(
                    f"  {arc.get('character', '?')}: "
                    f"start='{arc.get('start_state', '')}' -> "
                    f"final_moral_choice='{arc.get('final_moral_choice', '')}'"
                )
    arc_text = (
        "\n".join(arc_lines) if arc_lines else "No character arc plan available."
    )

    climax_start = max(1, round(total_chapters * 0.85))

    return render_prompt(
        "climax_integrity_checker",
        title=title,
        total_chapters=total_chapters,
        climax_start=climax_start,
        arc_text=arc_text,
        summaries_text=summaries_text,
    )


def build_loose_thread_resolver_prompt(
    title: str,
    all_summaries: list[str],
    character_state_log: list[str],
    continuity_audit: dict | None = None,
    resolution_report: dict | None = None,
) -> list[dict]:
    """
    Loose Thread Resolver: scans the complete manuscript to surface every
    unresolved narrative question, broken promise, or dangling setup element.
    Classifies each thread as:
      - "open"               → clearly needs resolution before the novel ends
      - "dangling"           → introduced but silently forgotten mid-story
      - "intentionally_open" → deliberate ambiguity / sequel hook

    Returns a JSON report with resolution recommendations.
    """
    summaries_block = "\n".join(
        f"Chapter {i + 1}: {s}" for i, s in enumerate(all_summaries)
    ) or "No chapter summaries available."

    state_block = (
        "\n".join(character_state_log)
        if character_state_log
        else "No character state log available."
    )

    audit_issues: list[str] = []
    if continuity_audit:
        for field in ("contradictions", "character_state_errors", "timeline_errors"):
            items = continuity_audit.get(field, [])
            if isinstance(items, list):
                audit_issues.extend(items)
    audit_block = (
        "\n".join(f"- {x}" for x in audit_issues)
        if audit_issues
        else "No continuity issues flagged."
    )

    unresolved_chars: list[str] = []
    if resolution_report:
        raw = resolution_report.get("unresolved_characters", [])
        if isinstance(raw, list):
            unresolved_chars = [str(x) for x in raw]
    unresolved_block = (
        "\n".join(f"- {c}" for c in unresolved_chars)
        if unresolved_chars
        else "No unresolved characters flagged."
    )

    return render_prompt(
        "loose_thread_resolver",
        title=title,
        summaries_block=summaries_block,
        state_block=state_block,
        audit_block=audit_block,
        unresolved_block=unresolved_block,
    )


def build_reader_immersion_tester_prompt(
    title: str,
    all_summaries: list[str],
    character_arc_plan: dict | None = None,
    thematic_report: dict | None = None,
) -> list[dict]:
    """
    Reader Immersion Tester: evaluates the completed manuscript from a reader's
    perspective, assessing pacing, tension curve, clarity of stakes, and
    moment-to-moment engagement. Identifies weak or confusing chapters and
    surfaces any immersion-breaking inconsistencies. Returns a JSON report.
    """
    summaries_block = "\n".join(
        f"Chapter {i + 1}: {s}" for i, s in enumerate(all_summaries)
    ) or "No chapter summaries available."

    arc_lines: list[str] = []
    if isinstance(character_arc_plan, dict):
        for arc in character_arc_plan.get("arcs", []):
            name = arc.get("character", "Unknown")
            start = arc.get("start_state", "")
            end = arc.get("final_state", "")
            arc_lines.append(f"- {name}: {start} → {end}")
    arc_block = "\n".join(arc_lines) if arc_lines else "No character arc plan available."

    theme_lines: list[str] = []
    if isinstance(thematic_report, dict):
        for item in thematic_report.get("themes", []):
            if isinstance(item, dict):
                theme_lines.append(
                    f"- {item.get('theme', item)}: payoff={item.get('payoff_present', '?')}"
                )
            else:
                theme_lines.append(f"- {item}")
    theme_block = "\n".join(theme_lines) if theme_lines else "No thematic payoff data available."

    return render_prompt(
        "reader_immersion_tester",
        title=title,
        summaries_block=summaries_block,
        arc_block=arc_block,
        theme_block=theme_block,
    )


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

ALLOWED_GENRES = {"Fantasy", "Sci-Fi", "Mystery", "Romance", "Horror", "Thriller", "Historical"}


def validate_outline_input(data: dict) -> tuple[bool, str]:
    """Validate the /generate_outline form data. Returns (ok, error_message)."""
    premise = data.get("premise", "").strip()
    if not premise:
        return False, "Story premise is required."
    if len(premise) > 2000:
        return False, "Story premise must be 2000 characters or fewer."

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
        title = call_llm(build_title_prompt(str(premise), genre), action="Generating Title").strip().strip('"')

        # 2. Generate outline
        outline_raw = call_llm(
            build_outline_prompt(
                str(premise), genre, chapters, word_count, special_events, special_instructions
            ),
            action="Generating Outline",
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

        # 2.5. Generate story architecture plan
        story_architecture = plan_story_architecture(
            title=title,
            premise=str(premise),
            genre=genre,
            chapter_list=chapter_list,
            special_instructions=special_instructions,
        )

        # 2.6. Generate technology rules (after outline generation)
        technology_rules = plan_technology_rules(
            title=title,
            premise=str(premise),
            genre=genre,
            chapter_list=chapter_list,
            special_instructions=special_instructions,
        )

        # 2.7. Generate theme reinforcement plan (after outline generation)
        theme_reinforcement = plan_theme_reinforcement(
            title=title,
            premise=str(premise),
            genre=genre,
            chapter_list=chapter_list,
            special_instructions=special_instructions,
        )

        # 3. Generate characters
        characters_raw = call_llm(
            build_characters_prompt(str(premise), genre, outline_text),
            action="Generating Characters",
            json_mode=True,
        )
        try:
            characters_data = parse_llm_json(characters_raw)
            character_list = characters_data.get("characters", [])
        except json.JSONDecodeError:
            character_list = []

        # 4. Generate master timeline (after outline + characters)
        master_timeline = plan_master_timeline(
            title=title,
            premise=str(premise),
            genre=genre,
            chapter_list=chapter_list,
            character_list=character_list,
            special_instructions=special_instructions,
        )

        # 5. Generate character fate registry (after characters, before drafting)
        character_fate_registry = plan_character_fate_registry(
            title=title,
            premise=str(premise),
            genre=genre,
            character_list=character_list,
            chapter_list=chapter_list,
            master_timeline=master_timeline,
            special_instructions=special_instructions,
        )

        # 6. Generate character arc plan (after character generation, before drafting)
        character_arc_plan = plan_character_arc_plan(
            title=title,
            premise=str(premise),
            genre=genre,
            character_list=character_list,
            chapter_list=chapter_list,
            special_instructions=special_instructions,
        )

        # 7. Generate antagonist motivation plan (after character generation, before drafting)
        antagonist_motivation_plan = plan_antagonist_motivation_plan(
            title=title,
            premise=str(premise),
            genre=genre,
            character_list=character_list,
            chapter_list=chapter_list,
            master_timeline=master_timeline,
            special_instructions=special_instructions,
        )

        # Store outline data in session
        session["title"] = title
        session["chapter_list"] = chapter_list
        session["character_list"] = character_list
        session["story_architecture"] = story_architecture
        session["master_timeline"] = master_timeline
        session["character_fate_registry"] = character_fate_registry
        session["character_arc_plan"] = character_arc_plan
        session["antagonist_motivation_plan"] = antagonist_motivation_plan
        session["technology_rules"] = technology_rules
        session["theme_reinforcement"] = theme_reinforcement

        return jsonify(
            {
                "title": title,
                "chapters": chapter_list,
                "characters": character_list,
                "story_architecture": story_architecture,
                "master_timeline": master_timeline,
                "character_fate_registry": character_fate_registry,
                "character_arc_plan": character_arc_plan,
                "antagonist_motivation_plan": antagonist_motivation_plan,
                "technology_rules": technology_rules,
                "theme_reinforcement": theme_reinforcement,
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
    session["story_architecture"] = plan_story_architecture(
        title=session["title"],
        premise=session.get("premise", ""),
        genre=session.get("genre", ""),
        chapter_list=session["chapter_list"],
        special_instructions=session.get("special_instructions", ""),
    )
    session["technology_rules"] = plan_technology_rules(
        title=session["title"],
        premise=session.get("premise", ""),
        genre=session.get("genre", ""),
        chapter_list=session["chapter_list"],
        special_instructions=session.get("special_instructions", ""),
    )
    session["master_timeline"] = plan_master_timeline(
        title=session["title"],
        premise=session.get("premise", ""),
        genre=session.get("genre", ""),
        chapter_list=session["chapter_list"],
        character_list=session["character_list"],
        special_instructions=session.get("special_instructions", ""),
    )
    session["character_fate_registry"] = plan_character_fate_registry(
        title=session["title"],
        premise=session.get("premise", ""),
        genre=session.get("genre", ""),
        character_list=session["character_list"],
        chapter_list=session["chapter_list"],
        master_timeline=session.get("master_timeline", {}),
        special_instructions=session.get("special_instructions", ""),
    )
    session["character_arc_plan"] = plan_character_arc_plan(
        title=session["title"],
        premise=session.get("premise", ""),
        genre=session.get("genre", ""),
        character_list=session["character_list"],
        chapter_list=session["chapter_list"],
        special_instructions=session.get("special_instructions", ""),
    )
    session["antagonist_motivation_plan"] = plan_antagonist_motivation_plan(
        title=session["title"],
        premise=session.get("premise", ""),
        genre=session.get("genre", ""),
        character_list=session["character_list"],
        chapter_list=session["chapter_list"],
        master_timeline=session.get("master_timeline", {}),
        special_instructions=session.get("special_instructions", ""),
    )
    session["theme_reinforcement"] = plan_theme_reinforcement(
        title=session["title"],
        premise=session.get("premise", ""),
        genre=session.get("genre", ""),
        chapter_list=session["chapter_list"],
        special_instructions=session.get("special_instructions", ""),
    )

    # Auto-save session state after outline approval
    save_session_state()

    return jsonify(
        {
            "status": "approved",
            "story_architecture": session["story_architecture"],
            "master_timeline": session["master_timeline"],
            "character_fate_registry": session["character_fate_registry"],
            "character_arc_plan": session["character_arc_plan"],
            "antagonist_motivation_plan": session["antagonist_motivation_plan"],
            "technology_rules": session["technology_rules"],
            "theme_reinforcement": session["theme_reinforcement"],
        }
    )


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
        "story_architecture": session.get("story_architecture", {}),
        "master_timeline": session.get("master_timeline", {}),
        "character_fate_registry": session.get("character_fate_registry", {}),
        "character_arc_plan": session.get("character_arc_plan", {}),
        "antagonist_motivation_plan": session.get("antagonist_motivation_plan", {}),
        "technology_rules": session.get("technology_rules", {}),
        "theme_reinforcement": session.get("theme_reinforcement", {}),
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
        6. Redundancy detector – collapse repeated beats into stronger events
        7. Structure agent – confirm chapter fits story architecture
        8. Operational distinctiveness – ensure each op differs in strategy and stakes
        9. Character agent – deepen arcs and fix out-of-character moments
       10. Character thread tracker – ensure no named character left static
       11. Synthesizer     – unify voice and theme after multi-pass edits
       12. Polish agent    – grammar, style, vivid language
       13. Anti-LLM agent  – strip robotic patterns and forbidden words
       14. Quality control – engagement, tension, pacing check
       15. Story momentum tracker – ensure stakes escalate vs prior chapters
       16. Summary         – 100-200 word continuity summary

    Post-chapter (after summary, before next chapter):
        A. Character State Updater – log definitive character states into running registry

    Post-manuscript (after all chapters):
        I.   Final consistency pass – arc, threads, thematic payoff (JSON)
        II.  Global Continuity Auditor – cross-chapter contradiction detection (JSON)
        III. Narrative Compression Editor – identify redundant sequences for consolidation (JSON)
        IV.  Character Resolution Validator – confirm every major character receives closure (JSON)
        V.   Thematic Payoff Analyzer – ensure all themes culminate in the final act (JSON)
        VI.  Climax Integrity Checker – verify protagonist makes definitive final moral decision (JSON)
        VII. Loose Thread Resolver – identify and close unresolved narrative questions (JSON)
        VIII.Reader Immersion Tester – evaluate pacing, tension, and engagement from reader POV (JSON)
    """
    _run_chapter_generation_internal(token, snap, [], [], 0)


def _run_all_chapter_agents(
    text: str,
    chapter_num: int,
    title: str,
    genre: str,
    total_chapters: int,
    chapter_outline_summary: str,
    characters_text: str,
    previous_summaries: str,
    chapter_architecture_context: str = "",
    chapter_timeline_context: str = "",
    chapter_fate_context: str = "",
    chapter_arc_context: str = "",
    chapter_antagonist_context: str = "",
    chapter_technology_context: str = "",
    chapter_theme_context: str = "",
    gatekeeper_brief: str = "",
    step_callback=None,
) -> tuple[str, str]:
    """
    Run all chapter refinement agents (post-draft) and return:
    (final_chapter_text, continuity_summary)
    """
    if step_callback:
        step_callback(f"Chapter {chapter_num}: refining dialogue")
    text = call_llm(build_dialog_agent_prompt(text, chapter_num, title), action=f"Chapter {chapter_num}: refining dialogue")

    if step_callback:
        step_callback(f"Chapter {chapter_num}: structuring scenes")
    text = call_llm(build_scene_agent_prompt(text, chapter_num, title), action=f"Chapter {chapter_num}: structuring scenes")

    if step_callback:
        step_callback(f"Chapter {chapter_num}: verifying continuity")
    text = call_llm(
        build_context_analyzer_prompt(
            text,
            previous_summaries,
            chapter_num,
            title,
            chapter_timeline_context,
            chapter_technology_context,
            chapter_theme_context,
        ),
        action=f"Chapter {chapter_num}: verifying continuity"
    )

    if step_callback:
        step_callback(f"Chapter {chapter_num}: editing")
    text = call_llm(build_editing_agent_prompt(text, chapter_outline_summary, chapter_num, title), action=f"Chapter {chapter_num}: editing")

    if step_callback:
        step_callback(f"Chapter {chapter_num}: removing redundancy")
    text = call_llm(
        build_narrative_redundancy_detector_prompt(
            text,
            previous_summaries,
            chapter_outline_summary,
            chapter_num,
            title,
        ),
        action=f"Chapter {chapter_num}: removing redundancy"
    )

    if step_callback:
        step_callback(f"Chapter {chapter_num}: checking structure")
    text = call_llm(
        build_structure_agent_prompt(
            text,
            chapter_num,
            total_chapters,
            chapter_outline_summary,
            chapter_architecture_context,
        ),
        action=f"Chapter {chapter_num}: checking structure"
    )

    
    if step_callback:
        step_callback(f"Chapter {chapter_num}: verifying operational distinctiveness")
    text = call_llm(
        build_operational_distinctiveness_prompt(
            text,
            previous_summaries,
            chapter_outline_summary,
            chapter_num,
            title,
        ),
        action=f"Chapter {chapter_num}: verifying operational distinctiveness"
    )

    if step_callback:
        step_callback(f"Chapter {chapter_num}: deepening characters")
    text = call_llm(
        build_character_agent_prompt(
            text,
            characters_text,
            chapter_num,
            title,
            chapter_fate_context,
            chapter_arc_context,
            chapter_antagonist_context,
        ),
        action=f"Chapter {chapter_num}: deepening characters"
    )

    if step_callback:
        step_callback(f"Chapter {chapter_num}: tracking character threads")
    text = call_llm(
        build_character_thread_tracker_prompt(
            text,
            characters_text,
            chapter_num,
            title,
            chapter_arc_context,
        ),
        action=f"Chapter {chapter_num}: tracking character threads"
    )

    if step_callback:
        step_callback(f"Chapter {chapter_num}: synthesizing")
    text = call_llm(build_synthesizer_prompt(text, chapter_num, title, genre), action=f"Chapter {chapter_num}: synthesizing")

    if step_callback:
        step_callback(f"Chapter {chapter_num}: polishing")
    text = call_llm(build_polish_agent_prompt(text, chapter_num, title, genre), action=f"Chapter {chapter_num}: polishing")

    if step_callback:
        step_callback(f"Chapter {chapter_num}: anti-LLM pass")
    text = call_llm(build_anti_llm_agent_prompt(text, chapter_num, title), action=f"Chapter {chapter_num}: anti-LLM pass")

    if step_callback:
        step_callback(f"Chapter {chapter_num}: quality control")
    text = call_llm(build_quality_controller_prompt(text, chapter_num, title), action=f"Chapter {chapter_num}: quality control")

    if step_callback:
        step_callback(f"Chapter {chapter_num}: tracking story momentum")
    text = call_llm(
        build_story_momentum_tracker_prompt(
            text,
            previous_summaries,
            chapter_num,
            title,
            total_chapters,
        ),
        action=f"Chapter {chapter_num}: tracking story momentum"
    )

    if step_callback:
        step_callback(f"Chapter {chapter_num}: summarising")
    summary = call_llm(build_chapter_summary_prompt(text, chapter_num), action=f"Chapter {chapter_num}: summarising")

    return text, summary


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
    story_architecture = normalise_story_architecture(
        snap.get("story_architecture", {}),
        chapter_list,
        total_chapters,
    )
    master_timeline = normalise_master_timeline(
        snap.get("master_timeline", {}),
        chapter_list,
        character_list,
    )
    character_fate_registry = normalise_character_fate_registry(
        snap.get("character_fate_registry", {}),
        character_list,
        total_chapters,
    )
    character_arc_plan = normalise_character_arc_plan(
        snap.get("character_arc_plan", {}),
        character_list,
        chapter_list,
    )
    antagonist_motivation_plan = normalise_antagonist_motivation_plan(
        snap.get("antagonist_motivation_plan", {}),
        character_list,
        chapter_list,
    )
    technology_rules = normalise_technology_rules(
        snap.get("technology_rules", {}),
        chapter_list,
    )
    theme_reinforcement = normalise_theme_reinforcement(
        snap.get("theme_reinforcement", {}),
        chapter_list,
    )

    target_per_chapter = max(500, word_count // total_chapters)
    characters_text = _format_characters(character_list)
    character_state_log: list[str] = []
    compression_guidance: str = ""  # Guidance from previous chapter's compression check

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
                "character_state_log": character_state_log,
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
            chapter_architecture_context = get_chapter_architecture_context(
                story_architecture,
                chapter_num,
            )
            chapter_timeline_context = get_chapter_timeline_context(
                master_timeline,
                chapter_num,
            )
            chapter_fate_context = get_chapter_fate_context(
                character_fate_registry,
                chapter_num,
            )
            chapter_arc_context = get_chapter_arc_context(
                character_arc_plan,
                chapter_num,
            )
            chapter_antagonist_context = get_chapter_antagonist_context(
                antagonist_motivation_plan,
                chapter_num,
            )
            chapter_technology_context = get_chapter_technology_context(
                technology_rules,
                chapter_num,
            )
            chapter_theme_context = get_chapter_theme_context(
                theme_reinforcement,
                chapter_num,
            )

            previous_summaries = "\n\n".join(
                f"Chapter {i+1}: {s}" for i, s in enumerate(summaries)
            )

            # Continuity Gatekeeper – validates chapter constraints immediately before Draft
            _set_step(f"Chapter {chapter_num}: continuity gatekeeper")
            gatekeeper_brief = run_continuity_gatekeeper(
                chapter_num=chapter_num,
                chapter_title=chapter_title,
                chapter_summary=chapter_outline_summary,
                previous_summaries=previous_summaries,
                chapter_timeline_context=chapter_timeline_context,
                chapter_fate_context=chapter_fate_context,
                chapter_arc_context=chapter_arc_context,
                character_state_log="\n\n".join(character_state_log),
            )

            # 1. Draft
            _set_step(f"Chapter {chapter_num}: drafting")
            text = call_llm(
                build_chapter_draft_prompt(
                    premise, genre, title, chapter_num, chapter_title,
                    chapter_outline_summary, characters_text,
                    previous_summaries, target_per_chapter, special_instructions,
                    chapter_architecture_context,
                    chapter_timeline_context,
                    chapter_fate_context,
                    chapter_arc_context,
                    chapter_antagonist_context,
                    chapter_technology_context,
                    chapter_theme_context,
                    gatekeeper_brief,
                    compression_guidance,
                ),
                action=f"Chapter {chapter_num}: drafting"
            )

            text, summary = _run_all_chapter_agents(
                text=text,
                chapter_num=chapter_num,
                title=title,
                genre=genre,
                total_chapters=total_chapters,
                chapter_outline_summary=chapter_outline_summary,
                characters_text=characters_text,
                previous_summaries=previous_summaries,
                chapter_architecture_context=chapter_architecture_context,
                chapter_timeline_context=chapter_timeline_context,
                chapter_fate_context=chapter_fate_context,
                chapter_arc_context=chapter_arc_context,
                chapter_antagonist_context=chapter_antagonist_context,
                chapter_technology_context=chapter_technology_context,
                chapter_theme_context=chapter_theme_context,
                gatekeeper_brief=gatekeeper_brief,
                step_callback=_set_step,
            )
            summaries.append(summary)

            # Post-chapter: Character State Updater – record definitive character states
            _set_step(f"Chapter {chapter_num}: updating character states")
            state_update = run_character_state_updater(
                chapter_text=text,
                chapter_summary=summary,
                characters_text=characters_text,
                chapter_num=chapter_num,
                title=title,
            )
            if state_update.strip():
                character_state_log.append(
                    f"--- After Chapter {chapter_num} ---\n{state_update}"
                )

            # Post-chapter: Compression Check – identify redundancy for next chapter
            _set_step(f"Chapter {chapter_num}: compression check")
            compression_guidance = run_per_chapter_compression_check(
                chapter_num=chapter_num,
                chapter_summary=summary,
                previous_summaries=previous_summaries,
                title=title,
            )

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
            action="Final consistency pass",
            json_mode=True,
        )
        try:
            consistency = parse_llm_json(consistency_raw)
        except json.JSONDecodeError:
            consistency = {"issues": [], "overall_assessment": ""}

        # --- Global Continuity Audit ---
        with _progress_lock:
            _progress_store[token]["step"] = "Global continuity audit"
        audit_raw = call_llm(
            build_global_continuity_auditor_prompt(
                title=title,
                all_summaries=summaries,
                character_state_log=character_state_log,
                master_timeline=master_timeline,
                character_fate_registry=character_fate_registry,
            ),
            action="Global continuity audit",
            json_mode=True,
        )
        try:
            global_audit = parse_llm_json(audit_raw)
        except json.JSONDecodeError:
            global_audit = {
                "contradictions": [],
                "character_state_errors": [],
                "timeline_errors": [],
                "location_errors": [],
                "overall_integrity": "unknown",
                "overall_assessment": "",
            }

        with _progress_lock:
            _progress_store[token]["status"] = "done"
            _progress_store[token]["consistency"] = consistency
            _progress_store[token]["global_continuity_audit"] = global_audit

        # --- Narrative Compression Edit ---
        with _progress_lock:
            _progress_store[token]["step"] = "Narrative compression analysis"
        compression_raw = call_llm(
            build_narrative_compression_editor_prompt(
                title=title,
                all_summaries=summaries,
                continuity_audit=global_audit,
            ),
            action="Narrative compression analysis",
            json_mode=True,
        )
        try:
            compression_report = parse_llm_json(compression_raw)
        except json.JSONDecodeError:
            compression_report = {
                "redundant_sequences": [],
                "emotional_beat_repetitions": [],
                "compression_priority": "unknown",
                "overall_assessment": "",
            }

        with _progress_lock:
            _progress_store[token]["narrative_compression_report"] = compression_report

        # --- Character Resolution Validation ---
        with _progress_lock:
            _progress_store[token]["step"] = "Character resolution validation"
        resolution_raw = call_llm(
            build_character_resolution_validator_prompt(
                title=title,
                all_summaries=summaries,
                character_arc_plan=character_arc_plan,
                character_fate_registry=character_fate_registry,
                character_state_log=character_state_log,
            ),
            action="Character resolution validation",
            json_mode=True,
        )
        try:
            resolution_report = parse_llm_json(resolution_raw)
        except json.JSONDecodeError:
            resolution_report = {
                "character_resolutions": [],
                "unresolved_characters": [],
                "resolution_integrity": "unknown",
                "overall_assessment": "",
            }

        with _progress_lock:
            _progress_store[token]["character_resolution_report"] = resolution_report

        # --- Thematic Payoff Analysis ---
        with _progress_lock:
            _progress_store[token]["step"] = "Thematic payoff analysis"
        thematic_raw = call_llm(
            build_thematic_payoff_analyzer_prompt(
                title=title,
                all_summaries=summaries,
                theme_reinforcement=theme_reinforcement,
                total_chapters=total_chapters,
            ),
            action="Thematic payoff analysis",
            json_mode=True,
        )
        try:
            thematic_report = parse_llm_json(thematic_raw)
        except json.JSONDecodeError:
            thematic_report = {
                "theme_payoffs": [],
                "abandoned_themes": [],
                "weak_payoffs": [],
                "thematic_integrity": "unknown",
                "overall_assessment": "",
            }

        with _progress_lock:
            _progress_store[token]["thematic_payoff_report"] = thematic_report

        # --- Climax Integrity Check ---
        with _progress_lock:
            _progress_store[token]["step"] = "Climax integrity check"
        climax_raw = call_llm(
            build_climax_integrity_checker_prompt(
                title=title,
                all_summaries=summaries,
                character_arc_plan=character_arc_plan,
                total_chapters=total_chapters,
            ),
            action="Climax integrity check",
            json_mode=True,
        )
        try:
            climax_report = parse_llm_json(climax_raw)
        except json.JSONDecodeError:
            climax_report = {
                "climax_decision_present": False,
                "decision_is_active": False,
                "moral_dimension_present": False,
                "arc_resolved": False,
                "protagonist_is_agent": False,
                "climax_chapter": None,
                "integrity_failures": [],
                "climax_integrity": "unknown",
                "overall_assessment": "",
            }

        with _progress_lock:
            _progress_store[token]["climax_integrity_report"] = climax_report

        # --- Loose Thread Resolution ---
        with _progress_lock:
            _progress_store[token]["step"] = "Loose thread resolution"
        threads_raw = call_llm(
            build_loose_thread_resolver_prompt(
                title=title,
                all_summaries=summaries,
                character_state_log=character_state_log,
                continuity_audit=global_audit,
                resolution_report=resolution_report,
            ),
            action="Loose thread resolution",
            json_mode=True,
        )
        try:
            threads_report = parse_llm_json(threads_raw)
        except json.JSONDecodeError:
            threads_report = {
                "unresolved_threads": [],
                "dangling_setup_elements": [],
                "intentionally_open_threads": [],
                "thread_integrity": "unknown",
                "overall_assessment": "",
            }

        with _progress_lock:
            _progress_store[token]["loose_thread_report"] = threads_report

        # --- Reader Immersion Testing ---
        with _progress_lock:
            _progress_store[token]["step"] = "Reader immersion testing"
        immersion_raw = call_llm(
            build_reader_immersion_tester_prompt(
                title=title,
                all_summaries=summaries,
                character_arc_plan=character_arc_plan,
                thematic_report=thematic_report,
            ),
            action="Reader immersion testing",
            json_mode=True,
        )
        try:
            immersion_report = parse_llm_json(immersion_raw)
        except json.JSONDecodeError:
            immersion_report = {
                "pacing_assessment": "unknown",
                "tension_curve": "unknown",
                "stakes_clarity": "unknown",
                "engagement_score": 0,
                "weak_chapters": [],
                "immersion_breaks": [],
                "reader_experience_highlights": [],
                "overall_rating": "unknown",
                "recommendations": [],
            }

        with _progress_lock:
            _progress_store[token]["reader_immersion_report"] = immersion_report

        # Auto-save final state
        _set_step("Complete")

    except (RuntimeError, requests.exceptions.RequestException, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.error("Chapter generation failed for token %s: %s", token, exc)
        with _progress_lock:
            _progress_store[token]["status"] = "error"
            _progress_store[token]["error"] = str(exc)
        
        # Auto-save error state
        _set_step(f"Error: {str(exc)}")


@app.route("/revise_chapter", methods=["POST"])
def revise_chapter():
    """
    Apply custom editor instructions to one generated chapter, then re-run all
    chapter agents on that updated material.

    Expects JSON: {
      "token": "<progress_token>",
      "chapter_number": <int>,
      "instructions": "<revision instructions>"
    }
    """
    data = request.get_json(silent=True) or {}
    token = data.get("token", "")
    instructions = str(data.get("instructions", "")).strip()

    try:
        chapter_number = int(data.get("chapter_number", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "Chapter number must be a valid number."}), 400

    if not token:
        return jsonify({"error": "Missing progress token."}), 400
    if chapter_number < 1:
        return jsonify({"error": "Chapter number must be at least 1."}), 400
    if not instructions:
        return jsonify({"error": "Revision instructions are required."}), 400

    with _progress_lock:
        progress_data = _progress_store.get(token)

    if not progress_data or progress_data.get("status") != "done":
        return jsonify({"error": "Novel generation not complete."}), 400

    chapters_done = list(progress_data.get("chapters_done", []))
    if not chapters_done:
        return jsonify({"error": "No generated chapters found."}), 400

    target_idx = next(
        (i for i, chapter in enumerate(chapters_done) if int(chapter.get("number", 0)) == chapter_number),
        None,
    )
    if target_idx is None:
        return jsonify({"error": "Selected chapter was not found."}), 404

    title = session.get("title", "Novel")
    genre = session.get("genre", "")
    total_chapters = int(session.get("chapters", len(chapters_done) or 1))
    chapter_list = session.get("chapter_list", [])
    character_list = session.get("character_list", [])
    special_instructions = session.get("special_instructions", "")
    story_architecture = normalise_story_architecture(
        session.get("story_architecture", {}),
        chapter_list,
        total_chapters,
    )
    master_timeline = normalise_master_timeline(
        session.get("master_timeline", {}),
        chapter_list,
        character_list,
    )
    character_fate_registry = normalise_character_fate_registry(
        session.get("character_fate_registry", {}),
        character_list,
        total_chapters,
    )
    character_arc_plan = normalise_character_arc_plan(
        session.get("character_arc_plan", {}),
        character_list,
        chapter_list,
    )
    antagonist_motivation_plan = normalise_antagonist_motivation_plan(
        session.get("antagonist_motivation_plan", {}),
        character_list,
        chapter_list,
    )
    technology_rules = normalise_technology_rules(
        session.get("technology_rules", {}),
        chapter_list,
    )
    theme_reinforcement = normalise_theme_reinforcement(
        session.get("theme_reinforcement", {}),
        chapter_list,
    )

    chapter_outline_summary = ""
    for chapter_outline in chapter_list:
        try:
            if int(chapter_outline.get("number", 0)) == chapter_number:
                chapter_outline_summary = chapter_outline.get("summary", "")
                break
        except (TypeError, ValueError):
            continue

    target_chapter = chapters_done[target_idx]
    previous_summaries = "\n\n".join(
        f"Chapter {c.get('number', i+1)}: {c.get('summary', '')}"
        for i, c in enumerate(chapters_done[:target_idx])
    )
    characters_text = _format_characters(character_list)
    chapter_architecture_context = get_chapter_architecture_context(
        story_architecture,
        chapter_number,
    )
    chapter_timeline_context = get_chapter_timeline_context(
        master_timeline,
        chapter_number,
    )
    chapter_fate_context = get_chapter_fate_context(
        character_fate_registry,
        chapter_number,
    )
    chapter_arc_context = get_chapter_arc_context(
        character_arc_plan,
        chapter_number,
    )
    chapter_antagonist_context = get_chapter_antagonist_context(
        antagonist_motivation_plan,
        chapter_number,
    )
    chapter_technology_context = get_chapter_technology_context(
        technology_rules,
        chapter_number,
    )
    chapter_theme_context = get_chapter_theme_context(
        theme_reinforcement,
        chapter_number,
    )

    gatekeeper_brief = run_continuity_gatekeeper(
        chapter_num=chapter_number,
        chapter_title=target_chapter.get("title", f"Chapter {chapter_number}"),
        chapter_summary=chapter_outline_summary,
        previous_summaries=previous_summaries,
        chapter_timeline_context=chapter_timeline_context,
        chapter_fate_context=chapter_fate_context,
        chapter_arc_context=chapter_arc_context,
    )

    try:
        revised_text = call_llm(
            build_chapter_revision_prompt(
                chapter_text=target_chapter.get("content", ""),
                chapter_num=chapter_number,
                title=title,
                chapter_outline_summary=chapter_outline_summary,
                revision_instructions=instructions,
                chapter_architecture_context=chapter_architecture_context,
                chapter_timeline_context=chapter_timeline_context,
                chapter_fate_context=chapter_fate_context,
                chapter_arc_context=chapter_arc_context,
                chapter_antagonist_context=chapter_antagonist_context,
                chapter_technology_context=chapter_technology_context,
                chapter_theme_context=chapter_theme_context,
                gatekeeper_brief=gatekeeper_brief,
            ),
            action=f"Chapter {chapter_number}: applying revision instructions"
        )

        revised_text, revised_summary = _run_all_chapter_agents(
            text=revised_text,
            chapter_num=chapter_number,
            title=title,
            genre=genre,
            total_chapters=total_chapters,
            chapter_outline_summary=chapter_outline_summary,
            characters_text=characters_text,
            previous_summaries=previous_summaries,
            chapter_architecture_context=chapter_architecture_context,
            chapter_timeline_context=chapter_timeline_context,
            chapter_fate_context=chapter_fate_context,
            chapter_arc_context=chapter_arc_context,
            chapter_antagonist_context=chapter_antagonist_context,
            chapter_technology_context=chapter_technology_context,
            chapter_theme_context=chapter_theme_context,
            gatekeeper_brief=gatekeeper_brief,
            step_callback=None,
        )

        chapters_done[target_idx]["content"] = revised_text
        chapters_done[target_idx]["summary"] = revised_summary

        all_summaries = [str(ch.get("summary", "")) for ch in chapters_done]
        consistency_raw = call_llm(
            build_consistency_pass_prompt(title, all_summaries, special_instructions),
            action="Final consistency pass after revision",
            json_mode=True,
        )
        try:
            consistency = parse_llm_json(consistency_raw)
        except json.JSONDecodeError:
            consistency = {"issues": [], "overall_assessment": ""}

        with _progress_lock:
            _progress_store[token]["status"] = "done"
            _progress_store[token]["step"] = f"Chapter {chapter_number}: revised"
            _progress_store[token]["chapters_done"] = chapters_done
            _progress_store[token]["consistency"] = consistency
            response_payload = dict(_progress_store[token])

        return jsonify(response_payload)

    except RuntimeError as exc:
        logger.error("Chapter revision failed for token %s: %s", token, exc)
        return jsonify({"error": str(exc)}), 502


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


@app.route("/export_editors_notes", methods=["POST"])
def export_editors_notes():
    """
    Export editor's notes into a Markdown file and return a download URL.
    Includes all diagnostic reports from post-generation audits.
    Expects JSON: { "token": "<progress_token>" }
    """
    data = request.get_json(silent=True) or {}
    token = data.get("token", "")

    with _progress_lock:
        progress_data = _progress_store.get(token)

    if not progress_data or progress_data.get("status") != "done":
        return jsonify({"error": "Novel generation not complete."}), 400

    title = session.get("title", "Novel")

    # Collect all diagnostic reports
    consistency = progress_data.get("consistency", {})
    global_continuity_audit = progress_data.get("global_continuity_audit", {})
    narrative_compression_report = progress_data.get("narrative_compression_report", {})
    character_resolution_report = progress_data.get("character_resolution_report", {})
    thematic_payoff_report = progress_data.get("thematic_payoff_report", {})
    climax_integrity_report = progress_data.get("climax_integrity_report", {})
    loose_thread_report = progress_data.get("loose_thread_report", {})
    reader_immersion_report = progress_data.get("reader_immersion_report", {})

    # Check if any reports are available
    has_content = any([
        consistency.get("overall_assessment") or consistency.get("issues"),
        global_continuity_audit,
        narrative_compression_report,
        character_resolution_report,
        thematic_payoff_report,
        climax_integrity_report,
        loose_thread_report,
        reader_immersion_report,
    ])

    if not has_content:
        return jsonify({"error": "No editor's notes are available for this novel."}), 400

    lines = [f"# {title} - Editor's Notes\n"]
    lines.append("This document contains all diagnostic reports from the novel generation process. ")
    lines.append("Use these notes to identify chapters that may need revision.\n")

    # 1. Consistency Pass
    if consistency:
        lines.append("---\n")
        lines.append("## 1. Consistency Pass\n")
        overall_assessment = (consistency.get("overall_assessment") or "").strip()
        if overall_assessment:
            lines.append(f"**Overall Assessment:** {overall_assessment}\n")
        issues = consistency.get("issues") or []
        if issues:
            lines.append("**Issues:**\n")
            for issue in issues:
                lines.append(f"- {issue}")
            lines.append("")

    # 2. Global Continuity Audit
    if global_continuity_audit:
        lines.append("---\n")
        lines.append("## 2. Global Continuity Audit\n")
        overall = (global_continuity_audit.get("overall_assessment") or "").strip()
        integrity = (global_continuity_audit.get("overall_integrity") or "").strip()
        if integrity:
            lines.append(f"**Overall Integrity:** {integrity}\n")
        if overall:
            lines.append(f"**Assessment:** {overall}\n")
        contradictions = global_continuity_audit.get("contradictions") or []
        if contradictions:
            lines.append("**Contradictions:**\n")
            for c in contradictions:
                if isinstance(c, dict):
                    chapters = c.get("chapters", [])
                    desc = c.get("description", str(c))
                    lines.append(f"- Chapters {chapters}: {desc}")
                else:
                    lines.append(f"- {c}")
            lines.append("")
        char_errors = global_continuity_audit.get("character_state_errors") or []
        if char_errors:
            lines.append("**Character State Errors:**\n")
            for e in char_errors:
                lines.append(f"- {e}" if isinstance(e, str) else f"- {e}")
            lines.append("")
        timeline_errors = global_continuity_audit.get("timeline_errors") or []
        if timeline_errors:
            lines.append("**Timeline Errors:**\n")
            for e in timeline_errors:
                lines.append(f"- {e}" if isinstance(e, str) else f"- {e}")
            lines.append("")

    # 3. Narrative Compression Report
    if narrative_compression_report:
        lines.append("---\n")
        lines.append("## 3. Narrative Compression Report\n")
        priority = (narrative_compression_report.get("compression_priority") or "").strip()
        overall = (narrative_compression_report.get("overall_assessment") or "").strip()
        if priority:
            lines.append(f"**Compression Priority:** {priority}\n")
        if overall:
            lines.append(f"**Assessment:** {overall}\n")
        redundant = narrative_compression_report.get("redundant_sequences") or []
        if redundant:
            lines.append("**Redundant Sequences:**\n")
            for r in redundant:
                if isinstance(r, dict):
                    chapters = r.get("chapters", [])
                    pattern = r.get("pattern", "")
                    rec = r.get("recommendation", "")
                    lines.append(f"- Chapters {chapters}: {pattern}")
                    if rec:
                        lines.append(f"  - *Recommendation:* {rec}")
                else:
                    lines.append(f"- {r}")
            lines.append("")
        emotional = narrative_compression_report.get("emotional_beat_repetitions") or []
        if emotional:
            lines.append("**Emotional Beat Repetitions:**\n")
            for e in emotional:
                if isinstance(e, dict):
                    chapters = e.get("chapters", [])
                    beat = e.get("beat", "")
                    rec = e.get("recommendation", "")
                    lines.append(f"- Chapters {chapters}: {beat}")
                    if rec:
                        lines.append(f"  - *Recommendation:* {rec}")
                else:
                    lines.append(f"- {e}")
            lines.append("")

    # 4. Character Resolution Report
    if character_resolution_report:
        lines.append("---\n")
        lines.append("## 4. Character Resolution Report\n")
        integrity = (character_resolution_report.get("resolution_integrity") or "").strip()
        overall = (character_resolution_report.get("overall_assessment") or "").strip()
        if integrity:
            lines.append(f"**Resolution Integrity:** {integrity}\n")
        if overall:
            lines.append(f"**Assessment:** {overall}\n")
        unresolved = character_resolution_report.get("unresolved_characters") or []
        if unresolved:
            lines.append("**Unresolved Characters:**\n")
            for u in unresolved:
                if isinstance(u, dict):
                    name = u.get("character", u.get("name", "Unknown"))
                    issue = u.get("issue", u.get("description", str(u)))
                    lines.append(f"- **{name}**: {issue}")
                else:
                    lines.append(f"- {u}")
            lines.append("")
        resolutions = character_resolution_report.get("character_resolutions") or []
        if resolutions:
            lines.append("**Character Resolutions:**\n")
            for r in resolutions:
                if isinstance(r, dict):
                    name = r.get("character", r.get("name", "Unknown"))
                    status = r.get("status", r.get("resolution", str(r)))
                    lines.append(f"- **{name}**: {status}")
                else:
                    lines.append(f"- {r}")
            lines.append("")

    # 5. Thematic Payoff Report
    if thematic_payoff_report:
        lines.append("---\n")
        lines.append("## 5. Thematic Payoff Report\n")
        integrity = (thematic_payoff_report.get("thematic_integrity") or "").strip()
        overall = (thematic_payoff_report.get("overall_assessment") or "").strip()
        if integrity:
            lines.append(f"**Thematic Integrity:** {integrity}\n")
        if overall:
            lines.append(f"**Assessment:** {overall}\n")
        abandoned = thematic_payoff_report.get("abandoned_themes") or []
        if abandoned:
            lines.append("**Abandoned Themes:**\n")
            for t in abandoned:
                if isinstance(t, dict):
                    theme = t.get("theme", t.get("name", str(t)))
                    reason = t.get("reason", t.get("description", ""))
                    lines.append(f"- **{theme}**" + (f": {reason}" if reason else ""))
                else:
                    lines.append(f"- {t}")
            lines.append("")
        weak = thematic_payoff_report.get("weak_payoffs") or []
        if weak:
            lines.append("**Weak Payoffs:**\n")
            for w in weak:
                if isinstance(w, dict):
                    theme = w.get("theme", w.get("name", str(w)))
                    issue = w.get("issue", w.get("description", ""))
                    lines.append(f"- **{theme}**" + (f": {issue}" if issue else ""))
                else:
                    lines.append(f"- {w}")
            lines.append("")

    # 6. Climax Integrity Report
    if climax_integrity_report:
        lines.append("---\n")
        lines.append("## 6. Climax Integrity Report\n")
        integrity = (climax_integrity_report.get("climax_integrity") or "").strip()
        overall = (climax_integrity_report.get("overall_assessment") or "").strip()
        climax_chapter = climax_integrity_report.get("climax_chapter")
        if climax_chapter:
            lines.append(f"**Climax Chapter:** {climax_chapter}\n")
        if integrity:
            lines.append(f"**Climax Integrity:** {integrity}\n")
        # Boolean checks
        checks = []
        if climax_integrity_report.get("climax_decision_present") is False:
            checks.append("Missing climax decision")
        if climax_integrity_report.get("decision_is_active") is False:
            checks.append("Decision is not active (protagonist passive)")
        if climax_integrity_report.get("moral_dimension_present") is False:
            checks.append("Missing moral dimension")
        if climax_integrity_report.get("arc_resolved") is False:
            checks.append("Character arc not resolved")
        if climax_integrity_report.get("protagonist_is_agent") is False:
            checks.append("Protagonist is not the agent of change")
        if checks:
            lines.append("**Failed Checks:**\n")
            for c in checks:
                lines.append(f"- {c}")
            lines.append("")
        failures = climax_integrity_report.get("integrity_failures") or []
        if failures:
            lines.append("**Integrity Failures:**\n")
            for f in failures:
                lines.append(f"- {f}" if isinstance(f, str) else f"- {f}")
            lines.append("")
        if overall:
            lines.append(f"**Assessment:** {overall}\n")

    # 7. Loose Thread Report
    if loose_thread_report:
        lines.append("---\n")
        lines.append("## 7. Loose Thread Report\n")
        integrity = (loose_thread_report.get("thread_integrity") or "").strip()
        overall = (loose_thread_report.get("overall_assessment") or "").strip()
        if integrity:
            lines.append(f"**Thread Integrity:** {integrity}\n")
        if overall:
            lines.append(f"**Assessment:** {overall}\n")
        unresolved = loose_thread_report.get("unresolved_threads") or []
        if unresolved:
            lines.append("**Unresolved Threads:**\n")
            for t in unresolved:
                if isinstance(t, dict):
                    thread = t.get("thread", t.get("description", str(t)))
                    chapters = t.get("chapters", t.get("introduced_in", ""))
                    lines.append(f"- {thread}" + (f" (Chapters: {chapters})" if chapters else ""))
                else:
                    lines.append(f"- {t}")
            lines.append("")
        dangling = loose_thread_report.get("dangling_setup_elements") or []
        if dangling:
            lines.append("**Dangling Setup Elements:**\n")
            for d in dangling:
                if isinstance(d, dict):
                    element = d.get("element", d.get("description", str(d)))
                    lines.append(f"- {element}")
                else:
                    lines.append(f"- {d}")
            lines.append("")
        intentional = loose_thread_report.get("intentionally_open_threads") or []
        if intentional:
            lines.append("**Intentionally Open Threads (for sequel):**\n")
            for t in intentional:
                if isinstance(t, dict):
                    thread = t.get("thread", t.get("description", str(t)))
                    lines.append(f"- {thread}")
                else:
                    lines.append(f"- {t}")
            lines.append("")

    # 8. Reader Immersion Report
    if reader_immersion_report:
        lines.append("---\n")
        lines.append("## 8. Reader Immersion Report\n")
        overall_rating = (reader_immersion_report.get("overall_rating") or "").strip()
        engagement_score = reader_immersion_report.get("engagement_score")
        pacing = (reader_immersion_report.get("pacing_assessment") or "").strip()
        tension = (reader_immersion_report.get("tension_curve") or "").strip()
        stakes = (reader_immersion_report.get("stakes_clarity") or "").strip()
        if overall_rating:
            lines.append(f"**Overall Rating:** {overall_rating}\n")
        if engagement_score is not None:
            lines.append(f"**Engagement Score:** {engagement_score}/10\n")
        if pacing:
            lines.append(f"**Pacing Assessment:** {pacing}\n")
        if tension:
            lines.append(f"**Tension Curve:** {tension}\n")
        if stakes:
            lines.append(f"**Stakes Clarity:** {stakes}\n")
        weak_chapters = reader_immersion_report.get("weak_chapters") or []
        if weak_chapters:
            lines.append("**Weak Chapters (need revision):**\n")
            for w in weak_chapters:
                if isinstance(w, dict):
                    chapter = w.get("chapter", w.get("number", "?"))
                    reason = w.get("reason", w.get("issue", str(w)))
                    lines.append(f"- **Chapter {chapter}**: {reason}")
                else:
                    lines.append(f"- {w}")
            lines.append("")
        breaks = reader_immersion_report.get("immersion_breaks") or []
        if breaks:
            lines.append("**Immersion Breaks:**\n")
            for b in breaks:
                if isinstance(b, dict):
                    chapter = b.get("chapter", "?")
                    desc = b.get("description", b.get("issue", str(b)))
                    lines.append(f"- Chapter {chapter}: {desc}")
                else:
                    lines.append(f"- {b}")
            lines.append("")
        recommendations = reader_immersion_report.get("recommendations") or []
        if recommendations:
            lines.append("**Recommendations:**\n")
            for r in recommendations:
                lines.append(f"- {r}" if isinstance(r, str) else f"- {r}")
            lines.append("")

    markdown_content = "\n".join(lines)

    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:80]
    safe_title = "_".join(safe_title.split()) or "Novel"
    filename = f"{safe_title}-Editors_Notes.md"
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
        for obj_str in json_objects[-10:]:
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
    app.run(debug=True, host=host, port=port)
