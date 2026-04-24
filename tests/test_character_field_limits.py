"""Tests for character attribute length enforcement (issue #165)."""

from __future__ import annotations

import pytest

from novelforge.validation import (
    CHARACTER_FIELD_LIMITS,
    truncate_character_field,
    validate_character_fields,
)
from novelforge.agents.chapter import (
    build_character_field_repair_prompt,
    build_characters_prompt,
)
from novelforge.agents.chapter.context import _format_characters


# ---------------------------------------------------------------------------
# Central schema
# ---------------------------------------------------------------------------

def test_central_schema_contains_new_fields():
    assert CHARACTER_FIELD_LIMITS["role"] == 512
    assert CHARACTER_FIELD_LIMITS["internal_flaw"] == 512
    assert CHARACTER_FIELD_LIMITS["personal_goal"] == 512
    # Fields that were already generous should stay generous.
    assert CHARACTER_FIELD_LIMITS["background"] >= 2000


# ---------------------------------------------------------------------------
# validate_character_fields
# ---------------------------------------------------------------------------

def test_validator_passes_for_compliant_character():
    char = {
        "name": "Aldric Holt",
        "age": 42,
        "role": "disillusioned medic",
        "background": "served in three wars",
        "arc": "finds meaning again",
        "internal_flaw": "cannot refuse a plea for help",
        "personal_goal": "find his estranged brother",
    }
    assert validate_character_fields(char) == {}


def test_validator_flags_oversize_role():
    char = {"name": "X", "role": "a" * 600}
    violations = validate_character_fields(char)
    assert "role" in violations
    assert violations["role"] == 600


def test_validator_skips_missing_fields():
    # Partial character dicts should not crash.
    assert validate_character_fields({"name": "X"}) == {}


def test_validator_accepts_integer_age():
    assert validate_character_fields({"age": 35}) == {}


def test_validator_flags_multiple_fields():
    char = {
        "role": "r" * 600,
        "internal_flaw": "f" * 600,
        "personal_goal": "ok goal",
    }
    v = validate_character_fields(char)
    assert set(v.keys()) == {"role", "internal_flaw"}


# ---------------------------------------------------------------------------
# truncate_character_field
# ---------------------------------------------------------------------------

def test_truncate_short_string_is_noop():
    assert truncate_character_field("short", 100) == "short"


def test_truncate_enforces_max_length_including_suffix():
    out = truncate_character_field("x" * 200, 50)
    assert len(out) <= 50


def test_truncate_prefers_word_boundary():
    value = " ".join(["alpha"] * 40)  # 239 chars
    out = truncate_character_field(value, 60)
    assert len(out) <= 60
    # Should not end mid-word (alpha) — must end with suffix char.
    assert out.endswith("…")


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def test_characters_prompt_includes_field_limits_block():
    msgs = build_characters_prompt(
        premise="test", genre="Thriller",
        chapters_count=10,
    )
    rendered = "\n".join(m["content"] for m in msgs)
    assert "role: max 512 characters" in rendered
    assert "internal_flaw: max 512 characters" in rendered
    assert "personal_goal: max 512 characters" in rendered


def test_field_repair_prompt_contains_limit_and_value():
    msgs = build_character_field_repair_prompt(
        character_name="Aldric",
        field_name="role",
        current_value="x" * 600,
        max_length=512,
    )
    rendered = "\n".join(m["content"] for m in msgs)
    assert "Aldric" in rendered
    assert "role" in rendered
    assert "512" in rendered
    assert "x" * 600 in rendered  # full current value passed


# ---------------------------------------------------------------------------
# internal_flaw/personal_goal flow into characters_text
# ---------------------------------------------------------------------------

def test_format_characters_surfaces_flaw_and_goal():
    chars = [{
        "name": "Mira",
        "age": 30,
        "role": "pilot",
        "background": "drifter",
        "arc": "becomes a leader",
        "internal_flaw": "distrusts authority",
        "personal_goal": "find her father's killer",
    }]
    text = _format_characters(chars)
    assert "Internal flaw: distrusts authority" in text
    assert "Personal goal: find her father's killer" in text


# ---------------------------------------------------------------------------
# End-to-end: /generate_outline repair path
# ---------------------------------------------------------------------------

def test_generate_outline_repairs_oversize_role(client, mock_llm, mocker):
    """When the LLM returns a character with an oversize role, the generation
    route must repair (or truncate) before persisting the character."""
    import json as _json

    oversize_role = "a very long role description " * 30  # ~870 chars

    def _responder(messages, *args, **kwargs):
        sys_content = messages[0].get("content", "") if messages else ""
        user_content = messages[-1].get("content", "") if messages else ""
        # Character generation call
        if "character development expert" in sys_content:
            return _json.dumps({
                "characters": [{
                    "name": "Aldric Holt",
                    "age": 42,
                    "background": "served in three wars",
                    "role": oversize_role,
                    "arc": "finds meaning",
                    "internal_flaw": "cannot refuse a plea",
                    "personal_goal": "find his brother",
                }],
            })
        # Field repair call — return a short, compliant value
        if "rewrite a single character attribute" in sys_content.lower() or \
           "rewrite this value" in user_content.lower():
            return "disillusioned medic turned reluctant smuggler"
        # Title
        if "title" in sys_content.lower() and "character" not in sys_content.lower():
            return "A Test Title"
        # Outline
        if "outline" in sys_content.lower() or "chapter_plan" in user_content.lower():
            return _json.dumps({
                "chapters": [
                    {"number": i, "title": f"Ch {i}", "summary": "summary"}
                    for i in range(1, 4)
                ],
            })
        # Default: minimal JSON or plain text for planning agents.
        return "{}"

    mock_llm.side_effect = _responder
    # Patch every reference too.
    for mod in (
        "novelforge.agents.base",
        "novelforge.agents.chapter",
        "novelforge.agents.chapter._helpers",
        "novelforge.agents.chapter.pipeline",
        "novelforge.agents.planning",
        "novelforge.routes.outline",
    ):
        mocker.patch(f"{mod}.call_llm", side_effect=_responder)

    resp = client.post("/generate_outline", json={
        "premise": "A rescue mission tests loyalties.",
        "genre": "Thriller",
        "chapters": 3,
        "word_count": 5000,
        "special_events": "",
        "special_instructions": "",
    })
    # The route should succeed (200) rather than fail on oversize role.
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert data["characters"], "expected at least one character"
    role = data["characters"][0]["role"]
    assert len(role) <= CHARACTER_FIELD_LIMITS["role"]


def test_generate_outline_truncates_when_repair_fails(client, mock_llm, mocker):
    """If the LLM repair attempt still returns an oversize value, the route
    must fall back to truncation so the character is still stored compliant."""
    import json as _json

    oversize = "x" * 900

    def _responder(messages, *args, **kwargs):
        sys_content = messages[0].get("content", "") if messages else ""
        user_content = messages[-1].get("content", "") if messages else ""
        if "character development expert" in sys_content:
            return _json.dumps({
                "characters": [{
                    "name": "Bad Bot",
                    "age": 30,
                    "background": "",
                    "role": oversize,
                    "arc": "",
                    "internal_flaw": "",
                    "personal_goal": "",
                }],
            })
        if "rewrite a single character attribute" in sys_content.lower() or \
           "rewrite this value" in user_content.lower():
            return "y" * 900  # still oversize — forces truncation fallback
        if "outline" in sys_content.lower() or "chapter_plan" in user_content.lower():
            return _json.dumps({
                "chapters": [
                    {"number": i, "title": f"Ch {i}", "summary": "s"}
                    for i in range(1, 4)
                ],
            })
        return "{}"

    mock_llm.side_effect = _responder
    for mod in (
        "novelforge.agents.base",
        "novelforge.agents.chapter",
        "novelforge.agents.chapter._helpers",
        "novelforge.agents.chapter.pipeline",
        "novelforge.agents.planning",
        "novelforge.routes.outline",
    ):
        mocker.patch(f"{mod}.call_llm", side_effect=_responder)

    resp = client.post("/generate_outline", json={
        "premise": "A rescue mission tests loyalties.",
        "genre": "Thriller",
        "chapters": 3,
        "word_count": 5000,
        "special_events": "",
        "special_instructions": "",
    })
    assert resp.status_code == 200, resp.get_json()
    role = resp.get_json()["characters"][0]["role"]
    assert len(role) <= CHARACTER_FIELD_LIMITS["role"]


# ---------------------------------------------------------------------------
# /approve_outline still enforces the same central limit
# ---------------------------------------------------------------------------

def test_approve_outline_rejects_oversize_role(client, app):
    with client.session_transaction() as sess:
        sess["title"] = "T"
        sess["premise"] = "p"
        sess["genre"] = "Thriller"
        sess["chapters"] = 3
        sess["word_count"] = 5000
        sess["chapter_list"] = [
            {"number": i, "title": f"Ch {i}", "summary": "s"}
            for i in range(1, 4)
        ]
        sess["character_list"] = []
    resp = client.post("/approve_outline", json={
        "title": "T",
        "chapters": [
            {"number": i, "title": f"Ch {i}", "summary": "s"}
            for i in range(1, 4)
        ],
        "characters": [{
            "name": "Aldric",
            "age": 40,
            "role": "x" * 600,
            "background": "bg",
            "arc": "arc",
        }],
    })
    assert resp.status_code == 400
    body = resp.get_json()
    assert "role" in body["error"]
    assert "512" in body["error"]
