"""Chapter agent pipeline and runner functions that wrap prompt builders with LLM calls."""

import logging
import re
import time
from collections.abc import Callable

from novelforge.llm.client import call_llm, parse_llm_json, ChapterTimeoutError, PER_CHAPTER_TIMEOUT
from novelforge.validation import (
    CHARACTER_FIELD_LIMITS,
    truncate_character_field,
    validate_character_fields,
)

from novelforge.agents.chapter._helpers import (
    PASS_FAILURE_KEY,
    _call_with_content_retry,
    _log_pass_failure,
    extract_named_characters,
    format_vocabulary_rules,
    scan_vocabulary_overuse,
)
from novelforge.agents.chapter.context import ChapterContext
from novelforge.agents.chapter.prompts import (
    build_anti_llm_agent_prompt,
    build_chapter_rhythm_classifier_prompt,
    build_chapter_summary_prompt,
    build_character_agent_prompt,
    build_character_field_repair_prompt,
    build_character_reconciliation_prompt,
    build_chapter_pattern_extractor_prompt,
    build_character_state_updater_prompt,
    build_context_analyzer_prompt,
    build_continuity_gatekeeper_prompt,
    build_copy_edit_prompt,
    build_editing_agent_prompt,
    build_human_oddities_prompt,
    build_metaphor_reduction_prompt,
    build_narrative_momentum_distinctiveness_prompt,
    build_operational_distinctiveness_prompt,
    build_per_chapter_compression_check_prompt,
    build_polish_agent_prompt,
    build_prose_refinement_agent_prompt,
    build_rhythm_compliance_verifier_prompt,
    build_quality_controller_prompt,
    build_scene_variety_compression_auditor_prompt,
    build_structure_agent_prompt,
    build_synthesizer_prompt,
    build_vocabulary_fix_prompt,
    build_voice_dialogue_differentiation_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runner functions (prompt + LLM call + error handling)
# ---------------------------------------------------------------------------

def run_scene_variety_compression_auditor(chapter_text: str, chapter_summary: str, chapter_num: int, title: str,
                                           previous_summaries: str = "",
                                           degraded_passes: list[dict] | None = None) -> str:
    """Run the scene variety audit, returning directives or empty string on failure."""
    try:
        return call_llm(
            build_scene_variety_compression_auditor_prompt(
                chapter_text=chapter_text, chapter_summary=chapter_summary,
                chapter_num=chapter_num, title=title,
                previous_summaries=previous_summaries,
            ),
            action=f"Chapter {chapter_num}: scene variety & compression audit",
        )
    except Exception as exc:
        failure_summary = _log_pass_failure(
            exc, pass_name="scene variety & compression auditor",
            chapter_num=chapter_num, chapter_title=title, optional=True,
        )
        if degraded_passes is not None:
            degraded_passes.append({
                "pass_name": "scene variety & compression auditor",
                "chapter_num": chapter_num,
                "failure_summary": failure_summary,
            })
        return ""


def run_per_chapter_compression_check(chapter_num: int, chapter_summary: str, previous_summaries: str, title: str,
                                       degraded_passes: list[dict] | None = None) -> str:
    """Run the post-chapter compression check, returning guidance or empty string on failure."""
    if chapter_num <= 1 or not previous_summaries.strip():
        return ""
    try:
        return call_llm(
            build_per_chapter_compression_check_prompt(
                chapter_num=chapter_num, chapter_summary=chapter_summary,
                previous_summaries=previous_summaries, title=title,
            ),
            action=f"Running Per-Chapter Compression Check for Chapter {chapter_num}"
        )
    except Exception as exc:
        failure_summary = _log_pass_failure(
            exc, pass_name="per-chapter compression check",
            chapter_num=chapter_num, chapter_title=title, optional=True,
        )
        if degraded_passes is not None:
            degraded_passes.append({
                "pass_name": "per-chapter compression check",
                "chapter_num": chapter_num,
                "failure_summary": failure_summary,
            })
        return ""


def run_character_state_updater(
    chapter_text: str, chapter_summary: str, characters_text: str,
    chapter_num: int, title: str,
    degraded_passes: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """Run the character state updater, returning ``(state_log, ledger_updates)``.

    ``state_log`` is the bullet-point character state text (empty string on
    failure). ``ledger_updates`` is a list of object-ledger delta records
    (empty list on failure or when no plot-critical object moved). The
    caller merges ledger_updates into a cumulative object ledger keyed by
    ``object_name``.

    If the LLM returns plain text instead of JSON (for example because the
    prompt change landed mid-run), the text is treated as the state log and
    the ledger updates are an empty list.
    """
    try:
        raw = call_llm(
            build_character_state_updater_prompt(
                chapter_text=chapter_text, chapter_summary=chapter_summary,
                characters_text=characters_text, chapter_num=chapter_num, title=title,
            ),
            action=f"Running Character State Updater for Chapter {chapter_num}",
            json_mode=True,
        )
    except Exception as exc:
        failure_summary = _log_pass_failure(
            exc, pass_name="character state updater",
            chapter_num=chapter_num, chapter_title=title, optional=True,
        )
        if degraded_passes is not None:
            degraded_passes.append({
                "pass_name": "character state updater",
                "chapter_num": chapter_num,
                "failure_summary": failure_summary,
            })
        return "", []

    # Tolerant parse: JSON preferred, plain text accepted as legacy fallback.
    try:
        data = parse_llm_json(raw)
    except Exception:
        data = None
    if isinstance(data, dict):
        state_log = str(data.get("character_state_log", "") or "").strip()
        ledger_updates_raw = data.get("object_ledger_updates", [])
        ledger_updates: list[dict] = []
        if isinstance(ledger_updates_raw, list):
            for item in ledger_updates_raw:
                if not isinstance(item, dict):
                    continue
                object_name = str(item.get("object_name", "")).strip()
                if not object_name:
                    continue
                ledger_updates.append({
                    "object_name": object_name,
                    "current_holder": str(item.get("current_holder", "")).strip(),
                    "last_transfer_method": str(item.get("last_transfer_method", "")).strip(),
                    "plot_critical": bool(item.get("plot_critical", True)),
                })
        return state_log, ledger_updates
    # Legacy / non-JSON response: keep the text and emit no ledger updates.
    return str(raw or "").strip(), []


def _apply_name_substitution(text: str, old_name: str, replacement: str) -> str:
    """Replace every whole-word occurrence of *old_name* with *replacement*.

    Handles possessive ``'s`` suffixes (``"Thomas's"`` → ``"the clerk's"``)
    and auto-capitalizes *replacement* when the match is at a sentence
    boundary so demoted role phrases read naturally at sentence starts.
    """
    if not old_name or not replacement:
        return text
    cap_replacement = replacement[0].upper() + replacement[1:]
    pattern = r"\b" + re.escape(old_name) + r"('s)?\b"

    def _replace(m: re.Match) -> str:
        start = m.start()
        before = text[:start].rstrip(" \t")
        at_sentence_start = (not before) or before[-1] in ".!?\n"
        out = cap_replacement if at_sentence_start else replacement
        return out + (m.group(1) or "")

    return re.sub(pattern, _replace, text)


def _repair_registered_character(char: dict) -> dict:
    """Enforce ``CHARACTER_FIELD_LIMITS`` on a newly-registered character.

    Attempts a single LLM rewrite per over-length field and falls back to
    :func:`truncate_character_field` if the repair itself exceeds the cap.
    This mirrors the behaviour of ``_enforce_character_field_limits`` in
    :mod:`novelforge.routes.outline`.
    """
    violations = validate_character_fields(char)
    for field_name, actual_len in violations.items():
        max_len = CHARACTER_FIELD_LIMITS[field_name]
        current_value = str(char.get(field_name, ""))
        logger.warning(
            "Reconciled character %r field %r is %d chars (limit %d); attempting repair",
            char.get("name", "?"), field_name, actual_len, max_len,
        )
        repaired: str | None = None
        try:
            repaired = call_llm(
                build_character_field_repair_prompt(
                    character_name=str(char.get("name", "")),
                    field_name=field_name,
                    current_value=current_value,
                    max_length=max_len,
                ),
                action=f"Reconciliation: repairing {field_name}",
            ).strip().strip('"').strip("'")
        except Exception as exc:
            logger.warning(
                "Reconciliation repair failed for %r.%s: %s",
                char.get("name", "?"), field_name, exc,
            )
        if repaired and len(repaired) <= max_len:
            char[field_name] = repaired
            continue
        source = repaired if repaired else current_value
        char[field_name] = truncate_character_field(source, max_len)
    return char


def run_character_reconciliation(
    chapter_text: str,
    character_list: list[dict],
    chapter_num: int,
    title: str,
    genre: str,
    degraded_passes: list[dict] | None = None,
) -> tuple[str, list[dict], dict]:
    """Reconcile newly-detected named characters in *chapter_text* against the canonical roster.

    Runs the pure-Python scanner, then — only if unknowns or variants are
    found — calls the LLM reconciliation agent to decide between REGISTER,
    RENAME_TO, and DEMOTE_TO_EXTRA for each detection. Applies decisions
    directly:

    * REGISTER → appends a new (field-limit-validated) character to the
      returned ``character_list``.
    * RENAME_TO → rewrites the prose, substituting the prose name with the
      canonical roster name.
    * DEMOTE_TO_EXTRA → rewrites the prose, substituting the prose name
      with a role phrase (e.g. ``"the clerk"``).

    Returns ``(updated_chapter_text, updated_character_list, log)`` where
    ``log`` is a dict containing the scan result and the applied decisions,
    suitable for progress-tracking and diagnostic export.
    """
    scan = extract_named_characters(chapter_text, character_list)
    unknowns: list[tuple[str, int]] = list(scan.get("unknown", []))
    variants: list[tuple[str, str, int]] = list(scan.get("variants", []))

    if not unknowns and not variants:
        return chapter_text, character_list, {"scan": scan, "decisions": []}

    try:
        raw = call_llm(
            build_character_reconciliation_prompt(
                chapter_text=chapter_text, chapter_num=chapter_num,
                title=title, genre=genre, character_list=character_list,
                unknowns=unknowns, variants=variants,
            ),
            action=f"Chapter {chapter_num}: character reconciliation",
            json_mode=True,
        )
        parsed = parse_llm_json(raw)
    except Exception as exc:
        failure_summary = _log_pass_failure(
            exc, pass_name="character reconciliation",
            chapter_num=chapter_num, chapter_title=title, optional=True,
        )
        if degraded_passes is not None:
            degraded_passes.append({
                "pass_name": "character reconciliation",
                "chapter_num": chapter_num,
                "failure_summary": failure_summary,
            })
        return chapter_text, character_list, {"scan": scan, "decisions": [], "error": failure_summary}

    if not isinstance(parsed, dict):
        return chapter_text, character_list, {"scan": scan, "decisions": []}
    decisions = parsed.get("decisions", [])
    if not isinstance(decisions, list):
        return chapter_text, character_list, {"scan": scan, "decisions": []}

    updated_text = chapter_text
    updated_list: list[dict] = list(character_list)
    applied: list[dict] = []
    existing_names = {str(c.get("name", "")).strip() for c in updated_list if isinstance(c, dict)}

    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        prose_name = str(decision.get("prose_name", "")).strip()
        resolution = str(decision.get("resolution", "")).strip().upper()
        if not prose_name or not resolution:
            continue

        if resolution == "REGISTER":
            profile = decision.get("profile")
            if not isinstance(profile, dict):
                continue
            new_name = str(profile.get("name") or decision.get("roster_name") or prose_name).strip()
            if not new_name or new_name in existing_names:
                continue
            profile["name"] = new_name
            profile.setdefault("age", "")
            for field_name in ("role", "background", "arc", "internal_flaw", "personal_goal"):
                profile.setdefault(field_name, "")
            profile = _repair_registered_character(profile)
            updated_list.append(profile)
            existing_names.add(new_name)
            applied.append({
                "prose_name": prose_name, "resolution": "REGISTER",
                "roster_name": new_name, "notes": decision.get("notes", ""),
            })
            logger.info(
                "Chapter %d: REGISTERED new character %r from prose (detected as %r)",
                chapter_num, new_name, prose_name,
            )

        elif resolution == "RENAME_TO":
            target = str(decision.get("roster_name", "")).strip()
            if not target:
                continue
            updated_text = _apply_name_substitution(updated_text, prose_name, target)
            applied.append({
                "prose_name": prose_name, "resolution": "RENAME_TO",
                "roster_name": target, "notes": decision.get("notes", ""),
            })
            logger.info(
                "Chapter %d: RENAMED prose name %r → roster %r",
                chapter_num, prose_name, target,
            )

        elif resolution == "DEMOTE_TO_EXTRA":
            replacement = str(decision.get("replacement", "")).strip()
            if not replacement:
                continue
            updated_text = _apply_name_substitution(updated_text, prose_name, replacement)
            applied.append({
                "prose_name": prose_name, "resolution": "DEMOTE_TO_EXTRA",
                "replacement": replacement, "notes": decision.get("notes", ""),
            })
            logger.info(
                "Chapter %d: DEMOTED walk-on %r → %r",
                chapter_num, prose_name, replacement,
            )

    return updated_text, updated_list, {"scan": scan, "decisions": applied}


def run_chapter_pattern_extractor(chapter_text: str, chapter_num: int, title: str,
                                  degraded_passes: list[dict] | None = None) -> dict:
    """Run the pattern extractor, returning a dict with procedural_types_dramatized and opening_style."""
    try:
        raw = call_llm(
            build_chapter_pattern_extractor_prompt(
                chapter_text=chapter_text, chapter_num=chapter_num, title=title,
            ),
            action=f"Running Chapter Pattern Extractor for Chapter {chapter_num}",
            json_mode=True,
        )
        result = parse_llm_json(raw)
        if not isinstance(result, dict):
            return {}
        return result
    except Exception as exc:
        failure_summary = _log_pass_failure(
            exc, pass_name="chapter pattern extractor",
            chapter_num=chapter_num, chapter_title=title, optional=True,
        )
        if degraded_passes is not None:
            degraded_passes.append({
                "pass_name": "chapter pattern extractor",
                "chapter_num": chapter_num,
                "failure_summary": failure_summary,
            })
        return {}


def run_continuity_gatekeeper(
    chapter_num: int, chapter_title: str, chapter_summary: str, previous_summaries: str,
    chapter_timeline_context: str = "", chapter_fate_context: str = "",
    chapter_arc_context: str = "", character_state_log: str = "",
    chapter_technology_context: str = "",
    chapter_rhythm_shape: str = "",
    object_ledger_context: str = "",
    degraded_passes: list[dict] | None = None,
) -> str:
    """Run the pre-chapter continuity gatekeeper, returning a brief or empty string on failure."""
    try:
        return call_llm(
            build_continuity_gatekeeper_prompt(
                chapter_num=chapter_num, chapter_title=chapter_title,
                chapter_summary=chapter_summary, previous_summaries=previous_summaries,
                chapter_timeline_context=chapter_timeline_context,
                chapter_fate_context=chapter_fate_context,
                chapter_arc_context=chapter_arc_context,
                character_state_log=character_state_log,
                chapter_technology_context=chapter_technology_context,
                chapter_rhythm_shape=chapter_rhythm_shape,
                object_ledger_context=object_ledger_context,
            ),
            action=f"Running Continuity Gatekeeper for Chapter {chapter_num}",
        )
    except Exception as exc:
        failure_summary = _log_pass_failure(
            exc, pass_name="continuity gatekeeper",
            chapter_num=chapter_num, chapter_title=chapter_title, optional=True,
        )
        if degraded_passes is not None:
            degraded_passes.append({
                "pass_name": "continuity gatekeeper",
                "chapter_num": chapter_num,
                "failure_summary": failure_summary,
            })
        return ""


def run_chapter_rhythm_classifier(
    chapter_num: int, chapter_title: str, chapter_summary: str, previous_summaries: str, title: str,
    chapter_architecture_context: str = "",
    rhythm_log: list[dict] | None = None,
    degraded_passes: list[dict] | None = None,
) -> dict:
    """Run the chapter rhythm classifier, returning a dict with shape recommendation or PASS_FAILURE_KEY on failure."""
    try:
        raw = call_llm(
            build_chapter_rhythm_classifier_prompt(
                chapter_num=chapter_num, chapter_title=chapter_title,
                chapter_summary=chapter_summary, previous_summaries=previous_summaries,
                title=title, chapter_architecture_context=chapter_architecture_context,
                rhythm_log=rhythm_log or [],
            ),
            action=f"Classifying chapter rhythm for Chapter {chapter_num}",
            json_mode=True,
        )
        result = parse_llm_json(raw)
        if not isinstance(result, dict):
            raise ValueError("LLM returned a JSON array instead of an object")
        return result
    except Exception as exc:
        failure_summary = _log_pass_failure(
            exc, pass_name="chapter rhythm classifier",
            chapter_num=chapter_num, chapter_title=title, optional=True,
        )
        if degraded_passes is not None:
            degraded_passes.append({
                "pass_name": "chapter rhythm classifier",
                "chapter_num": chapter_num,
                "failure_summary": failure_summary,
            })
        return {
            PASS_FAILURE_KEY: failure_summary,
            "recommended_shape_for_this_chapter": "",
            "recommendation_reason": "",
        }


# ---------------------------------------------------------------------------
# Main chapter agent pipeline
# ---------------------------------------------------------------------------

def _run_all_chapter_agents(
    text: str,
    chapter_num: int,
    title: str,
    genre: str,
    total_chapters: int,
    chapter_outline_summary: str,
    characters_text: str,
    previous_summaries: str,
    ctx: ChapterContext | None = None,
    step_callback: Callable[[str], None] | None = None,
    deadline: float = 0,
    degraded_passes: list[dict] | None = None,
    chapter_rhythm_shape: str = "",
    chapter_rhythm_reason: str = "",
    target_words: int = 0,
) -> tuple[str, str]:
    """
    Run all chapter refinement agents (post-draft) and return:
    (final_chapter_text, continuity_summary)

    Planning-agent context is passed via a ``ChapterContext`` dataclass.

    If *deadline* is non-zero (a ``time.monotonic()`` timestamp), each step
    checks the clock before calling the LLM and raises ``ChapterTimeoutError``
    if the deadline has passed.

    Optional pass failures are logged and, when *degraded_passes* is provided,
    appended to that list so the caller can surface them in progress metadata.
    """
    if ctx is None:
        ctx = ChapterContext()

    # Build the vocabulary-constraint block once for the whole pipeline.
    # Every prose-rewriting agent gets this injected into its system prompt
    # so forbidden words are never introduced by any agent in the chain.
    vocab_rules = format_vocabulary_rules(genre)

    def _check_deadline() -> None:
        """Raise ChapterTimeoutError if the per-chapter deadline has passed."""
        if deadline and time.monotonic() > deadline:
            raise ChapterTimeoutError(
                f"Chapter {chapter_num} exceeded the {PER_CHAPTER_TIMEOUT // 60}-minute time limit."
            )

    # Local shorthand: every agent call goes through the content-retry wrapper.
    # The wrapper also injects vocabulary constraints into the system message
    # so that every prose-rewriting agent is told about forbidden words.
    def _safe(build_msgs: Callable[[str], list[dict]], txt: str, *, action: str, json_mode: bool = False) -> str:
        """Call the LLM via the content-retry wrapper with vocabulary rules injected."""
        def _build_with_vocab_rules(t: str) -> list[dict]:
            messages = build_msgs(t)
            if vocab_rules and messages and messages[0].get("role") == "system":
                messages[0] = dict(messages[0])  # avoid mutating cached prompts
                messages[0]["content"] += f"\n\n{vocab_rules}"
            return messages
        return _call_with_content_retry(
            _build_with_vocab_rules, txt, action=action,
            chapter_num=chapter_num, title=title, json_mode=json_mode,
        )

    # Rhythm compliance verifier — ensure draft follows assigned rhythm
    if chapter_rhythm_shape:
        _check_deadline()
        if step_callback:
            step_callback(f"Chapter {chapter_num}: verifying rhythm compliance")
        text = _safe(
            lambda t: build_rhythm_compliance_verifier_prompt(
                t, chapter_num, title, chapter_rhythm_shape, chapter_rhythm_reason,
            ),
            text, action=f"Chapter {chapter_num}: rhythm compliance",
        )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: prose refinement (dialogue + scenes)")
    text = _safe(
        lambda t: build_prose_refinement_agent_prompt(t, chapter_num, title),
        text, action=f"Chapter {chapter_num}: prose refinement",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: voice & dialogue differentiation")
    text = _safe(
        lambda t: build_voice_dialogue_differentiation_prompt(
            t, chapter_num, title, characters_text, ctx.perspective_prompt,
        ),
        text, action=f"Chapter {chapter_num}: voice & dialogue differentiation",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: scene variety & compression audit")
    scene_audit_directives = run_scene_variety_compression_auditor(
        chapter_text=text, chapter_summary=chapter_outline_summary,
        chapter_num=chapter_num, title=title,
        previous_summaries=previous_summaries,
        degraded_passes=degraded_passes,
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: verifying continuity")
    text = _safe(
        lambda t: build_context_analyzer_prompt(
            t, previous_summaries, chapter_num, title,
            ctx.timeline, ctx.technology, ctx.theme,
        ),
        text, action=f"Chapter {chapter_num}: verifying continuity",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: editing")
    text = _safe(
        lambda t: build_editing_agent_prompt(t, chapter_outline_summary, chapter_num, title, scene_audit_directives),
        text, action=f"Chapter {chapter_num}: editing",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: momentum & distinctiveness check")
    text = _safe(
        lambda t: build_narrative_momentum_distinctiveness_prompt(
            t, previous_summaries, chapter_outline_summary, chapter_num, title, total_chapters,
            chapter_rhythm_shape=chapter_rhythm_shape,
        ),
        text, action=f"Chapter {chapter_num}: momentum & distinctiveness",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: human oddities")
    text = _safe(
        lambda t: build_human_oddities_prompt(
            t, chapter_num, title, total_chapters, characters_text,
        ),
        text, action=f"Chapter {chapter_num}: human oddities",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: checking structure")
    text = _safe(
        lambda t: build_structure_agent_prompt(
            t, chapter_num, total_chapters, chapter_outline_summary, ctx.architecture,
            chapter_rhythm_shape=chapter_rhythm_shape,
        ),
        text, action=f"Chapter {chapter_num}: checking structure",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: verifying operational distinctiveness")
    text = _safe(
        lambda t: build_operational_distinctiveness_prompt(
            t, previous_summaries, chapter_outline_summary, chapter_num, title,
            chapter_rhythm_shape=chapter_rhythm_shape,
        ),
        text, action=f"Chapter {chapter_num}: verifying operational distinctiveness",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: deepening characters")
    text = _safe(
        lambda t: build_character_agent_prompt(
            t, characters_text, chapter_num, title,
            ctx.fate, ctx.arc, ctx.antagonist, ctx.pov, ctx.perspective_prompt,
        ),
        text, action=f"Chapter {chapter_num}: deepening characters",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: synthesizing")
    text = _safe(
        lambda t: build_synthesizer_prompt(t, chapter_num, title, genre, ctx.perspective_prompt),
        text, action=f"Chapter {chapter_num}: synthesizing",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: polishing")
    text = _safe(
        lambda t: build_polish_agent_prompt(t, chapter_num, title, genre),
        text, action=f"Chapter {chapter_num}: polishing",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: anti-LLM pass")
    text = _safe(
        lambda t: build_anti_llm_agent_prompt(t, chapter_num, title, genre),
        text, action=f"Chapter {chapter_num}: anti-LLM pass",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: metaphor reduction")
    text = _safe(
        lambda t: build_metaphor_reduction_prompt(t, chapter_num, title),
        text, action=f"Chapter {chapter_num}: metaphor reduction",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: quality control")
    text = _safe(
        lambda t: build_quality_controller_prompt(t, chapter_num, title, genre, total_chapters),
        text, action=f"Chapter {chapter_num}: quality control",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: copy edit")
    text = _safe(
        lambda t: build_copy_edit_prompt(t, chapter_num, title),
        text, action=f"Chapter {chapter_num}: copy edit",
    )

    # Vocabulary diversity scan — pure Python, no LLM call.
    # Runs AFTER copy edit (the last prose-rewriting agent) so that no
    # subsequent agent can reintroduce forbidden words.
    _check_deadline()
    violations = scan_vocabulary_overuse(text, genre=genre)
    if violations:
        if step_callback:
            step_callback(f"Chapter {chapter_num}: fixing {len(violations)} vocabulary issues")
        logger.info("Chapter %d: vocabulary scan found %d violations", chapter_num, len(violations))
        text = _safe(
            lambda t: build_vocabulary_fix_prompt(t, chapter_num, title, violations),
            text, action=f"Chapter {chapter_num}: vocabulary fix-up",
        )

    # Length enforcement — expand under-length chapters after all refinement
    # passes have finished so that the expansion doesn't get trimmed.
    # Expansion uses _safe() so vocabulary rules and content-retry apply,
    # and a vocabulary scan runs afterwards to catch any reintroduced words.
    if target_words > 0:
        from novelforge.agents.chapter._helpers import check_chapter_length
        from novelforge.agents.chapter.prompts import build_chapter_expansion_prompt
        import novelforge.config as _cfg

        min_pct = _cfg.CHAPTER_MIN_LENGTH_PCT
        actual_wc, min_threshold, acceptable = check_chapter_length(text, target_words, min_pct)
        if not acceptable:
            for _exp_attempt in range(1, _cfg.MAX_EXPANSION_ATTEMPTS + 1):
                if actual_wc >= min_threshold:
                    break
                _check_deadline()
                if step_callback:
                    step_callback(
                        f"Chapter {chapter_num}: expanding ({actual_wc} words, "
                        f"need {min_threshold}, attempt {_exp_attempt})"
                    )
                logger.info(
                    "Chapter %d: expansion attempt %d/%d (%d words, need %d)",
                    chapter_num, _exp_attempt, _cfg.MAX_EXPANSION_ATTEMPTS,
                    actual_wc, min_threshold,
                )
                try:
                    _cur_wc = actual_wc
                    _cur_target = target_words
                    _cur_min = min_threshold
                    expanded = _safe(
                        lambda t: build_chapter_expansion_prompt(
                            chapter_text=t,
                            current_words=_cur_wc,
                            target_words=_cur_target,
                            min_words=_cur_min,
                        ),
                        text,
                        action=f"Chapter {chapter_num}: expansion (attempt {_exp_attempt})",
                    )
                    new_wc = len(expanded.split())
                    if new_wc > actual_wc:
                        text = expanded
                        actual_wc = new_wc
                        logger.info(
                            "Chapter %d: expansion attempt %d produced %d words",
                            chapter_num, _exp_attempt, new_wc,
                        )
                    else:
                        logger.warning(
                            "Chapter %d: expansion attempt %d did not increase length (%d → %d)",
                            chapter_num, _exp_attempt, actual_wc, new_wc,
                        )
                        break
                except Exception as exc:
                    logger.warning(
                        "Chapter %d: expansion attempt %d failed: %s: %s",
                        chapter_num, _exp_attempt, type(exc).__name__, exc,
                    )
                    break

            logger.info(
                "Chapter %d: post-expansion word count = %d (target=%d, min=%d)",
                chapter_num, actual_wc, target_words, min_threshold,
            )

            # Re-run vocabulary scan after expansion since the LLM may have
            # reintroduced forbidden or overused words.
            _check_deadline()
            post_violations = scan_vocabulary_overuse(text, genre=genre)
            if post_violations:
                if step_callback:
                    step_callback(
                        f"Chapter {chapter_num}: fixing {len(post_violations)} "
                        f"vocabulary issues (post-expansion)"
                    )
                logger.info(
                    "Chapter %d: post-expansion vocabulary scan found %d violations",
                    chapter_num, len(post_violations),
                )
                text = _safe(
                    lambda t: build_vocabulary_fix_prompt(
                        t, chapter_num, title, post_violations,
                    ),
                    text,
                    action=f"Chapter {chapter_num}: vocabulary fix-up (post-expansion)",
                )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: summarising")
    summary = _safe(
        lambda t: build_chapter_summary_prompt(t, chapter_num),
        text, action=f"Chapter {chapter_num}: summarising",
    )

    return text, summary
