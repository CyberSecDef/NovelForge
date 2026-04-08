"""Export, download, and illustration routes."""

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, abort, jsonify, request, send_file

from novelforge import limiter
from novelforge.progress import (
    progress_manager,
    set_correlation_token,
    clear_correlation_token,
)
import novelforge.config as config
from novelforge.llm.client import call_llm, parse_llm_json, friendly_llm_error
from novelforge.llm.image import call_image_api
from novelforge.agents.chapter import build_illustration_prompt_generator_prompt

logger = logging.getLogger(__name__)

export_bp = Blueprint("export", __name__)


# ---------------------------------------------------------------------------
# Export format functions
# ---------------------------------------------------------------------------

def _format_manuscript(title: str, chapters_done: list[dict]) -> str:
    """Format the completed novel as a clean Markdown manuscript."""
    lines = [f"# {title}\n"]
    for ch in chapters_done:
        lines.append(f"\n## Chapter {ch['number']}: {ch['title']}\n")
        lines.append(f"\n{ch['content']}\n")
    return "\n".join(lines)


def _md_text(text: object) -> str:
    """Escape a value for inline Markdown (list items, bold labels, headings).

    All line-ending variants (LF, CR+LF, bare CR) are replaced with a single
    space so that the content stays on one logical line and cannot break
    surrounding Markdown structure.
    """
    return str(text).replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")


def _md_table_cell(text: object) -> str:
    """Escape a value for a Markdown table cell.

    - Pipe characters are escaped as ``\\|`` to avoid splitting the column.
    - All line-ending variants are replaced with a space (raw newlines break
      the table row).
    """
    return (
        str(text)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", " ")
        .replace("|", r"\|")
    )


def _mermaid_node_label(text: object) -> str:
    """Escape a value for a Mermaid node label inside ``["…"]`` syntax.

    Double-quotes are replaced with the Mermaid HTML entity ``#quot;`` so they
    cannot close the label prematurely.  All line-ending variants are collapsed
    to a space.
    """
    return (
        str(text)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", " ")
        .replace('"', "#quot;")
    )


def _mermaid_edge_label(text: object) -> str:
    """Escape a value for a Mermaid edge label used with ``-- "…" -->`` syntax.

    Double-quotes are replaced with ``#quot;``; all line-ending variants become
    spaces.  Pipes do not need escaping in this quoting style.
    """
    return (
        str(text)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", " ")
        .replace('"', "#quot;")
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@export_bp.route("/export", methods=["POST"])
def export_novel() -> Response | tuple[Response, int]:
    """Compile the completed novel into a Markdown file and return a download URL."""
    data = request.get_json(silent=True) or {}
    token = data.get("token", "")

    progress_data = progress_manager.get(token)

    if not progress_data or progress_data.get("status") != "done":
        return jsonify({"error": "Novel generation not complete."}), 400

    title = (progress_data.get("snapshot") or {}).get("title", "Novel")
    chapters_done = progress_data.get("chapters_done", [])

    markdown_content = _format_manuscript(title, chapters_done)

    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:80]
    filename = f"{safe_title}.md"
    export_path = Path(config.EXPORT_DIR) / filename

    export_path.write_text(markdown_content, encoding="utf-8")

    return jsonify({"download_url": f"/download/{filename}"})


@export_bp.route("/export_editors_notes", methods=["POST"])
def export_editors_notes() -> Response | tuple[Response, int]:
    """Export editor's notes into a Markdown file and return a download URL."""
    data = request.get_json(silent=True) or {}
    token = data.get("token", "")

    progress_data = progress_manager.get(token)

    if not progress_data or progress_data.get("status") != "done":
        return jsonify({"error": "Novel generation not complete."}), 400

    title = (progress_data.get("snapshot") or {}).get("title", "Novel")
    consistency = progress_data.get("consistency") or {}
    global_continuity_audit = progress_data.get("global_continuity_audit") or {}
    narrative_compression_report = progress_data.get("narrative_compression_report") or {}
    character_resolution_report = progress_data.get("character_resolution_report") or {}
    thematic_payoff_report = progress_data.get("thematic_payoff_report") or {}
    climax_integrity_report = progress_data.get("climax_integrity_report") or {}
    loose_thread_report = progress_data.get("loose_thread_report") or {}
    reader_immersion_report = progress_data.get("reader_immersion_report") or {}
    pacing_heatmap = progress_data.get("pacing_heatmap") or {}
    character_relationship_map = progress_data.get("character_relationship_map") or {}

    has_content = any([
        consistency.get("overall_assessment") or consistency.get("issues"),
        global_continuity_audit, narrative_compression_report,
        character_resolution_report, thematic_payoff_report,
        climax_integrity_report, loose_thread_report,
        reader_immersion_report, pacing_heatmap,
        character_relationship_map.get("relationships"),
    ])

    if not has_content:
        return jsonify({"error": "No editor's notes are available for this novel."}), 400

    lines = [f"# {_md_text(title)} - Editor's Notes\n"]
    lines.append("This document contains all diagnostic reports from the novel generation process. ")
    lines.append("Use these notes to identify chapters that may need revision.\n")

    # 1. Consistency Pass
    if consistency:
        lines.append("---\n")
        lines.append("## 1. Consistency Pass\n")
        overall_assessment = (consistency.get("overall_assessment") or "").strip()
        if overall_assessment:
            lines.append(f"**Overall Assessment:** {_md_text(overall_assessment)}\n")
        issues = consistency.get("issues") or []
        if issues:
            lines.append("**Issues:**\n")
            for issue in issues:
                lines.append(f"- {_md_text(issue)}")
            lines.append("")

    # 2. Global Continuity Audit
    if global_continuity_audit:
        lines.append("---\n")
        lines.append("## 2. Global Continuity Audit\n")
        overall = (global_continuity_audit.get("overall_assessment") or "").strip()
        integrity = (global_continuity_audit.get("overall_integrity") or "").strip()
        if integrity:
            lines.append(f"**Overall Integrity:** {_md_text(integrity)}\n")
        if overall:
            lines.append(f"**Assessment:** {_md_text(overall)}\n")
        contradictions = global_continuity_audit.get("contradictions") or []
        if contradictions:
            lines.append("**Contradictions:**\n")
            for c in contradictions:
                if isinstance(c, dict):
                    chapters = c.get("chapters", [])
                    desc = c.get("description", str(c))
                    lines.append(f"- Chapters {chapters}: {_md_text(desc)}")
                else:
                    lines.append(f"- {_md_text(c)}")
            lines.append("")
        char_errors = global_continuity_audit.get("character_state_errors") or []
        if char_errors:
            lines.append("**Character State Errors:**\n")
            for e in char_errors:
                lines.append(f"- {_md_text(e)}")
            lines.append("")
        timeline_errors = global_continuity_audit.get("timeline_errors") or []
        if timeline_errors:
            lines.append("**Timeline Errors:**\n")
            for e in timeline_errors:
                lines.append(f"- {_md_text(e)}")
            lines.append("")

    # 3. Narrative Compression Report
    if narrative_compression_report:
        lines.append("---\n")
        lines.append("## 3. Narrative Compression Report\n")
        priority = (narrative_compression_report.get("compression_priority") or "").strip()
        overall = (narrative_compression_report.get("overall_assessment") or "").strip()
        if priority:
            lines.append(f"**Compression Priority:** {_md_text(priority)}\n")
        if overall:
            lines.append(f"**Assessment:** {_md_text(overall)}\n")
        redundant = narrative_compression_report.get("redundant_sequences") or []
        if redundant:
            lines.append("**Redundant Sequences:**\n")
            for r in redundant:
                if isinstance(r, dict):
                    chapters = r.get("chapters", [])
                    pattern = r.get("pattern", "")
                    rec = r.get("recommendation", "")
                    lines.append(f"- Chapters {chapters}: {_md_text(pattern)}")
                    if rec:
                        lines.append(f"  - *Recommendation:* {_md_text(rec)}")
                else:
                    lines.append(f"- {_md_text(r)}")
            lines.append("")
        emotional = narrative_compression_report.get("emotional_beat_repetitions") or []
        if emotional:
            lines.append("**Emotional Beat Repetitions:**\n")
            for e in emotional:
                if isinstance(e, dict):
                    chapters = e.get("chapters", [])
                    beat = e.get("beat", "")
                    rec = e.get("recommendation", "")
                    lines.append(f"- Chapters {chapters}: {_md_text(beat)}")
                    if rec:
                        lines.append(f"  - *Recommendation:* {_md_text(rec)}")
                else:
                    lines.append(f"- {_md_text(e)}")
            lines.append("")

    # 4-9: remaining reports (abbreviated for space but identical logic)
    if character_resolution_report:
        lines.append("---\n")
        lines.append("## 4. Character Resolution Report\n")
        integrity = (character_resolution_report.get("resolution_integrity") or "").strip()
        overall = (character_resolution_report.get("overall_assessment") or "").strip()
        if integrity:
            lines.append(f"**Resolution Integrity:** {_md_text(integrity)}\n")
        if overall:
            lines.append(f"**Assessment:** {_md_text(overall)}\n")
        unresolved = character_resolution_report.get("unresolved_characters") or []
        if unresolved:
            lines.append("**Unresolved Characters:**\n")
            for u in unresolved:
                if isinstance(u, dict):
                    name = u.get("character", u.get("name", "Unknown"))
                    issue = u.get("issue", u.get("description", str(u)))
                    lines.append(f"- **{_md_text(name)}**: {_md_text(issue)}")
                else:
                    lines.append(f"- {_md_text(u)}")
            lines.append("")
        resolutions = character_resolution_report.get("character_resolutions") or []
        if resolutions:
            lines.append("**Character Resolutions:**\n")
            for r in resolutions:
                if isinstance(r, dict):
                    name = r.get("character", r.get("name", "Unknown"))
                    status = r.get("status", r.get("resolution", str(r)))
                    lines.append(f"- **{_md_text(name)}**: {_md_text(status)}")
                else:
                    lines.append(f"- {_md_text(r)}")
            lines.append("")

    if thematic_payoff_report:
        lines.append("---\n")
        lines.append("## 5. Thematic Payoff Report\n")
        integrity = (thematic_payoff_report.get("thematic_integrity") or "").strip()
        overall = (thematic_payoff_report.get("overall_assessment") or "").strip()
        if integrity:
            lines.append(f"**Thematic Integrity:** {_md_text(integrity)}\n")
        if overall:
            lines.append(f"**Assessment:** {_md_text(overall)}\n")
        abandoned = thematic_payoff_report.get("abandoned_themes") or []
        if abandoned:
            lines.append("**Abandoned Themes:**\n")
            for t in abandoned:
                if isinstance(t, dict):
                    theme = t.get("theme", t.get("name", str(t)))
                    reason = t.get("reason", t.get("description", ""))
                    lines.append(f"- **{_md_text(theme)}**" + (f": {_md_text(reason)}" if reason else ""))
                else:
                    lines.append(f"- {_md_text(t)}")
            lines.append("")
        weak = thematic_payoff_report.get("weak_payoffs") or []
        if weak:
            lines.append("**Weak Payoffs:**\n")
            for w in weak:
                if isinstance(w, dict):
                    theme = w.get("theme", w.get("name", str(w)))
                    issue = w.get("issue", w.get("description", ""))
                    lines.append(f"- **{_md_text(theme)}**" + (f": {_md_text(issue)}" if issue else ""))
                else:
                    lines.append(f"- {_md_text(w)}")
            lines.append("")

    if climax_integrity_report:
        lines.append("---\n")
        lines.append("## 6. Climax Integrity Report\n")
        integrity = (climax_integrity_report.get("climax_integrity") or "").strip()
        overall = (climax_integrity_report.get("overall_assessment") or "").strip()
        climax_chapter = climax_integrity_report.get("climax_chapter")
        if climax_chapter:
            lines.append(f"**Climax Chapter:** {_md_text(climax_chapter)}\n")
        if integrity:
            lines.append(f"**Climax Integrity:** {_md_text(integrity)}\n")
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
                lines.append(f"- {_md_text(f)}")
            lines.append("")
        if overall:
            lines.append(f"**Assessment:** {_md_text(overall)}\n")

    if loose_thread_report:
        lines.append("---\n")
        lines.append("## 7. Loose Thread Report\n")
        integrity = (loose_thread_report.get("thread_integrity") or "").strip()
        overall = (loose_thread_report.get("overall_assessment") or "").strip()
        if integrity:
            lines.append(f"**Thread Integrity:** {_md_text(integrity)}\n")
        if overall:
            lines.append(f"**Assessment:** {_md_text(overall)}\n")
        unresolved = loose_thread_report.get("unresolved_threads") or []
        if unresolved:
            lines.append("**Unresolved Threads:**\n")
            for t in unresolved:
                if isinstance(t, dict):
                    thread = t.get("thread", t.get("description", str(t)))
                    chapters = t.get("chapters", t.get("introduced_in", ""))
                    lines.append(f"- {_md_text(thread)}" + (f" (Chapters: {chapters})" if chapters else ""))
                else:
                    lines.append(f"- {_md_text(t)}")
            lines.append("")
        dangling = loose_thread_report.get("dangling_setup_elements") or []
        if dangling:
            lines.append("**Dangling Setup Elements:**\n")
            for d in dangling:
                if isinstance(d, dict):
                    element = d.get("element", d.get("description", str(d)))
                    lines.append(f"- {_md_text(element)}")
                else:
                    lines.append(f"- {_md_text(d)}")
            lines.append("")
        intentional = loose_thread_report.get("intentionally_open_threads") or []
        if intentional:
            lines.append("**Intentionally Open Threads (for sequel):**\n")
            for t in intentional:
                if isinstance(t, dict):
                    thread = t.get("thread", t.get("description", str(t)))
                    lines.append(f"- {_md_text(thread)}")
                else:
                    lines.append(f"- {_md_text(t)}")
            lines.append("")

    if reader_immersion_report:
        lines.append("---\n")
        lines.append("## 8. Reader Immersion Report\n")
        overall_rating = (reader_immersion_report.get("overall_rating") or "").strip()
        engagement_score = reader_immersion_report.get("engagement_score")
        pacing = (reader_immersion_report.get("pacing_assessment") or "").strip()
        tension = (reader_immersion_report.get("tension_curve") or "").strip()
        stakes = (reader_immersion_report.get("stakes_clarity") or "").strip()
        if overall_rating:
            lines.append(f"**Overall Rating:** {_md_text(overall_rating)}\n")
        if engagement_score is not None:
            lines.append(f"**Engagement Score:** {engagement_score}/10\n")
        if pacing:
            lines.append(f"**Pacing Assessment:** {_md_text(pacing)}\n")
        if tension:
            lines.append(f"**Tension Curve:** {_md_text(tension)}\n")
        if stakes:
            lines.append(f"**Stakes Clarity:** {_md_text(stakes)}\n")
        weak_chapters = reader_immersion_report.get("weak_chapters") or []
        if weak_chapters:
            lines.append("**Weak Chapters (need revision):**\n")
            for w in weak_chapters:
                if isinstance(w, dict):
                    chapter = w.get("chapter", w.get("number", "?"))
                    reason = w.get("reason", w.get("issue", str(w)))
                    lines.append(f"- **Chapter {_md_text(chapter)}**: {_md_text(reason)}")
                else:
                    lines.append(f"- {_md_text(w)}")
            lines.append("")
        breaks = reader_immersion_report.get("immersion_breaks") or []
        if breaks:
            lines.append("**Immersion Breaks:**\n")
            for b in breaks:
                if isinstance(b, dict):
                    chapter = b.get("chapter", "?")
                    desc = b.get("description", b.get("issue", str(b)))
                    lines.append(f"- Chapter {_md_text(chapter)}: {_md_text(desc)}")
                else:
                    lines.append(f"- {_md_text(b)}")
            lines.append("")
        recommendations = reader_immersion_report.get("recommendations") or []
        if recommendations:
            lines.append("**Recommendations:**\n")
            for r in recommendations:
                lines.append(f"- {_md_text(r)}")
            lines.append("")

    if pacing_heatmap:
        lines.append("---\n")
        lines.append("## 9. Pacing & Tension Heatmap\n")
        overall = (pacing_heatmap.get("overall_pacing_assessment") or "").strip()
        if overall:
            lines.append(f"**Overall Pacing Assessment:** {_md_text(overall)}\n")
        metrics = pacing_heatmap.get("chapter_metrics") or []
        if metrics:
            def _bar(value: int) -> str:
                """Return a 10-character block bar representing a 0-100 percentage value."""
                clamped = max(0, min(100, int(value)))
                filled = round(clamped / 10)
                return "\u2588" * filled + "\u2591" * (10 - filled)

            lines.append("| Ch | Tension | Action | Emotion | Dialogue | Description |")
            lines.append("|---:|---------|--------|---------|----------|-------------|")
            for m in sorted(metrics, key=lambda x: x.get("chapter", 0)):
                if not isinstance(m, dict):
                    continue
                ch = m.get("chapter", "?")
                t = int(m.get("tension_score", 0))
                a = int(m.get("action_density", 0))
                e = int(m.get("emotional_intensity", 0))
                d = int(m.get("dialogue_ratio", 0))
                desc = int(m.get("description_ratio", 0))
                lines.append(
                    f"| {ch} "
                    f"| `{_bar(t)}` {t:3d} "
                    f"| `{_bar(a)}` {a:3d} "
                    f"| `{_bar(e)}` {e:3d} "
                    f"| `{_bar(d)}` {d:3d} "
                    f"| `{_bar(desc)}` {desc:3d} |"
                )
            lines.append("")
        flat_sections = pacing_heatmap.get("flat_sections") or []
        if flat_sections:
            lines.append("**Flat Sections (potential pacing issues):**\n")
            for fs in flat_sections:
                if isinstance(fs, dict):
                    chapters = fs.get("chapters", [])
                    issue = fs.get("issue", "")
                    lines.append(f"- Chapters {chapters}: {_md_text(issue)}")
                else:
                    lines.append(f"- {_md_text(fs)}")
            lines.append("")

    # 10. Character Relationship Map
    rel_chars = character_relationship_map.get("characters", [])
    rel_edges = character_relationship_map.get("relationships", [])
    if rel_edges:
        lines.append("---\n")
        lines.append("## 10. Character Relationship Map\n")

        # Render as a Mermaid diagram (viewable in GitHub, VS Code, etc.)
        lines.append("```mermaid")
        lines.append("graph LR")
        id_map: dict[str, str] = {}
        for idx, name in enumerate(rel_chars):
            cid = f"C{idx}"
            id_map[name] = cid
            lines.append(f'    {cid}["{_mermaid_node_label(name)}"]')
        seen: set[str] = set()
        for rel in rel_edges:
            from_id = id_map.get(str(rel.get("from", "")))
            to_id = id_map.get(str(rel.get("to", "")))
            if not from_id or not to_id:
                continue
            edge_key = f"{from_id}-{to_id}"
            if edge_key in seen:
                continue
            seen.add(edge_key)
            label = _mermaid_edge_label(rel.get("type", ""))
            lines.append(f'    {from_id} -- "{label}" --> {to_id}')
        lines.append("```\n")

        # Also render as a text table for plain Markdown readers
        lines.append("**Relationships:**\n")
        lines.append("| From | To | Type | Description |")
        lines.append("|------|------|------|-------------|")
        for rel in rel_edges:
            fr = rel.get("from", "?")
            to = rel.get("to", "?")
            rtype = rel.get("type", "?")
            desc = rel.get("label", "")
            lines.append(f"| {_md_table_cell(fr)} | {_md_table_cell(to)} | {_md_table_cell(rtype)} | {_md_table_cell(desc)} |")
        lines.append("")

    markdown_content = "\n".join(lines)

    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:80]
    safe_title = "_".join(safe_title.split()) or "Novel"
    filename = f"{safe_title}-Editors_Notes.md"
    export_path = Path(config.EXPORT_DIR) / filename

    export_path.write_text(markdown_content, encoding="utf-8")

    return jsonify({"download_url": f"/download/{filename}"})


@export_bp.route("/generate_illustrations", methods=["POST"])
@limiter.limit("2 per 10 minutes")
def generate_illustrations() -> Response | tuple[Response, int]:
    """Start background illustration generation; returns a job token for polling."""
    data = request.get_json(silent=True) or {}
    token = data.get("token", "")

    progress_data = progress_manager.get(token)

    if not progress_data or progress_data.get("status") != "done":
        return jsonify({"error": "Novel generation not complete."}), 400

    if not config.IMAGE_API_KEY:
        return jsonify({"error": "IMAGE_API_KEY not configured. Set it in your .env file."}), 400

    snapshot = progress_data.get("snapshot") or {}
    chapters_done = progress_data.get("chapters_done", [])

    illust_token = str(uuid.uuid4())
    progress_manager.create(illust_token, {
        "status": "running",
        "current": 0,
        "total": 0,
        "step": "Starting\u2026",
        "chapters_done": [],
        "error": None,
        "illustrations": [],
    })

    # Record the illustration job token on the novel entry so callers can
    # retrieve it from either side.
    try:
        progress_manager.update(token, {"illustration_token": illust_token})
    except KeyError:
        logger.warning(
            "Could not link illustration to novel (progress entry %s not found).",
            token,
        )

    thread = threading.Thread(
        target=_run_illustration_generation,
        args=(token, illust_token, snapshot, chapters_done),
        daemon=True,
    )
    thread.start()

    return jsonify({"illustration_token": illust_token})


def _run_illustration_generation(
    novel_token: str,
    illust_token: str,
    snapshot: dict[str, Any],
    chapters_done: list[dict[str, Any]],
) -> None:
    """Background worker: generate illustration prompts then images.

    Progress is tracked under *illust_token* and can be polled via the
    standard ``/progress/<token>`` endpoints.  Per-image outcomes are stored
    in the ``illustrations`` list so partial failures are visible.
    """
    set_correlation_token(illust_token)
    try:
        title = snapshot.get("title", "Novel")
        genre = snapshot.get("genre", "")
        premise = snapshot.get("premise", "")
        character_list = snapshot.get("character_list", [])
        all_summaries = [str(ch.get("summary", "")) for ch in chapters_done]

        # --- Step 1: generate illustration prompts via LLM (with retry) ---
        progress_manager.update(illust_token, {"step": "Generating illustration prompts\u2026"})

        raw: str | None = None
        for llm_attempt in range(1, 4):
            try:
                llm_prompt = build_illustration_prompt_generator_prompt(
                    title=title, genre=genre, premise=premise,
                    character_list=character_list, all_summaries=all_summaries,
                )
                raw = call_llm(llm_prompt, action="Generating illustration prompts", json_mode=True)
                break
            except RuntimeError:
                if llm_attempt == 3:
                    raise
                wait = 30 * llm_attempt
                logger.warning(
                    "Illustration LLM prompt failed (attempt %d/3) \u2013 retrying in %ds",
                    llm_attempt, wait,
                )
                progress_manager.update(illust_token, {
                    "step": f"Retrying prompt generation (attempt {llm_attempt}/3)\u2026",
                })
                time.sleep(wait)

        assert raw is not None  # guaranteed: final attempt raises if it fails
        prompt_data = parse_llm_json(raw)
        illustrations_spec = prompt_data.get("illustrations", []) if isinstance(prompt_data, dict) else []

        if not isinstance(illustrations_spec, list) or not illustrations_spec:
            progress_manager.update(illust_token, {
                "status": "error",
                "error": "LLM did not return valid illustration prompts.",
                "step": "Failed: no valid prompts",
            })
            return

        # --- Step 2: generate each image ---
        candidate_specs: list[dict[str, Any]] = [
            s for s in illustrations_spec[:2] if isinstance(s, dict)
        ]
        total = len(candidate_specs)
        progress_manager.update(illust_token, {
            "total": total,
            "step": f"Generating {total} image(s)\u2026",
        })

        results: list[dict[str, Any]] = []
        for idx, illust in enumerate(candidate_specs):
            art_prompt = str(illust.get("art_prompt", "")).strip()
            illust_type = str(illust.get("type", "chapter_scene"))
            chapter = illust.get("chapter")
            scene_desc = str(illust.get("scene_description", "")).strip()

            entry: dict[str, Any] = {
                "type": illust_type,
                "chapter": chapter,
                "scene_description": scene_desc,
                "art_prompt": art_prompt,
                "status": "pending",
                "image_url": None,
                "error": None,
            }

            if not art_prompt:
                entry["status"] = "image_failed"
                entry["error"] = "Empty art prompt"
                results.append(entry)
                progress_manager.update(illust_token, {
                    "current": idx + 1,
                    "step": f"Image {idx + 1}/{total}: skipped (empty prompt)",
                    "illustrations": list(results),
                })
                continue

            prefix = "cover" if illust_type == "cover" else f"ch{chapter or idx}"
            progress_manager.update(illust_token, {
                "step": f"Image {idx + 1}/{total}: calling image API\u2026",
            })
            try:
                filename = call_image_api(art_prompt, filename_prefix=prefix)
                if filename:
                    entry["status"] = "success"
                    entry["image_url"] = f"/illustrations/{filename}"
                else:
                    entry["status"] = "image_failed"
                    entry["error"] = "Image API returned no file"
            except Exception as exc:  # noqa: BLE001
                entry["status"] = "image_failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"

            results.append(entry)
            progress_manager.update(illust_token, {
                "current": idx + 1,
                "step": f"Image {idx + 1}/{total}: complete",
                "illustrations": list(results),
            })

        # --- Step 3: finalise ---
        successful = [r for r in results if r.get("status") == "success"]

        if not successful:
            progress_manager.update(illust_token, {
                "status": "error",
                "error": "Image generation failed for all prompts.",
                "step": "Failed",
                "illustrations": results,
            })
            return

        progress_manager.update(illust_token, {
            "status": "done",
            "step": "Complete",
            "illustrations": results,
        })

        # Mirror successful illustrations back onto the novel progress entry
        # so existing callers that read from the novel token still work.
        try:
            progress_manager.update(novel_token, {"illustrations": successful})
        except KeyError:
            pass

    except RuntimeError as exc:
        logger.error("Illustration generation failed: %s", exc)
        progress_manager.update(illust_token, {
            "status": "error",
            "error": friendly_llm_error(exc),
            "step": "Failed",
        })
    finally:
        clear_correlation_token()


@export_bp.route("/illustrations/<path:filename>")
def serve_illustration(filename: str) -> Response:
    """Serve a generated illustration image."""
    safe_filename = Path(filename).name
    img_path = Path(config.EXPORT_DIR) / "illustrations" / safe_filename
    if not img_path.exists():
        abort(404)
    return send_file(str(img_path), mimetype="image/png")


@export_bp.route("/download/<path:filename>")
def download_file(filename: str) -> Response:
    """Serve a generated export file."""
    safe_filename = Path(filename).name
    export_path = Path(config.EXPORT_DIR) / safe_filename
    if not export_path.exists():
        abort(404)
    return send_file(str(export_path), as_attachment=True, download_name=safe_filename)
