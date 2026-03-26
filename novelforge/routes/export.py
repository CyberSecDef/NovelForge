"""Export, download, and illustration routes."""

import json
import logging
import time
from pathlib import Path

from flask import Blueprint, Response, abort, jsonify, request, send_file, session

from novelforge import limiter
from novelforge.progress import _progress_store, _progress_lock
import novelforge.config as config
from novelforge.llm.client import call_llm, parse_llm_json, _friendly_llm_error
from novelforge.llm.image import call_image_api
from novelforge.agents.chapter import build_illustration_prompt_generator_prompt
from novelforge.session.persistence import save_session_state

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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@export_bp.route("/export", methods=["POST"])
def export_novel() -> Response | tuple[Response, int]:
    """Compile the completed novel into a Markdown file and return a download URL."""
    data = request.get_json(silent=True) or {}
    token = data.get("token", "")

    with _progress_lock:
        progress_data = _progress_store.get(token)

    if not progress_data or progress_data.get("status") != "done":
        return jsonify({"error": "Novel generation not complete."}), 400

    title = session.get("title", "Novel")
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

    with _progress_lock:
        progress_data = _progress_store.get(token)

    if not progress_data or progress_data.get("status") != "done":
        return jsonify({"error": "Novel generation not complete."}), 400

    title = session.get("title", "Novel")

    consistency = progress_data.get("consistency", {})
    global_continuity_audit = progress_data.get("global_continuity_audit", {})
    narrative_compression_report = progress_data.get("narrative_compression_report", {})
    character_resolution_report = progress_data.get("character_resolution_report", {})
    thematic_payoff_report = progress_data.get("thematic_payoff_report", {})
    climax_integrity_report = progress_data.get("climax_integrity_report", {})
    loose_thread_report = progress_data.get("loose_thread_report", {})
    reader_immersion_report = progress_data.get("reader_immersion_report", {})
    pacing_heatmap = progress_data.get("pacing_heatmap", {})

    has_content = any([
        consistency.get("overall_assessment") or consistency.get("issues"),
        global_continuity_audit, narrative_compression_report,
        character_resolution_report, thematic_payoff_report,
        climax_integrity_report, loose_thread_report,
        reader_immersion_report, pacing_heatmap,
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

    # 4-9: remaining reports (abbreviated for space but identical logic)
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

    if pacing_heatmap:
        lines.append("---\n")
        lines.append("## 9. Pacing & Tension Heatmap\n")
        overall = (pacing_heatmap.get("overall_pacing_assessment") or "").strip()
        if overall:
            lines.append(f"**Overall Pacing Assessment:** {overall}\n")
        metrics = pacing_heatmap.get("chapter_metrics") or []
        if metrics:
            def _bar(value: int) -> str:
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
                    lines.append(f"- Chapters {chapters}: {issue}")
                else:
                    lines.append(f"- {fs}")
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
    """Generate cover and chapter scene illustrations for the completed novel."""
    data = request.get_json(silent=True) or {}
    token = data.get("token", "")

    with _progress_lock:
        progress_data = _progress_store.get(token)

    if not progress_data or progress_data.get("status") != "done":
        return jsonify({"error": "Novel generation not complete."}), 400

    if not config.IMAGE_API_KEY:
        return jsonify({"error": "IMAGE_API_KEY not configured. Set it in your .env file."}), 400

    title = session.get("title", "Novel")
    genre = session.get("genre", "")
    premise = session.get("premise", "")
    character_list = session.get("character_list", [])
    chapters_done = progress_data.get("chapters_done", [])
    all_summaries = [str(ch.get("summary", "")) for ch in chapters_done]

    try:
        llm_prompt = build_illustration_prompt_generator_prompt(
            title=title, genre=genre, premise=premise,
            character_list=character_list, all_summaries=all_summaries,
        )
        raw = None
        for llm_attempt in range(1, 4):
            try:
                raw = call_llm(llm_prompt, action="Generating illustration prompts", json_mode=True)
                break
            except RuntimeError:
                if llm_attempt == 3:
                    raise
                wait = 30 * llm_attempt
                logger.warning("Illustration LLM prompt failed (attempt %d/3) – retrying in %ds", llm_attempt, wait)
                time.sleep(wait)

        assert raw is not None  # guaranteed by retry loop raising on final failure
        prompt_data = parse_llm_json(raw)
        illustrations = prompt_data.get("illustrations", [])

        if not isinstance(illustrations, list) or not illustrations:
            return jsonify({"error": "LLM did not return valid illustration prompts."}), 502

        results = []
        for idx, illust in enumerate(illustrations[:2]):
            if not isinstance(illust, dict):
                continue
            art_prompt = str(illust.get("art_prompt", "")).strip()
            if not art_prompt:
                continue

            illust_type = str(illust.get("type", "chapter_scene"))
            chapter = illust.get("chapter")
            scene_desc = str(illust.get("scene_description", "")).strip()

            prefix = "cover" if illust_type == "cover" else f"ch{chapter or idx}"
            filename = call_image_api(art_prompt, filename_prefix=prefix)

            if filename:
                results.append({
                    "type": illust_type,
                    "chapter": chapter,
                    "scene_description": scene_desc,
                    "art_prompt": art_prompt,
                    "image_url": f"/illustrations/{filename}",
                })

        if not results:
            return jsonify({"error": "Image generation failed for all prompts."}), 502

        with _progress_lock:
            _progress_store[token]["illustrations"] = results
        session["illustrations"] = results
        save_session_state()

        return jsonify({"illustrations": results})

    except RuntimeError as exc:
        logger.error("Illustration generation failed: %s", exc)
        return jsonify({"error": _friendly_llm_error(exc)}), 502


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
