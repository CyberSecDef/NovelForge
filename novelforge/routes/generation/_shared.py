"""Shared state for the generation route package."""

from flask import Blueprint

generation_bp = Blueprint("generation", __name__)

# Minimum seconds between time-based progress snapshot persists.
# Chapter completion and terminal states always trigger an unconditional write.
_PROGRESS_PERSIST_INTERVAL: float = 30.0

# Derived report fields that are invalidated when a chapter is revised.
# Any of these keys present in progress state may be stale after a revision and
# will be set to None so that consumers know they need to be regenerated.
_DERIVED_REPORT_FIELDS: tuple[str, ...] = (
    "global_continuity_audit",
    "narrative_compression_report",
    "character_resolution_report",
    "thematic_payoff_report",
    "climax_integrity_report",
    "loose_thread_report",
    "reader_immersion_report",
    "pacing_heatmap",
    "character_relationship_map",
)
