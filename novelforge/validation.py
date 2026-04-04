"""Input validation helpers for NovelForge."""

import novelforge.config as config

ALLOWED_GENRES = {
    "Adventure",
    "Contemporary Fiction",
    "Crime",
    "Dystopian",
    "Fantasy",
    "Gothic Fiction",
    "Historical Fiction",
    "Horror",
    "Literary Fiction",
    "Magical Realism",
    "Mystery",
    "Noir",
    "Paranormal",
    "Romance",
    "Satire Humor",
    "Science Fiction",
    "Speculative Fiction",
    "Thriller",
    "Urban Fantasy",
    "Western",
    "Young Adult",
}


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
        if chapters > config.MAX_CHAPTERS:
            return False, f"Number of chapters must be {config.MAX_CHAPTERS:,} or fewer."
    except (ValueError, TypeError):
        return False, "Chapters must be a valid number."

    try:
        word_count = int(data.get("word_count", 0))
        if word_count < 1000:
            return False, "Word count must be at least 1000."
        if word_count > config.MAX_WORD_COUNT:
            return False, f"Word count must be {config.MAX_WORD_COUNT:,} or fewer."
    except (ValueError, TypeError):
        return False, "Word count must be a valid number."

    special_events = data.get("special_events", "")
    if isinstance(special_events, str) and len(special_events) > 5000:
        return False, "Special events must be 5,000 characters or fewer."

    special_instructions = data.get("special_instructions", "")
    if isinstance(special_instructions, str) and len(special_instructions) > 5000:
        return False, "Special instructions must be 5,000 characters or fewer."

    return True, ""
