"""
Configuration for NovelForge.

Settings are read from environment variables. Copy .env.example to .env
and fill in your values before running the application.
"""

import logging
import os
import sys
from pathlib import Path

# Project root: parent of the novelforge/ package directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# LLM API endpoint (default: OpenAI chat completions)
LLM_API_URL = os.environ.get(
    "LLM_API_URL", "https://api.openai.com/v1/chat/completions"
)

# LLM API key – REQUIRED in production. Set the LLM_API_KEY environment variable.
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")

# LLM model name to request
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o")

# Image generation API endpoint (default: OpenAI image generations)
IMAGE_API_URL = os.environ.get("IMAGE_API_URL", "https://api.openai.com/v1/images/generations")

# Image generation API key – REQUIRED in production. Set the IMAGE_API_KEY environment variable.
IMAGE_API_KEY = os.environ.get("IMAGE_API_KEY", "")

# Image generation model name to request
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "gpt-image-1-mini")

# Image generation size
IMAGE_SIZE = os.environ.get("IMAGE_SIZE", "1024x1024")



# ---------------------------------------------------------------------------
# LLM tuning constants (override via environment variables)
# ---------------------------------------------------------------------------

# Retry and timeout settings for LLM API calls
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "5"))
LLM_RETRY_DELAY = int(os.environ.get("LLM_RETRY_DELAY", "5"))  # base seconds
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "240"))  # request timeout seconds
IMAGE_TIMEOUT = int(os.environ.get("IMAGE_TIMEOUT", "120"))  # image API timeout

# Circuit breaker: consecutive failures before tripping
LLM_CIRCUIT_BREAKER_THRESHOLD = int(os.environ.get("LLM_CIRCUIT_BREAKER_THRESHOLD", "3"))

# Per-chapter wall-clock timeout (seconds) – 60 minutes default
PER_CHAPTER_TIMEOUT = int(os.environ.get("PER_CHAPTER_TIMEOUT", "3600"))

# ---------------------------------------------------------------------------
# Input validation limits (override via environment variables)
# ---------------------------------------------------------------------------

MAX_CHAPTERS = int(os.environ.get("MAX_CHAPTERS", "100"))
MAX_WORD_COUNT = int(os.environ.get("MAX_WORD_COUNT", "500000"))

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------

# Flask secret key – override via SECRET_KEY environment variable in production
SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production")

# Directory where Flask-Session stores server-side session files
SESSION_FILE_DIR = str(PROJECT_ROOT / os.environ.get("SESSION_FILE_DIR", "sessions/flask"))

# Directory where exported novel files are stored temporarily
EXPORT_DIR = str(PROJECT_ROOT / os.environ.get("EXPORT_DIR", "exports"))

# Directory where novel session JSON files are stored
NOVELS_DIR = str(PROJECT_ROOT / os.environ.get("NOVELS_DIR", "sessions/novels"))

# Directory for log files
LOGS_DIR = str(PROJECT_ROOT / os.environ.get("LOGS_DIR", "logs"))


def validate_config(*, debug: bool = False) -> None:
    """Validate configuration at startup.

    In production (debug=False), missing critical values cause a hard exit.
    In development (debug=True), they produce warnings so the app can still
    start for UI work or testing.
    """
    _logger = logging.getLogger(__name__)
    errors: list[str] = []
    warnings: list[str] = []

    # LLM_API_KEY is required to do anything useful
    if not LLM_API_KEY:
        errors.append(
            "LLM_API_KEY is not set. Set the LLM_API_KEY environment variable."
        )

    # LLM_API_URL must look like a URL
    if not LLM_API_URL.startswith(("http://", "https://")):
        errors.append(
            f"LLM_API_URL is not a valid URL: {LLM_API_URL!r}"
        )

    # SECRET_KEY must be changed in production
    if SECRET_KEY == "change-me-in-production":
        if debug:
            warnings.append(
                "SECRET_KEY is using the default value. "
                "Set the SECRET_KEY environment variable before deploying."
            )
        else:
            errors.append(
                "SECRET_KEY is using the insecure default. "
                "Set the SECRET_KEY environment variable for production."
            )

    for msg in warnings:
        _logger.warning("CONFIG WARNING: %s", msg)

    if errors:
        if debug:
            # In development, warn but allow startup
            for msg in errors:
                _logger.warning("CONFIG WARNING: %s", msg)
        else:
            for msg in errors:
                _logger.error("CONFIG ERROR: %s", msg)
            sys.exit(1)
