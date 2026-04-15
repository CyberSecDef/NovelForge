"""
Configuration for NovelForge.

Settings are read from environment variables. Copy .env.example to .env
and fill in your values before running the application.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from the project-root .env file before any
# os.environ reads happen below.  Using an explicit path (derived from this
# file's location) keeps behaviour identical regardless of the working
# directory at startup.  ``override=False`` ensures that variables already
# present in the process environment (set by the deployment platform, CI,
# or the operator) are never replaced by .env values.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

# ---------------------------------------------------------------------------
# Public API – names exported by this module and re-exported by the top-level
# config.py backward-compatibility shim.  Keep this list in sync with any new
# public symbols added below.
# ---------------------------------------------------------------------------
__all__ = [
    # Helper
    "get_env_int",
    # Path
    "PROJECT_ROOT",
    # LLM provider configuration
    "ProviderConfig",
    "LLM_PROVIDERS",
    # Backward-compatible primary-provider aliases
    "LLM_API_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    # Image generation configuration
    "IMAGE_API_URL",
    "IMAGE_API_KEY",
    "IMAGE_MODEL",
    "IMAGE_SIZE",
    # LLM tuning constants
    "LLM_MAX_RETRIES",
    "LLM_RETRY_DELAY",
    "LLM_TIMEOUT",
    "IMAGE_TIMEOUT",
    "LLM_CIRCUIT_BREAKER_THRESHOLD",
    "PER_CHAPTER_TIMEOUT",
    # Input validation limits
    "MAX_CHAPTERS",
    "MAX_WORD_COUNT",
    # Chapter length enforcement
    "CHAPTER_MIN_LENGTH_PCT",
    "MAX_EXPANSION_ATTEMPTS",
    # Flask
    "SECRET_KEY",
    "SESSION_FILE_DIR",
    "EXPORT_DIR",
    "NOVELS_DIR",
    "LOGS_DIR",
    # Errors and startup helpers
    "ConfigurationError",
    "ensure_app_dirs",
    "validate_config",
]

# Collects errors discovered while parsing numeric environment variables at
# import time so that validate_config() can surface them all at once instead
# of crashing with a bare ValueError.
_CONFIG_PARSE_ERRORS: list[str] = []


def get_env_int(name: str, default: int, *, min_value: int | None = None) -> int:
    """Return the integer value of environment variable *name*.

    Falls back to *default* if the variable is unset or cannot be converted to
    ``int``.  If *min_value* is given, values below the minimum are rejected in
    the same way.  All problems are recorded in the module-level
    ``_CONFIG_PARSE_ERRORS`` list so that :func:`validate_config` can report
    them as structured errors rather than raising a raw :class:`ValueError` at
    import time.

    Returns:
        The parsed integer, or *default* when the value is missing, malformed,
        or out of the allowed range.

    Side effects:
        Appends a human-readable error string to ``_CONFIG_PARSE_ERRORS`` for
        every rejected value so the problem is visible at validation time.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        _CONFIG_PARSE_ERRORS.append(
            f"{name} must be an integer (got {raw!r}); using default {default}."
        )
        return default
    if min_value is not None and value < min_value:
        _CONFIG_PARSE_ERRORS.append(
            f"{name} must be >= {min_value} (got {value!r}); using default {default}."
        )
        return default
    return value

# Project root: parent of the novelforge/ package directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# LLM provider configuration (supports fallback chain)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderConfig:
    """A single LLM provider's connection settings."""
    url: str
    api_key: str
    model: str
    label: str  # human-readable label for logging (e.g. "primary", "provider_2")


# Maximum numbered fallback index scanned when building the provider list.
# Indices 2 … _MAX_PROVIDER_INDEX are checked; any with a URL set are included.
_MAX_PROVIDER_INDEX = 20


def _parse_llm_providers() -> list[ProviderConfig]:
    """
    Build an ordered list of LLM providers from environment variables.

    The primary provider uses LLM_API_URL / LLM_API_KEY / LLM_MODEL.
    Fallbacks use numbered suffixes: LLM_API_URL_2, LLM_API_KEY_2, LLM_MODEL_2, etc.
    A fallback is only added if at least its URL is set.

    All indices from 2 to :data:`_MAX_PROVIDER_INDEX` are scanned so that
    sparse numbering (e.g. provider 2 absent but provider 3 present) is
    discovered rather than silently dropped.  Any gaps found between index 2
    and the highest defined index are recorded in
    :data:`_CONFIG_PARSE_ERRORS` so that :func:`validate_config` can surface
    them to the operator.
    """
    providers: list[ProviderConfig] = []

    # Primary (no suffix)
    primary_url = os.environ.get("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
    primary_key = os.environ.get("LLM_API_KEY", "")
    primary_model = os.environ.get("LLM_MODEL", "gpt-4o")
    providers.append(ProviderConfig(
        url=primary_url, api_key=primary_key, model=primary_model, label="primary",
    ))

    # Discover all numbered fallbacks in range [2, _MAX_PROVIDER_INDEX]
    defined_indices = [
        idx for idx in range(2, _MAX_PROVIDER_INDEX + 1)
        if os.environ.get(f"LLM_API_URL_{idx}", "")
    ]

    if defined_indices:
        # Detect gaps: every index from 2 up to the highest defined one is
        # expected to be present.  Missing slots in that range are gaps.
        full_range = set(range(2, max(defined_indices) + 1))
        gap_indices = sorted(full_range - set(defined_indices))
        if gap_indices:
            _CONFIG_PARSE_ERRORS.append(
                "LLM provider numbering has gaps at index(es) "
                + ", ".join(str(i) for i in gap_indices)
                + ". All defined providers will still be used; "
                "consider renumbering them consecutively to avoid confusion."
            )

        for idx in defined_indices:
            url = os.environ.get(f"LLM_API_URL_{idx}", "")
            key = os.environ.get(f"LLM_API_KEY_{idx}", "")
            model = os.environ.get(f"LLM_MODEL_{idx}", primary_model)
            providers.append(ProviderConfig(
                url=url, api_key=key, model=model, label=f"provider_{idx}",
            ))

    return providers


LLM_PROVIDERS: list[ProviderConfig] = _parse_llm_providers()

# Backward-compatible aliases — always point to the primary provider
LLM_API_URL = LLM_PROVIDERS[0].url
LLM_API_KEY = LLM_PROVIDERS[0].api_key
LLM_MODEL = LLM_PROVIDERS[0].model

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
LLM_MAX_RETRIES = get_env_int("LLM_MAX_RETRIES", 5, min_value=0)
LLM_RETRY_DELAY = get_env_int("LLM_RETRY_DELAY", 5, min_value=0)  # base seconds
LLM_TIMEOUT = get_env_int("LLM_TIMEOUT", 240, min_value=1)  # request timeout seconds
IMAGE_TIMEOUT = get_env_int("IMAGE_TIMEOUT", 120, min_value=1)  # image API timeout

# Circuit breaker: consecutive failures before tripping
LLM_CIRCUIT_BREAKER_THRESHOLD = get_env_int("LLM_CIRCUIT_BREAKER_THRESHOLD", 3, min_value=1)

# Per-chapter wall-clock timeout (seconds) – 60 minutes default
PER_CHAPTER_TIMEOUT = get_env_int("PER_CHAPTER_TIMEOUT", 3600, min_value=1)

# ---------------------------------------------------------------------------
# Input validation limits (override via environment variables)
# ---------------------------------------------------------------------------

MAX_CHAPTERS = get_env_int("MAX_CHAPTERS", 100, min_value=1)
MAX_WORD_COUNT = get_env_int("MAX_WORD_COUNT", 500000, min_value=1)

# ---------------------------------------------------------------------------
# Chapter length enforcement (override via environment variables)
# ---------------------------------------------------------------------------


def _get_percentage_env(
    name: str, default: int, *, min_value: int = 0, max_value: int = 100
) -> int:
    """Return an integer percentage env var constrained to the given range."""
    value = get_env_int(name, default, min_value=min_value)
    if value > max_value:
        raise ValueError(
            f"{name} must be <= {max_value} (got {value})."
        )
    return value


# Minimum acceptable chapter word count as a percentage of the per-chapter target.
# Chapters below this threshold trigger an automatic expansion pass.
CHAPTER_MIN_LENGTH_PCT = _get_percentage_env(
    "CHAPTER_MIN_LENGTH_PCT", 85, min_value=50, max_value=100
)

# Maximum number of expansion attempts per chapter before accepting as-is.
MAX_EXPANSION_ATTEMPTS = get_env_int("MAX_EXPANSION_ATTEMPTS", 2, min_value=0)

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------

# Flask secret key – override via SECRET_KEY environment variable in production
SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production")


def _resolve_dir(env_var: str, default: str) -> str:
    """Return the absolute path for a directory env var, anchored to PROJECT_ROOT.

    Raises:
        ValueError: if the env var is set to an absolute path.  All directory
            env vars must be relative to *PROJECT_ROOT* to prevent arbitrary
            filesystem access.
    """
    raw = os.environ.get(env_var, default)
    if os.path.isabs(raw):
        raise ValueError(
            f"{env_var} must be a relative path (got {raw!r}). "
            "Directory env vars are resolved relative to the project root."
        )
    return str(PROJECT_ROOT / raw)


# Directory where Flask-Session stores server-side session files
SESSION_FILE_DIR = _resolve_dir("SESSION_FILE_DIR", "sessions/flask")

# Directory where exported novel files are stored temporarily
EXPORT_DIR = _resolve_dir("EXPORT_DIR", "exports")

# Directory where novel session JSON files are stored
NOVELS_DIR = _resolve_dir("NOVELS_DIR", "sessions/novels")

# Directory for log files
LOGS_DIR = _resolve_dir("LOGS_DIR", "logs")


class ConfigurationError(Exception):
    """Raised by validate_config() when one or more required settings are missing or invalid.

    Attributes:
        errors: Ordered list of human-readable error messages describing each
                configuration problem that was detected.
    """

    def __init__(self, errors: list[str]) -> None:
        """Initialise with a human-readable configuration error message."""
        self.errors = list(errors)
        super().__init__(
            "Configuration validation failed:\n"
            + "\n".join(f"  - {e}" for e in self.errors)
        )


def ensure_app_dirs() -> None:
    """Create all required application directories and verify they are writeable.

    Called at application start-up (before any logging handlers are attached or
    session stores are initialised) so that every subsequent write to these
    directories can succeed without a runtime ``FileNotFoundError`` or
    ``PermissionError``.

    Raises:
        PermissionError: if a directory exists but is not writeable by the
            current process.
    """
    required_dirs: list[Path] = [
        Path(SESSION_FILE_DIR),
        Path(EXPORT_DIR),
        Path(EXPORT_DIR) / "illustrations",
        Path(LOGS_DIR),
        Path(NOVELS_DIR),
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        if not os.access(directory, os.W_OK):
            raise PermissionError(
                f"Application directory is not writeable: {directory}"
            )


def validate_config(*, debug: bool = False) -> None:
    """Validate configuration at startup.

    In production (debug=False), missing critical values cause a hard exit.
    In development (debug=True), they produce warnings so the app can still
    start for UI work or testing.
    """
    _logger = logging.getLogger(__name__)
    errors: list[str] = []
    warnings: list[str] = []

    # Surface any numeric env-var parsing errors collected at import time
    errors.extend(_CONFIG_PARSE_ERRORS)

    # Validate all configured LLM providers
    for i, provider in enumerate(LLM_PROVIDERS):
        suffix = "" if i == 0 else f"_{i + 1}"
        label = provider.label

        if not provider.api_key:
            errors.append(
                f"LLM_API_KEY{suffix} is not set ({label}). "
                f"Set the LLM_API_KEY{suffix} environment variable."
            )

        if not provider.url.startswith(("http://", "https://")):
            errors.append(
                f"LLM_API_URL{suffix} is not a valid URL ({label}): {provider.url!r}"
            )

    if len(LLM_PROVIDERS) > 1:
        _logger.info(
            "LLM fallback chain configured: %s",
            " → ".join(p.label for p in LLM_PROVIDERS),
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
            raise ConfigurationError(errors)
