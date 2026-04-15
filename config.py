"""Backward-compatibility shim – re-exports the public config API from novelforge.config.

Prefer importing directly from ``novelforge.config`` in new code.  This module
exists only so that legacy callers using ``import config`` continue to work
without modification.

Supported public names are enumerated explicitly here (rather than via a
wildcard import) so that static-analysis tools, IDEs, and type checkers can
resolve every re-exported symbol precisely.
"""

from novelforge.config import (  # noqa: F401 – re-exports for backward compatibility
    CHAPTER_MIN_LENGTH_PCT,
    ConfigurationError,
    IMAGE_API_KEY,
    IMAGE_API_URL,
    IMAGE_MODEL,
    IMAGE_SIZE,
    IMAGE_TIMEOUT,
    LLM_API_KEY,
    LLM_API_URL,
    LLM_CIRCUIT_BREAKER_THRESHOLD,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_PROVIDERS,
    LLM_RETRY_DELAY,
    LLM_TIMEOUT,
    LOGS_DIR,
    MAX_CHAPTERS,
    MAX_EXPANSION_ATTEMPTS,
    MAX_WORD_COUNT,
    NOVELS_DIR,
    EXPORT_DIR,
    PER_CHAPTER_TIMEOUT,
    PROJECT_ROOT,
    ProviderConfig,
    SECRET_KEY,
    SESSION_FILE_DIR,
    ensure_app_dirs,
    get_env_int,
    validate_config,
)
