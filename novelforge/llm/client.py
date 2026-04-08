"""LLM API client: call_llm, circuit breaker, JSON parsing, usage tracking."""

import html
import json
import logging
import os
import random
import re
import threading
import time
from typing import Any, cast

import requests
from unidecode import unidecode

import novelforge.config as config

logger = logging.getLogger(__name__)

# Dedicated LLM logger — handler is attached lazily by setup_llm_logger(),
# which is called from create_app().  Keeping this at module level means
# llm_logger.info() calls work (silently, with no output) even before the
# handler is attached, which is the correct behaviour during import and in
# unit tests that never call create_app().
llm_logger = logging.getLogger("llm_requests")
llm_logger.propagate = False


def setup_llm_logger() -> None:
    """Attach a FileHandler to the ``llm_requests`` logger, idempotently.

    This function is safe to call multiple times (e.g. across repeated
    ``create_app()`` calls in tests): it checks for an existing
    ``FileHandler`` targeting ``logs/llm.log`` before adding a new one, so
    the handler is attached at most once per process.

    The log directory is created here rather than at import time so that
    filesystem side-effects are deferred until the app is actually
    initialised.
    """
    import os as _os
    from pathlib import Path as _Path

    log_path = _os.path.join(config.LOGS_DIR, "llm.log")
    _Path(config.LOGS_DIR).mkdir(parents=True, exist_ok=True)

    # Idempotency guard: skip if a FileHandler for this exact path already exists.
    norm_log_path = _os.path.normcase(_os.path.abspath(log_path))
    for handler in llm_logger.handlers:
        if isinstance(handler, logging.FileHandler) and _os.path.normcase(handler.baseFilename) == norm_log_path:
            return

    llm_logger.setLevel(logging.INFO)
    _handler = logging.FileHandler(log_path)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    llm_logger.addHandler(_handler)

# Thread-local LLM token usage accumulator
_llm_usage = threading.local()

# Read from config — all tuneable via environment variables
MAX_RETRIES = config.LLM_MAX_RETRIES
RETRY_DELAY = config.LLM_RETRY_DELAY
CIRCUIT_BREAKER_THRESHOLD = config.LLM_CIRCUIT_BREAKER_THRESHOLD
PER_CHAPTER_TIMEOUT = config.PER_CHAPTER_TIMEOUT


class LLMCircuitBreaker:
    """
    Thread-safe circuit breaker for LLM API calls.

    After CIRCUIT_BREAKER_THRESHOLD consecutive call_llm failures the breaker
    trips and immediately raises on subsequent calls until reset.
    """

    def __init__(self, threshold: int = CIRCUIT_BREAKER_THRESHOLD):
        """Initialise with the given failure threshold."""
        self._threshold = threshold
        self._consecutive_failures = 0
        self._tripped = False
        self._last_error: str = ""
        self._lock = threading.Lock()

    def record_success(self) -> None:
        """Record a successful call, resetting the failure counter."""
        with self._lock:
            self._consecutive_failures = 0
            self._tripped = False

    def record_failure(self, error: str) -> None:
        """Record a failed call; trip the breaker if threshold is reached."""
        with self._lock:
            self._consecutive_failures += 1
            self._last_error = error
            if self._consecutive_failures >= self._threshold:
                self._tripped = True
                logger.error(
                    "Circuit breaker tripped after %d consecutive LLM failures: %s",
                    self._consecutive_failures, error,
                )

    def check(self) -> None:
        """Raise if the breaker is tripped."""
        with self._lock:
            if self._tripped:
                raise CircuitBreakerError(
                    f"LLM API circuit breaker open — {self._consecutive_failures} "
                    f"consecutive failures. Last error: {self._last_error}"
                )

    def reset(self) -> None:
        """Manually reset the breaker to closed state."""
        with self._lock:
            self._consecutive_failures = 0
            self._tripped = False
            self._last_error = ""

    @property
    def is_tripped(self) -> bool:
        """Return True if the breaker is currently tripped."""
        with self._lock:
            return self._tripped

    @property
    def failure_count(self) -> int:
        """Return the current consecutive failure count."""
        with self._lock:
            return self._consecutive_failures


class CircuitBreakerError(RuntimeError):
    """Raised when the LLM circuit breaker is open."""
    pass


class ChapterTimeoutError(RuntimeError):
    """Raised when a single chapter exceeds the per-chapter time limit."""
    pass


class ContentRejectionError(Exception):
    """Raised when the LLM rejects a request due to content policy.

    Deliberately does NOT extend RuntimeError so that content-retry wrappers
    can catch it without interference from the broader RuntimeError handlers
    in the generation pipeline.
    """

    def __init__(self, message: str, *, status_code: int | None = None, response_body: str | None = None):
        """Initialise with message and optional status_code and response_body."""
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


# Patterns in API response bodies that indicate content-policy rejection
_CONTENT_REJECTION_PATTERNS = [
    "content_filter", "content_policy", "content_management",
    "content moderation", "content violation",
    "safety system", "safety filter",
    "flagged as", "policy violation",
    "harmful content", "inappropriate content",
    "violates our", "usage policy", "use policy",
    "responsible ai", "content standards",
    "output_moderation", "input_moderation",
]


def _is_content_rejection(status_code: int | None, response_body: str | None) -> bool:
    """Detect whether an API error is a content-policy rejection."""
    if not response_body:
        return False
    body_lower = response_body.lower()
    return any(pattern in body_lower for pattern in _CONTENT_REJECTION_PATTERNS)


# Per-provider circuit breakers (keyed by provider index).
#
# Intended scope: **process-level, per-provider**.
# One breaker exists per configured LLM provider and is shared across all
# concurrent requests.  This means that when a provider becomes unhealthy
# (enough consecutive failures), *all* requests stop trying that provider
# until it is healthy again — which is the correct behaviour for a
# process-level reliability guard.
#
# What must NOT happen is ad-hoc resets from request handlers or generation
# workers: resetting a breaker globally while another request is relying on
# the tripped state can cause repeated failures and unpredictable behaviour.
# The only legitimate callers of ``reset_circuit_breakers()`` are:
#   • the test-suite fixture in ``tests/conftest.py``, which resets state
#     between isolated test cases;
#   • explicit operator tooling / admin endpoints (if added in future).
_llm_circuit_breakers: dict[int, LLMCircuitBreaker] = {}
_llm_cb_lock = threading.Lock()


def _get_circuit_breaker(provider_idx: int) -> LLMCircuitBreaker:
    """Return the process-level circuit breaker for *provider_idx*.

    Breakers are created lazily and cached for the lifetime of the process.
    The returned object is shared across all concurrent requests; callers must
    not call ``reset()`` on it outside of test-suite teardown.
    """
    if provider_idx not in _llm_circuit_breakers:
        with _llm_cb_lock:
            if provider_idx not in _llm_circuit_breakers:
                _llm_circuit_breakers[provider_idx] = LLMCircuitBreaker()
    return _llm_circuit_breakers[provider_idx]


# Backward-compatible alias: points to the primary provider's breaker
_llm_circuit_breaker = _get_circuit_breaker(0)


class AllProvidersExhaustedError(RuntimeError):
    """Raised when every configured LLM provider has failed."""
    pass


def _safe_response_body(resp: requests.Response | None) -> str | None:
    """Extract response body as text, truncated to 4KB to avoid log bloat."""
    if resp is None:
        return None
    try:
        body = resp.text
        if len(body) > 4096:
            return body[:4096] + f"\n... [truncated, total {len(body)} chars]"
        return body
    except Exception:
        try:
            raw = resp.content
            return f"[binary, {len(raw)} bytes]: {raw[:512]!r}"
        except Exception:
            return "[could not read response body]"


def _format_exception_chain(exc: BaseException) -> list[str]:
    """Walk the exception __cause__ / __context__ chain and format each."""
    chain = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current and id(current) not in seen:
        seen.add(id(current))
        chain.append(f"{type(current).__module__}.{type(current).__qualname__}: {current}")
        current = current.__cause__ or current.__context__
    return chain


def _log_llm_error(
    *,
    action: str,
    attempt: int,
    status_code: int | None,
    response_headers: dict | None,
    response_body: str | None,
    error_type: str,
    error_message: str,
    url: str,
    model: str,
    prompt_messages_count: int,
    prompt_total_chars: int,
    exception_chain: list[str] | None = None,
) -> None:
    """Log a comprehensive error record to both the app logger and llm.log."""
    error_log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": "error",
        "action": action,
        "attempt": attempt,
        "max_retries": MAX_RETRIES,
        "error_type": error_type,
        "error_message": error_message,
        "status_code": status_code,
        "url": url,
        "model": model,
        "prompt_messages_count": prompt_messages_count,
        "prompt_total_chars": prompt_total_chars,
        "response_headers": response_headers,
        "response_body": response_body,
    }
    if exception_chain:
        error_log["exception_chain"] = exception_chain

    llm_logger.info(json.dumps(error_log))
    logger.error(
        "LLM error [%s] action=%s attempt=%d/%d status=%s: %s | response_body=%s",
        error_type, action, attempt, MAX_RETRIES, status_code,
        error_message, (response_body or "")[:500],
    )


def _to_ascii(text: str) -> str:
    """Transliterate Unicode to ASCII equivalents (curly quotes, em-dashes, accents, etc.)."""
    return cast(str, unidecode(text))


# Control characters that are invalid in JSON strings or rejected by LLM APIs.
# Preserves \t (0x09), \n (0x0a), and \r (0x0d) which are valid whitespace.
_CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    """Return a copy of *messages* with HTML entities unescaped, content transliterated to ASCII, and control characters stripped."""
    out = []
    for msg in messages:
        cleaned = dict(msg)
        if isinstance(cleaned.get("content"), str):
            # Unescape HTML entities that may be baked into older session data
            text = _to_ascii(html.unescape(cleaned["content"]))
            # Strip control characters that LLM APIs reject (NUL, backspace, etc.)
            cleaned["content"] = _CONTROL_CHAR_RE.sub('', text)
        out.append(cleaned)
    return out


def _retry_delay(attempt: int) -> float:
    """Compute retry delay with jitter to prevent thundering herd."""
    base = RETRY_DELAY * attempt
    return base * (0.5 + random.random())  # jitter: 50%-150% of base


def _build_request(
    provider: config.ProviderConfig,
    messages: list[dict],
    *,
    action: str,
    json_mode: bool,
) -> tuple[dict, dict]:
    """Build request headers and payload, log the outbound request.

    Returns ``(headers, payload)`` ready for ``requests.post()``.
    """
    ascii_messages = _sanitize_messages(messages)

    headers = {
        "Authorization": f"Bearer {provider.api_key}",
        "Content-Type": "application/json",
    }
    payload: dict = {
        "model": provider.model,
        "messages": ascii_messages,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    request_log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": "request",
        "action": action,
        "provider": provider.label,
        "url": provider.url,
        "headers": {
            "Authorization": "Bearer [REDACTED]" if provider.api_key else "None",
            "Content-Type": "application/json",
        },
        "payload": payload,
    }
    llm_logger.info(json.dumps(request_log))

    return headers, payload


def _handle_success(
    provider: config.ProviderConfig,
    resp: requests.Response,
    data: dict,
    *,
    action: str,
    attempt: int,
    prompt_msg_count: int,
    prompt_total_chars: int,
) -> str:
    """Process a successful LLM response: log, accumulate tokens, check content filter.

    Returns the extracted response content string.
    Raises ``ContentRejectionError`` if the output was blocked by a content filter.
    """
    response_log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": "response",
        "provider": provider.label,
        "status_code": resp.status_code,
        "headers": dict(resp.headers),
        "response": data,
    }
    llm_logger.info(json.dumps(response_log))

    # Accumulate token usage if available
    usage = data.get("usage")
    if usage:
        if not hasattr(_llm_usage, "prompt_tokens"):
            _llm_usage.prompt_tokens = 0
            _llm_usage.completion_tokens = 0
            _llm_usage.call_count = 0
        _llm_usage.prompt_tokens += int(usage.get("prompt_tokens", 0))
        _llm_usage.completion_tokens += int(usage.get("completion_tokens", 0))
        _llm_usage.call_count += 1

    # Check for output-level content filtering (HTTP 200 but generation blocked)
    choices = data.get("choices", [])
    if choices:
        finish_reason = choices[0].get("finish_reason", "")
        if finish_reason == "content_filter":
            body_preview = json.dumps(data, indent=2)[:2000]
            _log_llm_error(
                action=action, attempt=attempt, status_code=resp.status_code,
                response_headers=dict(resp.headers), response_body=body_preview,
                error_type="content_filter_output",
                error_message="LLM output blocked by content filter (finish_reason=content_filter)",
                url=provider.url, model=provider.model,
                prompt_messages_count=prompt_msg_count,
                prompt_total_chars=prompt_total_chars,
            )
            raise ContentRejectionError(
                f"LLM output blocked by content filter on {provider.label} "
                f"(finish_reason=content_filter). Response: {body_preview}",
                status_code=resp.status_code, response_body=body_preview,
            )

    content = _to_ascii(data["choices"][0]["message"]["content"])
    return _CONTROL_CHAR_RE.sub('', content)


def _classify_request_error(
    exc: requests.exceptions.RequestException,
    provider: config.ProviderConfig,
    *,
    action: str,
    attempt: int,
    prompt_msg_count: int,
    prompt_total_chars: int,
) -> RuntimeError | ContentRejectionError:
    """Classify a RequestException and return the appropriate typed error to raise.

    Logs the error details.  The caller is responsible for recording circuit
    breaker failures.
    """
    exc_resp = getattr(exc, "response", None)
    status = getattr(exc_resp, "status_code", None)
    resp_headers = dict(exc_resp.headers) if exc_resp is not None and hasattr(exc_resp, "headers") else None
    resp_body = _safe_response_body(exc_resp) if exc_resp is not None else None

    _log_llm_error(
        action=action, attempt=attempt, status_code=status,
        response_headers=resp_headers, response_body=resp_body,
        error_type=type(exc).__name__, error_message=str(exc),
        url=provider.url, model=provider.model,
        prompt_messages_count=prompt_msg_count,
        prompt_total_chars=prompt_total_chars,
        exception_chain=_format_exception_chain(exc),
    )

    if _is_content_rejection(status, resp_body):
        return ContentRejectionError(
            f"LLM request rejected by content policy on {provider.label} "
            f"(HTTP {status}). Response: {resp_body}",
            status_code=status, response_body=resp_body,
        )

    if status in (401, 403):
        err = f"LLM_AUTH_FAILURE: API key rejected on {provider.label}. Check your LLM_API_KEY setting."
    elif status == 400:
        err = f"LLM_BAD_REQUEST: {provider.label} rejected the request. The prompt may be too long or contain unsupported content."
    elif status == 404:
        err = f"LLM_ENDPOINT_NOT_FOUND: Endpoint not found on {provider.label}. Check your LLM_API_URL setting."
    else:
        err = f"LLM_REQUEST_FAILED: {provider.label}: {exc}"
    return RuntimeError(err)


def _call_single_provider(
    provider: config.ProviderConfig,
    provider_idx: int,
    messages: list[dict],
    *,
    action: str,
    json_mode: bool,
) -> str:
    """
    Call a single LLM provider with retries.  Returns the response content.

    Raises RuntimeError, ContentRejectionError, or CircuitBreakerError on
    failure — the caller (``call_llm``) decides whether to fall through to
    the next provider.
    """
    breaker = _get_circuit_breaker(provider_idx)
    breaker.check()

    headers, payload = _build_request(
        provider, messages, action=action, json_mode=json_mode,
    )
    prompt_msg_count = len(messages)
    prompt_total_chars = sum(len(str(m.get("content", ""))) for m in messages)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                provider.url,
                headers=headers,
                json=payload,
                timeout=config.LLM_TIMEOUT,
            )

            # Retryable server errors
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = _retry_delay(attempt)
                logger.warning(
                    "LLM API [%s] returned %s – retry %d/%d in %.1fs",
                    provider.label, resp.status_code, attempt, MAX_RETRIES, wait,
                )
                _log_llm_error(
                    action=action, attempt=attempt, status_code=resp.status_code,
                    response_headers=dict(resp.headers), response_body=_safe_response_body(resp),
                    error_type="retryable", error_message=f"HTTP {resp.status_code}",
                    url=provider.url, model=provider.model,
                    prompt_messages_count=prompt_msg_count,
                    prompt_total_chars=prompt_total_chars,
                )
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            content = _handle_success(
                provider, resp, data,
                action=action, attempt=attempt,
                prompt_msg_count=prompt_msg_count,
                prompt_total_chars=prompt_total_chars,
            )
            breaker.record_success()
            return content

        except requests.exceptions.Timeout:
            logger.warning(
                "LLM request timed out [%s] (attempt %d/%d)",
                provider.label, attempt, MAX_RETRIES,
            )
            _log_llm_error(
                action=action, attempt=attempt, status_code=None,
                response_headers=None, response_body=None,
                error_type="timeout",
                error_message=f"Request timed out after {config.LLM_TIMEOUT}s",
                url=provider.url, model=provider.model,
                prompt_messages_count=prompt_msg_count,
                prompt_total_chars=prompt_total_chars,
            )
            if attempt == MAX_RETRIES:
                err = (
                    f"LLM_TIMEOUT: Provider {provider.label} timed out on all "
                    f"{MAX_RETRIES} attempts. Your progress is saved."
                )
                breaker.record_failure(err)
                raise RuntimeError(err)
            time.sleep(_retry_delay(attempt))

        except requests.exceptions.RequestException as exc:
            typed_error = _classify_request_error(
                exc, provider,
                action=action, attempt=attempt,
                prompt_msg_count=prompt_msg_count,
                prompt_total_chars=prompt_total_chars,
            )
            breaker.record_failure(str(typed_error))
            raise typed_error from exc

    # All retries exhausted (rate-limited)
    err = (
        f"LLM_RATE_LIMITED: {provider.label} is overloaded after {MAX_RETRIES} "
        f"attempts. Your progress is saved."
    )
    breaker.record_failure(err)
    raise RuntimeError(err)


def call_llm(messages: list[dict], *, action: str = "", json_mode: bool = False) -> str:
    """
    Call the configured LLM API and return the assistant message content.

    Tries each provider in the fallback chain. For each provider, retries
    up to MAX_RETRIES times on transient errors (429, 5xx).

    All failure types — including ContentRejectionError — are retried on the
    next provider before propagating. This allows progressively less
    restrictive providers to handle content that stricter ones reject.

    Raises AllProvidersExhaustedError if every provider fails.
    Raises ContentRejectionError if every provider rejects due to content policy.
    """
    action = action if action else "Updating Content"
    providers = config.LLM_PROVIDERS
    last_error: Exception | None = None

    for idx, provider in enumerate(providers):
        breaker = _get_circuit_breaker(idx)
        if breaker.is_tripped:
            logger.info(
                "Skipping tripped provider %s, trying next", provider.label,
            )
            continue
        try:
            if idx > 0:
                logger.warning(
                    "Falling back to provider %s for action: %s",
                    provider.label, action,
                )
            return _call_single_provider(
                provider, idx, messages, action=action, json_mode=json_mode,
            )
        except (RuntimeError, CircuitBreakerError, ContentRejectionError) as exc:
            last_error = exc
            logger.warning(
                "Provider %s failed for '%s': %s — trying next provider",
                provider.label, action, exc,
            )
            continue

    # Every provider exhausted — preserve the original exception type
    if isinstance(last_error, ContentRejectionError):
        raise last_error
    raise AllProvidersExhaustedError(
        f"All {len(providers)} LLM provider(s) failed for '{action}'. "
        f"Last error: {last_error}"
    )


def reset_llm_usage() -> None:
    """Reset the thread-local token usage accumulator."""
    _llm_usage.prompt_tokens = 0
    _llm_usage.completion_tokens = 0
    _llm_usage.call_count = 0


def get_llm_usage() -> dict:
    """Return current accumulated token usage and reset."""
    usage = {
        "prompt_tokens": getattr(_llm_usage, "prompt_tokens", 0),
        "completion_tokens": getattr(_llm_usage, "completion_tokens", 0),
        "total_tokens": getattr(_llm_usage, "prompt_tokens", 0) + getattr(_llm_usage, "completion_tokens", 0),
        "call_count": getattr(_llm_usage, "call_count", 0),
    }
    reset_llm_usage()
    return usage


def reset_circuit_breakers() -> None:
    """Reset all per-provider LLM circuit breakers.

    **Intended for test-suite teardown and explicit lifecycle management only.**

    This function resets the process-level, per-provider breaker state for
    *every* provider at once.  Calling it from a request handler or a
    background generation worker is unsafe: it clears state that other
    concurrent requests may be relying on, causing unpredictable cross-request
    interference.

    Legitimate callers:
    • ``tests/conftest.py`` fixture — isolates state between test cases.
    • An explicit admin/operator endpoint (if added in future).
    """
    for cb in _llm_circuit_breakers.values():
        cb.reset()


def friendly_llm_error(exc: Exception) -> str:
    """
    Convert a RuntimeError from call_llm into a user-friendly message.

    The error string from call_llm is prefixed with a tag like LLM_TIMEOUT:
    which is stripped here. If no known tag is found, a generic message is used.
    """
    msg = str(exc)
    # Strip the machine tag prefix and return the human-readable part
    for tag in (
        "LLM_TIMEOUT:", "LLM_AUTH_FAILURE:", "LLM_BAD_REQUEST:",
        "LLM_ENDPOINT_NOT_FOUND:", "LLM_RATE_LIMITED:", "LLM_REQUEST_FAILED:",
    ):
        if msg.startswith(tag):
            return msg[len(tag):].strip()
    # CircuitBreakerError, ChapterTimeoutError, AllProvidersExhaustedError — already friendly
    if isinstance(exc, (CircuitBreakerError, ChapterTimeoutError, AllProvidersExhaustedError)):
        return msg
    # Fallback
    return "Something went wrong with the AI service. Your progress is saved — please try again."


def parse_llm_json(response: str) -> dict[str, Any] | list[Any]:
    """
    Parse JSON from LLM response, handling common irregularities:
    - Markdown code fences (```json, ```)
    - Extra whitespace
    - BOM characters
    - Text before/after the JSON object or array

    Returns a ``dict`` when the top-level JSON value is an object, or a
    ``list`` when it is an array.  Callers that only expect one of these
    forms should check ``isinstance(result, dict)`` /
    ``isinstance(result, list)`` as appropriate.

    Raises json.JSONDecodeError if parsing fails after all strategies.
    """
    raw_preview = response[:200] if len(response) > 200 else response
    strategy = "direct"

    # Strip BOM if present
    if response.startswith('\ufeff'):
        response = response[1:]
        strategy = "bom-stripped"
        logger.debug("parse_llm_json: stripped BOM")

    # Remove markdown code fences
    response = response.strip()

    if response.startswith('```'):
        strategy = "markdown-fence"
        logger.debug("parse_llm_json: stripping markdown code fences")
        first_newline = response.find('\n')
        if first_newline != -1:
            response = response[first_newline + 1:]
        if response.endswith('```'):
            response = response[:-3]

    response = response.strip()

    # First attempt: try parsing cleaned response directly
    try:
        result: dict[str, Any] | list[Any] = json.loads(response)
        logger.debug("parse_llm_json: parsed via strategy=%s", strategy)
        return result
    except json.JSONDecodeError:
        pass

    # Fallback: extract JSON by finding first { or [ and last } or ]
    start_brace = response.find('{')
    start_bracket = response.find('[')

    if start_brace == -1 and start_bracket == -1:
        logger.warning(
            "parse_llm_json: no JSON structure found in response (strategy=%s): %s",
            strategy, raw_preview,
        )
        return cast(dict[str, Any] | list[Any], json.loads(response))  # Will raise JSONDecodeError

    if start_brace == -1:
        start = start_bracket
        end_char = ']'
    elif start_bracket == -1:
        start = start_brace
        end_char = '}'
    else:
        start = min(start_brace, start_bracket)
        end_char = '}' if start == start_brace else ']'

    end = response.rfind(end_char)

    if end != -1 and end > start:
        extracted = response[start:end + 1]
        skipped_before = response[:start].strip()
        skipped_after = response[end + 1:].strip()
        if skipped_before or skipped_after:
            logger.info(
                "parse_llm_json: extracted JSON via heuristic (chars %d-%d); "
                "skipped prefix=%r suffix=%r",
                start, end, skipped_before[:80], skipped_after[:80],
            )
        strategy = "heuristic-extraction"
        response = extracted

    try:
        result = json.loads(response)
        logger.debug("parse_llm_json: parsed via strategy=%s", strategy)
        return result
    except json.JSONDecodeError:
        logger.warning(
            "parse_llm_json: all strategies failed (last=%s). Raw response: %s",
            strategy, raw_preview,
        )
        raise
