"""LLM API client: call_llm, circuit breaker, JSON parsing, usage tracking."""

import json
import logging
import os
import threading
import time

import requests

import novelforge.config as config

logger = logging.getLogger(__name__)

# Set up dedicated LLM logger that writes JSON objects to logs/llm.log
llm_logger = logging.getLogger("llm_requests")
llm_logger.setLevel(logging.INFO)
llm_handler = logging.FileHandler(os.path.join(config.LOGS_DIR, "llm.log"))
llm_handler.setFormatter(logging.Formatter("%(message)s"))
llm_logger.addHandler(llm_handler)
llm_logger.propagate = False

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
        self._threshold = threshold
        self._consecutive_failures = 0
        self._tripped = False
        self._last_error: str = ""
        self._lock = threading.Lock()

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._tripped = False

    def record_failure(self, error: str) -> None:
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
        with self._lock:
            self._consecutive_failures = 0
            self._tripped = False
            self._last_error = ""

    @property
    def is_tripped(self) -> bool:
        with self._lock:
            return self._tripped

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._consecutive_failures


class CircuitBreakerError(RuntimeError):
    """Raised when the LLM circuit breaker is open."""
    pass


class ChapterTimeoutError(RuntimeError):
    """Raised when a single chapter exceeds the per-chapter time limit."""
    pass


_llm_circuit_breaker = LLMCircuitBreaker()


def call_llm(messages: list[dict], *, action: str = "", json_mode: bool = False) -> str:
    """
    Call the configured LLM API and return the assistant message content.

    Retries up to MAX_RETRIES times on transient errors (429, 5xx).
    Raises RuntimeError on persistent failure.
    Raises CircuitBreakerError if the breaker is tripped.
    """
    _llm_circuit_breaker.check()

    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: dict = {
        "model": config.LLM_MODEL,
        "messages": messages,
    }
    action = action if action else "Updating Content"

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    # Log the request (sanitize API key)
    request_log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": "request",
        "action": action,
        "url": config.LLM_API_URL,
        "headers": {
            "Authorization": f"Bearer {config.LLM_API_KEY[:8]}..." if config.LLM_API_KEY else "None",
            "Content-Type": "application/json",
        },
        "payload": payload,
    }
    llm_logger.info(json.dumps(request_log, indent=2))

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                config.LLM_API_URL,
                headers=headers,
                json=payload,
                timeout=config.LLM_TIMEOUT,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = RETRY_DELAY * attempt
                logger.warning(
                    "LLM API returned %s – retry %d/%d in %ds",
                    resp.status_code, attempt, MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()

            # Log the response
            response_log = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": "response",
                "status_code": resp.status_code,
                "headers": dict(resp.headers),
                "response": data,
            }
            llm_logger.info(json.dumps(response_log, indent=2))

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

            _llm_circuit_breaker.record_success()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            logger.warning("LLM request timed out (attempt %d/%d)", attempt, MAX_RETRIES)
            if attempt == MAX_RETRIES:
                err = "LLM_TIMEOUT: The AI service is taking too long. Your progress is saved — you can resume when it recovers."
                _llm_circuit_breaker.record_failure(err)
                raise RuntimeError(err)
            time.sleep(RETRY_DELAY * attempt)
        except requests.exceptions.RequestException as exc:
            # Log the error
            error_log = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": "error",
                "error": str(exc),
            }
            llm_logger.info(json.dumps(error_log))

            # Map HTTP status codes to user-friendly messages
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403):
                err = "LLM_AUTH_FAILURE: API key rejected. Check your LLM_API_KEY setting."
            elif status == 400:
                err = "LLM_BAD_REQUEST: The AI service rejected the request. The prompt may be too long or contain unsupported content."
            elif status == 404:
                err = "LLM_ENDPOINT_NOT_FOUND: The API endpoint was not found. Check your LLM_API_URL setting."
            else:
                err = f"LLM_REQUEST_FAILED: {exc}"
            _llm_circuit_breaker.record_failure(err)
            raise RuntimeError(err) from exc

    err = "LLM_RATE_LIMITED: The AI service is overloaded. Your progress is saved — please wait a few minutes and try again."
    _llm_circuit_breaker.record_failure(err)
    raise RuntimeError(err)


def _reset_llm_usage() -> None:
    """Reset the thread-local token usage accumulator."""
    _llm_usage.prompt_tokens = 0
    _llm_usage.completion_tokens = 0
    _llm_usage.call_count = 0


def _get_llm_usage() -> dict:
    """Return current accumulated token usage and reset."""
    usage = {
        "prompt_tokens": getattr(_llm_usage, "prompt_tokens", 0),
        "completion_tokens": getattr(_llm_usage, "completion_tokens", 0),
        "total_tokens": getattr(_llm_usage, "prompt_tokens", 0) + getattr(_llm_usage, "completion_tokens", 0),
        "call_count": getattr(_llm_usage, "call_count", 0),
    }
    _reset_llm_usage()
    return usage


def _friendly_llm_error(exc: Exception) -> str:
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
    # CircuitBreakerError or ChapterTimeoutError — already friendly
    if isinstance(exc, (CircuitBreakerError, ChapterTimeoutError)):
        return msg
    # Fallback
    return "Something went wrong with the AI service. Your progress is saved — please try again."


def parse_llm_json(response: str) -> dict:
    """
    Parse JSON from LLM response, handling common irregularities:
    - Markdown code fences (```json, ```)
    - Extra whitespace
    - BOM characters
    - Text before/after the JSON object

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
        result = json.loads(response)
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
        return json.loads(response)  # Will raise JSONDecodeError

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
