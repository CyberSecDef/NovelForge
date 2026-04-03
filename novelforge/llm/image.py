"""Image generation API client."""

import base64
import json
import logging
import time
import uuid
from pathlib import Path

import requests

import novelforge.config as config
from novelforge.llm.client import MAX_RETRIES, RETRY_DELAY, llm_logger

logger = logging.getLogger(__name__)


def _log_image_request(prompt: str, url: str, model: str) -> None:
    """Log an outgoing image generation request to the LLM log."""
    request_log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": "image_request",
        "url": url,
        "model": model,
        "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
    }
    llm_logger.info(json.dumps(request_log, indent=2))


def _send_image_request(
    url: str,
    headers: dict,
    payload: dict,
    timeout: int,
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY,
) -> dict | None:
    """
    Transport layer: POST to the image API with retry logic.

    Retries on 429 and 5xx responses using linear back-off.  Returns the
    parsed JSON response dict on success, or None on unrecoverable failure.
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = retry_delay * attempt
                logger.warning(
                    "Image API returned %s – retry %d/%d in %ds",
                    resp.status_code, attempt, max_retries, wait,
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            logger.warning("Image API timed out (attempt %d/%d)", attempt, max_retries)
            if attempt == max_retries:
                return None
            time.sleep(retry_delay * attempt)
        except requests.exceptions.RequestException as exc:
            logger.error("Image API request failed: %s", exc)
            return None
    return None


def _extract_image_entry(data: dict) -> dict | None:
    """
    Parse an image API response and return the first image entry dict.

    Expects an OpenAI-compatible payload with a top-level ``"data"`` list
    whose first element contains either a ``"url"`` or ``"b64_json"`` key.

    Returns the image entry dict on success, or None if the response is
    malformed or missing a recognised image field.
    """
    data_list = data.get("data")
    if not isinstance(data_list, list) or len(data_list) == 0:
        logger.error(
            "Image API returned unexpected structure (missing 'data' array): %s", data
        )
        return None
    image_entry = data_list[0]
    if not isinstance(image_entry, dict):
        logger.error("Image API returned non-dict image entry: %s", image_entry)
        return None
    if "b64_json" not in image_entry and "url" not in image_entry:
        logger.error("Image API returned no url or b64_json: %s", data)
        return None
    return image_entry


def _download_image(url: str) -> bytes | None:
    """
    Download raw image bytes from a remote URL.

    Returns the image bytes on success, or None on any network failure.
    """
    try:
        img_resp = requests.get(url, timeout=60)
        img_resp.raise_for_status()
        return img_resp.content
    except requests.exceptions.RequestException as exc:
        logger.error("Failed to download image from %s: %s", url, exc)
        return None


def _persist_image(img_bytes: bytes, save_path: Path) -> bool:
    """
    Write image bytes to *save_path*, creating parent directories as needed.

    Returns True on success, False if either directory creation or the file
    write raises an OSError.
    """
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error(
            "Failed to create illustrations directory %s: %r",
            save_path.parent,
            exc,
        )
        return False
    try:
        save_path.write_bytes(img_bytes)
    except OSError as exc:
        logger.error("Failed to write image to %s: %r", save_path, exc)
        return False
    return True


def call_image_api(prompt: str, *, filename_prefix: str = "illustration") -> str | None:
    """
    Call the configured image generation API and save the result to disk.

    Supports OpenAI-compatible image generation endpoints that return either
    a URL or base64-encoded image data.

    Returns the saved filename (relative to illustrations dir) or None on failure.
    """
    if not config.IMAGE_API_KEY:
        logger.warning("IMAGE_API_KEY not set — skipping image generation")
        return None

    headers = {
        "Authorization": f"Bearer {config.IMAGE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.IMAGE_MODEL,
        "prompt": prompt,
        "n": 1,
        "size": config.IMAGE_SIZE,
    }

    _log_image_request(prompt, config.IMAGE_API_URL, config.IMAGE_MODEL)

    data = _send_image_request(
        config.IMAGE_API_URL,
        headers,
        payload,
        config.IMAGE_TIMEOUT,
    )
    if data is None:
        return None

    image_entry = _extract_image_entry(data)
    if image_entry is None:
        return None

    if "b64_json" in image_entry:
        img_bytes = base64.b64decode(image_entry["b64_json"])
    else:
        img_bytes = _download_image(image_entry["url"])
        if img_bytes is None:
            return None

    unique_id = uuid.uuid4().hex[:8]
    filename = f"{filename_prefix}_{unique_id}.png"
    save_path = Path(config.EXPORT_DIR) / "illustrations" / filename

    if not _persist_image(img_bytes, save_path):
        return None

    logger.info("Saved illustration to %s", save_path)
    return filename
