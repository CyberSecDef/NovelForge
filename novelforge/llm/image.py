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

    # Log the request
    request_log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": "image_request",
        "url": config.IMAGE_API_URL,
        "model": config.IMAGE_MODEL,
        "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
    }
    llm_logger.info(json.dumps(request_log, indent=2))

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                config.IMAGE_API_URL,
                headers=headers,
                json=payload,
                timeout=config.IMAGE_TIMEOUT,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = RETRY_DELAY * attempt
                logger.warning(
                    "Image API returned %s – retry %d/%d in %ds",
                    resp.status_code, attempt, MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()

            # Generate unique filename
            unique_id = uuid.uuid4().hex[:8]
            filename = f"{filename_prefix}_{unique_id}.png"
            save_path = Path(config.EXPORT_DIR) / "illustrations" / filename

            # Extract image data — OpenAI returns either url or b64_json
            image_data = data.get("data", [{}])[0]

            if "b64_json" in image_data:
                # Decode base64 and save
                img_bytes = base64.b64decode(image_data["b64_json"])
                save_path.write_bytes(img_bytes)
            elif "url" in image_data:
                # Download from URL
                img_resp = requests.get(image_data["url"], timeout=60)
                img_resp.raise_for_status()
                save_path.write_bytes(img_resp.content)
            else:
                logger.error("Image API returned no url or b64_json: %s", data)
                return None

            logger.info("Saved illustration to %s", save_path)
            return filename

        except requests.exceptions.Timeout:
            logger.warning("Image API timed out (attempt %d/%d)", attempt, MAX_RETRIES)
            if attempt == MAX_RETRIES:
                return None
            time.sleep(RETRY_DELAY * attempt)
        except requests.exceptions.RequestException as exc:
            logger.error("Image API request failed: %s", exc)
            return None

    return None
