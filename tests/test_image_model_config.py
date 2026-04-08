"""Tests that IMAGE_MODEL default is the correct 'gpt-image-1' model name.

The model 'gpt-image-1-mini' does not exist in the OpenAI API; using it
causes every image generation request to fail with a 4xx error and the
UI shows "No illustrations were generated."
"""

import base64
import pathlib
from unittest.mock import MagicMock, patch


def _make_api_response(payload: dict) -> MagicMock:
    """Build a mock requests.Response for the image generation POST call."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


class TestImageModelDefault:
    """Verify that the default IMAGE_MODEL is 'gpt-image-1' (not 'gpt-image-1-mini')."""

    def test_default_image_model_in_source(self):
        """config.py source must use 'gpt-image-1' as the fallback, not 'gpt-image-1-mini'."""
        config_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "novelforge"
            / "config.py"
        )
        source = config_path.read_text()
        assert '"gpt-image-1-mini"' not in source, (
            "Found invalid model 'gpt-image-1-mini' in novelforge/config.py. "
            "This model does not exist in the OpenAI API and causes all image "
            "generation requests to fail. Use 'gpt-image-1' instead."
        )
        assert '"gpt-image-1"' in source, (
            "novelforge/config.py should use 'gpt-image-1' as the IMAGE_MODEL default."
        )

    def test_model_name_sent_in_api_request(self, tmp_path, monkeypatch):
        """call_image_api must send 'gpt-image-1' as the model in the POST payload."""
        import novelforge.config as config
        import novelforge.llm.image as image_mod

        monkeypatch.setattr(config, "IMAGE_API_KEY", "test-key")
        monkeypatch.setattr(config, "IMAGE_API_URL", "https://api.openai.com/v1/images/generations")
        monkeypatch.setattr(config, "IMAGE_MODEL", "gpt-image-1")
        monkeypatch.setattr(config, "IMAGE_SIZE", "1024x1024")
        monkeypatch.setattr(config, "EXPORT_DIR", str(tmp_path))

        b64_data = base64.b64encode(b"fake_png_bytes").decode()
        api_resp = _make_api_response({"data": [{"b64_json": b64_data}]})

        post_calls = []

        def fake_post(url, headers=None, json=None, timeout=None, **kwargs):
            post_calls.append({"url": url, "json": json})
            return api_resp

        with patch.object(image_mod.requests, "post", side_effect=fake_post):
            result = image_mod.call_image_api("a scene", filename_prefix="ch1")

        assert result is not None, "call_image_api should succeed with valid model"
        assert len(post_calls) == 1
        assert post_calls[0]["json"]["model"] == "gpt-image-1", (
            f"Expected model='gpt-image-1' in POST body, got {post_calls[0]['json']['model']!r}"
        )

    def test_invalid_model_name_gpt_image_1_mini_returns_none(self, tmp_path, monkeypatch):
        """Sending 'gpt-image-1-mini' to the API triggers a 400 error → returns None."""
        import requests as requests_lib

        import novelforge.config as config
        import novelforge.llm.image as image_mod

        monkeypatch.setattr(config, "IMAGE_API_KEY", "test-key")
        monkeypatch.setattr(config, "IMAGE_API_URL", "https://api.openai.com/v1/images/generations")
        monkeypatch.setattr(config, "IMAGE_MODEL", "gpt-image-1-mini")
        monkeypatch.setattr(config, "IMAGE_SIZE", "1024x1024")
        monkeypatch.setattr(config, "EXPORT_DIR", str(tmp_path))

        # Simulate the OpenAI API response for an unknown model: 400 Bad Request.
        error_resp = MagicMock()
        error_resp.status_code = 400
        error_resp.raise_for_status.side_effect = requests_lib.exceptions.HTTPError(
            "400 Client Error: Bad Request", response=error_resp
        )

        with patch.object(image_mod.requests, "post", return_value=error_resp):
            result = image_mod.call_image_api("a scene", filename_prefix="ch1")

        assert result is None, (
            "call_image_api should return None when the API rejects an invalid model name"
        )
