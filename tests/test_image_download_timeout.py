"""Tests that call_image_api uses config.IMAGE_TIMEOUT for the image download request."""

import base64
import json
from unittest.mock import MagicMock, patch


def _make_api_response(payload: dict) -> MagicMock:
    """Build a mock requests.Response for the image generation POST call."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _make_download_response(content: bytes = b"fake_image_data") -> MagicMock:
    """Build a mock streaming requests.Response for the image download GET call."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.iter_content.return_value = iter([content])
    return resp


class TestImageDownloadUsesConfigTimeout:
    """Verify that requests.get for image URL download uses config.IMAGE_TIMEOUT."""

    def test_download_uses_config_image_timeout(self, tmp_path, monkeypatch):
        import novelforge.config as config
        import novelforge.llm.image as image_mod

        # Configure necessary settings
        monkeypatch.setattr(config, "IMAGE_API_KEY", "test-key")
        monkeypatch.setattr(config, "IMAGE_API_URL", "https://api.example.com/images")
        monkeypatch.setattr(config, "IMAGE_MODEL", "dall-e-3")
        monkeypatch.setattr(config, "IMAGE_SIZE", "1024x1024")
        monkeypatch.setattr(config, "IMAGE_TIMEOUT", 120)
        monkeypatch.setattr(config, "EXPORT_DIR", str(tmp_path))

        api_resp = _make_api_response({
            "data": [{"url": "https://cdn.example.com/image.png"}]
        })
        download_resp = _make_download_response(b"\x89PNG\r\n")

        get_calls = []

        def fake_get(url, **kwargs):
            get_calls.append({"url": url, "kwargs": kwargs})
            return download_resp

        with patch.object(image_mod.requests, "post", return_value=api_resp), \
             patch.object(image_mod.requests, "get", side_effect=fake_get):
            result = image_mod.call_image_api("a brave hero", filename_prefix="test")

        assert result is not None, "call_image_api should return a filename on success"
        assert len(get_calls) == 1, "requests.get should be called once for the download"

        actual_timeout = get_calls[0]["kwargs"].get("timeout")
        assert actual_timeout == 120, (
            f"Expected timeout=config.IMAGE_TIMEOUT (120), got {actual_timeout!r}"
        )

    def test_download_timeout_reflects_updated_config(self, tmp_path, monkeypatch):
        """Changing IMAGE_TIMEOUT should change the download timeout accordingly."""
        import novelforge.config as config
        import novelforge.llm.image as image_mod

        monkeypatch.setattr(config, "IMAGE_API_KEY", "test-key")
        monkeypatch.setattr(config, "IMAGE_API_URL", "https://api.example.com/images")
        monkeypatch.setattr(config, "IMAGE_MODEL", "dall-e-3")
        monkeypatch.setattr(config, "IMAGE_SIZE", "1024x1024")
        monkeypatch.setattr(config, "IMAGE_TIMEOUT", 300)
        monkeypatch.setattr(config, "EXPORT_DIR", str(tmp_path))

        api_resp = _make_api_response({
            "data": [{"url": "https://cdn.example.com/image.png"}]
        })
        download_resp = _make_download_response(b"\x89PNG\r\n")

        get_calls = []

        def fake_get(url, **kwargs):
            get_calls.append({"url": url, "kwargs": kwargs})
            return download_resp

        with patch.object(image_mod.requests, "post", return_value=api_resp), \
             patch.object(image_mod.requests, "get", side_effect=fake_get):
            image_mod.call_image_api("sunset scene", filename_prefix="cover")

        actual_timeout = get_calls[0]["kwargs"].get("timeout")
        assert actual_timeout == 300, (
            f"Expected timeout=config.IMAGE_TIMEOUT (300), got {actual_timeout!r}"
        )
