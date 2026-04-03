"""Unit tests for novelforge.llm.image helper functions."""

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from novelforge.llm.image import (
    _download_image,
    _extract_image_entry,
    _persist_image,
    _send_image_request,
    call_image_api,
)


# ---------------------------------------------------------------------------
# _extract_image_entry
# ---------------------------------------------------------------------------


class TestExtractImageEntry:
    """Tests for the response-parsing helper."""

    def test_returns_entry_with_b64_json(self):
        data = {"data": [{"b64_json": "abc123"}]}
        result = _extract_image_entry(data)
        assert result == {"b64_json": "abc123"}

    def test_returns_entry_with_url(self):
        data = {"data": [{"url": "https://example.com/img.png"}]}
        result = _extract_image_entry(data)
        assert result == {"url": "https://example.com/img.png"}

    def test_missing_data_key_returns_none(self):
        result = _extract_image_entry({})
        assert result is None

    def test_empty_data_list_returns_none(self):
        result = _extract_image_entry({"data": []})
        assert result is None

    def test_data_not_a_list_returns_none(self):
        result = _extract_image_entry({"data": "not-a-list"})
        assert result is None

    def test_non_dict_entry_returns_none(self):
        result = _extract_image_entry({"data": ["not-a-dict"]})
        assert result is None

    def test_entry_without_url_or_b64_returns_none(self):
        result = _extract_image_entry({"data": [{"revised_prompt": "something"}]})
        assert result is None

    def test_entry_with_both_fields_is_accepted(self):
        entry = {"url": "https://example.com/img.png", "b64_json": "abc"}
        result = _extract_image_entry({"data": [entry]})
        assert result == entry


# ---------------------------------------------------------------------------
# _download_image
# ---------------------------------------------------------------------------


class TestDownloadImage:
    """Tests for the remote URL download helper."""

    def test_success_returns_bytes(self, mocker):
        mock_resp = MagicMock()
        mock_resp.content = b"fake-image-bytes"
        mock_resp.raise_for_status = MagicMock()
        mocker.patch("novelforge.llm.image.requests.get", return_value=mock_resp)

        result = _download_image("https://example.com/img.png")
        assert result == b"fake-image-bytes"

    def test_http_error_returns_none(self, mocker):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
        mocker.patch("novelforge.llm.image.requests.get", return_value=mock_resp)

        result = _download_image("https://example.com/img.png")
        assert result is None

    def test_connection_error_returns_none(self, mocker):
        mocker.patch(
            "novelforge.llm.image.requests.get",
            side_effect=requests.exceptions.ConnectionError("refused"),
        )

        result = _download_image("https://example.com/img.png")
        assert result is None

    def test_timeout_returns_none(self, mocker):
        mocker.patch(
            "novelforge.llm.image.requests.get",
            side_effect=requests.exceptions.Timeout("timed out"),
        )

        result = _download_image("https://example.com/img.png")
        assert result is None


# ---------------------------------------------------------------------------
# _persist_image
# ---------------------------------------------------------------------------


class TestPersistImage:
    """Tests for the filesystem persistence helper."""

    def test_success_writes_bytes(self, tmp_path):
        save_path = tmp_path / "illustrations" / "cover_abc12345.png"
        result = _persist_image(b"image-data", save_path)

        assert result is True
        assert save_path.read_bytes() == b"image-data"

    def test_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "img.png"
        result = _persist_image(b"data", nested)

        assert result is True
        assert nested.exists()

    def test_mkdir_failure_returns_false(self, tmp_path, mocker):
        save_path = tmp_path / "illustrations" / "img.png"
        mocker.patch.object(Path, "mkdir", side_effect=OSError("permission denied"))

        result = _persist_image(b"data", save_path)
        assert result is False

    def test_write_failure_returns_false(self, tmp_path, mocker):
        save_path = tmp_path / "illustrations" / "img.png"
        # Let mkdir succeed but write_bytes fail
        mocker.patch.object(Path, "write_bytes", side_effect=OSError("disk full"))

        result = _persist_image(b"data", save_path)
        assert result is False


# ---------------------------------------------------------------------------
# _send_image_request
# ---------------------------------------------------------------------------


class TestSendImageRequest:
    """Tests for the transport/retry helper."""

    def test_success_returns_json(self, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"b64_json": "abc"}]}
        mock_resp.raise_for_status = MagicMock()
        mocker.patch("novelforge.llm.image.requests.post", return_value=mock_resp)
        mocker.patch("novelforge.llm.image.time.sleep")

        result = _send_image_request(
            "https://api.example.com/images",
            {},
            {},
            timeout=30,
            max_retries=1,
        )
        assert result == {"data": [{"b64_json": "abc"}]}

    def test_retries_on_429_then_succeeds(self, mocker):
        rate_limit_resp = MagicMock()
        rate_limit_resp.status_code = 429

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"data": [{"url": "https://example.com/img.png"}]}
        ok_resp.raise_for_status = MagicMock()

        mocker.patch(
            "novelforge.llm.image.requests.post",
            side_effect=[rate_limit_resp, ok_resp],
        )
        mocker.patch("novelforge.llm.image.time.sleep")

        result = _send_image_request(
            "https://api.example.com/images",
            {},
            {},
            timeout=30,
            max_retries=3,
            retry_delay=0,
        )
        assert result == {"data": [{"url": "https://example.com/img.png"}]}

    def test_retries_on_500_then_succeeds(self, mocker):
        server_error = MagicMock()
        server_error.status_code = 500

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"data": [{"b64_json": "xyz"}]}
        ok_resp.raise_for_status = MagicMock()

        mocker.patch(
            "novelforge.llm.image.requests.post",
            side_effect=[server_error, ok_resp],
        )
        mocker.patch("novelforge.llm.image.time.sleep")

        result = _send_image_request(
            "https://api.example.com/images",
            {},
            {},
            timeout=30,
            max_retries=3,
            retry_delay=0,
        )
        assert result == {"data": [{"b64_json": "xyz"}]}

    def test_returns_none_when_retries_exhausted_on_429(self, mocker):
        rate_limit_resp = MagicMock()
        rate_limit_resp.status_code = 429

        mocker.patch("novelforge.llm.image.requests.post", return_value=rate_limit_resp)
        mocker.patch("novelforge.llm.image.time.sleep")

        result = _send_image_request(
            "https://api.example.com/images",
            {},
            {},
            timeout=30,
            max_retries=2,
            retry_delay=0,
        )
        assert result is None

    def test_returns_none_when_timeout_persists(self, mocker):
        mocker.patch(
            "novelforge.llm.image.requests.post",
            side_effect=requests.exceptions.Timeout("timed out"),
        )
        mocker.patch("novelforge.llm.image.time.sleep")

        result = _send_image_request(
            "https://api.example.com/images",
            {},
            {},
            timeout=30,
            max_retries=1,
        )
        assert result is None

    def test_request_exception_returns_none(self, mocker):
        mocker.patch(
            "novelforge.llm.image.requests.post",
            side_effect=requests.exceptions.ConnectionError("refused"),
        )
        mocker.patch("novelforge.llm.image.time.sleep")

        result = _send_image_request(
            "https://api.example.com/images",
            {},
            {},
            timeout=30,
            max_retries=3,
        )
        assert result is None


# ---------------------------------------------------------------------------
# call_image_api (orchestrator)
# ---------------------------------------------------------------------------


class TestCallImageApi:
    """End-to-end tests for the public orchestrator function."""

    def test_no_api_key_returns_none(self, mocker):
        mocker.patch("novelforge.llm.image.config.IMAGE_API_KEY", "")
        result = call_image_api("A hero at sunset")
        assert result is None

    def test_b64_response_saves_file(self, tmp_path, mocker):
        mocker.patch("novelforge.llm.image.config.IMAGE_API_KEY", "test-key")
        mocker.patch("novelforge.llm.image.config.IMAGE_API_URL", "https://api.example.com/images")
        mocker.patch("novelforge.llm.image.config.IMAGE_MODEL", "dall-e-3")
        mocker.patch("novelforge.llm.image.config.IMAGE_SIZE", "1024x1024")
        mocker.patch("novelforge.llm.image.config.IMAGE_TIMEOUT", 30)
        mocker.patch("novelforge.llm.image.config.EXPORT_DIR", str(tmp_path))

        raw_png = b"\x89PNG fake image bytes"
        encoded = base64.b64encode(raw_png).decode()
        mocker.patch(
            "novelforge.llm.image._send_image_request",
            return_value={"data": [{"b64_json": encoded}]},
        )

        filename = call_image_api("A hero at sunset", filename_prefix="cover")

        assert filename is not None
        assert filename.startswith("cover_")
        assert filename.endswith(".png")
        saved = tmp_path / "illustrations" / filename
        assert saved.read_bytes() == raw_png

    def test_url_response_downloads_and_saves_file(self, tmp_path, mocker):
        mocker.patch("novelforge.llm.image.config.IMAGE_API_KEY", "test-key")
        mocker.patch("novelforge.llm.image.config.IMAGE_API_URL", "https://api.example.com/images")
        mocker.patch("novelforge.llm.image.config.IMAGE_MODEL", "dall-e-3")
        mocker.patch("novelforge.llm.image.config.IMAGE_SIZE", "1024x1024")
        mocker.patch("novelforge.llm.image.config.IMAGE_TIMEOUT", 30)
        mocker.patch("novelforge.llm.image.config.EXPORT_DIR", str(tmp_path))

        img_bytes = b"downloaded-image-bytes"
        mocker.patch(
            "novelforge.llm.image._send_image_request",
            return_value={"data": [{"url": "https://cdn.example.com/img.png"}]},
        )
        mocker.patch(
            "novelforge.llm.image._download_image",
            return_value=img_bytes,
        )

        filename = call_image_api("A stormy sea")

        assert filename is not None
        assert filename.endswith(".png")
        saved = tmp_path / "illustrations" / filename
        assert saved.read_bytes() == img_bytes

    def test_transport_failure_returns_none(self, mocker):
        mocker.patch("novelforge.llm.image.config.IMAGE_API_KEY", "test-key")
        mocker.patch("novelforge.llm.image.config.IMAGE_API_URL", "https://api.example.com/images")
        mocker.patch("novelforge.llm.image.config.IMAGE_MODEL", "dall-e-3")
        mocker.patch("novelforge.llm.image.config.IMAGE_SIZE", "1024x1024")
        mocker.patch("novelforge.llm.image.config.IMAGE_TIMEOUT", 30)
        mocker.patch("novelforge.llm.image._send_image_request", return_value=None)

        result = call_image_api("A prompt")
        assert result is None

    def test_malformed_response_returns_none(self, mocker):
        mocker.patch("novelforge.llm.image.config.IMAGE_API_KEY", "test-key")
        mocker.patch("novelforge.llm.image.config.IMAGE_API_URL", "https://api.example.com/images")
        mocker.patch("novelforge.llm.image.config.IMAGE_MODEL", "dall-e-3")
        mocker.patch("novelforge.llm.image.config.IMAGE_SIZE", "1024x1024")
        mocker.patch("novelforge.llm.image.config.IMAGE_TIMEOUT", 30)
        mocker.patch(
            "novelforge.llm.image._send_image_request",
            return_value={"not_data": []},
        )

        result = call_image_api("A prompt")
        assert result is None

    def test_url_download_failure_returns_none(self, mocker):
        mocker.patch("novelforge.llm.image.config.IMAGE_API_KEY", "test-key")
        mocker.patch("novelforge.llm.image.config.IMAGE_API_URL", "https://api.example.com/images")
        mocker.patch("novelforge.llm.image.config.IMAGE_MODEL", "dall-e-3")
        mocker.patch("novelforge.llm.image.config.IMAGE_SIZE", "1024x1024")
        mocker.patch("novelforge.llm.image.config.IMAGE_TIMEOUT", 30)
        mocker.patch(
            "novelforge.llm.image._send_image_request",
            return_value={"data": [{"url": "https://cdn.example.com/img.png"}]},
        )
        mocker.patch("novelforge.llm.image._download_image", return_value=None)

        result = call_image_api("A prompt")
        assert result is None

    def test_persistence_failure_returns_none(self, tmp_path, mocker):
        mocker.patch("novelforge.llm.image.config.IMAGE_API_KEY", "test-key")
        mocker.patch("novelforge.llm.image.config.IMAGE_API_URL", "https://api.example.com/images")
        mocker.patch("novelforge.llm.image.config.IMAGE_MODEL", "dall-e-3")
        mocker.patch("novelforge.llm.image.config.IMAGE_SIZE", "1024x1024")
        mocker.patch("novelforge.llm.image.config.IMAGE_TIMEOUT", 30)
        mocker.patch("novelforge.llm.image.config.EXPORT_DIR", str(tmp_path))

        raw_png = b"\x89PNG fake"
        encoded = base64.b64encode(raw_png).decode()
        mocker.patch(
            "novelforge.llm.image._send_image_request",
            return_value={"data": [{"b64_json": encoded}]},
        )
        mocker.patch("novelforge.llm.image._persist_image", return_value=False)

        result = call_image_api("A prompt")
        assert result is None
