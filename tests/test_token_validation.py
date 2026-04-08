"""
Tests for progress-token format validation.

Covers the acceptance criteria from the issue:
  - Malformed tokens receive an "Invalid progress token." 400 response
    instead of a misleading "Novel generation not complete." message.
  - Valid UUID-format tokens that are unknown receive "not complete" (400)
    or "Unknown token" (404) from the respective endpoints.
  - _is_valid_token correctly accepts / rejects token strings.
"""

import json
import uuid

import pytest

from novelforge.routes.generation._shared import _is_valid_token, _UUID_RE
from novelforge.routes.export import _is_valid_token as _export_is_valid_token
from novelforge.progress import progress_manager


# ---------------------------------------------------------------------------
# Unit tests for _is_valid_token
# ---------------------------------------------------------------------------

class TestIsValidToken:
    """_is_valid_token must accept well-formed UUIDs and reject everything else."""

    def test_valid_uuid_accepted(self):
        assert _is_valid_token(str(uuid.uuid4()))

    def test_all_zeros_uuid_accepted(self):
        # UUID(int=0) is technically valid format (though not v4)
        assert _is_valid_token("00000000-0000-0000-0000-000000000000")

    def test_uppercase_rejected(self):
        # Our regex requires lowercase hex
        assert not _is_valid_token("AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA")

    def test_missing_dashes_rejected(self):
        assert not _is_valid_token("00000000000040008000000000000001")

    def test_too_short_rejected(self):
        assert not _is_valid_token("00000000-0000-4000-8000-00000000001")

    def test_too_long_rejected(self):
        assert not _is_valid_token("00000000-0000-4000-8000-0000000000011")

    def test_human_readable_string_rejected(self):
        assert not _is_valid_token("fake-token")

    def test_empty_string_rejected(self):
        assert not _is_valid_token("")

    def test_none_like_string_rejected(self):
        assert not _is_valid_token("None")

    def test_export_module_helper_matches(self):
        """Both copies of _is_valid_token must behave identically."""
        valid = str(uuid.uuid4())
        assert _is_valid_token(valid) == _export_is_valid_token(valid)
        assert _is_valid_token("bad-token") == _export_is_valid_token("bad-token")
        assert _is_valid_token("") == _export_is_valid_token("")


# ---------------------------------------------------------------------------
# HTTP-route tests: malformed token → 400 "Invalid progress token."
# ---------------------------------------------------------------------------

_MALFORMED_TOKENS = [
    "fake-token",
    "incomplete-novel",
    "not-a-uuid",
    "",
    "00000000-0000-0000-0000",          # too short
    "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA",  # uppercase
]

# Subset of malformed tokens that are valid URL path segments (non-empty).
_MALFORMED_URL_TOKENS = [t for t in _MALFORMED_TOKENS if t]


class TestExportNovelInvalidToken:
    @pytest.mark.parametrize("bad_token", _MALFORMED_TOKENS)
    def test_rejects_malformed_token(self, client, bad_token):
        r = client.post(
            "/export",
            data=json.dumps({"token": bad_token}),
            content_type="application/json",
        )
        assert r.status_code == 400
        assert r.get_json()["error"] == "Invalid progress token."

    def test_valid_but_unknown_uuid_gives_not_complete(self, client):
        """A well-formed UUID that has no progress entry returns 'not complete'."""
        unknown = str(uuid.uuid4())
        r = client.post(
            "/export",
            data=json.dumps({"token": unknown}),
            content_type="application/json",
        )
        assert r.status_code == 400
        assert "not complete" in r.get_json()["error"].lower()


class TestExportEditorsNotesInvalidToken:
    @pytest.mark.parametrize("bad_token", _MALFORMED_TOKENS)
    def test_rejects_malformed_token(self, client, bad_token):
        r = client.post(
            "/export_editors_notes",
            data=json.dumps({"token": bad_token}),
            content_type="application/json",
        )
        assert r.status_code == 400
        assert r.get_json()["error"] == "Invalid progress token."

    def test_valid_but_unknown_uuid_gives_not_complete(self, client):
        unknown = str(uuid.uuid4())
        r = client.post(
            "/export_editors_notes",
            data=json.dumps({"token": unknown}),
            content_type="application/json",
        )
        assert r.status_code == 400
        assert "not complete" in r.get_json()["error"].lower()


class TestGenerateIllustrationsInvalidToken:
    @pytest.mark.parametrize("bad_token", _MALFORMED_TOKENS)
    def test_rejects_malformed_token(self, client, bad_token):
        r = client.post(
            "/generate_illustrations",
            data=json.dumps({"token": bad_token}),
            content_type="application/json",
        )
        assert r.status_code == 400
        assert r.get_json()["error"] == "Invalid progress token."

    def test_valid_but_unknown_uuid_gives_not_complete(self, client):
        unknown = str(uuid.uuid4())
        r = client.post(
            "/generate_illustrations",
            data=json.dumps({"token": unknown}),
            content_type="application/json",
        )
        assert r.status_code == 400
        assert "not complete" in r.get_json()["error"].lower()


class TestReviseChapterInvalidToken:
    @pytest.mark.parametrize("bad_token", [t for t in _MALFORMED_TOKENS if t])
    def test_rejects_malformed_token(self, client, bad_token):
        r = client.post(
            "/revise_chapter",
            data=json.dumps({
                "token": bad_token,
                "chapter_number": 1,
                "instructions": "Add drama.",
            }),
            content_type="application/json",
        )
        assert r.status_code == 400
        assert r.get_json()["error"] == "Invalid progress token."

    def test_rejects_empty_token(self, client):
        """Empty token is caught by the earlier 'missing token' guard."""
        r = client.post(
            "/revise_chapter",
            data=json.dumps({
                "token": "",
                "chapter_number": 1,
                "instructions": "Add drama.",
            }),
            content_type="application/json",
        )
        assert r.status_code == 400
        assert "missing" in r.get_json()["error"].lower()

    def test_valid_but_unknown_uuid_gives_not_complete(self, client):
        unknown = str(uuid.uuid4())
        r = client.post(
            "/revise_chapter",
            data=json.dumps({
                "token": unknown,
                "chapter_number": 1,
                "instructions": "Add drama.",
            }),
            content_type="application/json",
        )
        assert r.status_code == 400
        assert "not complete" in r.get_json()["error"].lower()


class TestProgressEndpointInvalidToken:
    @pytest.mark.parametrize("bad_token", _MALFORMED_URL_TOKENS)
    def test_lightweight_rejects_malformed_token(self, client, bad_token):
        r = client.get(f"/progress/{bad_token}")
        assert r.status_code == 400
        assert r.get_json()["error"] == "Invalid progress token."

    @pytest.mark.parametrize("bad_token", _MALFORMED_URL_TOKENS)
    def test_full_rejects_malformed_token(self, client, bad_token):
        r = client.get(f"/progress/{bad_token}/full")
        assert r.status_code == 400
        assert r.get_json()["error"] == "Invalid progress token."

    def test_valid_but_unknown_uuid_gives_404(self, client):
        unknown = str(uuid.uuid4())
        r = client.get(f"/progress/{unknown}")
        assert r.status_code == 404

    def test_valid_but_unknown_uuid_full_gives_404(self, client):
        unknown = str(uuid.uuid4())
        r = client.get(f"/progress/{unknown}/full")
        assert r.status_code == 404
