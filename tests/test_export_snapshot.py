"""
Tests verifying that export and illustration routes use the token's
generation snapshot rather than the live Flask session.

Covers the acceptance criteria from the issue:
  - Export routes no longer depend on mutable session values for
    token-scoped artifacts.
  - Given the same token, results are reproducible regardless of the
    current session state.
  - Token/session mismatches do not corrupt exported output.
"""

import json
import pytest

from novelforge.progress import progress_manager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _done_token(token: str, title: str, **extra) -> None:
    """Seed the progress_manager with a completed generation entry."""
    progress_manager.create(token, {
        "status": "done",
        "current": 2,
        "total": 2,
        "step": "Complete",
        "error": None,
        "snapshot": {
            "title": title,
            "genre": "Fantasy",
            "premise": "A hero embarks on a quest.",
            "character_list": [
                {"name": "Alice", "role": "Protagonist"},
            ],
        },
        "chapters_done": [
            {"number": 1, "title": "Ch1", "content": "Chapter 1 text.", "summary": "S1"},
            {"number": 2, "title": "Ch2", "content": "Chapter 2 text.", "summary": "S2"},
        ],
        "consistency": {"overall_assessment": "Good.", "issues": []},
        **extra,
    })


# ---------------------------------------------------------------------------
# /export  –  title comes from snapshot
# ---------------------------------------------------------------------------

class TestExportUsesSnapshot:
    """The /export route must derive the manuscript title from the token snapshot."""

    def test_title_from_snapshot_not_session(self, client, tmp_path, monkeypatch):
        """Even if the session title differs, the exported file reflects the snapshot."""
        import novelforge.config as config
        monkeypatch.setattr(config, "EXPORT_DIR", str(tmp_path))

        token = "snap-export-title"
        _done_token(token, title="Snapshot Novel Title")

        # Deliberately set a *different* title in the live session.
        with client.session_transaction() as sess:
            sess["title"] = "STALE SESSION TITLE"

        r = client.post(
            "/export",
            data=json.dumps({"token": token}),
            content_type="application/json",
        )
        assert r.status_code == 200
        url = r.get_json()["download_url"]

        # The downloaded file must use the snapshot title, not the session title.
        r2 = client.get(url)
        assert r2.status_code == 200
        content = r2.data.decode("utf-8")
        assert "Snapshot Novel Title" in content
        assert "STALE SESSION TITLE" not in content

    def test_title_reproducible_across_session_changes(self, client, tmp_path, monkeypatch):
        """Calling /export twice with the same token always yields the same title."""
        import novelforge.config as config
        monkeypatch.setattr(config, "EXPORT_DIR", str(tmp_path))

        token = "snap-export-repro"
        _done_token(token, title="Reproducible Title")

        # First export with one session title.
        with client.session_transaction() as sess:
            sess["title"] = "Session Title A"
        r1 = client.post("/export", data=json.dumps({"token": token}),
                         content_type="application/json")
        assert r1.status_code == 200
        url1 = r1.get_json()["download_url"]
        content1 = client.get(url1).data.decode("utf-8")

        # Change the session title and export again.
        with client.session_transaction() as sess:
            sess["title"] = "Session Title B"
        r2 = client.post("/export", data=json.dumps({"token": token}),
                         content_type="application/json")
        assert r2.status_code == 200
        url2 = r2.get_json()["download_url"]
        content2 = client.get(url2).data.decode("utf-8")

        # Both exports must use the snapshot title.
        assert "Reproducible Title" in content1
        assert "Reproducible Title" in content2
        assert "Session Title A" not in content1
        assert "Session Title B" not in content2

    def test_no_snapshot_falls_back_to_default(self, client, tmp_path, monkeypatch):
        """When a token has no snapshot the title defaults to 'Novel'."""
        import novelforge.config as config
        monkeypatch.setattr(config, "EXPORT_DIR", str(tmp_path))

        token = "snap-export-no-snap"
        progress_manager.create(token, {
            "status": "done",
            "current": 1,
            "total": 1,
            "step": "Complete",
            "error": None,
            "chapters_done": [
                {"number": 1, "title": "Ch1", "content": "Text.", "summary": "S"},
            ],
        })
        # Session sets a title but snapshot is absent – route should still work.
        with client.session_transaction() as sess:
            sess["title"] = "Session Only Title"

        r = client.post("/export", data=json.dumps({"token": token}),
                        content_type="application/json")
        assert r.status_code == 200
        url = r.get_json()["download_url"]
        content = client.get(url).data.decode("utf-8")
        # Falls back to "Novel" heading, not the session title.
        assert "# Novel" in content
        assert "Session Only Title" not in content


# ---------------------------------------------------------------------------
# /export_editors_notes  –  title comes from snapshot
# ---------------------------------------------------------------------------

class TestEditorsNotesUsesSnapshot:
    """The /export_editors_notes route must derive the title from the token snapshot."""

    def test_filename_reflects_snapshot_title(self, client, tmp_path, monkeypatch):
        """The exported filename uses the snapshot title, not the live session title."""
        import novelforge.config as config
        monkeypatch.setattr(config, "EXPORT_DIR", str(tmp_path))

        token = "snap-notes-title"
        _done_token(token, title="Snapshot Notes Title")

        with client.session_transaction() as sess:
            sess["title"] = "STALE SESSION"

        r = client.post(
            "/export_editors_notes",
            data=json.dumps({"token": token}),
            content_type="application/json",
        )
        assert r.status_code == 200
        payload = r.get_json()
        assert "download_url" in payload
        # Filename must contain the snapshot title, not the stale session value.
        assert "Snapshot_Notes_Title" in payload["download_url"]
        assert "STALE" not in payload["download_url"]

    def test_heading_reflects_snapshot_title(self, client, tmp_path, monkeypatch):
        """The H1 heading inside the notes file uses the snapshot title."""
        import novelforge.config as config
        monkeypatch.setattr(config, "EXPORT_DIR", str(tmp_path))

        token = "snap-notes-heading"
        _done_token(token, title="Correct Notes Title")

        with client.session_transaction() as sess:
            sess["title"] = "Wrong Session Title"

        r = client.post(
            "/export_editors_notes",
            data=json.dumps({"token": token}),
            content_type="application/json",
        )
        url = r.get_json()["download_url"]
        content = client.get(url).data.decode("utf-8")

        assert "Correct Notes Title" in content
        assert "Wrong Session Title" not in content


# ---------------------------------------------------------------------------
# /generate_illustrations  –  metadata comes from snapshot
# ---------------------------------------------------------------------------

class TestIllustrationsUsesSnapshot:
    """The /generate_illustrations route must use the token snapshot for all
    metadata (title, genre, premise, character_list) passed to the LLM."""

    def test_illustration_prompt_uses_snapshot_not_session(
        self, client, mock_llm, monkeypatch
    ):
        """The LLM is called with snapshot data even when the session differs.

        The route now returns immediately with an illustration_token; the
        background worker is run synchronously via a thread mock so assertions
        can be made right away.
        """
        import threading
        import novelforge.config as config
        import novelforge.routes.export as export_module

        monkeypatch.setattr(config, "IMAGE_API_KEY", "test-key")

        # Capture what build_illustration_prompt_generator_prompt receives.
        captured: dict = {}

        original_build = export_module.build_illustration_prompt_generator_prompt

        def capturing_build(**kwargs):
            captured.update(kwargs)
            return original_build(**kwargs)

        monkeypatch.setattr(
            export_module,
            "build_illustration_prompt_generator_prompt",
            capturing_build,
        )

        # Stub call_image_api so no real HTTP request is made.
        monkeypatch.setattr(
            export_module,
            "call_image_api",
            lambda prompt, filename_prefix="": f"{filename_prefix}.png",
        )

        # Run background thread synchronously so assertions work immediately.
        class SyncThread:
            def __init__(self, target=None, args=(), daemon=True, **kw):
                self._target = target
                self._args = args
            def start(self):
                if self._target:
                    self._target(*self._args)

        monkeypatch.setattr(export_module.threading, "Thread", SyncThread)

        token = "snap-illust-meta"
        _done_token(
            token,
            title="Snapshot Illustration Novel",
            # genre / premise / character_list already set in _done_token helper
        )

        # Put conflicting values in the live session.
        with client.session_transaction() as sess:
            sess["title"] = "WRONG TITLE"
            sess["genre"] = "WRONG GENRE"
            sess["premise"] = "WRONG PREMISE"
            sess["character_list"] = [{"name": "WRONG CHARACTER"}]

        r = client.post(
            "/generate_illustrations",
            data=json.dumps({"token": token}),
            content_type="application/json",
        )
        assert r.status_code == 200
        # The route returns a job token, not illustrations directly.
        assert "illustration_token" in r.get_json()

        # The prompt builder must have received snapshot values.
        assert captured.get("title") == "Snapshot Illustration Novel"
        assert captured.get("genre") == "Fantasy"
        assert captured.get("premise") == "A hero embarks on a quest."
        assert any(c.get("name") == "Alice" for c in captured.get("character_list", []))
        # Must NOT have received the stale session values.
        assert captured.get("title") != "WRONG TITLE"
        assert captured.get("genre") != "WRONG GENRE"

