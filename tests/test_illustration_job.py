"""
Tests for the async illustration-generation background job.

Acceptance criteria covered:
- Route returns quickly with a job token (no sleep in request thread).
- Job progress is visible and pollable via /progress/<illustration_token>.
- Partial failures are recorded per-image with a "status" field.
- LLM retry logic runs inside the background worker, not the request handler.
- Full success, retry, and partial-failure flows are exercised.
"""

import json
import pytest

from novelforge.progress import progress_manager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _done_novel(token: str, **extra) -> None:
    """Seed progress_manager with a completed novel entry."""
    progress_manager.create(token, {
        "status": "done",
        "current": 2,
        "total": 2,
        "step": "Complete",
        "error": None,
        "snapshot": {
            "title": "Test Novel",
            "genre": "Fantasy",
            "premise": "A hero's journey.",
            "character_list": [{"name": "Hero", "role": "Protagonist"}],
        },
        "chapters_done": [
            {"number": 1, "title": "Ch1", "content": "Text1", "summary": "Sum1"},
            {"number": 2, "title": "Ch2", "content": "Text2", "summary": "Sum2"},
        ],
        **extra,
    })


class _SyncThread:
    """Synchronous stand-in for threading.Thread used in tests."""

    def __init__(self, target=None, args=(), daemon=True, **kw):
        self._target = target
        self._args = args

    def start(self):
        if self._target:
            self._target(*self._args)


def _patch_sync_thread(monkeypatch):
    import novelforge.routes.export as export_module
    monkeypatch.setattr(export_module.threading, "Thread", _SyncThread)


# ---------------------------------------------------------------------------
# Route contract: returns illustration_token immediately
# ---------------------------------------------------------------------------

class TestGenerateIllustrationsRoute:
    """The /generate_illustrations endpoint must return a job token quickly."""

    def test_returns_illustration_token(self, client, mock_llm, monkeypatch):
        import novelforge.config as config
        import novelforge.routes.export as export_module

        monkeypatch.setattr(config, "IMAGE_API_KEY", "key")
        monkeypatch.setattr(export_module, "call_image_api",
                            lambda p, filename_prefix="": "img.png")
        monkeypatch.setattr(export_module.threading, "Thread", _SyncThread)

        token = "route-token-test"
        _done_novel(token)

        r = client.post("/generate_illustrations",
                        data=json.dumps({"token": token}),
                        content_type="application/json")
        assert r.status_code == 200
        body = r.get_json()
        assert "illustration_token" in body
        assert body["illustration_token"]  # non-empty string

    def test_rejects_missing_token(self, client):
        r = client.post("/generate_illustrations",
                        data=json.dumps({}),
                        content_type="application/json")
        assert r.status_code == 400

    def test_rejects_incomplete_novel(self, client):
        progress_manager.create("incomplete-novel", {
            "status": "running",
            "current": 1,
            "total": 3,
            "step": "Running",
            "chapters_done": [],
            "error": None,
        })
        r = client.post("/generate_illustrations",
                        data=json.dumps({"token": "incomplete-novel"}),
                        content_type="application/json")
        assert r.status_code == 400

    def test_rejects_missing_image_api_key(self, client, monkeypatch):
        import novelforge.config as config
        monkeypatch.setattr(config, "IMAGE_API_KEY", "")

        token = "no-key-token"
        _done_novel(token)
        r = client.post("/generate_illustrations",
                        data=json.dumps({"token": token}),
                        content_type="application/json")
        assert r.status_code == 400
        assert "IMAGE_API_KEY" in r.get_json()["error"]

    def test_novel_token_records_illustration_token(self, client, mock_llm, monkeypatch):
        """After the route is called, the novel progress entry stores illustration_token."""
        import novelforge.config as config
        import novelforge.routes.export as export_module

        monkeypatch.setattr(config, "IMAGE_API_KEY", "key")
        monkeypatch.setattr(export_module, "call_image_api",
                            lambda p, filename_prefix="": "img.png")
        monkeypatch.setattr(export_module.threading, "Thread", _SyncThread)

        token = "novel-with-illust-link"
        _done_novel(token)

        r = client.post("/generate_illustrations",
                        data=json.dumps({"token": token}),
                        content_type="application/json")
        illust_token = r.get_json()["illustration_token"]

        novel_state = progress_manager.get(token)
        assert novel_state is not None
        assert novel_state.get("illustration_token") == illust_token


# ---------------------------------------------------------------------------
# Background worker: success flow
# ---------------------------------------------------------------------------

class TestIllustrationJobSuccess:
    """Full success: LLM returns prompts, image API generates all images."""

    def test_job_reaches_done_status(self, client, mock_llm, monkeypatch):
        import novelforge.config as config
        import novelforge.routes.export as export_module

        monkeypatch.setattr(config, "IMAGE_API_KEY", "key")
        monkeypatch.setattr(export_module, "call_image_api",
                            lambda p, filename_prefix="": f"{filename_prefix}_ok.png")
        _patch_sync_thread(monkeypatch)

        token = "success-novel"
        _done_novel(token)

        r = client.post("/generate_illustrations",
                        data=json.dumps({"token": token}),
                        content_type="application/json")
        illust_token = r.get_json()["illustration_token"]

        state = progress_manager.get(illust_token)
        assert state is not None
        assert state["status"] == "done"
        assert state["step"] == "Complete"

    def test_all_images_have_success_status(self, client, mock_llm, monkeypatch):
        import novelforge.config as config
        import novelforge.routes.export as export_module

        monkeypatch.setattr(config, "IMAGE_API_KEY", "key")
        monkeypatch.setattr(export_module, "call_image_api",
                            lambda p, filename_prefix="": f"{filename_prefix}_ok.png")
        _patch_sync_thread(monkeypatch)

        token = "all-success"
        _done_novel(token)

        r = client.post("/generate_illustrations",
                        data=json.dumps({"token": token}),
                        content_type="application/json")
        illust_token = r.get_json()["illustration_token"]

        state = progress_manager.get(illust_token)
        illustrations = state.get("illustrations", [])
        assert len(illustrations) >= 1
        for img in illustrations:
            assert img["status"] == "success"
            assert img["image_url"] is not None
            assert img["error"] is None

    def test_successful_images_mirrored_to_novel_token(
        self, client, mock_llm, monkeypatch
    ):
        """Successful illustrations are copied onto the novel progress entry."""
        import novelforge.config as config
        import novelforge.routes.export as export_module

        monkeypatch.setattr(config, "IMAGE_API_KEY", "key")
        monkeypatch.setattr(export_module, "call_image_api",
                            lambda p, filename_prefix="": f"{filename_prefix}_ok.png")
        _patch_sync_thread(monkeypatch)

        token = "mirror-test"
        _done_novel(token)

        client.post("/generate_illustrations",
                    data=json.dumps({"token": token}),
                    content_type="application/json")

        novel_state = progress_manager.get(token)
        assert novel_state is not None
        mirrored = novel_state.get("illustrations", [])
        assert len(mirrored) >= 1
        for img in mirrored:
            assert img["status"] == "success"

    def test_progress_pollable_after_completion(self, client, mock_llm, monkeypatch):
        """Illustration job token is reachable via /progress/<token>."""
        import novelforge.config as config
        import novelforge.routes.export as export_module

        monkeypatch.setattr(config, "IMAGE_API_KEY", "key")
        monkeypatch.setattr(export_module, "call_image_api",
                            lambda p, filename_prefix="": "file.png")
        _patch_sync_thread(monkeypatch)

        token = "poll-test"
        _done_novel(token)

        r = client.post("/generate_illustrations",
                        data=json.dumps({"token": token}),
                        content_type="application/json")
        illust_token = r.get_json()["illustration_token"]

        # /progress/<token> must return the lightweight status.
        r2 = client.get(f"/progress/{illust_token}")
        assert r2.status_code == 200
        data = r2.get_json()
        assert "status" in data
        assert data["status"] == "done"

        # /progress/<token>/full must return full payload including illustrations.
        r3 = client.get(f"/progress/{illust_token}/full")
        assert r3.status_code == 200
        full = r3.get_json()
        assert "illustrations" in full


# ---------------------------------------------------------------------------
# Background worker: LLM retry flow
# ---------------------------------------------------------------------------

class TestIllustrationJobRetry:
    """LLM prompt-generation retries inside the worker, not the request thread."""

    def test_retry_succeeds_on_second_attempt(self, client, monkeypatch):
        """Worker retries the LLM call and succeeds on the second attempt."""
        import novelforge.config as config
        import novelforge.routes.export as export_module

        monkeypatch.setattr(config, "IMAGE_API_KEY", "key")
        monkeypatch.setattr(export_module, "call_image_api",
                            lambda p, filename_prefix="": "img.png")
        _patch_sync_thread(monkeypatch)

        # Suppress actual sleep so the test runs quickly.
        monkeypatch.setattr(export_module.time, "sleep", lambda s: None)

        call_count = {"n": 0}
        good_response = json.dumps({
            "illustrations": [
                {
                    "type": "cover",
                    "chapter": None,
                    "scene_description": "A brave hero",
                    "art_prompt": "Hero at the gates",
                }
            ]
        })

        def flaky_llm(messages, *, action="", json_mode=False):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("Temporary LLM failure")
            return good_response

        monkeypatch.setattr(export_module, "call_llm", flaky_llm)

        token = "retry-novel"
        _done_novel(token)

        r = client.post("/generate_illustrations",
                        data=json.dumps({"token": token}),
                        content_type="application/json")
        illust_token = r.get_json()["illustration_token"]

        state = progress_manager.get(illust_token)
        assert state is not None
        assert state["status"] == "done"
        assert call_count["n"] == 2  # failed once, succeeded once

    def test_all_retries_exhausted_sets_error_status(self, client, monkeypatch):
        """Worker marks job as error when all three LLM attempts fail."""
        import novelforge.config as config
        import novelforge.routes.export as export_module

        monkeypatch.setattr(config, "IMAGE_API_KEY", "key")
        _patch_sync_thread(monkeypatch)
        monkeypatch.setattr(export_module.time, "sleep", lambda s: None)
        monkeypatch.setattr(export_module, "call_llm",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                RuntimeError("LLM always fails")))

        token = "retry-exhausted"
        _done_novel(token)

        r = client.post("/generate_illustrations",
                        data=json.dumps({"token": token}),
                        content_type="application/json")
        illust_token = r.get_json()["illustration_token"]

        state = progress_manager.get(illust_token)
        assert state is not None
        assert state["status"] == "error"
        assert state["error"]

    def test_request_does_not_sleep(self, client, monkeypatch):
        """Confirm time.sleep is never called inside the request handler."""
        import novelforge.config as config
        import novelforge.routes.export as export_module

        monkeypatch.setattr(config, "IMAGE_API_KEY", "key")
        # Prevent the background thread from running during timing check.
        monkeypatch.setattr(export_module.threading, "Thread",
                            lambda *a, **kw: type("T", (), {"start": lambda s: None, "daemon": True})())

        sleep_called = {"n": 0}
        real_sleep = export_module.time.sleep

        def tracking_sleep(s):
            sleep_called["n"] += 1
            real_sleep(s)

        monkeypatch.setattr(export_module.time, "sleep", tracking_sleep)

        token = "no-sleep-route"
        _done_novel(token)

        client.post("/generate_illustrations",
                    data=json.dumps({"token": token}),
                    content_type="application/json")

        assert sleep_called["n"] == 0, (
            "time.sleep must not be called inside the request handler"
        )


# ---------------------------------------------------------------------------
# Background worker: partial-failure flow
# ---------------------------------------------------------------------------

class TestIllustrationJobPartialFailure:
    """When some images fail and at least one succeeds, job status is 'done'."""

    def test_partial_success_job_is_done(self, client, mock_llm, monkeypatch):
        """One image fails, one succeeds → status 'done', not 'error'."""
        import novelforge.config as config
        import novelforge.routes.export as export_module
        import json as _json

        monkeypatch.setattr(config, "IMAGE_API_KEY", "key")
        _patch_sync_thread(monkeypatch)

        # Return two illustration specs.
        two_specs = _json.dumps({
            "illustrations": [
                {
                    "type": "cover",
                    "chapter": None,
                    "scene_description": "Cover",
                    "art_prompt": "Epic cover art",
                },
                {
                    "type": "chapter_scene",
                    "chapter": 1,
                    "scene_description": "Scene",
                    "art_prompt": "Scene art",
                },
            ]
        })
        monkeypatch.setattr(export_module, "call_llm",
                            lambda *a, **kw: two_specs)

        call_n = {"n": 0}

        def partial_image_api(prompt, filename_prefix=""):
            call_n["n"] += 1
            if call_n["n"] == 1:
                return None  # first image fails
            return f"{filename_prefix}_ok.png"

        monkeypatch.setattr(export_module, "call_image_api", partial_image_api)

        token = "partial-fail"
        _done_novel(token)

        r = client.post("/generate_illustrations",
                        data=json.dumps({"token": token}),
                        content_type="application/json")
        illust_token = r.get_json()["illustration_token"]

        state = progress_manager.get(illust_token)
        assert state is not None
        assert state["status"] == "done"

        illustrations = state.get("illustrations", [])
        statuses = [img["status"] for img in illustrations]
        assert "image_failed" in statuses
        assert "success" in statuses

    def test_all_images_fail_sets_error_status(self, client, mock_llm, monkeypatch):
        """All images fail → job status is 'error' with error message."""
        import novelforge.config as config
        import novelforge.routes.export as export_module

        monkeypatch.setattr(config, "IMAGE_API_KEY", "key")
        _patch_sync_thread(monkeypatch)

        monkeypatch.setattr(export_module, "call_image_api",
                            lambda p, filename_prefix="": None)

        token = "all-fail"
        _done_novel(token)

        r = client.post("/generate_illustrations",
                        data=json.dumps({"token": token}),
                        content_type="application/json")
        illust_token = r.get_json()["illustration_token"]

        state = progress_manager.get(illust_token)
        assert state is not None
        assert state["status"] == "error"
        assert "Image generation failed" in (state.get("error") or "")

        # All individual entries should reflect the failure.
        for img in state.get("illustrations", []):
            assert img["status"] == "image_failed"

    def test_per_image_error_field_populated_on_failure(
        self, client, mock_llm, monkeypatch
    ):
        """Each failed image entry has a non-empty 'error' field."""
        import novelforge.config as config
        import novelforge.routes.export as export_module

        monkeypatch.setattr(config, "IMAGE_API_KEY", "key")
        _patch_sync_thread(monkeypatch)
        monkeypatch.setattr(export_module, "call_image_api",
                            lambda p, filename_prefix="": None)

        token = "per-img-err"
        _done_novel(token)

        r = client.post("/generate_illustrations",
                        data=json.dumps({"token": token}),
                        content_type="application/json")
        illust_token = r.get_json()["illustration_token"]

        state = progress_manager.get(illust_token)
        for img in state.get("illustrations", []):
            if img["status"] == "image_failed":
                assert img["error"], "Failed image must have a non-empty error message"
