"""
Concurrency and threading tests for NovelForge.

Tests progress store thread safety, correlation ID isolation,
concurrent generation requests, and lock correctness.
"""

import json
import threading
import time
import pytest

from novelforge.progress import (
    progress_manager,
    set_correlation_token, get_correlation_token, clear_correlation_token,
)


def _make_chapters(n: int) -> list[dict]:
    return [
        {
            "number": i + 1,
            "title": f"Chapter {i + 1}",
            "content": f"Content for chapter {i + 1}.",
            "summary": f"Summary {i + 1}.",
            "word_count": 3000 + i * 100,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Stable UUIDs used by lock-identity tests (must be valid UUID format)
# ---------------------------------------------------------------------------
_LOCK_UUID_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
_LOCK_UUID_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
_LOCK_UUID_C = "cccccccc-cccc-cccc-cccc-cccccccccccc"

class TestProgressStoreThreadSafety:
    """Verify ProgressManager handles concurrent reads and writes correctly."""

    def setup_method(self):
        """Clean progress store before each test."""
        progress_manager.clear()

    def teardown_method(self):
        progress_manager.clear()

    def test_concurrent_writes_no_data_loss(self):
        """Multiple threads writing different tokens should not lose entries."""
        num_threads = 20
        barrier = threading.Barrier(num_threads)
        errors = []

        def writer(idx):
            try:
                barrier.wait(timeout=5)
                token = f"token-{idx}"
                progress_manager.create(token, {
                    "status": "running",
                    "current": 0,
                    "total": 10,
                    "step": "",
                    "chapters_done": [],
                    "error": None,
                })
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert len(progress_manager.keys()) == num_threads
        for i in range(num_threads):
            assert progress_manager.get(f"token-{i}") is not None

    def test_concurrent_read_write(self):
        """Readers and writers accessing the store concurrently should not crash."""
        token = "shared-token"
        progress_manager.create(token, {
            "status": "running", "current": 0, "total": 50,
            "step": "", "chapters_done": [], "error": None,
        })

        errors = []
        stop = threading.Event()

        def writer():
            try:
                for i in range(50):
                    progress_manager.update(token, {"current": i + 1, "step": f"Step {i + 1}"})
                    time.sleep(0.001)
                stop.set()
            except Exception as e:
                errors.append(("writer", e))
                stop.set()

        def reader():
            try:
                while not stop.is_set():
                    data = progress_manager.get(token) or {}
                    # Verify data consistency — current should be an int
                    if "current" in data:
                        assert isinstance(data["current"], int)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(("reader", e))

        writer_t = threading.Thread(target=writer)
        readers = [threading.Thread(target=reader) for _ in range(5)]

        for r in readers:
            r.start()
        writer_t.start()

        writer_t.join(timeout=10)
        for r in readers:
            r.join(timeout=10)

        assert not errors
        assert progress_manager.get(token)["current"] == 50

    def test_concurrent_chapter_append(self):
        """Simulates multiple chapters being appended from a background thread."""
        token = "append-test"
        progress_manager.create(token, {
            "status": "running", "current": 0, "total": 10,
            "step": "", "chapters_done": [], "error": None,
        })

        def generate():
            local_chapters: list[dict] = []
            for i in range(10):
                chapter = {"number": i + 1, "title": f"Ch{i+1}", "content": f"Text {i+1}"}
                local_chapters.append(chapter)
                progress_manager.update(token, {
                    "chapters_done": list(local_chapters),
                    "current": i + 1,
                })
                time.sleep(0.002)
            progress_manager.update(token, {"status": "done"})

        poll_results = []

        def poller():
            while True:
                data = progress_manager.get(token) or {}
                done = list(data.get("chapters_done", []))
                poll_results.append(len(done))
                if data.get("status") == "done":
                    break
                time.sleep(0.005)

        gen_t = threading.Thread(target=generate)
        poll_t = threading.Thread(target=poller)

        gen_t.start()
        poll_t.start()
        gen_t.join(timeout=10)
        poll_t.join(timeout=10)

        final = progress_manager.get(token)
        assert final["status"] == "done"
        assert len(final["chapters_done"]) == 10

        # Poll results should show monotonically increasing chapter counts
        for i in range(1, len(poll_results)):
            assert poll_results[i] >= poll_results[i - 1], \
                f"Chapter count went backwards: {poll_results[i-1]} -> {poll_results[i]}"

    def test_two_tokens_isolated(self):
        """Two generation tokens should not interfere with each other."""
        for tok in ("novel-a", "novel-b"):
            progress_manager.create(tok, {
                "status": "running", "current": 0, "total": 10,
                "step": "", "chapters_done": [], "error": None,
            })

        def gen(token, count):
            local_chapters: list[dict] = []
            for i in range(count):
                local_chapters.append({"number": i + 1})
                progress_manager.update(token, {
                    "chapters_done": list(local_chapters),
                    "current": i + 1,
                })
                time.sleep(0.001)
            progress_manager.update(token, {"status": "done"})

        t_a = threading.Thread(target=gen, args=("novel-a", 5))
        t_b = threading.Thread(target=gen, args=("novel-b", 8))
        t_a.start()
        t_b.start()
        t_a.join(timeout=10)
        t_b.join(timeout=10)

        data_a = progress_manager.get("novel-a")
        data_b = progress_manager.get("novel-b")
        assert len(data_a["chapters_done"]) == 5
        assert len(data_b["chapters_done"]) == 8
        assert data_a["status"] == "done"
        assert data_b["status"] == "done"


class TestCorrelationIDIsolation:
    """Verify thread-local correlation tokens don't leak between threads."""

    def test_tokens_isolated_across_threads(self):
        """Each thread should see only its own correlation token."""
        results = {}
        barrier = threading.Barrier(3)

        def worker(name, token):
            set_correlation_token(token)
            barrier.wait(timeout=5)
            # After all threads set their token, read back
            time.sleep(0.01)
            results[name] = get_correlation_token()
            clear_correlation_token()

        threads = [
            threading.Thread(target=worker, args=("a", "token-aaa")),
            threading.Thread(target=worker, args=("b", "token-bbb")),
            threading.Thread(target=worker, args=("c", "token-ccc")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert results["a"] == "token-aaa"
        assert results["b"] == "token-bbb"
        assert results["c"] == "token-ccc"

    def test_main_thread_unaffected(self):
        """Setting a correlation token in a child thread should not affect the main thread."""
        clear_correlation_token()
        assert get_correlation_token() == ""

        def child():
            set_correlation_token("child-token")
            time.sleep(0.01)

        t = threading.Thread(target=child)
        t.start()
        t.join(timeout=5)

        assert get_correlation_token() == ""

    def test_clear_only_affects_current_thread(self):
        """Clearing in one thread should not clear another thread's token."""
        results = {}
        ready = threading.Event()
        cleared = threading.Event()

        def thread_a():
            set_correlation_token("a-token")
            ready.set()
            cleared.wait(timeout=5)
            results["a"] = get_correlation_token()

        def thread_b():
            set_correlation_token("b-token")
            ready.wait(timeout=5)
            clear_correlation_token()
            cleared.set()
            results["b"] = get_correlation_token()

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start()
        tb.start()
        ta.join(timeout=10)
        tb.join(timeout=10)

        assert results["a"] == "a-token"  # Unaffected by thread_b's clear
        assert results["b"] == ""          # Cleared


class TestConcurrentGenerationRequests:
    """Test behavior when the generation endpoint is hit concurrently."""

    def setup_method(self):
        progress_manager.clear()

    def teardown_method(self):
        progress_manager.clear()

    def _setup_session_data(self, client):
        with client.session_transaction() as sess:
            sess["premise"] = "A test"
            sess["genre"] = "Fantasy"
            sess["chapters"] = 3
            sess["word_count"] = 10000
            sess["title"] = "Test"
            sess["chapter_list"] = [
                {"number": 1, "title": "Ch1", "summary": "S1"},
                {"number": 2, "title": "Ch2", "summary": "S2"},
                {"number": 3, "title": "Ch3", "summary": "S3"},
            ]
            sess["character_list"] = []
            sess["special_instructions"] = ""
            sess["story_architecture"] = {}
            sess["master_timeline"] = {}
            sess["character_fate_registry"] = {}
            sess["character_arc_plan"] = {}
            sess["antagonist_motivation_plan"] = {}
            sess["technology_rules"] = {}
            sess["theme_reinforcement"] = {}
            sess["pov_focal_character_plan"] = {}

    def test_first_generation_request_succeeds(self, client, monkeypatch):
        """First POST /generate_chapters should return 200 with a token."""
        import novelforge.routes.generation as gen_mod
        monkeypatch.setattr(gen_mod.threading, "Thread",
                            lambda *a, **kw: type("FakeThread", (), {"start": lambda s: None, "daemon": True})())

        self._setup_session_data(client)
        r = client.post("/generate_chapters", data=json.dumps({}),
                        content_type="application/json")
        assert r.status_code == 200
        token = r.get_json()["token"]
        assert token
        data = progress_manager.get(token)
        assert data is not None
        assert data["status"] == "running"

    def test_duplicate_generation_request_blocked_with_409(self, client, monkeypatch):
        """Second POST /generate_chapters while first is still running returns 409."""
        import novelforge.routes.generation as gen_mod
        monkeypatch.setattr(gen_mod.threading, "Thread",
                            lambda *a, **kw: type("FakeThread", (), {"start": lambda s: None, "daemon": True})())

        self._setup_session_data(client)
        r1 = client.post("/generate_chapters", data=json.dumps({}),
                         content_type="application/json")
        assert r1.status_code == 200
        token1 = r1.get_json()["token"]

        # Second request from same session while first is still "running"
        r2 = client.post("/generate_chapters", data=json.dumps({}),
                         content_type="application/json")
        assert r2.status_code == 409
        body2 = r2.get_json()
        assert body2["error_code"] == "generation_in_progress"
        # Returns the existing token so the client can attach to it
        assert body2["token"] == token1

        # Only one entry in progress store
        assert progress_manager.get(token1) is not None
        assert len(progress_manager.keys()) == 1

    def test_new_generation_allowed_after_previous_completes(self, client, monkeypatch):
        """POST /generate_chapters is allowed once the previous generation finishes."""
        import novelforge.routes.generation as gen_mod
        monkeypatch.setattr(gen_mod.threading, "Thread",
                            lambda *a, **kw: type("FakeThread", (), {"start": lambda s: None, "daemon": True})())

        self._setup_session_data(client)
        r1 = client.post("/generate_chapters", data=json.dumps({}),
                         content_type="application/json")
        assert r1.status_code == 200
        token1 = r1.get_json()["token"]

        # Simulate the first generation completing
        progress_manager.update(token1, {"status": "done"})

        # Session data persists; no need to re-seed before the second request
        r2 = client.post("/generate_chapters", data=json.dumps({}),
                         content_type="application/json")
        assert r2.status_code == 200
        token2 = r2.get_json()["token"]
        assert token2 != token1

        data2 = progress_manager.get(token2)
        assert data2 is not None
        assert data2["status"] == "running"

    def test_rapid_repeat_calls_all_blocked_after_first(self, client, monkeypatch):
        """Rapid repeated calls from the same session are blocked after the first."""
        import novelforge.routes.generation as gen_mod
        monkeypatch.setattr(gen_mod.threading, "Thread",
                            lambda *a, **kw: type("FakeThread", (), {"start": lambda s: None, "daemon": True})())

        self._setup_session_data(client)
        responses = []
        for _ in range(5):
            r = client.post("/generate_chapters", data=json.dumps({}),
                            content_type="application/json")
            responses.append(r.status_code)

        assert responses[0] == 200
        assert all(s == 409 for s in responses[1:])
        # Only one entry in the progress store
        assert len(progress_manager.keys()) == 1


class TestCircuitBreakerThreadSafety:
    """Verify circuit breaker behaves correctly under concurrent access."""

    def test_concurrent_failures_trip_breaker(self):
        from novelforge.llm.client import LLMCircuitBreaker, CircuitBreakerError

        breaker = LLMCircuitBreaker(threshold=3)
        barrier = threading.Barrier(5)

        def fail_once():
            barrier.wait(timeout=5)
            breaker.record_failure("concurrent error")

        threads = [threading.Thread(target=fail_once) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert breaker.is_tripped
        assert breaker.failure_count >= 3
        with pytest.raises(CircuitBreakerError):
            breaker.check()

    def test_success_resets_during_concurrent_failures(self):
        from novelforge.llm.client import LLMCircuitBreaker

        breaker = LLMCircuitBreaker(threshold=5)
        breaker.record_failure("err1")
        breaker.record_failure("err2")
        assert breaker.failure_count == 2

        # A concurrent success should reset
        def succeed():
            breaker.record_success()

        t = threading.Thread(target=succeed)
        t.start()
        t.join(timeout=5)

        assert breaker.failure_count == 0
        assert not breaker.is_tripped


class TestPersistenceLock:
    """Verify concurrent session persistence writes do not clobber each other.

    These tests cover the interleaved scenarios described in the issue:
    - ``save_session_state()`` (request thread) vs ``persist_completed_chapters()``
      (background thread) operating on the same session file at the same time.
    - Multiple concurrent ``persist_completed_chapters()`` calls.
    """

    def setup_method(self):
        progress_manager.clear()

    def teardown_method(self):
        progress_manager.clear()

    # ------------------------------------------------------------------
    # Helper: write an initial session file without Flask request context
    # ------------------------------------------------------------------

    def _write_initial_file(self, path, session_id: str, extra: dict | None = None) -> None:
        state = {
            "session_id": session_id,
            "title": "Race Test",
            "premise": "Test premise",
            "genre": "Fantasy",
            "chapters": 10,
            "word_count": 80000,
            "special_events": "",
            "special_instructions": "",
            "chapter_list": [],
            "character_list": [],
            "story_architecture": {},
            "master_timeline": {},
            "character_fate_registry": {},
            "character_arc_plan": {},
            "antagonist_motivation_plan": {},
            "technology_rules": {},
            "theme_reinforcement": {},
            "pov_focal_character_plan": {},
            "progress_token": "",
            "completed_chapters": [],
            "illustrations": [],
            "voice_seed": {},
        }
        if extra:
            state.update(extra)
        (path / f"{session_id}.json").write_text(json.dumps(state), encoding="utf-8")

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_concurrent_persist_calls_produce_valid_json(self, app):
        """Many concurrent persist_completed_chapters() calls must not corrupt the file."""
        import novelforge.config as config
        from pathlib import Path
        from novelforge.session.persistence import (
            save_session_state, persist_completed_chapters, get_session_id,
        )

        sessions_dir = Path(config.NOVELS_DIR)

        with app.test_request_context():
            import flask
            sess = flask.session
            sess["title"] = "Concurrent Test"
            sess["premise"] = "Test"
            sess["genre"] = "Fantasy"
            sess["chapters"] = 20
            sess["word_count"] = 80000
            sess["special_instructions"] = ""
            sess["special_events"] = ""
            sess["chapter_list"] = []
            sess["character_list"] = []
            save_session_state()
            session_id = get_session_id()

        session_file = sessions_dir / f"{session_id}.json"
        assert session_file.exists()

        n = 20
        errors: list[Exception] = []
        barrier = threading.Barrier(n)

        def persist_worker(i: int) -> None:
            try:
                barrier.wait(timeout=5)
                persist_completed_chapters(session_id, _make_chapters(i + 1))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=persist_worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"

        # File must be valid JSON with a non-empty completed_chapters list
        state = json.loads(session_file.read_text(encoding="utf-8"))
        assert isinstance(state["completed_chapters"], list)
        assert len(state["completed_chapters"]) > 0

    def test_save_and_persist_interleave_no_data_loss(self, app, monkeypatch):
        """save_session_state and persist_completed_chapters serialise correctly.

        The test injects a deliberate delay *inside* the persist read-modify-write
        to force the two writers to overlap in time.  With the lock in place the
        final file must be valid JSON and must never show zero chapters once
        persist_completed_chapters has written at least one.
        """
        import novelforge.config as config
        from pathlib import Path
        import novelforge.session.persistence as persistence_mod
        from novelforge.session.persistence import (
            save_session_state, persist_completed_chapters, get_session_id,
        )

        sessions_dir = Path(config.NOVELS_DIR)

        # Create initial session
        with app.test_request_context():
            import flask
            sess = flask.session
            sess["title"] = "Interleave Test"
            sess["premise"] = "Test"
            sess["genre"] = "Fantasy"
            sess["chapters"] = 5
            sess["word_count"] = 50000
            sess["special_instructions"] = ""
            sess["special_events"] = ""
            sess["chapter_list"] = []
            sess["character_list"] = []
            save_session_state()
            session_id = get_session_id()

        session_file = sessions_dir / f"{session_id}.json"

        # Inject a sleep between the file-read and _atomic_write inside
        # persist_completed_chapters so that save_session_state can race it.
        original_atomic_write = persistence_mod._atomic_write
        write_call_count = [0]

        def slow_atomic_write(filepath: Path, content: str) -> None:
            write_call_count[0] += 1
            # First write comes from persist_completed_chapters (background);
            # introduce a 50 ms delay to hold the lock long enough for the
            # concurrent save_session_state() call to queue behind it.
            if write_call_count[0] == 1:
                time.sleep(0.05)
            original_atomic_write(filepath, content)

        monkeypatch.setattr(persistence_mod, "_atomic_write", slow_atomic_write)

        errors: list[Exception] = []
        persist_done = threading.Event()

        def bg_persist() -> None:
            try:
                persist_completed_chapters(session_id, _make_chapters(5))
            except Exception as exc:
                errors.append(exc)
            finally:
                persist_done.set()

        bg = threading.Thread(target=bg_persist)
        bg.start()

        # Allow 10 ms for the background thread to acquire the lock and start
        # its file read before the request thread attempts its write.
        time.sleep(0.01)

        with app.test_request_context():
            import flask
            sess = flask.session
            sess["session_id"] = session_id
            sess["title"] = "Interleave Test"
            sess["premise"] = "Test"
            sess["genre"] = "Fantasy"
            sess["chapters"] = 5
            sess["word_count"] = 50000
            sess["special_instructions"] = ""
            sess["special_events"] = ""
            sess["chapter_list"] = []
            sess["character_list"] = []
            save_session_state()

        persist_done.wait(timeout=10)
        bg.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"

        # File must be valid JSON
        state = json.loads(session_file.read_text(encoding="utf-8"))
        assert state["title"] == "Interleave Test"
        assert isinstance(state["completed_chapters"], list)

    def test_get_session_lock_returns_same_object_for_same_id(self):
        """_get_session_lock must return the identical Lock for the same session_id."""
        from novelforge.session.persistence import _get_session_lock

        lock1 = _get_session_lock(_LOCK_UUID_A)
        lock2 = _get_session_lock(_LOCK_UUID_A)
        assert lock1 is lock2

    def test_get_session_lock_different_sessions_get_different_locks(self):
        """Different session IDs must not share a lock."""
        from novelforge.session.persistence import _get_session_lock

        assert _get_session_lock(_LOCK_UUID_A) is not _get_session_lock(_LOCK_UUID_B)

    def test_lock_registry_thread_safe(self):
        """Concurrent first-access calls for the same session_id yield the same lock."""
        from novelforge.session.persistence import _get_session_lock

        n = 30
        results: list[object] = []
        barrier = threading.Barrier(n)

        def fetch() -> None:
            barrier.wait(timeout=5)
            results.append(_get_session_lock(_LOCK_UUID_C))

        threads = [threading.Thread(target=fetch) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(results) == n
        # All threads must have received the same lock object
        assert len(set(id(lk) for lk in results)) == 1

    def test_concurrent_persist_and_save_file_always_valid_json(self, app):
        """Concurrent save_session_state + persist_completed_chapters never corrupt the file."""
        import novelforge.config as config
        from pathlib import Path
        from novelforge.session.persistence import (
            save_session_state, persist_completed_chapters, get_session_id,
        )

        sessions_dir = Path(config.NOVELS_DIR)

        with app.test_request_context():
            import flask
            sess = flask.session
            sess["title"] = "JSON Integrity Test"
            sess["premise"] = "Test"
            sess["genre"] = "Sci-Fi"
            sess["chapters"] = 10
            sess["word_count"] = 70000
            sess["special_instructions"] = ""
            sess["special_events"] = ""
            sess["chapter_list"] = []
            sess["character_list"] = []
            save_session_state()
            session_id = get_session_id()

        session_file = sessions_dir / f"{session_id}.json"
        errors: list[Exception] = []
        barrier = threading.Barrier(2)

        def bg_persist() -> None:
            try:
                barrier.wait(timeout=5)
                for i in range(1, 6):
                    persist_completed_chapters(session_id, _make_chapters(i))
                    # 5 ms gap between persists so save_session_state() calls
                    # from the main thread can interleave between them.
                    time.sleep(0.005)
            except Exception as exc:
                errors.append(exc)

        bg = threading.Thread(target=bg_persist)
        bg.start()

        barrier.wait(timeout=5)
        with app.test_request_context():
            import flask
            sess = flask.session
            sess["session_id"] = session_id
            sess["title"] = "JSON Integrity Test"
            sess["premise"] = "Test"
            sess["genre"] = "Sci-Fi"
            sess["chapters"] = 10
            sess["word_count"] = 70000
            sess["special_instructions"] = ""
            sess["special_events"] = ""
            sess["chapter_list"] = []
            sess["character_list"] = []
            for _ in range(5):
                save_session_state()
                # 5 ms gap mirrors the background-thread cadence so writes overlap.
                time.sleep(0.005)

        bg.join(timeout=10)
        assert not errors, f"Thread errors: {errors}"

        # File must be parseable and have a valid schema
        raw = session_file.read_text(encoding="utf-8")
        state = json.loads(raw)
        assert "completed_chapters" in state
        assert isinstance(state["completed_chapters"], list)
        assert "title" in state

