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
    _progress_store, _progress_lock,
    set_correlation_token, get_correlation_token, clear_correlation_token,
)


class TestProgressStoreThreadSafety:
    """Verify _progress_store handles concurrent reads and writes correctly."""

    def setup_method(self):
        """Clean progress store before each test."""
        with _progress_lock:
            _progress_store.clear()

    def teardown_method(self):
        with _progress_lock:
            _progress_store.clear()

    def test_concurrent_writes_no_data_loss(self):
        """Multiple threads writing different tokens should not lose entries."""
        num_threads = 20
        barrier = threading.Barrier(num_threads)
        errors = []

        def writer(idx):
            try:
                barrier.wait(timeout=5)
                token = f"token-{idx}"
                with _progress_lock:
                    _progress_store[token] = {
                        "status": "running",
                        "current": 0,
                        "total": 10,
                        "thread": idx,
                    }
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        with _progress_lock:
            assert len(_progress_store) == num_threads
            for i in range(num_threads):
                assert f"token-{i}" in _progress_store

    def test_concurrent_read_write(self):
        """Readers and writers accessing the store concurrently should not crash."""
        token = "shared-token"
        with _progress_lock:
            _progress_store[token] = {"status": "running", "current": 0, "total": 50}

        errors = []
        stop = threading.Event()

        def writer():
            try:
                for i in range(50):
                    with _progress_lock:
                        _progress_store[token]["current"] = i + 1
                        _progress_store[token]["step"] = f"Step {i + 1}"
                    time.sleep(0.001)
                stop.set()
            except Exception as e:
                errors.append(("writer", e))
                stop.set()

        def reader():
            try:
                while not stop.is_set():
                    with _progress_lock:
                        data = dict(_progress_store.get(token, {}))
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
        with _progress_lock:
            assert _progress_store[token]["current"] == 50

    def test_concurrent_chapter_append(self):
        """Simulates multiple chapters being appended from a background thread."""
        token = "append-test"
        with _progress_lock:
            _progress_store[token] = {
                "status": "running",
                "current": 0,
                "total": 10,
                "chapters_done": [],
            }

        def generate():
            for i in range(10):
                chapter = {"number": i + 1, "title": f"Ch{i+1}", "content": f"Text {i+1}"}
                with _progress_lock:
                    _progress_store[token]["chapters_done"].append(chapter)
                    _progress_store[token]["current"] = i + 1
                time.sleep(0.002)
            with _progress_lock:
                _progress_store[token]["status"] = "done"

        poll_results = []

        def poller():
            while True:
                with _progress_lock:
                    data = dict(_progress_store.get(token, {}))
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

        with _progress_lock:
            assert _progress_store[token]["status"] == "done"
            assert len(_progress_store[token]["chapters_done"]) == 10

        # Poll results should show monotonically increasing chapter counts
        for i in range(1, len(poll_results)):
            assert poll_results[i] >= poll_results[i - 1], \
                f"Chapter count went backwards: {poll_results[i-1]} -> {poll_results[i]}"

    def test_two_tokens_isolated(self):
        """Two generation tokens should not interfere with each other."""
        with _progress_lock:
            _progress_store["novel-a"] = {"status": "running", "current": 0, "chapters_done": []}
            _progress_store["novel-b"] = {"status": "running", "current": 0, "chapters_done": []}

        def gen(token, count):
            for i in range(count):
                with _progress_lock:
                    _progress_store[token]["chapters_done"].append({"number": i + 1})
                    _progress_store[token]["current"] = i + 1
                time.sleep(0.001)
            with _progress_lock:
                _progress_store[token]["status"] = "done"

        t_a = threading.Thread(target=gen, args=("novel-a", 5))
        t_b = threading.Thread(target=gen, args=("novel-b", 8))
        t_a.start()
        t_b.start()
        t_a.join(timeout=10)
        t_b.join(timeout=10)

        with _progress_lock:
            assert len(_progress_store["novel-a"]["chapters_done"]) == 5
            assert len(_progress_store["novel-b"]["chapters_done"]) == 8
            assert _progress_store["novel-a"]["status"] == "done"
            assert _progress_store["novel-b"]["status"] == "done"


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
        with _progress_lock:
            _progress_store.clear()

    def teardown_method(self):
        with _progress_lock:
            _progress_store.clear()

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
        with _progress_lock:
            assert token in _progress_store
            assert _progress_store[token]["status"] == "running"

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
        with _progress_lock:
            assert token1 in _progress_store
            assert len(_progress_store) == 1

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
        with _progress_lock:
            _progress_store[token1]["status"] = "done"

        # Session data persists; no need to re-seed before the second request
        r2 = client.post("/generate_chapters", data=json.dumps({}),
                         content_type="application/json")
        assert r2.status_code == 200
        token2 = r2.get_json()["token"]
        assert token2 != token1

        with _progress_lock:
            assert token2 in _progress_store
            assert _progress_store[token2]["status"] == "running"

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
        with _progress_lock:
            assert len(_progress_store) == 1


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
