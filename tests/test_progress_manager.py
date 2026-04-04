"""
Tests for the ProgressManager API.

Covers: create/get/update/delete lifecycle, typed ProgressState schema,
state transitions, invalid patch rejection, and concurrent updates.
"""

import threading
import time
import pytest

from novelforge.progress import progress_manager, ProgressState, _VALID_STATUSES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> dict:
    """Return a minimal valid progress state for use in tests."""
    base: dict = {
        "status": "running",
        "current": 0,
        "total": 5,
        "step": "Preparing…",
        "chapters_done": [],
        "error": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fixture: clean store per test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_store():
    """Wipe the progress store before and after every test."""
    progress_manager.clear()
    yield
    progress_manager.clear()


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------

class TestCreateAndGet:
    def test_create_and_get_round_trip(self):
        progress_manager.create("tok1", _base_state())
        data = progress_manager.get("tok1")
        assert data is not None
        assert data["status"] == "running"
        assert data["current"] == 0
        assert data["total"] == 5
        assert data["chapters_done"] == []

    def test_get_unknown_token_returns_none(self):
        assert progress_manager.get("nonexistent") is None

    def test_get_returns_a_copy_not_the_original(self):
        """Mutating the returned dict must not affect the stored state."""
        progress_manager.create("tok2", _base_state())
        copy = progress_manager.get("tok2")
        copy["step"] = "MUTATED"
        # Original must be unchanged
        assert progress_manager.get("tok2")["step"] == "Preparing…"

    def test_create_overwrites_existing_entry(self):
        """create() should overwrite any existing entry for the same token."""
        progress_manager.create("tok3", _base_state(step="first"))
        progress_manager.create("tok3", _base_state(step="second"))
        assert progress_manager.get("tok3")["step"] == "second"

    def test_create_stores_extra_optional_fields(self):
        state = _base_state(snapshot={"title": "My Novel"}, _live=True)
        progress_manager.create("tok4", state)
        data = progress_manager.get("tok4")
        assert data["snapshot"] == {"title": "My Novel"}
        assert data["_live"] is True


class TestUpdate:
    def test_update_single_field(self):
        progress_manager.create("u1", _base_state())
        progress_manager.update("u1", {"step": "Chapter 1: drafting"})
        assert progress_manager.get("u1")["step"] == "Chapter 1: drafting"

    def test_update_multiple_fields_atomically(self):
        progress_manager.create("u2", _base_state())
        progress_manager.update("u2", {"current": 3, "step": "Chapter 3: complete"})
        data = progress_manager.get("u2")
        assert data["current"] == 3
        assert data["step"] == "Chapter 3: complete"

    def test_update_unknown_token_raises_key_error(self):
        with pytest.raises(KeyError, match="nonexistent"):
            progress_manager.update("nonexistent", {"step": "x"})

    def test_update_preserves_unpatched_fields(self):
        progress_manager.create("u3", _base_state(total=10))
        progress_manager.update("u3", {"current": 1})
        data = progress_manager.get("u3")
        assert data["total"] == 10   # untouched
        assert data["current"] == 1  # updated


class TestDelete:
    def test_delete_removes_entry(self):
        progress_manager.create("d1", _base_state())
        progress_manager.delete("d1")
        assert progress_manager.get("d1") is None

    def test_delete_unknown_token_is_no_op(self):
        """delete() must not raise when the token is absent."""
        progress_manager.delete("does-not-exist")


# ---------------------------------------------------------------------------
# Typed schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_create_invalid_status_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid status"):
            progress_manager.create("bad", _base_state(status="pending"))

    def test_create_empty_status_raises_value_error(self):
        with pytest.raises(ValueError):
            progress_manager.create("bad2", _base_state(status=""))

    def test_update_invalid_status_raises_value_error(self):
        progress_manager.create("val1", _base_state())
        with pytest.raises(ValueError, match="Invalid status"):
            progress_manager.update("val1", {"status": "invalid"})

    def test_all_valid_statuses_accepted(self):
        for status in _VALID_STATUSES:
            token = f"status-{status}"
            progress_manager.create(token, _base_state(status=status))
            assert progress_manager.get(token)["status"] == status

    def test_progress_state_typeddict_has_expected_keys(self):
        """Verify the TypedDict schema exposes the documented keys."""
        annotations = ProgressState.__annotations__
        assert "status" in annotations
        assert "current" in annotations
        assert "total" in annotations
        assert "step" in annotations
        assert "chapters_done" in annotations
        assert "error" in annotations


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

class TestStateTransitions:
    def test_running_to_done(self):
        progress_manager.create("t1", _base_state(status="running"))
        progress_manager.update("t1", {"status": "done"})
        assert progress_manager.get("t1")["status"] == "done"

    def test_running_to_error(self):
        progress_manager.create("t2", _base_state(status="running"))
        progress_manager.update("t2", {
            "status": "error",
            "error": "LLM call failed",
            "error_code": "runtime",
        })
        data = progress_manager.get("t2")
        assert data["status"] == "error"
        assert data["error"] == "LLM call failed"
        assert data["error_code"] == "runtime"

    def test_chapter_progress_increments(self):
        progress_manager.create("t3", _base_state(total=3))
        for i in range(1, 4):
            progress_manager.update("t3", {
                "current": i,
                "step": f"Chapter {i}: complete",
                "chapters_done": [{"number": j} for j in range(1, i + 1)],
            })
        data = progress_manager.get("t3")
        assert data["current"] == 3
        assert len(data["chapters_done"]) == 3

    def test_done_state_can_carry_reports(self):
        """Completed generation should allow report fields to be stored."""
        progress_manager.create("t4", _base_state(status="done"))
        progress_manager.update("t4", {
            "consistency": {"issues": [], "overall_assessment": "Good"},
            "global_continuity_audit": {"overall_integrity": "high"},
        })
        data = progress_manager.get("t4")
        assert data["consistency"]["overall_assessment"] == "Good"
        assert data["global_continuity_audit"]["overall_integrity"] == "high"


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

class TestQueryHelpers:
    def test_list_active_returns_only_running_tokens(self):
        progress_manager.create("a", _base_state(status="running"))
        progress_manager.create("b", _base_state(status="done"))
        progress_manager.create("c", _base_state(status="error"))
        active = progress_manager.list_active()
        assert "a" in active
        assert "b" not in active
        assert "c" not in active

    def test_keys_returns_all_tokens(self):
        for i in range(4):
            progress_manager.create(f"k{i}", _base_state())
        assert set(progress_manager.keys()) == {"k0", "k1", "k2", "k3"}

    def test_snapshot_returns_shallow_copies(self):
        progress_manager.create("s1", _base_state(step="init"))
        snap = progress_manager.snapshot()
        snap["s1"]["step"] = "MUTATED"
        # Original in the store must be unchanged
        assert progress_manager.get("s1")["step"] == "init"

    def test_clear_removes_all_entries(self):
        for i in range(3):
            progress_manager.create(f"c{i}", _base_state())
        progress_manager.clear()
        assert progress_manager.keys() == []


# ---------------------------------------------------------------------------
# Concurrent updates (thread safety)
# ---------------------------------------------------------------------------

class TestConcurrentUpdates:
    def test_concurrent_create_no_data_loss(self):
        """20 threads creating different tokens should all succeed."""
        n = 20
        barrier = threading.Barrier(n)
        errors: list = []

        def worker(i: int) -> None:
            try:
                barrier.wait(timeout=5)
                progress_manager.create(f"ct{i}", _base_state())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert len(progress_manager.keys()) == n

    def test_concurrent_updates_to_single_token(self):
        """50 sequential updates from one thread plus 5 concurrent readers."""
        token = "concurrent-token"
        progress_manager.create(token, _base_state(total=50))
        errors: list = []
        stop = threading.Event()

        def writer() -> None:
            try:
                for i in range(50):
                    progress_manager.update(token, {"current": i + 1})
                    time.sleep(0.001)
                stop.set()
            except Exception as exc:
                errors.append(exc)
                stop.set()

        def reader() -> None:
            try:
                while not stop.is_set():
                    data = progress_manager.get(token) or {}
                    if "current" in data:
                        assert isinstance(data["current"], int)
                    time.sleep(0.001)
            except Exception as exc:
                errors.append(exc)

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

    def test_concurrent_update_raises_for_unknown_token(self):
        """Concurrent updates to a missing token should raise KeyError, not corrupt state."""
        errors: list = []

        def try_update(i: int) -> None:
            try:
                progress_manager.update(f"ghost-{i}", {"step": "oops"})
            except KeyError:
                pass  # expected
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=try_update, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert not errors
