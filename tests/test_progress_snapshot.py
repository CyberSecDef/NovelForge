"""
Tests for progress snapshot persistence in the generation worker.

Covers:
- Snapshots are written atomically (mkstemp + os.replace, no partial files).
- In-memory step updates are not throttled; only disk I/O is throttled.
- Disk writes are skipped within _PROGRESS_PERSIST_INTERVAL unless forced.
- Chapter completion always triggers a forced persist regardless of interval.
- Terminal states (success and every error variant) always trigger a forced persist.
- The progress file contains the expected JSON structure.
- No leftover .tmp files after a successful or failed write.
"""

import json
import os
import time

import pytest

import novelforge.config as config
from novelforge.progress import progress_manager
from novelforge.routes.generation import _PROGRESS_PERSIST_INTERVAL


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _seed_snap(session_id: str | None = None, chapters: int = 1) -> dict:
    snap: dict = {
        "title": "Test Novel",
        "genre": "Fantasy",
        "chapters": chapters,
        "word_count": 3000,
        "special_instructions": "",
        "premise": "A hero sets out on a quest",
        "chapter_list": [
            {"number": i + 1, "title": f"Ch{i + 1}", "summary": "Setup"}
            for i in range(chapters)
        ],
        "character_list": [],
        "story_architecture": {},
        "master_timeline": {},
        "character_fate_registry": {},
        "character_arc_plan": {},
        "antagonist_motivation_plan": {},
        "technology_rules": {},
        "theme_reinforcement": {},
        "pov_focal_character_plan": {},
        "narrative_perspective": "third_person",
        "voice_seed": {},
    }
    if session_id:
        snap["session_id"] = session_id
    return snap


def _create_progress(token: str, total: int = 1) -> None:
    progress_manager.create(token, {
        "status": "running",
        "current": 0,
        "total": total,
        "step": "Preparing\u2026",
        "chapters_done": [],
        "error": None,
    })


# ---------------------------------------------------------------------------
# Fixture: clean progress store and isolated NOVELS_DIR
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_store():
    progress_manager.clear()
    yield
    progress_manager.clear()


# ---------------------------------------------------------------------------
# Atomic write and file-content tests
# ---------------------------------------------------------------------------

class TestProgressSnapshotAtomicity:
    """Progress snapshots use the atomic mkstemp+rename pattern."""

    def test_snapshot_file_is_valid_json(self, mock_llm, monkeypatch, tmp_path):
        """After a full generation run, the progress file is valid JSON."""
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))
        from novelforge.routes.generation import _run_chapter_generation_internal

        token = "test-atomic-json"
        _create_progress(token)
        snap = _seed_snap()

        _run_chapter_generation_internal(token, snap, [], [], 0)

        save_file = tmp_path / f"{token}_progress.json"
        assert save_file.exists(), "Progress file must exist after generation"
        data = json.loads(save_file.read_text(encoding="utf-8"))
        assert data["token"] == token
        assert "snapshot" in data
        assert "chapters_done" in data
        assert "summaries" in data
        assert "progress" in data

    def test_no_leftover_tmp_files(self, mock_llm, monkeypatch, tmp_path):
        """No .tmp files remain in NOVELS_DIR after a successful generation run."""
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))
        from novelforge.routes.generation import _run_chapter_generation_internal

        token = "test-no-tmp"
        _create_progress(token)
        snap = _seed_snap()

        _run_chapter_generation_internal(token, snap, [], [], 0)

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Leftover .tmp files found: {tmp_files}"

    def test_snapshot_contains_completed_chapters(self, mock_llm, monkeypatch, tmp_path):
        """The progress file written at chapter completion includes chapter content."""
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))
        from novelforge.routes.generation import _run_chapter_generation_internal

        token = "test-chapters-in-snap"
        _create_progress(token)
        snap = _seed_snap(chapters=1)

        _run_chapter_generation_internal(token, snap, [], [], 0)

        save_file = tmp_path / f"{token}_progress.json"
        data = json.loads(save_file.read_text(encoding="utf-8"))
        assert len(data["chapters_done"]) == 1
        assert data["chapters_done"][0]["number"] == 1


# ---------------------------------------------------------------------------
# Throttle tests
# ---------------------------------------------------------------------------

class TestProgressSnapshotThrottling:
    """Disk writes are throttled; forced writes bypass the interval."""

    def test_set_step_updates_in_memory_always(self, mock_llm, monkeypatch, tmp_path):
        """_set_step updates the in-memory progress entry on every call."""
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))
        from novelforge.routes.generation import _run_chapter_generation_internal

        token = "test-in-memory-always"
        _create_progress(token)
        snap = _seed_snap()

        step_labels: list[str] = []
        original_update = type(progress_manager).update

        def spy_update(self_pm, tok, patch):
            original_update(self_pm, tok, patch)
            if tok == token and "step" in patch:
                step_labels.append(patch["step"])

        monkeypatch.setattr(type(progress_manager), "update", spy_update)
        _run_chapter_generation_internal(token, snap, [], [], 0)

        # Multiple step transitions must have been recorded in memory
        assert len(step_labels) > 5, (
            f"Expected many in-memory step updates; got {len(step_labels)}"
        )

    def test_fewer_disk_writes_than_step_updates(self, mock_llm, monkeypatch, tmp_path):
        """The number of disk writes must be less than the number of step updates."""
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))
        from novelforge.routes.generation import _run_chapter_generation_internal

        token = "test-throttle-disk"
        _create_progress(token)
        snap = _seed_snap()

        step_labels: list[str] = []
        original_update = type(progress_manager).update

        def spy_update(self_pm, tok, patch):
            original_update(self_pm, tok, patch)
            if tok == token and "step" in patch:
                step_labels.append(patch["step"])

        monkeypatch.setattr(type(progress_manager), "update", spy_update)

        # Patch os.replace to count actual file renames (i.e. completed writes)
        rename_count: list[int] = [0]
        real_replace = os.replace

        def counting_replace(src, dst):
            if token in str(dst):
                rename_count[0] += 1
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", counting_replace)

        _run_chapter_generation_internal(token, snap, [], [], 0)

        assert len(step_labels) > 0
        assert rename_count[0] < len(step_labels), (
            f"Expected fewer disk writes ({rename_count[0]}) than "
            f"step updates ({len(step_labels)}); throttle not working"
        )

    def test_persist_interval_constant_is_positive(self):
        """The persist interval must be a positive number."""
        assert isinstance(_PROGRESS_PERSIST_INTERVAL, float)
        assert _PROGRESS_PERSIST_INTERVAL > 0


# ---------------------------------------------------------------------------
# Force-write on chapter completion
# ---------------------------------------------------------------------------

class TestForceWriteOnChapterCompletion:
    """A progress snapshot is always force-written when a chapter finishes."""

    def test_snapshot_written_at_chapter_completion(self, mock_llm, monkeypatch, tmp_path):
        """After each chapter completes, the progress file must be present and valid."""
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))
        from novelforge.routes.generation import _run_chapter_generation_internal

        token = "test-force-chapter"
        _create_progress(token, total=1)
        snap = _seed_snap(chapters=1)

        completion_snapshots: list[dict] = []
        real_replace = os.replace

        def capture_replace(src, dst):
            if token in str(dst) and str(dst).endswith("_progress.json"):
                with open(src, encoding="utf-8") as fh:
                    try:
                        completion_snapshots.append(json.load(fh))
                    except json.JSONDecodeError:
                        pass
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", capture_replace)
        _run_chapter_generation_internal(token, snap, [], [], 0)

        # At least one write with chapters_done populated must exist
        with_chapters = [s for s in completion_snapshots if s.get("chapters_done")]
        assert len(with_chapters) >= 1, (
            "Expected at least one force-write after chapter completion with chapters_done"
        )


# ---------------------------------------------------------------------------
# Force-write on terminal states (success and all error branches)
# ---------------------------------------------------------------------------

class TestForceWriteOnTerminalStates:
    """Progress snapshot is always force-written on success and error."""

    def _collect_final_snapshot(self, monkeypatch, tmp_path, token: str) -> list[dict]:
        """Return list of snapshots captured from atomic writes for *token*."""
        written: list[dict] = []
        real_replace = os.replace

        def capture_replace(src, dst):
            if token in str(dst) and str(dst).endswith("_progress.json"):
                with open(src, encoding="utf-8") as fh:
                    try:
                        written.append(json.load(fh))
                    except json.JSONDecodeError:
                        pass
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", capture_replace)
        return written

    def test_snapshot_written_on_success(self, mock_llm, monkeypatch, tmp_path):
        """A forced persist occurs after the 'Complete' step."""
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))
        from novelforge.routes.generation import _run_chapter_generation_internal

        token = "test-success-snap"
        _create_progress(token)
        snap = _seed_snap()

        written = self._collect_final_snapshot(monkeypatch, tmp_path, token)
        _run_chapter_generation_internal(token, snap, [], [], 0)

        complete_writes = [w for w in written if w.get("progress", {}).get("step") == "Complete"]
        assert complete_writes, "Expected a force-write with step='Complete'"

    def test_snapshot_written_on_runtime_error(self, mock_llm, mocker, monkeypatch, tmp_path):
        """A forced persist occurs when a RuntimeError terminates the worker."""
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))
        from novelforge.routes.generation import _run_chapter_generation_internal

        token = "test-error-snap"
        _create_progress(token)
        snap = _seed_snap()

        mocker.patch("novelforge.routes.generation.call_llm", side_effect=RuntimeError("boom"))

        written = self._collect_final_snapshot(monkeypatch, tmp_path, token)
        _run_chapter_generation_internal(token, snap, [], [], 0)

        error_writes = [
            w for w in written
            if w.get("progress", {}).get("status") == "error"
        ]
        assert error_writes, "Expected a force-write after a RuntimeError with status='error'"

    def test_snapshot_written_on_content_rejection(self, mock_llm, mocker, monkeypatch, tmp_path):
        """A forced persist occurs for ContentRejectionError."""
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))
        from novelforge.routes.generation import _run_chapter_generation_internal
        import novelforge.routes.generation as gen_mod

        token = "test-content-rejection"
        _create_progress(token)
        snap = _seed_snap()

        # ContentRejectionError has an inner retry loop that exhausts 3 attempts;
        # patching with side_effect ensures all draft calls raise and the outer
        # except ContentRejectionError handler is reached.
        #
        # Use gen_mod.ContentRejectionError (the class bound in generation.py's own
        # namespace) so the raised instance matches what generation.py's except
        # clauses check, regardless of any module-reload in other tests.
        mocker.patch(
            "novelforge.routes.generation.call_llm",
            side_effect=gen_mod.ContentRejectionError("policy"),
        )

        written = self._collect_final_snapshot(monkeypatch, tmp_path, token)
        _run_chapter_generation_internal(token, snap, [], [], 0)

        error_writes = [
            w for w in written
            if w.get("progress", {}).get("status") == "error"
        ]
        assert error_writes, "Expected a force-write after ContentRejectionError"

    def test_snapshot_written_on_circuit_breaker(self, mock_llm, mocker, monkeypatch, tmp_path):
        """A forced persist occurs for CircuitBreakerError."""
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))
        from novelforge.routes.generation import _run_chapter_generation_internal
        from novelforge.llm.client import CircuitBreakerError

        token = "test-circuit-breaker"
        _create_progress(token)
        snap = _seed_snap()

        mocker.patch(
            "novelforge.routes.generation.call_llm",
            side_effect=CircuitBreakerError("cb"),
        )

        written = self._collect_final_snapshot(monkeypatch, tmp_path, token)
        _run_chapter_generation_internal(token, snap, [], [], 0)

        error_writes = [
            w for w in written
            if w.get("progress", {}).get("status") == "error"
        ]
        assert error_writes, "Expected a force-write after CircuitBreakerError"

    def test_snapshot_written_on_all_providers_exhausted(self, mock_llm, mocker, monkeypatch, tmp_path):
        """A forced persist occurs for AllProvidersExhaustedError."""
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))
        from novelforge.routes.generation import _run_chapter_generation_internal
        from novelforge.llm.client import AllProvidersExhaustedError

        token = "test-all-providers"
        _create_progress(token)
        snap = _seed_snap()

        mocker.patch(
            "novelforge.routes.generation.call_llm",
            side_effect=AllProvidersExhaustedError("all"),
        )

        written = self._collect_final_snapshot(monkeypatch, tmp_path, token)
        _run_chapter_generation_internal(token, snap, [], [], 0)

        error_writes = [
            w for w in written
            if w.get("progress", {}).get("status") == "error"
        ]
        assert error_writes, "Expected a force-write after AllProvidersExhaustedError"


# ---------------------------------------------------------------------------
# Crash-recovery: resume from persisted snapshot
# ---------------------------------------------------------------------------

class TestProgressSnapshotCrashRecovery:
    """A persisted snapshot contains enough state to resume generation."""

    def test_snapshot_contains_resume_fields(self, mock_llm, monkeypatch, tmp_path):
        """After generation completes, the snapshot has all fields needed to resume."""
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))
        from novelforge.routes.generation import _run_chapter_generation_internal

        token = "test-resume-fields"
        _create_progress(token, total=1)
        snap = _seed_snap(chapters=1)

        _run_chapter_generation_internal(token, snap, [], [], 0)

        save_file = tmp_path / f"{token}_progress.json"
        data = json.loads(save_file.read_text(encoding="utf-8"))

        for field in ("token", "snapshot", "chapters_done", "summaries", "character_state_log", "progress"):
            assert field in data, (
                f"Missing field {field!r} in progress snapshot. "
                f"Present fields: {list(data.keys())}"
            )

    def test_snapshot_progress_field_has_status(self, mock_llm, monkeypatch, tmp_path):
        """The 'progress' sub-object in the snapshot includes a 'status' field."""
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))
        from novelforge.routes.generation import _run_chapter_generation_internal

        token = "test-progress-status"
        _create_progress(token)
        snap = _seed_snap()

        _run_chapter_generation_internal(token, snap, [], [], 0)

        save_file = tmp_path / f"{token}_progress.json"
        data = json.loads(save_file.read_text(encoding="utf-8"))
        assert "status" in data["progress"], "progress.status must be present in snapshot"
