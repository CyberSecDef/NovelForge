"""Tests for novelforge.config.ensure_app_dirs()."""

import os
import stat
from pathlib import Path

import pytest


class TestEnsureAppDirs:
    """Validate that ensure_app_dirs() creates directories and checks writability."""

    def test_creates_all_required_directories(self, tmp_path, monkeypatch):
        import novelforge.config as cfg

        monkeypatch.setattr(cfg, "SESSION_FILE_DIR", str(tmp_path / "sessions/flask"))
        monkeypatch.setattr(cfg, "EXPORT_DIR", str(tmp_path / "exports"))
        monkeypatch.setattr(cfg, "LOGS_DIR", str(tmp_path / "logs"))
        monkeypatch.setattr(cfg, "NOVELS_DIR", str(tmp_path / "sessions/novels"))

        cfg.ensure_app_dirs()

        assert (tmp_path / "sessions" / "flask").is_dir()
        assert (tmp_path / "exports").is_dir()
        assert (tmp_path / "exports" / "illustrations").is_dir()
        assert (tmp_path / "logs").is_dir()
        assert (tmp_path / "sessions" / "novels").is_dir()

    def test_idempotent_when_dirs_already_exist(self, tmp_path, monkeypatch):
        import novelforge.config as cfg

        monkeypatch.setattr(cfg, "SESSION_FILE_DIR", str(tmp_path / "sessions/flask"))
        monkeypatch.setattr(cfg, "EXPORT_DIR", str(tmp_path / "exports"))
        monkeypatch.setattr(cfg, "LOGS_DIR", str(tmp_path / "logs"))
        monkeypatch.setattr(cfg, "NOVELS_DIR", str(tmp_path / "sessions/novels"))

        cfg.ensure_app_dirs()
        # Calling a second time must not raise
        cfg.ensure_app_dirs()

        assert (tmp_path / "logs").is_dir()

    def test_raises_permission_error_on_non_writeable_dir(self, tmp_path, monkeypatch):
        import novelforge.config as cfg

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True)
        # Remove write permission from the directory
        logs_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)

        monkeypatch.setattr(cfg, "SESSION_FILE_DIR", str(tmp_path / "sessions/flask"))
        monkeypatch.setattr(cfg, "EXPORT_DIR", str(tmp_path / "exports"))
        monkeypatch.setattr(cfg, "LOGS_DIR", str(logs_dir))
        monkeypatch.setattr(cfg, "NOVELS_DIR", str(tmp_path / "sessions/novels"))

        try:
            with pytest.raises(PermissionError, match="not writeable"):
                cfg.ensure_app_dirs()
        finally:
            # Restore permissions so tmp_path cleanup works
            logs_dir.chmod(stat.S_IRWXU)

    def test_creates_nested_dirs_with_parents(self, tmp_path, monkeypatch):
        import novelforge.config as cfg

        # Use deeply nested paths that don't exist yet
        monkeypatch.setattr(cfg, "SESSION_FILE_DIR", str(tmp_path / "a/b/c/sessions"))
        monkeypatch.setattr(cfg, "EXPORT_DIR", str(tmp_path / "x/y/exports"))
        monkeypatch.setattr(cfg, "LOGS_DIR", str(tmp_path / "p/q/logs"))
        monkeypatch.setattr(cfg, "NOVELS_DIR", str(tmp_path / "r/s/novels"))

        cfg.ensure_app_dirs()

        assert (tmp_path / "a" / "b" / "c" / "sessions").is_dir()
        assert (tmp_path / "x" / "y" / "exports").is_dir()
        assert (tmp_path / "x" / "y" / "exports" / "illustrations").is_dir()
        assert (tmp_path / "p" / "q" / "logs").is_dir()
        assert (tmp_path / "r" / "s" / "novels").is_dir()
