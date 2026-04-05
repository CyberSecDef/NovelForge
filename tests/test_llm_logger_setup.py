"""Tests for idempotent LLM logger setup in novelforge.llm.client."""

import importlib
import logging


class TestLlmLoggerNoImportTimeHandlers:
    """Module-level code in novelforge.llm.client must not attach file handlers."""

    def test_import_does_not_attach_file_handler(self):
        """Reloading the module must not attach a FileHandler to llm_requests.

        ``importlib.reload`` re-executes the module-level code without a full
        sys.modules teardown, which is exactly what we need here: we want to
        verify that the module body itself no longer contains any handler-setup
        side effect.  Handlers accumulated by earlier create_app() calls in
        this test session are stripped before the reload so the assertion
        starts from a known clean state.
        """
        import novelforge.llm.client as client_module

        llm_logger = logging.getLogger("llm_requests")
        # Strip any handlers that may have been added by earlier create_app() calls
        # in the test session so we start from a known clean state.
        for h in list(llm_logger.handlers):
            if isinstance(h, logging.FileHandler):
                llm_logger.removeHandler(h)
                h.close()

        # Re-execute the module-level code; this is the action under test.
        importlib.reload(client_module)

        file_handlers = [h for h in llm_logger.handlers if isinstance(h, logging.FileHandler)]
        assert file_handlers == [], (
            "Re-executing novelforge.llm.client module-level code must not attach a FileHandler; "
            f"found {file_handlers!r}"
        )


class TestSetupLlmLoggerIdempotency:
    """setup_llm_logger() must be safe to call multiple times."""

    def _remove_file_handlers(self, logger_name: str) -> list[logging.FileHandler]:
        """Remove and return all FileHandlers on the named logger."""
        lg = logging.getLogger(logger_name)
        removed = [h for h in lg.handlers if isinstance(h, logging.FileHandler)]
        for h in removed:
            lg.removeHandler(h)
            h.close()
        return removed

    def test_setup_attaches_exactly_one_file_handler(self, tmp_path, monkeypatch):
        import novelforge.config as cfg
        from novelforge.llm.client import setup_llm_logger

        monkeypatch.setattr(cfg, "LOGS_DIR", str(tmp_path / "logs"))
        self._remove_file_handlers("llm_requests")

        try:
            setup_llm_logger()

            llm_logger = logging.getLogger("llm_requests")
            file_handlers = [h for h in llm_logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) == 1, (
                f"Expected exactly 1 FileHandler after setup_llm_logger(), got {len(file_handlers)}"
            )
        finally:
            self._remove_file_handlers("llm_requests")

    def test_repeated_calls_do_not_add_duplicate_handlers(self, tmp_path, monkeypatch):
        import novelforge.config as cfg
        from novelforge.llm.client import setup_llm_logger

        monkeypatch.setattr(cfg, "LOGS_DIR", str(tmp_path / "logs"))
        self._remove_file_handlers("llm_requests")

        try:
            setup_llm_logger()
            setup_llm_logger()
            setup_llm_logger()

            llm_logger = logging.getLogger("llm_requests")
            file_handlers = [h for h in llm_logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) == 1, (
                f"Expected exactly 1 FileHandler after 3 setup_llm_logger() calls, "
                f"got {len(file_handlers)}"
            )
        finally:
            self._remove_file_handlers("llm_requests")

    def test_create_app_repeated_does_not_duplicate_handlers(self, tmp_path, monkeypatch):
        """Multiple create_app() calls must not multiply FileHandlers on llm_requests."""
        import novelforge.config as cfg

        monkeypatch.setattr(cfg, "NOVELS_DIR", str(tmp_path / "novels"))
        monkeypatch.setattr(cfg, "LOGS_DIR", str(tmp_path / "logs"))
        (tmp_path / "novels").mkdir()
        self._remove_file_handlers("llm_requests")

        try:
            from novelforge import create_app

            create_app(testing=True)
            create_app(testing=True)
            create_app(testing=True)

            llm_logger = logging.getLogger("llm_requests")
            file_handlers = [h for h in llm_logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) == 1, (
                f"Expected exactly 1 FileHandler after 3 create_app() calls, "
                f"got {len(file_handlers)}"
            )
        finally:
            self._remove_file_handlers("llm_requests")
