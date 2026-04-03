"""Tests for idempotent logging bootstrap in the NovelForge app factory."""

import logging

import pytest

from novelforge.progress import (
    CorrelationFilter,
    CorrelationFormatter,
    set_correlation_token,
    clear_correlation_token,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_bootstrap(monkeypatch):
    """Reset the module-level sentinel and remove any CorrelationFilter from root.

    This gives each test a clean slate without relying on per-test try/finally
    blocks for filter cleanup.
    """
    import novelforge as nf
    monkeypatch.setattr(nf, "_logging_bootstrapped", False)
    root = logging.getLogger()
    for f in list(root.filters):
        if isinstance(f, CorrelationFilter):
            root.removeFilter(f)


# ---------------------------------------------------------------------------
# _bootstrap_logging unit tests
# ---------------------------------------------------------------------------

class TestBootstrapLogging:
    def test_sets_root_level_to_info(self, monkeypatch):
        _reset_bootstrap(monkeypatch)
        from novelforge import _bootstrap_logging

        root = logging.getLogger()
        original_level = root.level
        try:
            _bootstrap_logging()
            assert root.level == logging.INFO
        finally:
            root.setLevel(original_level)

    def test_attaches_correlation_filter_once(self, monkeypatch):
        _reset_bootstrap(monkeypatch)
        from novelforge import _bootstrap_logging

        root = logging.getLogger()
        original_filters = root.filters[:]
        try:
            _bootstrap_logging()
            correlation_filters = [f for f in root.filters if isinstance(f, CorrelationFilter)]
            assert len(correlation_filters) == 1
        finally:
            root.filters = original_filters

    def test_idempotent_does_not_duplicate_correlation_filter(self, monkeypatch):
        _reset_bootstrap(monkeypatch)
        from novelforge import _bootstrap_logging

        root = logging.getLogger()
        original_filters = root.filters[:]
        try:
            _bootstrap_logging()
            _bootstrap_logging()  # second call – should be a no-op
            _bootstrap_logging()  # third call
            correlation_filters = [f for f in root.filters if isinstance(f, CorrelationFilter)]
            assert len(correlation_filters) == 1, (
                f"Expected exactly 1 CorrelationFilter, found {len(correlation_filters)}"
            )
        finally:
            root.filters = original_filters

    def test_sentinel_prevents_re_entry(self, monkeypatch):
        _reset_bootstrap(monkeypatch)
        from novelforge import _bootstrap_logging
        import novelforge as nf

        _bootstrap_logging()
        assert nf._logging_bootstrapped is True

        # Even after a second call the sentinel stays True.
        _bootstrap_logging()
        assert nf._logging_bootstrapped is True

    def test_does_not_add_handler_when_handlers_already_present(self, monkeypatch):
        """_bootstrap_logging must not add a StreamHandler when one exists."""
        _reset_bootstrap(monkeypatch)
        from novelforge import _bootstrap_logging

        root = logging.getLogger()
        original_handlers = root.handlers[:]
        original_filters = root.filters[:]

        # Ensure at least one handler is present before bootstrap.
        test_handler = logging.NullHandler()
        root.addHandler(test_handler)
        try:
            handler_count_before = len(root.handlers)
            _bootstrap_logging()
            assert len(root.handlers) == handler_count_before, (
                "_bootstrap_logging should not add a handler when one already exists"
            )
        finally:
            root.handlers = original_handlers
            root.filters = original_filters


# ---------------------------------------------------------------------------
# Repeated create_app() tests
# ---------------------------------------------------------------------------

class TestRepeatedCreateApp:
    """Verify that calling create_app() multiple times is safe."""

    def test_two_create_app_calls_return_independent_apps(self, tmp_path, monkeypatch):
        import novelforge.config as cfg

        monkeypatch.setattr(cfg, "NOVELS_DIR", str(tmp_path / "novels"))
        (tmp_path / "novels").mkdir()

        from novelforge import create_app

        app1 = create_app(testing=True)
        app2 = create_app(testing=True)

        assert app1 is not app2
        assert app1.config["TESTING"] is True
        assert app2.config["TESTING"] is True

    def test_correlation_filter_not_duplicated_after_repeated_create_app(
        self, tmp_path, monkeypatch
    ):
        import novelforge.config as cfg

        monkeypatch.setattr(cfg, "NOVELS_DIR", str(tmp_path / "novels"))
        (tmp_path / "novels").mkdir()

        # Reset sentinel and remove pre-existing CorrelationFilters.
        _reset_bootstrap(monkeypatch)

        root = logging.getLogger()
        original_filters = root.filters[:]
        try:
            from novelforge import create_app

            create_app(testing=True)
            create_app(testing=True)
            create_app(testing=True)

            correlation_filters = [f for f in root.filters if isinstance(f, CorrelationFilter)]
            assert len(correlation_filters) == 1, (
                f"Expected exactly 1 CorrelationFilter after 3 create_app calls, "
                f"found {len(correlation_filters)}"
            )
        finally:
            root.filters = original_filters

    def test_root_logger_level_stable_after_repeated_create_app(
        self, tmp_path, monkeypatch
    ):
        import novelforge.config as cfg

        monkeypatch.setattr(cfg, "NOVELS_DIR", str(tmp_path / "novels"))
        (tmp_path / "novels").mkdir()

        root = logging.getLogger()
        original_level = root.level
        original_filters = root.filters[:]

        _reset_bootstrap(monkeypatch)

        try:
            from novelforge import create_app

            create_app(testing=True)
            level_after_first = root.level

            create_app(testing=True)
            level_after_second = root.level

            assert level_after_first == logging.INFO
            assert level_after_second == logging.INFO
        finally:
            root.setLevel(original_level)
            root.filters = original_filters


# ---------------------------------------------------------------------------
# CorrelationFilter structured-metadata tests
# ---------------------------------------------------------------------------

def _make_record(msg: str = "test message") -> logging.LogRecord:
    """Return a minimal LogRecord with the given message."""
    return logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )


class TestCorrelationFilterStructuredMetadata:
    """CorrelationFilter must attach token as metadata without mutating record.msg."""

    def setup_method(self):
        clear_correlation_token()

    def teardown_method(self):
        clear_correlation_token()

    def test_msg_not_mutated_when_token_set(self):
        set_correlation_token("tok-001")
        record = _make_record("original message")
        original_msg = record.msg

        cf = CorrelationFilter()
        cf.filter(record)

        assert record.msg == original_msg, (
            "CorrelationFilter must not modify record.msg"
        )

    def test_msg_not_mutated_when_no_token(self):
        record = _make_record("no token here")
        original_msg = record.msg

        cf = CorrelationFilter()
        cf.filter(record)

        assert record.msg == original_msg

    def test_correlation_token_attribute_set_when_token_present(self):
        set_correlation_token("tok-abc")
        record = _make_record()

        CorrelationFilter().filter(record)

        assert record.correlation_token == "tok-abc"  # type: ignore[attr-defined]

    def test_correlation_token_attribute_empty_when_no_token(self):
        record = _make_record()

        CorrelationFilter().filter(record)

        assert record.correlation_token == ""  # type: ignore[attr-defined]

    def test_no_double_prefixing_with_multiple_filter_passes(self):
        """Applying the filter multiple times (e.g. multiple handlers) must not duplicate."""
        set_correlation_token("tok-dup")
        record = _make_record("some log line")
        original_msg = record.msg

        cf = CorrelationFilter()
        cf.filter(record)
        cf.filter(record)
        cf.filter(record)

        assert record.msg == original_msg, (
            "Multiple filter passes must not duplicate content in record.msg"
        )
        assert record.correlation_token == "tok-dup"  # type: ignore[attr-defined]

    def test_token_present_in_formatted_output(self):
        """CorrelationFormatter renders the token in formatted text without mutating msg."""
        set_correlation_token("tok-fmt")
        record = _make_record("hello world")

        CorrelationFilter().filter(record)
        formatter = CorrelationFormatter(fmt="%(correlation_token)s %(message)s")
        output = formatter.format(record)

        assert "tok-fmt" in output, "Formatted output must contain the correlation token"
        assert "hello world" in output, "Formatted output must contain the original message"
        assert record.msg == "hello world", "record.msg must not be mutated after formatting"

    def test_formatter_output_unchanged_when_no_token(self):
        """CorrelationFormatter output is identical to base Formatter when no token is set."""
        record = _make_record("plain message")

        CorrelationFilter().filter(record)
        fmt_str = "%(message)s"
        result = CorrelationFormatter(fmt=fmt_str).format(record)
        expected = logging.Formatter(fmt=fmt_str).format(record)

        assert result == expected

    def test_formatter_does_not_leave_record_mutated(self):
        """CorrelationFormatter.format() must restore record state even after rendering."""
        set_correlation_token("tok-restore")
        record = _make_record("restore check")

        CorrelationFilter().filter(record)
        formatter = CorrelationFormatter(fmt="%(message)s")
        formatter.format(record)

        # After formatting the record.msg and record.args must be untouched.
        assert record.msg == "restore check"
        assert record.args == ()
