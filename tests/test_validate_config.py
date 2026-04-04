"""Tests for novelforge.config.validate_config() and ConfigurationError."""

import pytest
import novelforge.config as cfg


# ---------------------------------------------------------------------------
# get_env_int helper
# ---------------------------------------------------------------------------

class TestGetEnvInt:
    """Unit tests for the get_env_int() helper."""

    def setup_method(self):
        """Clear the parse-error accumulator before every test."""
        cfg._CONFIG_PARSE_ERRORS.clear()

    def teardown_method(self):
        """Restore clean accumulator state after every test."""
        cfg._CONFIG_PARSE_ERRORS.clear()

    def test_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("NF_TEST_INT", raising=False)
        assert cfg.get_env_int("NF_TEST_INT", 42) == 42
        assert cfg._CONFIG_PARSE_ERRORS == []

    def test_returns_parsed_value_when_valid(self, monkeypatch):
        monkeypatch.setenv("NF_TEST_INT", "7")
        assert cfg.get_env_int("NF_TEST_INT", 42) == 7
        assert cfg._CONFIG_PARSE_ERRORS == []

    def test_returns_default_for_non_integer_string(self, monkeypatch):
        monkeypatch.setenv("NF_TEST_INT", "abc")
        result = cfg.get_env_int("NF_TEST_INT", 42)
        assert result == 42
        assert len(cfg._CONFIG_PARSE_ERRORS) == 1
        assert "NF_TEST_INT" in cfg._CONFIG_PARSE_ERRORS[0]
        assert "abc" in cfg._CONFIG_PARSE_ERRORS[0]

    def test_returns_default_for_float_string(self, monkeypatch):
        monkeypatch.setenv("NF_TEST_INT", "3.14")
        result = cfg.get_env_int("NF_TEST_INT", 10)
        assert result == 10
        assert len(cfg._CONFIG_PARSE_ERRORS) == 1
        assert "NF_TEST_INT" in cfg._CONFIG_PARSE_ERRORS[0]

    def test_returns_default_for_empty_string(self, monkeypatch):
        monkeypatch.setenv("NF_TEST_INT", "")
        result = cfg.get_env_int("NF_TEST_INT", 99)
        assert result == 99
        assert len(cfg._CONFIG_PARSE_ERRORS) == 1

    def test_returns_default_below_min_value(self, monkeypatch):
        monkeypatch.setenv("NF_TEST_INT", "0")
        result = cfg.get_env_int("NF_TEST_INT", 5, min_value=1)
        assert result == 5
        assert len(cfg._CONFIG_PARSE_ERRORS) == 1
        assert "NF_TEST_INT" in cfg._CONFIG_PARSE_ERRORS[0]
        assert ">= 1" in cfg._CONFIG_PARSE_ERRORS[0]

    def test_accepts_value_exactly_at_min_value(self, monkeypatch):
        monkeypatch.setenv("NF_TEST_INT", "1")
        result = cfg.get_env_int("NF_TEST_INT", 5, min_value=1)
        assert result == 1
        assert cfg._CONFIG_PARSE_ERRORS == []

    def test_accepts_negative_when_no_min_value(self, monkeypatch):
        monkeypatch.setenv("NF_TEST_INT", "-3")
        result = cfg.get_env_int("NF_TEST_INT", 0)
        assert result == -3
        assert cfg._CONFIG_PARSE_ERRORS == []

    def test_error_message_contains_default_value(self, monkeypatch):
        monkeypatch.setenv("NF_TEST_INT", "bad")
        cfg.get_env_int("NF_TEST_INT", 123)
        assert "123" in cfg._CONFIG_PARSE_ERRORS[0]

    def test_multiple_bad_vars_accumulate_errors(self, monkeypatch):
        monkeypatch.setenv("NF_TEST_A", "nope")
        monkeypatch.setenv("NF_TEST_B", "also_nope")
        cfg.get_env_int("NF_TEST_A", 1)
        cfg.get_env_int("NF_TEST_B", 2)
        assert len(cfg._CONFIG_PARSE_ERRORS) == 2
        assert any("NF_TEST_A" in e for e in cfg._CONFIG_PARSE_ERRORS)
        assert any("NF_TEST_B" in e for e in cfg._CONFIG_PARSE_ERRORS)


# ---------------------------------------------------------------------------
# validate_config surfaces parse errors
# ---------------------------------------------------------------------------

class TestValidateConfigSurfacesParseErrors:
    """Numeric parsing errors collected at import time appear in validate_config()."""

    def setup_method(self):
        cfg._CONFIG_PARSE_ERRORS.clear()

    def teardown_method(self):
        cfg._CONFIG_PARSE_ERRORS.clear()

    def _good_providers(self):
        return [
            cfg.ProviderConfig(
                url="https://api.example.com/v1",
                api_key="sk-valid",
                model="gpt-4",
                label="primary",
            )
        ]

    def test_parse_errors_appear_in_exception(self, monkeypatch):
        monkeypatch.setenv("LLM_TIMEOUT", "bad")
        cfg.get_env_int("LLM_TIMEOUT", 240)
        monkeypatch.setattr(cfg, "LLM_PROVIDERS", self._good_providers())
        monkeypatch.setattr(cfg, "SECRET_KEY", "a-valid-secret-key")

        with pytest.raises(cfg.ConfigurationError) as exc_info:
            cfg.validate_config(debug=False)

        assert any("LLM_TIMEOUT" in e for e in exc_info.value.errors)

    def test_parse_errors_included_with_other_errors(self, monkeypatch):
        monkeypatch.setenv("MAX_CHAPTERS", "x")
        cfg.get_env_int("MAX_CHAPTERS", 100)
        monkeypatch.setattr(cfg, "LLM_PROVIDERS", [
            cfg.ProviderConfig(
                url="https://api.example.com/v1",
                api_key="",
                model="gpt-4",
                label="primary",
            )
        ])
        monkeypatch.setattr(cfg, "SECRET_KEY", "a-valid-secret-key")

        with pytest.raises(cfg.ConfigurationError) as exc_info:
            cfg.validate_config(debug=False)

        errors = exc_info.value.errors
        assert any("MAX_CHAPTERS" in e for e in errors)
        assert any("LLM_API_KEY" in e for e in errors)

    def test_parse_errors_logged_as_warnings_in_debug(self, monkeypatch):
        monkeypatch.setenv("LLM_RETRY_DELAY", "slow")
        cfg.get_env_int("LLM_RETRY_DELAY", 5)
        monkeypatch.setattr(cfg, "LLM_PROVIDERS", self._good_providers())
        monkeypatch.setattr(cfg, "SECRET_KEY", "a-valid-secret-key")

        # Must not raise in debug mode even with parse errors
        cfg.validate_config(debug=True)

    def test_no_errors_when_parse_errors_empty(self, monkeypatch):
        monkeypatch.setattr(cfg, "LLM_PROVIDERS", self._good_providers())
        monkeypatch.setattr(cfg, "SECRET_KEY", "a-valid-secret-key")

        # Must not raise – accumulator is empty
        cfg.validate_config(debug=False)


class TestConfigurationError:
    """Validate the ConfigurationError exception structure."""

    def test_stores_errors_list(self):
        exc = cfg.ConfigurationError(["error one", "error two"])
        assert exc.errors == ["error one", "error two"]

    def test_str_contains_all_messages(self):
        exc = cfg.ConfigurationError(["missing key", "bad url"])
        msg = str(exc)
        assert "missing key" in msg
        assert "bad url" in msg

    def test_empty_errors_list(self):
        exc = cfg.ConfigurationError([])
        assert exc.errors == []

    def test_is_exception_subclass(self):
        assert issubclass(cfg.ConfigurationError, Exception)


class TestValidateConfigProduction:
    """validate_config() raises ConfigurationError in production (debug=False)."""

    def test_raises_when_api_key_missing(self, monkeypatch):
        monkeypatch.setattr(cfg, "LLM_PROVIDERS", [
            cfg.ProviderConfig(
                url="https://api.example.com/v1",
                api_key="",
                model="gpt-4",
                label="primary",
            )
        ])
        monkeypatch.setattr(cfg, "SECRET_KEY", "a-valid-secret-key")

        with pytest.raises(cfg.ConfigurationError) as exc_info:
            cfg.validate_config(debug=False)

        errors = exc_info.value.errors
        assert any("LLM_API_KEY" in e for e in errors)

    def test_raises_when_url_invalid(self, monkeypatch):
        monkeypatch.setattr(cfg, "LLM_PROVIDERS", [
            cfg.ProviderConfig(
                url="not-a-valid-url",
                api_key="sk-valid",
                model="gpt-4",
                label="primary",
            )
        ])
        monkeypatch.setattr(cfg, "SECRET_KEY", "a-valid-secret-key")

        with pytest.raises(cfg.ConfigurationError) as exc_info:
            cfg.validate_config(debug=False)

        errors = exc_info.value.errors
        assert any("LLM_API_URL" in e for e in errors)

    def test_raises_when_secret_key_is_default(self, monkeypatch):
        monkeypatch.setattr(cfg, "LLM_PROVIDERS", [
            cfg.ProviderConfig(
                url="https://api.example.com/v1",
                api_key="sk-valid",
                model="gpt-4",
                label="primary",
            )
        ])
        monkeypatch.setattr(cfg, "SECRET_KEY", "change-me-in-production")

        with pytest.raises(cfg.ConfigurationError) as exc_info:
            cfg.validate_config(debug=False)

        errors = exc_info.value.errors
        assert any("SECRET_KEY" in e for e in errors)

    def test_raises_with_multiple_errors(self, monkeypatch):
        monkeypatch.setattr(cfg, "LLM_PROVIDERS", [
            cfg.ProviderConfig(
                url="bad-url",
                api_key="",
                model="gpt-4",
                label="primary",
            )
        ])
        monkeypatch.setattr(cfg, "SECRET_KEY", "change-me-in-production")

        with pytest.raises(cfg.ConfigurationError) as exc_info:
            cfg.validate_config(debug=False)

        # All three problems should be reported
        errors = exc_info.value.errors
        assert len(errors) >= 3
        assert any("LLM_API_KEY" in e for e in errors)
        assert any("LLM_API_URL" in e for e in errors)
        assert any("SECRET_KEY" in e for e in errors)

    def test_does_not_raise_when_config_valid(self, monkeypatch):
        monkeypatch.setattr(cfg, "LLM_PROVIDERS", [
            cfg.ProviderConfig(
                url="https://api.example.com/v1",
                api_key="sk-valid",
                model="gpt-4",
                label="primary",
            )
        ])
        monkeypatch.setattr(cfg, "SECRET_KEY", "a-very-secure-secret-key")

        # Must not raise
        cfg.validate_config(debug=False)

    def test_error_message_contains_provider_label(self, monkeypatch):
        monkeypatch.setattr(cfg, "LLM_PROVIDERS", [
            cfg.ProviderConfig(
                url="https://api.example.com/v1",
                api_key="",
                model="gpt-4",
                label="primary",
            )
        ])
        monkeypatch.setattr(cfg, "SECRET_KEY", "a-valid-secret-key")

        with pytest.raises(cfg.ConfigurationError) as exc_info:
            cfg.validate_config(debug=False)

        assert any("primary" in e for e in exc_info.value.errors)


class TestValidateConfigDebug:
    """validate_config() does NOT raise in debug/development mode."""

    def test_no_raise_when_api_key_missing_in_debug(self, monkeypatch):
        monkeypatch.setattr(cfg, "LLM_PROVIDERS", [
            cfg.ProviderConfig(
                url="https://api.example.com/v1",
                api_key="",
                model="gpt-4",
                label="primary",
            )
        ])
        monkeypatch.setattr(cfg, "SECRET_KEY", "change-me-in-production")

        # Must not raise even with all-invalid config
        cfg.validate_config(debug=True)

    def test_no_raise_when_secret_key_default_in_debug(self, monkeypatch):
        monkeypatch.setattr(cfg, "LLM_PROVIDERS", [
            cfg.ProviderConfig(
                url="https://api.example.com/v1",
                api_key="sk-valid",
                model="gpt-4",
                label="primary",
            )
        ])
        monkeypatch.setattr(cfg, "SECRET_KEY", "change-me-in-production")

        # Must not raise – in debug mode SECRET_KEY check is a warning only
        cfg.validate_config(debug=True)


class TestValidateConfigFallbackProviders:
    """Errors from fallback LLM providers are also collected."""

    def test_raises_for_fallback_provider_missing_key(self, monkeypatch):
        monkeypatch.setattr(cfg, "LLM_PROVIDERS", [
            cfg.ProviderConfig(
                url="https://api.example.com/v1",
                api_key="sk-valid",
                model="gpt-4",
                label="primary",
            ),
            cfg.ProviderConfig(
                url="https://api2.example.com/v1",
                api_key="",
                model="gpt-4",
                label="provider_2",
            ),
        ])
        monkeypatch.setattr(cfg, "SECRET_KEY", "a-valid-secret-key")

        with pytest.raises(cfg.ConfigurationError) as exc_info:
            cfg.validate_config(debug=False)

        errors = exc_info.value.errors
        assert any("provider_2" in e for e in errors)
        assert any("LLM_API_KEY_2" in e for e in errors)


# ---------------------------------------------------------------------------
# _parse_llm_providers – gap detection / non-contiguous numbering
# ---------------------------------------------------------------------------

class TestParseLlmProviders:
    """Unit tests for _parse_llm_providers() gap detection behaviour."""

    def setup_method(self):
        cfg._CONFIG_PARSE_ERRORS.clear()

    def teardown_method(self):
        cfg._CONFIG_PARSE_ERRORS.clear()

    def test_contiguous_providers_no_gap_warning(self, monkeypatch):
        """Providers numbered 2, 3 without gaps produce no parse errors."""
        monkeypatch.setenv("LLM_API_URL_2", "https://api2.example.com/v1")
        monkeypatch.setenv("LLM_API_URL_3", "https://api3.example.com/v1")

        providers = cfg._parse_llm_providers()

        assert any(p.label == "provider_2" for p in providers)
        assert any(p.label == "provider_3" for p in providers)
        assert cfg._CONFIG_PARSE_ERRORS == []

    def test_gap_in_numbering_emits_parse_error(self, monkeypatch):
        """A missing index between two defined providers is recorded as an error."""
        monkeypatch.delenv("LLM_API_URL_2", raising=False)
        monkeypatch.setenv("LLM_API_URL_3", "https://api3.example.com/v1")

        cfg._parse_llm_providers()

        assert len(cfg._CONFIG_PARSE_ERRORS) == 1
        assert "2" in cfg._CONFIG_PARSE_ERRORS[0]

    def test_provider_after_gap_is_not_dropped(self, monkeypatch):
        """A provider defined after a gap is still included in the returned list."""
        monkeypatch.delenv("LLM_API_URL_2", raising=False)
        monkeypatch.setenv("LLM_API_URL_3", "https://api3.example.com/v1")

        providers = cfg._parse_llm_providers()

        labels = [p.label for p in providers]
        assert "provider_3" in labels
        assert "provider_2" not in labels

    def test_multiple_gaps_all_reported(self, monkeypatch):
        """Multiple gap indices are all mentioned in the single error message."""
        monkeypatch.delenv("LLM_API_URL_2", raising=False)
        monkeypatch.delenv("LLM_API_URL_3", raising=False)
        monkeypatch.setenv("LLM_API_URL_4", "https://api4.example.com/v1")

        cfg._parse_llm_providers()

        assert len(cfg._CONFIG_PARSE_ERRORS) == 1
        error = cfg._CONFIG_PARSE_ERRORS[0]
        assert "2" in error
        assert "3" in error

    def test_all_providers_after_multiple_gaps_included(self, monkeypatch):
        """All providers defined after multiple gaps are still returned."""
        monkeypatch.delenv("LLM_API_URL_2", raising=False)
        monkeypatch.delenv("LLM_API_URL_3", raising=False)
        monkeypatch.setenv("LLM_API_URL_4", "https://api4.example.com/v1")
        monkeypatch.setenv("LLM_API_URL_5", "https://api5.example.com/v1")

        providers = cfg._parse_llm_providers()

        labels = [p.label for p in providers]
        assert "provider_4" in labels
        assert "provider_5" in labels
        assert "provider_2" not in labels
        assert "provider_3" not in labels

    def test_no_fallback_providers_produces_no_errors(self, monkeypatch):
        """When no numbered fallbacks are defined at all, no error is emitted."""
        for idx in range(2, cfg._MAX_PROVIDER_INDEX + 1):
            monkeypatch.delenv(f"LLM_API_URL_{idx}", raising=False)

        cfg._parse_llm_providers()

        assert cfg._CONFIG_PARSE_ERRORS == []

    def test_gap_error_surfaced_by_validate_config(self, monkeypatch):
        """Gaps discovered during provider parsing appear in validate_config errors."""
        monkeypatch.delenv("LLM_API_URL_2", raising=False)
        monkeypatch.setenv("LLM_API_URL_3", "https://api3.example.com/v1")
        monkeypatch.setenv("LLM_API_KEY_3", "sk-valid3")

        # Re-run parser so errors land in _CONFIG_PARSE_ERRORS for this test
        providers = cfg._parse_llm_providers()
        monkeypatch.setattr(cfg, "LLM_PROVIDERS", providers)
        monkeypatch.setattr(cfg, "SECRET_KEY", "a-valid-secret-key")

        with pytest.raises(cfg.ConfigurationError) as exc_info:
            cfg.validate_config(debug=False)

        assert any("gap" in e.lower() or "2" in e for e in exc_info.value.errors)
