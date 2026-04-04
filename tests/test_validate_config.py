"""Tests for novelforge.config.validate_config() and ConfigurationError."""

import pytest
import novelforge.config as cfg


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
