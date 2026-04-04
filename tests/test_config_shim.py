"""Regression tests for the top-level config.py backward-compatibility shim.

Verifies that every name declared in ``novelforge.config.__all__`` is
accessible via the top-level ``config`` module so that legacy callers
continue to work without modification.
"""

import importlib
import types

import novelforge.config as _src


class TestConfigShimExplicitExports:
    """All public names from novelforge.config must be importable from config."""

    def _load_shim(self) -> types.ModuleType:
        """Return a freshly imported reference to the top-level config shim."""
        import config as shim  # noqa: PLC0415
        return shim

    def test_shim_exposes_all_public_names(self):
        """Every name in novelforge.config.__all__ is present in config (the shim)."""
        shim = self._load_shim()
        missing = [name for name in _src.__all__ if not hasattr(shim, name)]
        assert missing == [], (
            f"config shim is missing the following names from novelforge.config.__all__: "
            f"{missing}"
        )

    def test_shim_names_are_identical_objects(self):
        """Shim attributes must be the same objects as those in novelforge.config."""
        shim = self._load_shim()
        for name in _src.__all__:
            assert getattr(shim, name) is getattr(_src, name), (
                f"config.{name} is not the same object as novelforge.config.{name}"
            )

    def test_shim_has_no_wildcard_import(self):
        """The top-level config.py must not use a wildcard import statement."""
        import pathlib
        source = (pathlib.Path(__file__).resolve().parent.parent / "config.py").read_text()
        assert "import *" not in source, (
            "config.py still contains a wildcard import ('import *'). "
            "Replace it with explicit re-exports."
        )

    def test_novelforge_config_defines_all(self):
        """novelforge.config must define __all__ so the public surface is explicit."""
        assert hasattr(_src, "__all__"), "novelforge.config does not define __all__"
        assert isinstance(_src.__all__, list)
        assert len(_src.__all__) > 0

    def test_all_contains_core_public_names(self):
        """Spot-check that essential public names are present in __all__."""
        essential = {
            "ConfigurationError",
            "validate_config",
            "ensure_app_dirs",
            "ProviderConfig",
            "LLM_PROVIDERS",
            "SECRET_KEY",
            "PROJECT_ROOT",
        }
        missing = essential - set(_src.__all__)
        assert missing == set(), (
            f"novelforge.config.__all__ is missing essential public names: {missing}"
        )
