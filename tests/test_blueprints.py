"""Tests for the declarative blueprint registry in novelforge.routes."""

import importlib

import pytest
from flask import Blueprint

from novelforge.routes import BLUEPRINT_REGISTRY


def _load_blueprints_from_registry() -> list[Blueprint]:
    """Return Blueprint objects in registry order."""
    blueprints = []
    for module_path, attr in BLUEPRINT_REGISTRY:
        module = importlib.import_module(module_path)
        blueprints.append(getattr(module, attr))
    return blueprints


class TestBlueprintRegistry:
    """Validate the declarative blueprint registry."""

    # Names expected to be registered by the application.
    EXPECTED_BLUEPRINT_NAMES = {"outline", "generation", "export", "sessions"}

    def test_registry_contains_all_expected_blueprints(self):
        """Every expected blueprint name must appear in the registry."""
        registered_names = {bp.name for bp in _load_blueprints_from_registry()}
        assert registered_names == self.EXPECTED_BLUEPRINT_NAMES, (
            f"Registry mismatch. Expected {self.EXPECTED_BLUEPRINT_NAMES}, "
            f"got {registered_names}."
        )

    def test_registry_has_no_duplicate_names(self):
        """Blueprint names in the registry must be unique."""
        names = [bp.name for bp in _load_blueprints_from_registry()]
        assert len(names) == len(set(names)), (
            f"Duplicate blueprint names detected: {names}"
        )

    def test_all_registry_entries_are_importable(self):
        """Every (module_path, attr) entry must resolve to a Flask Blueprint."""
        for module_path, attr in BLUEPRINT_REGISTRY:
            module = importlib.import_module(module_path)
            bp = getattr(module, attr, None)
            assert bp is not None, f"{module_path}.{attr} not found"
            assert isinstance(bp, Blueprint), (
                f"{module_path}.{attr} is not a Flask Blueprint"
            )

    def test_register_blueprints_registers_all_in_app(self, app):
        """After app creation, all registry blueprints must be registered."""
        registered_names = {bp.name for bp in app.blueprints.values()}
        assert self.EXPECTED_BLUEPRINT_NAMES.issubset(registered_names), (
            f"Missing blueprints in app: "
            f"{self.EXPECTED_BLUEPRINT_NAMES - registered_names}"
        )

    def test_registration_order_matches_registry(self, app):
        """Blueprints must be registered in the order declared in BLUEPRINT_REGISTRY."""
        registry_names = [bp.name for bp in _load_blueprints_from_registry()]
        app_blueprint_names = list(app.blueprints.keys())
        # All registry names must appear in the app in registry order
        filtered = [n for n in app_blueprint_names if n in registry_names]
        assert filtered == registry_names, (
            f"Registration order mismatch. "
            f"Expected {registry_names}, got {filtered}."
        )
