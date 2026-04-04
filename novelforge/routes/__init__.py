"""Route blueprints for NovelForge."""

import importlib

from flask import Flask

# Declarative registry: each entry is (dotted_module_path, blueprint_attribute).
# Blueprints are registered in the order listed here; add new route modules in
# this tuple rather than inside register_blueprints().  Order matters when URL
# rules could overlap across blueprints.
BLUEPRINT_REGISTRY: tuple[tuple[str, str], ...] = (
    ("novelforge.routes.outline",    "outline_bp"),
    ("novelforge.routes.generation", "generation_bp"),
    ("novelforge.routes.export",     "export_bp"),
    ("novelforge.routes.sessions",   "sessions_bp"),
)


def register_blueprints(app: Flask) -> None:
    """Import and register every blueprint listed in BLUEPRINT_REGISTRY."""
    for module_path, attr in BLUEPRINT_REGISTRY:
        module = importlib.import_module(module_path)
        bp = getattr(module, attr)
        app.register_blueprint(bp)
